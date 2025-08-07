#   Adapted from:
#   https://github.com/ZikangZhou/QCNet (Apache License 2.0)

from typing import Dict, Mapping, Optional, List

import torch
import torch.nn as nn
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import subgraph
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform

from nuplan.planning.training.modeling.types import FeaturesType

from nuplan_zigned.training.modeling.layers.qcmae_attention_layer import AttentionLayer
from nuplan_zigned.training.modeling.layers.fourier_embedding import FourierEmbedding
from nuplan_zigned.utils.utils import angle_between_2d_vectors
from nuplan_zigned.utils.utils import wrap_angle
from nuplan_zigned.utils.weight_init import weight_init


class QCMAEAgentEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_features: List[str],
                 agent_features: List[str],
                 pretrain: bool,
                 prob_pretrain_mask: float,
                 prob_pretrain_mask_mask: float,
                 prob_pretrain_mask_random: float,
                 prob_pretrain_mask_unchanged: float) -> None:
        super(QCMAEAgentEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.map_features = map_features
        self.agent_features = agent_features
        self.pretrain = pretrain
        self.prob_pretrain_mask = prob_pretrain_mask
        self.prob_pretrain_mask_mask = prob_pretrain_mask_mask
        self.prob_pretrain_mask_random = prob_pretrain_mask_random
        self.prob_pretrain_mask_unchanged = prob_pretrain_mask_unchanged

        input_dim_x_a = 4
        input_dim_r_t = 4
        input_dim_r_pl2a = 3
        input_dim_r_a2a = 3

        self.type_a_emb = nn.Embedding(7, hidden_dim)
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        if not self.pretrain:
            self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                               num_freq_bands=num_freq_bands)
            self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                              num_freq_bands=num_freq_bands)
        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        if self.pretrain:
            self.ff_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, input_dim_x_a),
            )
            self.categorical = Categorical(torch.tensor(
                [self.prob_pretrain_mask_mask, self.prob_pretrain_mask_random, self.prob_pretrain_mask_unchanged]))
        else:
            self.pl2a_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
            )
            self.a2a_attn_layers = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                                bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
            )
        self.apply(weight_init)

    def forward(self,
                features: FeaturesType,
                map_enc: Dict[str, List[torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
        map_data = features['vector_set_map'].map_data
        agent_data = features['generic_agents'].agent_data
        if len(agent_data['num_nodes']) > features['generic_agents'].batch_size:
            num_nodes = [sum(agent_data['num_nodes'])]  # special treatment for ritp_actor
            ritp_actor = True
        else:
            num_nodes = agent_data['num_nodes']
            ritp_actor = False
        if self.pretrain:
            agent_enc = {'x_a_predicted': [],
                         'x_a_before_masking': []}
        else:
            agent_enc = {'x_a': []}

        for sample_idx in range(features['generic_agents'].batch_size):
            mask = agent_data['valid_mask'][sample_idx][:, :self.num_historical_steps].bool().contiguous()
            pos_a = agent_data['position'][sample_idx][:, :self.num_historical_steps, :self.input_dim].contiguous()
            motion_vector_a = torch.cat([pos_a.new_zeros(num_nodes[sample_idx], 1, self.input_dim),
                                         pos_a[:, 1:] - pos_a[:, :-1]], dim=1)
            head_a = agent_data['heading'][sample_idx][:, :self.num_historical_steps].contiguous()
            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
            pos_pl = map_data['map_polygon']['position'][sample_idx][:, :self.input_dim].contiguous()
            orient_pl = map_data['map_polygon']['orientation'][sample_idx].contiguous()
            vel = agent_data['velocity'][sample_idx][:, :self.num_historical_steps, :self.input_dim].contiguous()
            length = width = height = None
            categorical_embs = [
                self.type_a_emb(agent_data['type'][sample_idx].long()).repeat_interleave(repeats=self.num_historical_steps,
                                                                                dim=0),
            ]

            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
                 torch.norm(vel[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2])], dim=-1)

            if self.pretrain:
                x_a_before_masking = x_a.clone()
                pretrain_mask = torch.bernoulli(self.prob_pretrain_mask * x_a.new_ones(x_a.size()[0:2])).bool() & mask
                pretrain_mask_ = self.categorical.sample(x_a.size()[0:2]).to(pretrain_mask.device)
                pretrain_mask_mask = pretrain_mask & (pretrain_mask_ == 0)  # [MASK], 2023
                pretrain_mask_mask = pretrain_mask_mask.unsqueeze(-1).repeat(1, 1, x_a.size(-1))
                pretrain_mask_random = pretrain_mask & (pretrain_mask_ == 1)  # random token
                pretrain_mask_random = pretrain_mask_random.unsqueeze(-1).repeat(1, 1, x_a.size(-1))
                uniform = Uniform(x_a[mask].min().clone(), x_a[mask].max().clone())
                x_a = x_a.masked_fill(pretrain_mask_mask, 2023.)
                x_a[pretrain_mask_random] = \
                uniform.sample(x_a.size()[0:2]).unsqueeze(-1).repeat(1, 1, x_a.size(-1)).to(pretrain_mask.device)[
                    pretrain_mask_random]
            x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs)
            x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)

            pos_t = pos_a.reshape(-1, self.input_dim)
            head_t = head_a.reshape(-1)
            head_vector_t = head_vector_a.reshape(-1, 2)
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
            edge_index_t = dense_to_sparse(mask_t)[0]
            edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
            edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]
            rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
            rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])  # |heading| < pi
            r_t = torch.stack(
                [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
                 rel_head_t,
                 edge_index_t[0] - edge_index_t[1]], dim=-1)
            r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

            if self.pretrain:
                for i in range(self.num_layers):
                    x_a = x_a.reshape(-1, self.hidden_dim)
                    x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
                x_a = self.ff_mlp(x_a.reshape(-1, self.num_historical_steps,
                                      self.hidden_dim))
                x_a_predicted = x_a[pretrain_mask]
                x_a_before_masking = x_a_before_masking[pretrain_mask]

                agent_enc['x_a_predicted'].append(x_a_predicted)
                agent_enc['x_a_before_masking'].append(x_a_before_masking)

            else:
                pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
                head_s = head_a.transpose(0, 1).reshape(-1)
                head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
                mask_s = mask.transpose(0, 1).reshape(-1)
                pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
                orient_pl = orient_pl.repeat(self.num_historical_steps)
                agents_ids = agent_data['id'][sample_idx] * self.num_historical_steps
                ego_id_mask = torch.tensor([agent_id == 'AV' for agent_id in agents_ids], dtype=torch.bool, device=mask_s.device)
                ego_mask = mask_s & ego_id_mask
                not_ego_mask = mask_s & (~ego_id_mask)
                polygons_types = map_data['map_polygon']['type'][sample_idx]
                not_route_lane_mask = polygons_types != self.map_features.index('ROUTE_LANES')
                not_route_lane_mask = not_route_lane_mask.repeat(self.num_historical_steps)

                if ritp_actor:
                    # special treatment for ritp_actor
                    batch_s, batch_pl = [], []
                    idx_s, idx_pl = 0, 0
                    for t in range(self.num_historical_steps):
                        for i in range(len(agent_data['num_nodes'])):
                            batch_s.append(
                                torch.full(size=(agent_data['num_nodes'][i],),
                                           fill_value=idx_s,
                                           dtype=torch.long,
                                           device=pos_a.device)
                            )
                            batch_pl.append(
                                torch.full(size=(map_data['map_polygon']['num_nodes'][i],),
                                           fill_value=idx_pl,
                                           dtype=torch.long,
                                           device=pos_a.device)
                            )
                            idx_s += 1
                            idx_pl += 1
                    batch_s = torch.cat(batch_s, dim=0)
                    batch_pl = torch.cat(batch_pl, dim=0)
                else:
                    batch_s = torch.arange(self.num_historical_steps,
                                           device=pos_a.device).repeat_interleave(num_nodes[sample_idx])
                    batch_pl = torch.arange(self.num_historical_steps,
                                            device=pos_pl.device).repeat_interleave(map_data['map_polygon']['num_nodes'][sample_idx])
                edge_index_pl2a = radius(x=pos_s[:, :2],
                                         y=pos_pl[:, :2],
                                         r=self.pl2a_radius,
                                         batch_x=batch_s,
                                         batch_y=batch_pl,
                                         max_num_neighbors=200)  # edge_index_pl2a[0] are indices of pos_pl alone dim[0]，edge_index_pl2a[1] are indices of pos_s alone dim[0]. If the number of actual neighbors is greater than `max_num_neighbors`, returned neighbors are picked randomly.
                edge_index_pl2a = edge_index_pl2a[:, (
                                                             (ego_mask[edge_index_pl2a[1]]) |
                                                             (not_ego_mask[edge_index_pl2a[1]] & not_route_lane_mask[edge_index_pl2a[0]])
                                                     )]  # only ego attends to the route lanes
                rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
                rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
                r_pl2a = torch.stack(
                    [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
                     rel_orient_pl2a], dim=-1)
                r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
                edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                              max_num_neighbors=200)
                edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
                rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
                rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
                r_a2a = torch.stack(
                    [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                     rel_head_a2a], dim=-1)
                r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

                for i in range(self.num_layers):
                    x_a = x_a.reshape(-1, self.hidden_dim)
                    x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
                    x_a = x_a.reshape(-1, self.num_historical_steps,
                                      self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                    x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'][sample_idx].transpose(0, 1).reshape(-1, self.hidden_dim), x_a),
                                                   r_pl2a,
                                                   edge_index_pl2a)
                    x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
                    x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)

                agent_enc['x_a'].append(x_a)
        if self.pretrain:
            return {**agent_enc}
        else:
            return {**map_enc, **agent_enc}
