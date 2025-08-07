#   Adapted from:
#   https://github.com/ZikangZhou/QCNet (Apache License 2.0)

import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.utils import dense_to_sparse

from nuplan.planning.training.modeling.types import FeaturesType

from nuplan_zigned.training.modeling.layers.qcmae_attention_layer import AttentionLayer
from nuplan_zigned.training.modeling.layers.fourier_embedding import FourierEmbedding
from nuplan_zigned.training.modeling.layers.mlp_layer import MLPLayer
from nuplan_zigned.utils.utils import angle_between_2d_vectors
from nuplan_zigned.utils.utils import wrap_angle
from nuplan_zigned.utils.graph import bipartite_dense_to_sparse
from nuplan_zigned.utils.weight_init import weight_init


class QCMAEDecoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 finetune_range: str=None) -> None:
        super(QCMAEDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.finetune_range = finetune_range

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(self,
                features: FeaturesType,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        map_data = features['vector_set_map'].map_data
        agent_data = features['generic_agents'].agent_data
        if len(agent_data['num_nodes']) > features['generic_agents'].batch_size:
            ritp_actor = True
        else:
            ritp_actor = False
        pred = {
            'loc_propose_pos': [],
            'scale_propose_pos': [],
            'loc_propose_head': [],
            'conc_propose_head': [],
            'loc_refine_pos': [],
            'scale_refine_pos': [],
            'loc_refine_head': [],
            'conc_refine_head': [],
            'pi': [],
        }

        for sample_idx in range(features['generic_agents'].batch_size):
            pos_m = agent_data['position'][sample_idx][:, self.num_historical_steps - 1, :self.input_dim]
            head_m = agent_data['heading'][sample_idx][:, self.num_historical_steps - 1]
            head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

            x_t = scene_enc['x_a'][sample_idx].reshape(-1, self.hidden_dim)
            x_pl = scene_enc['x_pl'][sample_idx][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
            x_a = scene_enc['x_a'][sample_idx][:, -1].repeat(self.num_modes, 1)
            m = self.mode_emb.weight.repeat(scene_enc['x_a'][sample_idx].size(0), 1)

            mask_src = agent_data['valid_mask'][sample_idx][:, :self.num_historical_steps].bool().contiguous()
            mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
            mask_dst = agent_data['predict_mask'][sample_idx].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

            pos_t = agent_data['position'][sample_idx][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
            head_t = agent_data['heading'][sample_idx][:, :self.num_historical_steps].reshape(-1)  # 202310181530 debugged
            edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
            rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
            rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
            r_t2m = torch.stack(
                [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
                 rel_head_t2m,
                 (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
            r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
            edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
            r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

            pos_pl = map_data['map_polygon']['position'][sample_idx][:, :self.input_dim]
            orient_pl = map_data['map_polygon']['orientation'][sample_idx]
            if ritp_actor:
                batch_m = torch.arange(len(agent_data['num_nodes']), device=pos_m.device)
                batch_m = batch_m.repeat_interleave(torch.tensor(agent_data['num_nodes'], device=pos_m.device))
                batch_pl = torch.arange(len(agent_data['num_nodes']), device=pos_pl.device)
                batch_pl = batch_pl.repeat_interleave(torch.tensor(map_data['map_polygon']['num_nodes'], device=pos_pl.device))
            else:
                batch_m = None
                batch_pl = None
            edge_index_pl2m = radius(
                x=pos_m[:, :2],
                y=pos_pl[:, :2],
                r=self.pl2m_radius,
                batch_x=batch_m,
                batch_y=batch_pl,
                max_num_neighbors=200)  # max_num_neighbors actually limits the number of x
            edge_index_pl2m = edge_index_pl2m[:, edge_index_pl2m[0] < 200]  # limit the number of polygons
            edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
            rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
            rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
            r_pl2m = torch.stack(
                [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
                 rel_orient_pl2m], dim=-1)
            r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
            if ritp_actor:
                edge_index_pl2m = torch.cat(
                    [edge_index_pl2m + i * edge_index_pl2m.new_tensor(
                        [[sum(map_data['map_polygon']['num_nodes'])],
                         [sum(agent_data['num_nodes'])]]
                    ) for i in range(self.num_modes)],
                    dim=1)
            else:
                edge_index_pl2m = torch.cat(
                    [edge_index_pl2m + i * edge_index_pl2m.new_tensor(
                        [[map_data['map_polygon']['num_nodes'][sample_idx]],
                         [agent_data['num_nodes'][sample_idx]]]
                    ) for i in range(self.num_modes)],
                    dim=1)
            r_pl2m = r_pl2m.repeat(self.num_modes, 1)

            edge_index_a2m = radius_graph(
                x=pos_m[:, :2],
                r=self.a2m_radius,
                batch=batch_m,
                loop=False,
                max_num_neighbors=200)
            edge_index_a2m = edge_index_a2m[:, edge_index_a2m[0] < 100]  # limit the number of agents
            edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
            rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
            rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
            r_a2m = torch.stack(
                [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
                 rel_head_a2m], dim=-1)
            r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
            if ritp_actor:
                edge_index_a2m = torch.cat(
                    [edge_index_a2m + i * edge_index_a2m.new_tensor([sum(agent_data['num_nodes'])])
                     for i in range(self.num_modes)], dim=1)
            else:
                edge_index_a2m = torch.cat(
                    [edge_index_a2m + i * edge_index_a2m.new_tensor([agent_data['num_nodes'][sample_idx]])
                     for i in range(self.num_modes)], dim=1)
            r_a2m = r_a2m.repeat(self.num_modes, 1)

            edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

            locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
            scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
            locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
            concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
            for t in range(self.num_recurrent_steps):
                for i in range(self.num_layers):
                    m = m.reshape(-1, self.hidden_dim)
                    m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                    m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                    m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                    m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                    m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim)
                locs_propose_pos[t] = self.to_loc_propose_pos(m)
                scales_propose_pos[t] = self.to_scale_propose_pos(m)
                if self.output_head:
                    locs_propose_head[t] = self.to_loc_propose_head(m)
                    concs_propose_head[t] = self.to_conc_propose_head(m)
            loc_propose_pos = torch.cumsum(
                torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                dim=-2)
            scale_propose_pos = torch.cumsum(
                F.elu_(
                    torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                    alpha=1.0) +
                1.0,
                dim=-2) + 0.1

            if self.finetune_range is not None:
                if self.finetune_range == 'full':
                    pass
                elif self.finetune_range == 'decoder':
                    x_t = x_t.detach()
                    x_pl = x_pl.detach()
                    x_a = x_a.detach()
                    r_t2m = r_t2m.detach()
                    r_pl2m = r_pl2m.detach()
                    r_a2m = r_a2m.detach()
                elif self.finetune_range == 'refine_module':
                    x_t = x_t.detach()
                    x_pl = x_pl.detach()
                    x_a = x_a.detach()
                    r_t2m = r_t2m.detach()
                    r_pl2m = r_pl2m.detach()
                    r_a2m = r_a2m.detach()
                elif self.finetune_range == 'mlp':
                    x_t = x_t.detach()
                    x_pl = x_pl.detach()
                    x_a = x_a.detach()
                    r_t2m = r_t2m.detach()
                    r_pl2m = r_pl2m.detach()
                    r_a2m = r_a2m.detach()

            if self.output_head:
                loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                                dim=-2)
                conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                        dim=-2) + 0.02)
                m = self.y_emb(torch.cat([loc_propose_pos.detach(),
                                          wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
            else:
                loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
                                                              self.num_future_steps, 1))
                conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                                 self.num_future_steps, 1))
                m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
            m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
            m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
            for i in range(self.num_layers):
                m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)

            if self.finetune_range is not None:
                if self.finetune_range == 'mlp':
                    m = m.detach()

            loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
            loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()

            scale_refine_pos = F.elu_(
                self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) + 1.0 + 0.1
            if self.output_head:
                loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
                loc_refine_head = loc_refine_head + loc_propose_head.detach()
                conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
            else:
                loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                            1))
                conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                               self.num_future_steps, 1))
            pi = self.to_pi(m).squeeze(-1)

            pred['loc_propose_pos'].append(loc_propose_pos)
            pred['scale_propose_pos'].append(scale_propose_pos)
            pred['loc_propose_head'].append(loc_propose_head)
            pred['conc_propose_head'].append(conc_propose_head)
            pred['loc_refine_pos'].append(loc_refine_pos)
            pred['scale_refine_pos'].append(scale_refine_pos)
            pred['loc_refine_head'].append(loc_refine_head)
            pred['conc_refine_head'].append(conc_refine_head)
            pred['pi'].append(pi)

        return pred