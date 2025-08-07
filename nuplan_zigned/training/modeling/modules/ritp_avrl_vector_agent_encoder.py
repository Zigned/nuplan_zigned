from typing import Dict, Mapping, Optional, List, Any, Union

import torch
import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex

from nuplan_zigned.training.modeling.layers.avrl_attention_layer import AttentionLayer
from nuplan_zigned.training.modeling.layers.fourier_embedding import FourierEmbedding
from nuplan_zigned.utils.weight_init import weight_init
from nuplan_zigned.utils.utils import (
    wrap_angle,
    angle_between_2d_vectors,
)


class RewardAgentEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 num_poses: int,
                 only_ego_attends_map: bool,
                 gated_attention: bool,
                 gate_has_dropout: bool,
                 ) -> None:
        super(RewardAgentEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_poses = num_poses
        self.only_ego_attends_map = only_ego_attends_map
        self.gated_attention = gated_attention
        self.gate_has_dropout = gate_has_dropout

        if input_dim == 2:
            input_dim_x_a = 2
            input_dim_r_pl2a = 3
            input_dim_r_a2a = 3
        else:
            raise ValueError('{} is not a valid dimension'.format(input_dim))

        self._agent_types = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT']

        self.type_a_emb = nn.Embedding(7, hidden_dim)
        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2a_emb = FourierEmbedding(input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)

        self.pl2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True, gated_attention=gated_attention,
                            gate_has_dropout=gate_has_dropout) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True, gated_attention=gated_attention,
                            gate_has_dropout=gate_has_dropout) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self,
                features: FeaturesType,
                map_enc: Dict[str, List[Union[torch.Tensor, Any]]]) -> Dict[str, List[Union[torch.Tensor, Any]]]:
        agent_data = features['generic_agents'].agent_data
        map_data = features['vector_set_map'].map_data
        batch_size = features['generic_agents'].batch_size
        device = map_data['map_polygon']['position'][0].device

        output = {
            'x_a': [],
        }

        for sample_idx in range(batch_size):
            num_polygons = [detail['num_polygons'] for detail in map_data['num_pl_detail']]
            num_agents = agent_data['num_nodes'][sample_idx]
            num_poses = len(num_agents) // len(num_polygons)
            num_polygons = [[num_pl] * num_poses for num_pl in num_polygons]
            num_polygons = sum(num_polygons, [])

            edge_index_pl2a = []
            edge_index_a2a = []
            total_num_pl = 0
            total_num_a = 0
            for i in range(len(num_polygons)):
                num_pl = num_polygons[i]
                num_a = num_agents[i]
                if not self.only_ego_attends_map:
                    edge_index_pl2a.append(torch.stack(
                        [
                            torch.arange(total_num_pl, total_num_pl + num_pl, dtype=torch.long).repeat((num_a,)),
                            torch.arange(total_num_a, total_num_a + num_a, dtype=torch.long).repeat_interleave(repeats=num_pl, dim=0),
                        ],
                        dim=0,
                    ).to(device))
                else:
                    # only ego attends map
                    edge_index_pl2a.append(torch.stack(
                        [
                            torch.arange(total_num_pl, total_num_pl + num_pl, dtype=torch.long),
                            torch.arange(total_num_a, total_num_a + 1, dtype=torch.long).repeat_interleave(repeats=num_pl, dim=0),
                        ],
                        dim=0
                    ).to(device))

                edge_index_a2a.append(torch.stack(
                    [torch.arange(total_num_a + 1, total_num_a + num_a),
                     torch.tensor(total_num_a).repeat_interleave(torch.tensor(num_a - 1))]
                ).to(device))

                total_num_pl += num_pl
                total_num_a += num_a if not self.only_ego_attends_map else 1

            edge_index_pl2a = torch.cat(edge_index_pl2a, dim=1)
            edge_index_a2a = torch.cat(edge_index_a2a, dim=1)

            pos_a = agent_data['position'][sample_idx].contiguous()  # float64
            head_a = agent_data['heading'][sample_idx].contiguous().float()
            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
            pos_pl = map_data['map_polygon']['position'][sample_idx][:, :self.input_dim].contiguous()
            orient_pl = map_data['map_polygon']['orientation'][sample_idx].contiguous().float()
            vel = agent_data['velocity'][sample_idx].contiguous().float()
            agents_type = agent_data['type'][sample_idx].contiguous()
            categorical_embs = [self.type_a_emb(agents_type.long())]

            x_a = torch.stack(
                [torch.norm(vel, p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel)],
                dim=-1
            )
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)

            rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_a[edge_index_pl2a[1]]
            rel_pos_pl2a = rel_pos_pl2a.float()
            rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_a[edge_index_pl2a[1]])
            r_pl2a = torch.stack(
                [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a),
                 rel_orient_pl2a], dim=-1)
            r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

            rel_pos_a2a = pos_a[edge_index_a2a[0]] - pos_a[edge_index_a2a[1]]
            rel_pos_a2a = rel_pos_a2a.float()
            rel_head_a2a = wrap_angle(head_a[edge_index_a2a[0]] - head_a[edge_index_a2a[1]])
            r_a2a = torch.stack(
                [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                 rel_head_a2a], dim=-1)
            r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

            for i in range(self.num_layers):
                # if i_pose == 0:
                #     store_dropout_mask = True
                #     use_frozen_dropout_mask = False
                # else:
                #     store_dropout_mask = False
                #     use_frozen_dropout_mask = True
                # self.pl2a_attn_layers[i].bools['store_dropout_mask'] = store_dropout_mask
                # self.pl2a_attn_layers[i].bools['use_frozen_dropout_mask'] = use_frozen_dropout_mask
                # self.a2a_attn_layers[i].bools['store_dropout_mask'] = store_dropout_mask
                # self.a2a_attn_layers[i].bools['use_frozen_dropout_mask'] = use_frozen_dropout_mask

                x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'][sample_idx], x_a),
                                               r_pl2a,
                                               edge_index_pl2a)
                x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)

            output['x_a'].append(x_a[agent_data['av_index']])

        return output
