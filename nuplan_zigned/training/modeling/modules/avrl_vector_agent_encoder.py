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
        generic_agents = features['generic_agents']
        map_data = features['vector_set_map'].map_data
        num_pl_detail = map_data['num_pl_detail']
        batch_size = generic_agents.batch_size
        num_trajs = len(generic_agents.agents['VEHICLE'][0])
        num_poses = self.num_poses

        output = {
            'x_a': [],
        }
        num_agents_detail = []
        num_pl2a_edge_index_detail = []

        for sample_idx in range(batch_size):
            agents_data = []
            agents_type = []
            ego_idx = []
            num_agents_detail.append({})
            num_pl2a_edge_index_detail.append({})
            edge_index_pl2a = []
            idx_offset_pl = 0
            idx_offset_agent = 0
            for i_traj in range(num_trajs):
                if i_traj not in num_agents_detail[sample_idx].keys():
                    num_agents_detail[sample_idx][i_traj] = {}
                if i_traj not in num_pl2a_edge_index_detail[sample_idx].keys():
                    num_pl2a_edge_index_detail[sample_idx][i_traj] = {}
                for i_pose in range(num_poses):
                    if i_pose not in num_agents_detail[sample_idx][i_traj].keys():
                        num_agents_detail[sample_idx][i_traj][i_pose] = {}
                    if i_pose not in num_pl2a_edge_index_detail[sample_idx][i_traj].keys():
                        num_pl2a_edge_index_detail[sample_idx][i_traj][i_pose] = {}
                    ego_idx.append(sum([data.shape[0] for data in agents_data]))
                    agents_data.append(generic_agents.ego[sample_idx][i_traj][i_pose])
                    agents_type.append(torch.tensor([self._agent_types.index('VEHICLE'), ], dtype=torch.uint8))
                    num_a = 1
                    num_agents_detail[sample_idx][i_traj][i_pose]['EGO'] = 1
                    for agent_type in self._agent_types:
                        agent_data = generic_agents.agents[agent_type][sample_idx][i_traj][i_pose]
                        if agent_data.shape[0] > 0:
                            agents_data.append(agent_data)
                            agents_type.append(torch.full((agent_data.shape[0],), self._agent_types.index(agent_type), dtype=torch.uint8))
                            num_a += agent_data.shape[0]
                        num_agents_detail[sample_idx][i_traj][i_pose][agent_type] = agent_data.shape[0]
                    num_pl = num_pl_detail[sample_idx][i_traj][i_pose]['num_polygons']
                    if not self.only_ego_attends_map:
                        edge_index_pl2a.append(
                            torch.stack(
                                [
                                    torch.arange(idx_offset_pl, idx_offset_pl + num_pl, dtype=torch.long).repeat((num_a, )),
                                    torch.arange(idx_offset_agent, idx_offset_agent + num_a, dtype=torch.long).repeat_interleave(repeats=num_pl, dim=0),
                                ],
                                dim=0
                            )
                        )
                    else:
                        # only ego attends map
                        edge_index_pl2a.append(
                            torch.stack(
                                [
                                    torch.arange(idx_offset_pl, idx_offset_pl + num_pl, dtype=torch.long),
                                    torch.arange(idx_offset_agent, idx_offset_agent + 1,
                                                 dtype=torch.long).repeat_interleave(repeats=num_pl, dim=0),
                                ],
                                dim=0
                            )
                        )
                    idx_offset_pl += num_pl
                    idx_offset_agent += num_a

            agents_data = torch.cat(agents_data, dim=0)
            agents_type = torch.cat(agents_type, dim=0).to(agents_data.device)
            edge_index_pl2a = torch.cat(edge_index_pl2a, dim=1).to(agents_data.device)

            pos_a = agents_data[:, [AgentFeatureIndex.x(), AgentFeatureIndex.y()]].contiguous()
            head_a = agents_data[:, AgentFeatureIndex.heading()].contiguous()
            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
            pos_pl = map_data['map_polygon'][sample_idx]['position'][:, :self.input_dim].contiguous()
            orient_pl = map_data['map_polygon'][sample_idx]['orientation'].contiguous()
            vel = agents_data[:, [AgentFeatureIndex.vx(), AgentFeatureIndex.vy()]].contiguous()
            categorical_embs = [self.type_a_emb(agents_type.long())]

            x_a = torch.stack(
                [torch.norm(vel, p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel)],
                dim=-1
            )
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)

            rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_a[edge_index_pl2a[1]]
            rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_a[edge_index_pl2a[1]])
            r_pl2a = torch.stack(
                [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a),
                 rel_orient_pl2a], dim=-1)
            r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)

            agents_idx = [torch.arange(ego_idx[i] + 1, ego_idx[i + 1]) for i in range(len(ego_idx) - 1)]
            agents_idx.append(torch.arange(ego_idx[-1] + 1, agents_data.shape[0]))
            agents_idx_dim = [index.shape[0] for index in agents_idx]
            edge_index_a2a = torch.stack(
                [torch.cat(agents_idx),
                 torch.tensor(ego_idx).repeat_interleave(torch.tensor(agents_idx_dim))]
            ).to(agents_data.device)
            rel_pos_a2a = pos_a[edge_index_a2a[0]] - pos_a[edge_index_a2a[1]]
            rel_head_a2a = wrap_angle(head_a[edge_index_a2a[0]] - head_a[edge_index_a2a[1]])
            r_a2a = torch.stack(
                [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                 rel_head_a2a], dim=-1)
            r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)

            for i in range(self.num_layers):
                x_a = self.pl2a_attn_layers[i]((map_enc['x_pl'][sample_idx], x_a), r_pl2a,
                                               edge_index_pl2a)
                x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)

            output['x_a'].append(x_a[ego_idx])

        return output
