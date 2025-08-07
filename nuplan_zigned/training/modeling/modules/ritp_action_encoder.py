import math
from typing import Dict, List, Mapping, Optional

import numpy as np
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


class RITPActionEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_t2action_steps: int,
                 pl2action_radius: float,
                 a2action_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(RITPActionEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_t2m_steps = num_t2action_steps
        self.pl2m_radius = pl2action_radius
        self.a2m_radius = a2action_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=2, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.to_critic = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(self,
                features: FeaturesType,
                scene_enc: Dict[str, torch.Tensor],
                action: List[torch.Tensor]) -> List[torch.Tensor]:
        map_data = features['vector_set_map'].map_data
        agent_data = features['generic_agents'].agent_data
        ego_index = agent_data['av_index']
        critic = []
        batch_size = len(action)

        for sample_idx in range(batch_size):
            ego_idx = ego_index[sample_idx]
            pos_m = agent_data['position'][sample_idx][ego_idx:ego_idx + 1, self.num_historical_steps - 1, :self.input_dim]
            pos_m_all = agent_data['position'][sample_idx][:, self.num_historical_steps - 1, :self.input_dim]
            head_m = agent_data['heading'][sample_idx][ego_idx:ego_idx + 1, self.num_historical_steps - 1]
            head_m_all = agent_data['heading'][sample_idx][:, self.num_historical_steps - 1]
            head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)
            head_vector_m_all = torch.stack([head_m_all.cos(), head_m_all.sin()], dim=-1)

            x_t = scene_enc['x_a'][sample_idx][ego_idx]
            x_pl = scene_enc['x_pl'][sample_idx][:, self.num_historical_steps - 1]
            x_a = scene_enc['x_a'][sample_idx][:, -1]

            mask_src = agent_data['valid_mask'][sample_idx][ego_idx:ego_idx + 1, :self.num_historical_steps].bool().contiguous()
            mask_src[ego_idx:ego_idx + 1, :self.num_historical_steps - self.num_t2m_steps] = False
            mask_src_all = agent_data['valid_mask'][sample_idx][:, :self.num_historical_steps].bool().contiguous()
            mask_src_all[:, :self.num_historical_steps - self.num_t2m_steps] = False
            mask_dst = agent_data['predict_mask'][sample_idx][ego_idx:ego_idx + 1].any(dim=-1, keepdim=True)
            mask_dst_all = agent_data['predict_mask'][sample_idx].any(dim=-1, keepdim=True)

            pos_t = agent_data['position'][sample_idx][ego_idx, :self.num_historical_steps, :self.input_dim]
            head_t = agent_data['heading'][sample_idx][ego_idx, :self.num_historical_steps]
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

            pos_pl = map_data['map_polygon']['position'][sample_idx][:, :self.input_dim]
            orient_pl = map_data['map_polygon']['orientation'][sample_idx]
            edge_index_pl2m = radius(
                x=pos_m[:, :2],
                y=pos_pl[:, :2],
                r=self.pl2m_radius,
                batch_x=None,
                batch_y=None,
                max_num_neighbors=200)
            edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
            rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
            rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
            r_pl2m = torch.stack(
                [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
                 rel_orient_pl2m], dim=-1)
            r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)

            if len(action[sample_idx].shape) == 2:
                edge_index_a2m = radius_graph(
                    x=pos_m[:, :2],
                    r=self.a2m_radius,
                    batch=None,
                    loop=False,
                    max_num_neighbors=200)
                edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
                rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
                rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
                r_a2m = torch.stack(
                    [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
                     rel_head_a2m], dim=-1)
                r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)

                m = self.y_emb(action[sample_idx][:, 0:2])
                m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
                m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
                for i in range(self.num_layers):
                    m = self.t2m_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                    m = m.reshape(-1, 1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                    m = self.pl2m_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                    m = self.a2m_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                    m = m.reshape(1, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                critic.append(self.to_critic(m).squeeze())

            else:  # len(action[sample_idx].shape) == 3
                edge_index_a2m = radius_graph(
                    x=pos_m_all[:, :2],
                    r=self.a2m_radius,
                    batch=None,
                    loop=False,
                    max_num_neighbors=200)
                edge_index_a2m = edge_index_a2m[:, mask_src_all[:, -1][edge_index_a2m[0]] & mask_dst_all[edge_index_a2m[1], 0]]
                edge_index_a2m = edge_index_a2m[:, edge_index_a2m[1] == ego_idx]  # only ego attends others
                rel_pos_a2m = pos_m_all[edge_index_a2m[0]] - pos_m_all[edge_index_a2m[1]]
                rel_head_a2m = wrap_angle(head_m_all[edge_index_a2m[0]] - head_m_all[edge_index_a2m[1]])
                r_a2m = torch.stack(
                    [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=head_vector_m_all[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
                     rel_head_a2m], dim=-1)
                r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)

                num_trajs = action[sample_idx].shape[0]
                split_size = 5000
                split_size_ = num_trajs % split_size
                if num_trajs < split_size:
                    split_size = num_trajs
                r_t2m, r_t2m_ = (
                    r_t2m.unsqueeze(1).expand(-1, split_size, -1).reshape((-1, self.hidden_dim)),  # repeat_interleave,
                    r_t2m.unsqueeze(1).expand(-1, split_size_, -1).reshape((-1, self.hidden_dim))  # repeat_interleave
                )
                edge_index_t2m, edge_index_t2m_ = (
                    torch.stack((
                        edge_index_t2m[0].reshape((-1, 1)).expand(-1, split_size).reshape((-1,)),  # repeat_interleave
                        torch.arange(0, split_size, dtype=torch.int32, device=edge_index_t2m.device)
                        .reshape((1, -1))
                        .expand(self.num_t2m_steps, -1).reshape((-1,))  # repeat
                    )),
                    torch.stack((
                        edge_index_t2m[0].reshape((-1, 1)).expand(-1, split_size_).reshape((-1,)),  # repeat_interleave
                        torch.arange(0, split_size_, dtype=torch.int32, device=edge_index_t2m.device)
                        .reshape((1, -1))
                        .expand(self.num_t2m_steps, -1).reshape((-1,))  # repeat
                    ))
                )
                r_pl2m, r_pl2m_ = (
                    r_pl2m.unsqueeze(0).expand(split_size, -1, -1).reshape((-1, self.hidden_dim)),  # repeat
                    r_pl2m.unsqueeze(0).expand(split_size_, -1, -1).reshape((-1, self.hidden_dim)),  # repeat
                )
                edge_index_pl2m, edge_index_pl2m_ = (
                    torch.stack((
                        edge_index_pl2m[0].reshape((-1, 1)).expand(-1, split_size).reshape((-1,)),  # repeat_interleave
                        torch.arange(0, split_size, dtype=torch.int32, device=edge_index_pl2m.device)
                        .reshape((1, -1))
                        .expand(edge_index_pl2m.shape[1], -1).reshape((-1,))  # repeat
                    )),
                    torch.stack((
                        edge_index_pl2m[0].reshape((-1, 1)).expand(-1, split_size_).reshape((-1,)),  # repeat_interleave
                        torch.arange(0, split_size_, dtype=torch.int32, device=edge_index_pl2m.device)
                        .reshape((1, -1))
                        .expand(edge_index_pl2m.shape[1], -1).reshape((-1,))  # repeat
                    ))
                )
                r_a2m, r_a2m_ = (
                    r_a2m.unsqueeze(0).expand(split_size, -1, -1).reshape((-1, self.hidden_dim)),  # repeat
                    r_a2m.unsqueeze(0).expand(split_size_, -1, -1).reshape((-1, self.hidden_dim)),  # repeat
                )
                edge_index_a2m, edge_index_a2m_ = (
                    torch.stack((
                        edge_index_a2m[0].reshape((1, -1)).expand(split_size, -1).reshape((-1,)),  # repeat
                        torch.arange(0, split_size, dtype=torch.int32, device=edge_index_a2m.device)
                        .reshape((-1, 1))
                        .expand(-1, edge_index_a2m.shape[1]).reshape((-1,))  # repeat_interleave
                    )),
                    torch.stack((
                        edge_index_a2m[0].reshape((1, -1)).expand(split_size_, -1).reshape((-1,)),  # repeat
                        torch.arange(0, split_size_, dtype=torch.int32, device=edge_index_a2m.device)
                        .reshape((-1, 1))
                        .expand(-1, edge_index_a2m.shape[1]).reshape((-1,))  # repeat_interleave
                    )),
                )

                m = [self.y_emb(a) for a in torch.split(action[sample_idx][:, :, 0:2], split_size)]  # avoid out-of-memory issue
                m = [tensor.transpose(0, 1) for tensor in m]
                m = [self.traj_emb(tensor, self.traj_emb_h0.unsqueeze(1).repeat(1, tensor.size(1), 1))[1].squeeze(0) for tensor in m]

                for i in range(self.num_layers):
                    m = [
                        self.t2m_attn_layers[i]((x_t, q), r_t2m, edge_index_t2m)
                        if q.shape[0] == split_size
                        else self.t2m_attn_layers[i]((x_t, q), r_t2m_, edge_index_t2m_)
                        for q in m
                    ]
                    m = [
                        self.pl2m_attn_layers[i]((x_pl, q), r_pl2m, edge_index_pl2m)
                        if q.shape[0] == split_size
                        else self.pl2m_attn_layers[i]((x_pl, q), r_pl2m_, edge_index_pl2m_)
                        for q in m
                    ]
                    m = [
                        self.a2m_attn_layers[i]((x_a, q), r_a2m, edge_index_a2m)
                        if q.shape[0] == split_size
                        else self.a2m_attn_layers[i]((x_a, q), r_a2m_, edge_index_a2m_)
                        for q in m
                    ]
                critic.append(
                    torch.cat([
                        self.to_critic(tensor).squeeze(-1)
                        for tensor in m
                    ], dim=0)
                )

        return critic