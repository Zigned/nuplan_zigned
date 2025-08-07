from typing import List, Dict, Optional, Union, Any

import torch
import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType

from nuplan_zigned.training.modeling.layers.avrl_attention_layer import AttentionLayer
from nuplan_zigned.training.modeling.layers.fourier_embedding import FourierEmbedding
from nuplan_zigned.utils.weight_init import weight_init
from nuplan_zigned.utils.utils import (
    wrap_angle,
    angle_between_2d_vectors,
)


class RewardMapEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 num_poses: int,
                 gated_attention: bool,
                 gate_has_dropout: bool,
                 ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.num_poses = num_poses
        self.gated_attention = gated_attention
        self.gate_has_dropout = gate_has_dropout

        if input_dim == 2:
            input_dim_x_pt = 1
            input_dim_x_pl = 0
            input_dim_r_pt2pl = 3
            input_dim_r_pl2pl = 3
        else:
            raise ValueError('{} is not a valid dimension'.format(input_dim))

        self.type_pt_emb = nn.Embedding(6, hidden_dim)
        self.side_pt_emb = nn.Embedding(4, hidden_dim)
        self.tl_pt_emb = nn.Embedding(4, hidden_dim)
        self.type_pl_emb = nn.Embedding(6, hidden_dim)
        self.int_pl_emb = nn.Embedding(3, hidden_dim)
        self.type_pl2pl_emb = nn.Embedding(5, hidden_dim)
        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2pl_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True, gated_attention=gated_attention,
                            gate_has_dropout=gate_has_dropout) for _ in range(num_layers)]
        )
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True, gated_attention=gated_attention,
                            gate_has_dropout=gate_has_dropout) for _ in range(num_layers)]
        )
        self.apply(weight_init)

    def forward(self, features: FeaturesType) -> Dict[str, List[Union[torch.Tensor, Any]]]:
        output = {
            'x_pt': [],
            'x_pl': [],
        }
        for sample_idx in range(features['vector_set_map'].batch_size):
            map_data = features['vector_set_map'].map_data
            pos_pt = map_data['map_point'][sample_idx]['position'][:, :self.input_dim].contiguous()
            orient_pt = map_data['map_point'][sample_idx]['orientation'].contiguous()
            pos_pl = map_data['map_polygon'][sample_idx]['position'][:, :self.input_dim].contiguous()
            orient_pl = map_data['map_polygon'][sample_idx]['orientation'].contiguous()
            orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)

            x_pt = map_data['map_point'][sample_idx]['magnitude'].unsqueeze(-1)
            x_pl = None
            x_pt_categorical_embs = [self.type_pt_emb(map_data['map_point'][sample_idx]['type'].long()),
                                     self.side_pt_emb(map_data['map_point'][sample_idx]['side'].long()),
                                     self.tl_pt_emb(map_data['map_point'][sample_idx]['tl_statuses'].long())]
            x_pl_categorical_embs = [self.type_pl_emb(map_data['map_polygon'][sample_idx]['type'].long()),
                                     self.int_pl_emb(map_data['map_polygon'][sample_idx]['is_intersection'].long())]

            edge_index_pt2pl = map_data['map_point', 'to', 'map_polygon'][sample_idx]['edge_index']

            x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=x_pt_categorical_embs)
            x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs)

            rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - pos_pl[edge_index_pt2pl[1]]
            rel_orient_pt2pl = wrap_angle(orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]])
            r_pt2pl = torch.stack(
                [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                          nbr_vector=rel_pos_pt2pl[:, :2]),
                 rel_orient_pt2pl], dim=-1)
            r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None)

            edge_index_pl2pl = map_data['map_polygon', 'to', 'map_polygon'][sample_idx]['edge_index']
            type_pl2pl = map_data['map_polygon', 'to', 'map_polygon'][sample_idx]['type']
            rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
            rel_orient_pl2pl = wrap_angle(orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]])
            r_pl2pl = torch.stack(
                [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                          nbr_vector=rel_pos_pl2pl[:, :2]),
                 rel_orient_pl2pl], dim=-1)
            r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=[self.type_pl2pl_emb(type_pl2pl.long())])

            for i in range(self.num_layers):
                x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_index_pt2pl)
                x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)

            output['x_pt'].append(x_pt)
            output['x_pl'].append(x_pl)

        return output