#   Adapted from:
#   https://github.com/ZikangZhou/QCNet (Apache License 2.0)

from typing import Dict, List

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform

from nuplan.planning.training.modeling.types import FeaturesType

from nuplan_zigned.training.modeling.layers.qcmae_attention_layer import AttentionLayer
from nuplan_zigned.training.modeling.layers.fourier_embedding import FourierEmbedding
from nuplan_zigned.utils.utils import angle_between_2d_vectors
from nuplan_zigned.utils.utils import wrap_angle
from nuplan_zigned.utils.graph import merge_edges
from nuplan_zigned.utils.weight_init import weight_init


class QCMAEMapEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 pretrain: bool,
                 prob_pretrain_mask: float,
                 prob_pretrain_mask_mask: float,
                 prob_pretrain_mask_random: float,
                 prob_pretrain_mask_unchanged: float) -> None:
        super(QCMAEMapEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.pretrain = pretrain
        self.prob_pretrain_mask = prob_pretrain_mask
        self.prob_pretrain_mask_mask = prob_pretrain_mask_mask
        self.prob_pretrain_mask_random = prob_pretrain_mask_random
        self.prob_pretrain_mask_unchanged = prob_pretrain_mask_unchanged

        if input_dim == 2:
            input_dim_x_pt = 1
            input_dim_x_pl = 0
            input_dim_r_pt2pl = 3
            input_dim_r_pl2pl = 3
        else:
            raise ValueError('{} is not a valid dimension'.format(input_dim))

        self.type_pt_emb = nn.Embedding(17, hidden_dim)
        self.side_pt_emb = nn.Embedding(4, hidden_dim)
        self.type_pl_emb = nn.Embedding(6, hidden_dim)
        self.int_pl_emb = nn.Embedding(3, hidden_dim)
        self.tl_pl_emb = nn.Embedding(4, hidden_dim)

        self.type_pl2pl_emb = nn.Embedding(5, hidden_dim)
        self.x_pt_emb = FourierEmbedding(input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.x_pl_emb = FourierEmbedding(input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2pl_emb = FourierEmbedding(input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.r_pl2pl_emb = FourierEmbedding(input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2pl_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        if self.pretrain:
            self.ff_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, 2),
            )
            self.categorical = Categorical(torch.tensor(
                [self.prob_pretrain_mask_mask, self.prob_pretrain_mask_random, self.prob_pretrain_mask_unchanged]))
        self.apply(weight_init)

    def forward(self, features: FeaturesType) -> Dict[str, List[torch.Tensor]]:
        data = features['vector_set_map'].map_data
        if self.pretrain:
            map_enc = {'x_pl_predicted': [],
                       'x_pl_before_masking': []}
        else:
            map_enc = {'x_pt': [],
                       'x_pl': []}

        for sample_idx in range(features['vector_set_map'].batch_size):
            edge_index_pt2pl = data['map_point', 'to', 'map_polygon']['edge_index'][sample_idx].long()
            # num_pt = data['map_point']['position'][sample_idx].shape[0]
            # num_pl = data['map_polygon']['position'][sample_idx].shape[0]
            # pt_mask = torch.zeros((num_pt,), device=edge_index_pt2pl.device, dtype=torch.bool)
            # pl_mask = torch.zeros((num_pl,), device=edge_index_pt2pl.device, dtype=torch.bool)
            # pt_mask[edge_index_pt2pl[0]] = True
            # pl_mask[edge_index_pt2pl[1]] = True

            pos_pt = data['map_point']['position'][sample_idx][:, :self.input_dim].contiguous()
            orient_pt = data['map_point']['orientation'][sample_idx].contiguous()
            pos_pl = data['map_polygon']['position'][sample_idx][:, :self.input_dim].contiguous()
            orient_pl = data['map_polygon']['orientation'][sample_idx].contiguous()
            orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)

            if self.input_dim == 2:
                x_pt = data['map_point']['magnitude'][sample_idx].unsqueeze(-1)
                x_pl = None
            elif self.input_dim == 3:
                x_pt = torch.stack([data['map_point']['magnitude'][sample_idx], data['map_point']['height'][sample_idx]], dim=-1)
                x_pl = data['map_polygon']['height'][sample_idx].unsqueeze(-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            x_pt_categorical_embs = [self.type_pt_emb(data['map_point']['type'][sample_idx].long()),
                                     self.side_pt_emb(data['map_point']['side'][sample_idx].long())]
            x_pl_categorical_embs = [self.type_pl_emb(data['map_polygon']['type'][sample_idx].long()),
                                     self.int_pl_emb(data['map_polygon']['is_intersection'][sample_idx].long()),
                                     self.tl_pl_emb(data['map_polygon']['tl_statuses'][sample_idx].long())]

            if self.pretrain:
                x_pl_before_masking = data['map_polygon']['position'][sample_idx][:, 0:self.input_dim]
                pretrain_mask = torch.bernoulli(self.prob_pretrain_mask * x_pl_before_masking.new_ones(x_pl_before_masking.size(0))).bool()
                pretrain_mask_ = self.categorical.sample(x_pl_before_masking.size()[0:1]).to(pretrain_mask.device)
                pretrain_mask_mask = pretrain_mask & (pretrain_mask_ == 0)  # [MASK], 2023
                pretrain_mask_mask = pretrain_mask_mask
                pretrain_mask_random = pretrain_mask & (pretrain_mask_ == 1)  # random token
                pretrain_mask_random = pretrain_mask_random
                uniform = Uniform(x_pt.min().clone(), x_pt.max().clone())
                num_points = []
                for item in torch.unique(edge_index_pt2pl[1]):
                    num_points.append((edge_index_pt2pl[1] == item).sum())
                num_points = torch.tensor(num_points, device=edge_index_pt2pl.device)
                x_pt = x_pt.masked_fill(pretrain_mask_mask.repeat_interleave(num_points).unsqueeze(-1), 2023.)
                x_pt[pretrain_mask_random.repeat_interleave(num_points)] = \
                    uniform.sample(x_pt.size())[pretrain_mask_random.repeat_interleave(num_points)].to(pretrain_mask.device)

            x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=x_pt_categorical_embs)
            x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs)

            rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - pos_pl[edge_index_pt2pl[1]]
            rel_orient_pt2pl = wrap_angle(orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]])
            if self.input_dim == 2:
                r_pt2pl = torch.stack(
                    [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                              nbr_vector=rel_pos_pt2pl[:, :2]),
                     rel_orient_pt2pl], dim=-1)
            elif self.input_dim == 3:
                r_pt2pl = torch.stack(
                    [torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                                              nbr_vector=rel_pos_pt2pl[:, :2]),
                     rel_pos_pt2pl[:, -1],
                     rel_orient_pt2pl], dim=-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None)

            edge_index_pl2pl = data['map_polygon', 'to', 'map_polygon']['edge_index'][sample_idx].long()
            edge_index_pl2pl_radius = radius_graph(x=pos_pl[:, :2], r=self.pl2pl_radius,
                                                   batch=None,
                                                   loop=False, max_num_neighbors=200)
            type_pl2pl = data['map_polygon', 'to', 'map_polygon']['type'][sample_idx].long()
            type_pl2pl_radius = type_pl2pl.new_zeros(edge_index_pl2pl_radius.size(1), dtype=torch.uint8)
            edge_index_pl2pl, type_pl2pl = merge_edges(edge_indices=[edge_index_pl2pl_radius, edge_index_pl2pl],
                                                       edge_attrs=[type_pl2pl_radius, type_pl2pl], reduce='max')
            rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
            rel_orient_pl2pl = wrap_angle(orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]])
            if self.input_dim == 2:
                r_pl2pl = torch.stack(
                    [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                              nbr_vector=rel_pos_pl2pl[:, :2]),
                     rel_orient_pl2pl], dim=-1)
            elif self.input_dim == 3:
                r_pl2pl = torch.stack(
                    [torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                     angle_between_2d_vectors(ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                                              nbr_vector=rel_pos_pl2pl[:, :2]),
                     rel_pos_pl2pl[:, -1],
                     rel_orient_pl2pl], dim=-1)
            else:
                raise ValueError('{} is not a valid dimension'.format(self.input_dim))
            r_pl2pl = self.r_pl2pl_emb(continuous_inputs=r_pl2pl, categorical_embs=[self.type_pl2pl_emb(type_pl2pl.long())])

            for i in range(self.num_layers):
                x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_index_pt2pl)
                x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)

            if self.pretrain:
                x_pl = self.ff_mlp(x_pl.reshape(-1, self.hidden_dim))
                x_pl_predicted = x_pl[pretrain_mask]
                x_pl_before_masking = x_pl_before_masking[pretrain_mask]

                map_enc['x_pl_predicted'].append(x_pl_predicted)
                map_enc['x_pl_before_masking'].append(x_pl_before_masking)

            else:
                x_pl = x_pl.repeat_interleave(repeats=self.num_historical_steps,
                                              dim=0).reshape(-1, self.num_historical_steps, self.hidden_dim)

                map_enc['x_pt'].append(x_pt)
                map_enc['x_pl'].append(x_pl)

        return map_enc