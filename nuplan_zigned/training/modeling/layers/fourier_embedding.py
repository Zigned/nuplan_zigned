#   Heavily borrowed from:
#   https://github.com/ZikangZhou/QCNet (Apache License 2.0)

import math
from typing import List, Optional

import torch
import torch.nn as nn

from nuplan_zigned.utils.weight_init import weight_init


class FourierEmbedding(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
                for _ in range(input_dim)])
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(self,
                continuous_inputs: Optional[torch.Tensor] = None,
                categorical_embs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError('Both continuous_inputs and categorical_embs are None')
            return self.to_out(x)
        else:
            if len(continuous_inputs.shape) == 3:
                num_trajs = continuous_inputs.shape[0]
                num_poses = continuous_inputs.shape[1]
                continuous_inputs = continuous_inputs.reshape((-1, self.input_dim))
            else:
                num_trajs, num_poses = None, None
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
            # Warning: if your data are noisy, don't use learnable sinusoidal embedding
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
            continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
            if num_trajs is not None:
                return self.to_out(x).reshape((num_trajs, num_poses, -1))
            else:
                return self.to_out(x)


class OriFourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super(OriFourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
        x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
        continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
        for i in range(self.input_dim):
            continuous_embs[i] = self.mlps[i](x[..., i, :])
        x = torch.stack(continuous_embs).sum(dim=0)
        return self.to_out(x)