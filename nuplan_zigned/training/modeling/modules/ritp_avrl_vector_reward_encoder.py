from dataclasses import dataclass
from typing import Dict, List, Tuple, cast, Union, Any

import torch
import torch.nn as nn

from nuplan_zigned.utils.weight_init import weight_init
from nuplan_zigned.training.modeling.layers.avrl_dropout import MyDropout


class RewardEncoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float,
                 num_poses: int,
                 ) -> None:
        super(RewardEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_poses = num_poses

        self.fc = nn.Sequential(
            MyDropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim * 4),
            # MyDropout(p=dropout),
            nn.ReLU(inplace=True),
            MyDropout(p=dropout),
            nn.Linear(hidden_dim * 4, 1),
        )

        self.apply(weight_init)

    def forward(self, scene_enc: Dict[str, List[Union[torch.Tensor, Any]]]) -> Dict[str, torch.Tensor]:
        x_a = scene_enc['x_a'][0]

        reward_enc = self.fc(x_a)

        return {'reward_enc': reward_enc}


