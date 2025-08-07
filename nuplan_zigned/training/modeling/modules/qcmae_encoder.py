#   Adapted from:
#   https://github.com/ZikangZhou/QCNet (Apache License 2.0)

from typing import Dict, Optional, List

import torch
import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType

from nuplan_zigned.training.modeling.modules.qcmae_agent_encoder import QCMAEAgentEncoder
from nuplan_zigned.training.modeling.modules.qcmae_map_encoder import QCMAEMapEncoder


class QCMAEEncoder(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_features: List[str],
                 agent_features: List[str],
                 pretrain: bool,
                 pretrain_map_encoder: bool,
                 pretrain_agent_encoder: bool,
                 prob_pretrain_mask: List[float],
                 prob_pretrain_mask_mask: float,
                 prob_pretrain_mask_random: float,
                 prob_pretrain_mask_unchanged: float) -> None:
        super(QCMAEEncoder, self).__init__()
        if not pretrain or (pretrain and pretrain_map_encoder):
            self.map_encoder = QCMAEMapEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                pl2pl_radius=pl2pl_radius,
                num_freq_bands=num_freq_bands,
                num_layers=num_map_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                pretrain=pretrain,
                prob_pretrain_mask=prob_pretrain_mask[1],
                prob_pretrain_mask_mask=prob_pretrain_mask_mask,
                prob_pretrain_mask_random=prob_pretrain_mask_random,
                prob_pretrain_mask_unchanged=prob_pretrain_mask_unchanged,
            )
        if not pretrain or (pretrain and pretrain_agent_encoder):
            self.agent_encoder = QCMAEAgentEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                time_span=time_span,
                pl2a_radius=pl2a_radius,
                a2a_radius=a2a_radius,
                num_freq_bands=num_freq_bands,
                num_layers=num_agent_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                map_features=map_features,
                agent_features=agent_features,
                pretrain=pretrain,
                prob_pretrain_mask=prob_pretrain_mask[0],
                prob_pretrain_mask_mask=prob_pretrain_mask_mask,
                prob_pretrain_mask_random=prob_pretrain_mask_random,
                prob_pretrain_mask_unchanged=prob_pretrain_mask_unchanged,
            )
        self.pretrain = pretrain
        self.pretrain_map_encoder = pretrain_map_encoder
        self.pretrain_agent_encoder = pretrain_agent_encoder

    def forward(self, data: FeaturesType) -> Dict[str, torch.Tensor]:
        if self.pretrain:
            if self.pretrain_map_encoder and not self.pretrain_agent_encoder:
                map_enc = self.map_encoder(data)
                return {**map_enc}
            elif self.pretrain_agent_encoder and not self.pretrain_map_encoder:
                map_enc = None
                agent_enc = self.agent_encoder(data, map_enc)
                return {**agent_enc}
            elif self.pretrain_map_encoder and self.pretrain_agent_encoder:
                map_enc = self.map_encoder(data)
                agent_enc = self.agent_encoder(data, map_enc)
                return {**map_enc, **agent_enc}
        else:
            map_enc = self.map_encoder(data)
            agent_enc = self.agent_encoder(data, map_enc)
            return {**map_enc, **agent_enc}
