import logging
import copy
from typing import Dict, List, Tuple, cast, Union, Any, Optional

import torch
import torch.nn as nn

from nuplan.planning.training.modeling.types import FeaturesType

from nuplan_zigned.training.modeling.modules.qcmae_agent_encoder import QCMAEAgentEncoder
from nuplan_zigned.training.modeling.modules.qcmae_map_encoder import QCMAEMapEncoder
from nuplan_zigned.training.modeling.modules.ritp_action_encoder import RITPActionEncoder
from nuplan_zigned.utils.weight_init import weight_init

logger = logging.getLogger(__name__)


class Critic(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 pl2pl_radius: float,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_t2action_steps: int,
                 pl2action_radius: float,
                 a2action_radius: float,
                 num_freq_bands: int,
                 num_map_layers: int,
                 num_agent_layers: int,
                 num_action_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_features: List[str],
                 agent_features: List[str],
                 pretrained_model_dir: str,
                 map_location: str,
                 ) -> None:
        super(Critic, self).__init__()
        self.critic_map_encoder1 = QCMAEMapEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pretrain=False,
            prob_pretrain_mask=0.,
            prob_pretrain_mask_mask=0.,
            prob_pretrain_mask_random=0.,
            prob_pretrain_mask_unchanged=0.,
        )
        self.critic_agent_encoder1 = QCMAEAgentEncoder(
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
            pretrain=False,
            prob_pretrain_mask=0.,
            prob_pretrain_mask_mask=0.,
            prob_pretrain_mask_random=0.,
            prob_pretrain_mask_unchanged=0.,
        )
        self.critic_action_encoder1 = RITPActionEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                num_t2action_steps=num_t2action_steps,
                pl2action_radius=pl2action_radius,
                a2action_radius=a2action_radius,
                num_freq_bands=num_freq_bands,
                num_layers=num_action_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
            )
        self.apply(weight_init)

        self.critic_map_encoder1.to(device=torch.device(map_location))
        self.critic_agent_encoder1.to(device=torch.device(map_location))
        self.critic_action_encoder1.to(device=torch.device(map_location))

        if pretrained_model_dir is not None:
            pretrained_model_dict = torch.load(pretrained_model_dir, map_location=map_location)
            map_encoder_dict = self.critic_map_encoder1.state_dict()
            model_dict = {}
            for k, v in pretrained_model_dict['state_dict'].items():
                if k.replace('model.encoder.map_encoder.', '', 1) in list(map_encoder_dict.keys()):
                    model_dict[k.replace('model.encoder.map_encoder.', '', 1)] = v
            assert len(model_dict) > 0
            map_encoder_dict.update(model_dict)
            self.critic_map_encoder1.load_state_dict(map_encoder_dict)

            agent_encoder_dict = self.critic_agent_encoder1.state_dict()
            model_dict = {}
            for k, v in pretrained_model_dict['state_dict'].items():
                if k.replace('model.encoder.agent_encoder.', '', 1) in list(agent_encoder_dict.keys()):
                    model_dict[k.replace('model.encoder.agent_encoder.', '', 1)] = v
            assert len(model_dict) > 0
            agent_encoder_dict.update(model_dict)
            self.critic_agent_encoder1.load_state_dict(agent_encoder_dict)

            logger.info(f'\nCritic has been initialized with pretrained QCMAE: {pretrained_model_dir}\n')

        self.critic_map_encoder2 = copy.deepcopy(self.critic_map_encoder1)
        self.critic_agent_encoder2 = copy.deepcopy(self.critic_agent_encoder1)
        self.critic_action_encoder2 = copy.deepcopy(self.critic_action_encoder1)

    def forward(self,
                features: FeaturesType,
                action: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        map_enc1 = self.critic_map_encoder1(features)
        scene_enc1 = self.critic_agent_encoder1(features, map_enc1)
        q1 = self.critic_action_encoder1(features, scene_enc1, action)

        map_enc2 = self.critic_map_encoder2(features)
        scene_enc2 = self.critic_agent_encoder2(features, map_enc2)
        q2 = self.critic_action_encoder2(features, scene_enc2, action)

        return q1, q2

    def Q1(self,
           features: FeaturesType,
           action: List[torch.Tensor]) -> List[torch.Tensor]:

        map_enc1 = self.critic_map_encoder1(features)
        scene_enc1 = self.critic_agent_encoder1(features, map_enc1)
        q1 = self.critic_action_encoder1(features, scene_enc1, action)

        return q1