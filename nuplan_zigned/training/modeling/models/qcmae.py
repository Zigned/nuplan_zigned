import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, cast, Any, Optional, Union

import torch
from pytorch_lightning.utilities import device_parser

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType

from nuplan_zigned.training.preprocessing.feature_builders.qcmae_vector_set_map_feature_builder import VectorSetMapFeatureBuilder
from nuplan_zigned.training.preprocessing.feature_builders.qcmae_generic_agents_feature_builder import GenericAgentsFeatureBuilder
from nuplan_zigned.training.preprocessing.target_builders.qcmae_agents_trajectories_target_builder import AgentTrajectoryTargetBuilder
from nuplan_zigned.training.modeling.modules.qcmae_encoder import QCMAEEncoder
from nuplan_zigned.training.modeling.modules.qcmae_decoder import QCMAEDecoder
from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories
from nuplan_zigned.training.preprocessing.features.qcmae_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.features.qcmae_generic_agents import GenericAgents

logger = logging.getLogger(__name__)


@dataclass
class FeatureParams:
    """
    Parameters for QCMAE features.
    """
    dim: int
    num_past_steps: int
    num_future_steps: int
    radius: float
    map_features: List[str]
    max_elements: Dict[str, int]
    max_points: Dict[str, int]
    max_agents: Dict[str, int]
    predict_unseen_agents: bool
    vector_repr: bool
    interpolation_method: str
    agent_features: List[str]
    past_trajectory_sampling: TrajectorySampling


@dataclass
class TargetParams:
    """
    Parameters for QCMAE targets.
        agent_featrues: agents of interested types to be predicted.
        future_trajectory_sampling: Sampling parameters for future trajectory.
    """
    agent_featrues: List[str]
    num_past_steps: int
    future_trajectory_sampling: TrajectorySampling


@dataclass
class ModelParams:
    """
    Parameters for QCMAE
    """
    input_dim: int
    hidden_dim: int
    output_dim: int
    output_head: bool
    num_past_steps: int
    num_future_steps: int
    num_modes: int
    num_recurrent_steps: int
    num_freq_bands: int
    num_map_layers: int
    num_agent_layers: int
    num_dec_layers: int
    num_heads: int
    head_dim: int
    dropout: float
    pl2pl_radius: int
    time_span: int
    pl2a_radius: int
    a2a_radius: int
    num_t2m_steps: int
    pl2m_radius: int
    a2m_radius: int
    self_distillation: bool
    sd_temperature: float
    sd_alpha: float
    sd_margin: float
    pretrain: bool
    pretrain_map_encoder: bool
    pretrain_agent_encoder: bool
    prob_pretrain_mask: List[float]
    prob_pretrain_mask_mask: float
    prob_pretrain_mask_random: float
    prob_pretrain_mask_unchanged: float
    pretrained_model_dir: str
    ckpt_path: str
    gpus: Optional[Union[List[int], str, int]]


class QCMAE(TorchModuleWrapper):
    """
    Wrapper around transformer-based reward model that consumes ego, agent and map data in vector format
    and regresses a scaler reward.
    """
    def __init__(
        self,
        model_name: str,
        feature_params: FeatureParams,
        target_params: TargetParams,
        model_params: ModelParams,
    ):
        """
        Initialize model.
        """
        super().__init__(
            feature_builders=[
                VectorSetMapFeatureBuilder(
                    dim=feature_params.dim,
                    radius=feature_params.radius,
                    map_features=feature_params.map_features,
                    max_elements=feature_params.max_elements,
                    max_points=feature_params.max_points,
                    interpolation_method=feature_params.interpolation_method,
                ),
                GenericAgentsFeatureBuilder(
                    agent_features=feature_params.agent_features,
                    trajectory_sampling=feature_params.past_trajectory_sampling,
                    num_future_steps=feature_params.num_future_steps,
                    a2a_radius=model_params.a2a_radius,
                    max_agents=feature_params.max_agents,
                ),
            ],
            target_builders=[AgentTrajectoryTargetBuilder(agent_featrues=target_params.agent_featrues,
                                                          num_past_steps=target_params.num_past_steps,
                                                          future_trajectory_sampling=target_params.future_trajectory_sampling)],
            future_trajectory_sampling=target_params.future_trajectory_sampling,
        )
        self.model_name = model_name
        self.encoder = QCMAEEncoder(
            input_dim=model_params.input_dim,
            hidden_dim=model_params.hidden_dim,
            num_historical_steps=model_params.num_past_steps + 1,
            pl2pl_radius=model_params.pl2pl_radius,
            time_span=model_params.time_span,
            pl2a_radius=model_params.pl2a_radius,
            a2a_radius=model_params.a2a_radius,
            num_freq_bands=model_params.num_freq_bands,
            num_map_layers=model_params.num_map_layers,
            num_agent_layers=model_params.num_agent_layers,
            num_heads=model_params.num_heads,
            head_dim=model_params.head_dim,
            dropout=model_params.dropout,
            map_features=feature_params.map_features,
            agent_features=feature_params.agent_features,
            pretrain=model_params.pretrain,
            pretrain_map_encoder=model_params.pretrain_map_encoder,
            pretrain_agent_encoder=model_params.pretrain_agent_encoder,
            prob_pretrain_mask=model_params.prob_pretrain_mask,
            prob_pretrain_mask_mask=model_params.prob_pretrain_mask_mask,
            prob_pretrain_mask_random=model_params.prob_pretrain_mask_random,
            prob_pretrain_mask_unchanged=model_params.prob_pretrain_mask_unchanged,
        )
        if not model_params.pretrain:
            self.decoder = QCMAEDecoder(
                input_dim=model_params.input_dim,
                hidden_dim=model_params.hidden_dim,
                output_dim=model_params.output_dim,
                output_head=model_params.output_head,
                num_historical_steps=model_params.num_past_steps + 1,
                num_future_steps=model_params.num_future_steps,
                num_modes=model_params.num_modes,
                num_recurrent_steps=model_params.num_recurrent_steps,
                num_t2m_steps=model_params.num_t2m_steps,
                pl2m_radius=model_params.pl2m_radius,
                a2m_radius=model_params.a2m_radius,
                num_freq_bands=model_params.num_freq_bands,
                num_layers=model_params.num_dec_layers,
                num_heads=model_params.num_heads,
                head_dim=model_params.head_dim,
                dropout=model_params.dropout,
            )

        self.pretrain = model_params.pretrain
        self.pretrain_map_encoder = model_params.pretrain_map_encoder
        self.pretrain_agent_encoder = model_params.pretrain_agent_encoder

        self.self_distillation = model_params.self_distillation
        self.last_data = None

        self.gpus = model_params.gpus
        try:
            self.parallel_device_ids = device_parser.parse_gpu_ids(self.gpus)
            self.map_location = f'cuda:{self.parallel_device_ids[0]}'
            self.device = torch.device(self.map_location)
        except:
            self.map_location = 'cpu'
            self.device = torch.device('cpu')

        if model_params.pretrained_model_dir is not None:
            pretrained_model_dict = torch.load(model_params.pretrained_model_dir, map_location=self.map_location)
            encoder_dict = self.encoder.state_dict()
            model_dict = {}
            for k, v in pretrained_model_dict['state_dict'].items():
                if k.replace('model.encoder.', '', 1) in list(encoder_dict.keys()):
                    model_dict[k.replace('model.encoder.', '', 1)] = v
            assert len(model_dict) > 0
            encoder_dict.update(model_dict)
            self.encoder.load_state_dict(encoder_dict)
            logger.info(f'\npretrained model loaded: {model_params.pretrained_model_dir}\n')

        if model_params.ckpt_path is not None:
            ckpt_model_dict = torch.load(model_params.ckpt_path, map_location=self.map_location)
            encoder_dict = self.encoder.state_dict()
            model_dict = {}
            for k, v in ckpt_model_dict['state_dict'].items():
                if k.replace('model.encoder.', '', 1) in list(encoder_dict.keys()):
                    model_dict[k.replace('model.encoder.', '', 1)] = v
            assert len(model_dict) > 0
            encoder_dict.update(model_dict)
            self.encoder.load_state_dict(encoder_dict)
            decoder_dict = self.decoder.state_dict()
            model_dict = {}
            for k, v in ckpt_model_dict['state_dict'].items():
                if k.replace('model.decoder.', '', 1) in list(decoder_dict.keys()):
                    model_dict[k.replace('model.decoder.', '', 1)] = v
            assert len(model_dict) > 0
            decoder_dict.update(model_dict)
            self.decoder.load_state_dict(decoder_dict)
            logger.info(f'\ncheck point loaded: {model_params.ckpt_path}\n')

    def forward(self, features: FeaturesType) -> Dict[str, Any]:
        """
        Predict rewards.
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                        }
        :return: pred: prediction from network
                        {
                            "agents_trajectories": AgentsTrajectories,
                        }
        """
        features_before_collating = features

        # reset self.last_data when starting validation
        if self.last_data is not None:
            if self.last_data['vector_set_map'].batch_size > features['vector_set_map'].batch_size:
                self.last_data = None

        if self.last_data is not None:  # self-distillation
            features = {
                'vector_set_map': VectorSetMap.collate(batch=[self.last_data['vector_set_map'], features['vector_set_map']]),
                'generic_agents': GenericAgents.collate(batch=[self.last_data['generic_agents'], features['generic_agents']])
            }

        scene_enc = self.encoder(features)
        if self.pretrain is False:
            pred = self.decoder(features, scene_enc)
        else:
            pred = scene_enc

        if self.self_distillation:
            self.last_data = features_before_collating

        return {'agents_trajectories': AgentsTrajectories(trajectories=pred,
                                                          track_token_ids=features['generic_agents'].agent_data['id'],
                                                          objects_types=features['generic_agents'].agent_data['type'],
                                                          predict_mask=features['generic_agents'].agent_data['predict_mask'])}