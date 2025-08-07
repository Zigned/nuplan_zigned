import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union

import numpy as np
import torch
from pytorch_lightning.utilities import device_parser
from torch.nn.utils.rnn import pad_sequence
import copy

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.common.actor_state.state_representation import StateSE2

from nuplan_zigned.training.preprocessing.feature_builders.avrl_vector_set_map_feature_builder import VectorSetMapFeatureBuilder
from nuplan_zigned.training.preprocessing.feature_builders.avrl_generic_agents_feature_builder import GenericAgentsFeatureBuilder
from nuplan_zigned.training.modeling.modules.avrl_vector_map_encoder import RewardMapEncoder
from nuplan_zigned.training.modeling.modules.avrl_vector_agent_encoder import RewardAgentEncoder
from nuplan_zigned.training.modeling.modules.ritp_avrl_vector_agent_encoder import RewardAgentEncoder as RITPRewardAgentEncoder
from nuplan_zigned.training.modeling.modules.ritp_avrl_vector_map_encoder import RewardMapEncoder as RITPRewardMapEncoder
from nuplan_zigned.training.modeling.modules.ritp_avrl_vector_reward_encoder import RewardEncoder as RITPRewardEncoder
from nuplan_zigned.training.modeling.modules.avrl_vector_reward_encoder import RewardEncoder
from nuplan_zigned.training.preprocessing.features.reward import Reward
from nuplan_zigned.training.preprocessing.features.qcmae_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.features.qcmae_generic_agents import GenericAgents

logger = logging.getLogger(__name__)


@dataclass
class RewardFormerFeatureParams:
    """
    Parameters for AVRL vector RewardFormer features.
        feature_types: List of feature types (agent and map) supported by model. Used in type embedding layer.
        total_max_points: maximum number of points per element, to maintain fixed sized features.
        feature_dimension: feature size, to maintain fixed sized features.
        agent_features: Agent features to request from agent feature builder.
        ego_dimension: Feature dimensionality to keep from ego features.
        agent_dimension: Feature dimensionality to keep from agent features.
        max_agents: maximum number of agents, to maintain fixed sized features.
        map_features: Map features to request from vector set map feature builder.
        max_elements: Maximum number of elements to extract per map feature layer.
        max_points: Maximum number of points per feature to extract per map feature layer.
        vector_set_map_feature_radius: The query radius scope relative to the current ego-pose.
        interpolation_method: Interpolation method to apply when interpolating to maintain fixed size map elements.
        disable_map: whether to ignore map.
        disable_agents: whether to ignore agents.
        num_poses: number of poses in trajectory in addition to initial state. used for trajectory sampling.
        time_horizon: [s] time horizon of all poses. used for trajectory sampling.
        frenet_radius: [m] The minimum query radius scope relative to the current ego-pose. Will be adjusted according to speed. used for trajectory sampling.
    """

    feature_types: Dict[str, int]
    total_max_points: int
    feature_dimension: int
    agent_features: List[str]
    ego_dimension: int
    agent_dimension: int
    max_agents: int
    map_features: List[str]
    max_elements: Dict[str, int]
    max_points: Dict[str, int]
    vector_set_map_feature_radius: int
    interpolation_method: str
    disable_map: bool
    disable_agents: bool
    num_poses: int
    time_horizon: float
    frenet_radius: float

    def __post_init__(self) -> None:
        """
        Sanitize feature parameters.
        :raise AssertionError if parameters invalid.
        """
        if not self.total_max_points > 0:
            raise AssertionError(f"Total max points must be >0! Got: {self.total_max_points}")

        if not self.feature_dimension >= 2:
            raise AssertionError(f"Feature dimension must be >=2! Got: {self.feature_dimension}")

        # sanitize feature types
        for feature_name in ["NONE", "EGO"]:
            if feature_name not in self.feature_types:
                raise AssertionError(f"{feature_name} must be among feature types! Got: {self.feature_types}")

        self._sanitize_agent_features()
        self._sanitize_map_features()

    def _sanitize_agent_features(self) -> None:
        """
        Sanitize agent feature parameters.
        :raise AssertionError if parameters invalid.
        """
        if "EGO" in self.agent_features:
            raise AssertionError("EGO must not be among agent features!")
        for feature_name in self.agent_features:
            if feature_name not in self.feature_types:
                raise AssertionError(f"Agent feature {feature_name} not in feature_types: {self.feature_types}!")

    def _sanitize_map_features(self) -> None:
        """
        Sanitize map feature parameters.
        :raise AssertionError if parameters invalid.
        """
        for feature_name in self.map_features:
            if feature_name not in self.feature_types:
                raise AssertionError(f"Map feature {feature_name} not in feature_types: {self.feature_types}!")
            if feature_name not in self.max_elements:
                raise AssertionError(f"Map feature {feature_name} not in max_elements: {self.max_elements.keys()}!")
            if feature_name not in self.max_points:
                raise AssertionError(f"Map feature {feature_name} not in max_points types: {self.max_points.keys()}!")


@dataclass
class RewardFormerParams:
    """
    Parameters for AVRL vector RewardFormer
    """
    dropout: float
    sigma_prior: float
    num_training_scenarios: int
    batch_size: int
    num_poses: int
    input_dim: int
    hidden_dim: int
    num_freq_bands: int
    num_layers: int
    num_heads: int
    head_dim: int
    only_ego_attends_map: bool
    gated_attention: bool
    gate_has_dropout: bool
    estimate_u_r_stats: bool
    validate: bool
    simulate: bool
    num_samples: int
    lambda_u_r: float
    precision_tau: float
    model_dir: Optional[str]
    u_r_stats_dir: Optional[str]
    gpus: Optional[Union[List[int], str, int]]


class RewardFormer(TorchModuleWrapper):
    """
    Wrapper around transformer-based reward model that consumes ego, agent and map data in vector format
    and regresses a scaler reward.
    """
    def __init__(
        self,
        feature_params: RewardFormerFeatureParams,
        target_builders: List[AbstractTargetBuilder],
        model_name: str,
        future_trajectory_sampling: TrajectorySampling,
        model_params: RewardFormerParams,
    ):
        """
        Initialize model.
        :param feature_params: agent and map feature parameters.
        :param target_builders: list of builders for targets.
        :param model_name: name of the model (e.g. resnet_50, efficientnet_b3).
        :param num_features_per_pose: number of features per single pose
        :param future_trajectory_sampling: parameters of predicted trajectory
        :param dropout: probability of an element to be zeroed
        :param sigma_prior: standard deviation of the prior distribution p(\omega)
        """
        super().__init__(
            feature_builders=[
                VectorSetMapFeatureBuilder(
                    map_features=feature_params.map_features,
                    max_elements=feature_params.max_elements,
                    max_points=feature_params.max_points,
                    radius=feature_params.vector_set_map_feature_radius,
                    interpolation_method=feature_params.interpolation_method,
                    num_poses=feature_params.num_poses,
                    time_horizon=feature_params.time_horizon,
                    frenet_radius=feature_params.frenet_radius,
                ),
                GenericAgentsFeatureBuilder(
                    agent_features=feature_params.agent_features,
                    num_poses=feature_params.num_poses,
                    time_horizon=feature_params.time_horizon,
                    max_agents=feature_params.max_agents,
                ),
            ],
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )
        if model_params.simulate:
            self.map_encoder = RITPRewardMapEncoder(
                input_dim=model_params.input_dim,
                hidden_dim=model_params.hidden_dim,
                num_freq_bands=model_params.num_freq_bands,
                num_layers=model_params.num_layers,
                num_heads=model_params.num_heads,
                head_dim=model_params.head_dim,
                dropout=model_params.dropout,
                num_poses=model_params.num_poses,
                gated_attention=model_params.gated_attention,
                gate_has_dropout=model_params.gate_has_dropout,
            )
            self.agent_encoder = RITPRewardAgentEncoder(
                input_dim=model_params.input_dim,
                hidden_dim=model_params.hidden_dim,
                num_freq_bands=model_params.num_freq_bands,
                num_layers=model_params.num_layers,
                num_heads=model_params.num_heads,
                head_dim=model_params.head_dim,
                dropout=model_params.dropout,
                num_poses=model_params.num_poses,
                only_ego_attends_map=model_params.only_ego_attends_map,
                gated_attention=model_params.gated_attention,
                gate_has_dropout=model_params.gate_has_dropout,
            )
            self.reward_encoder = RITPRewardEncoder(
                hidden_dim=model_params.hidden_dim,
                num_layers=model_params.num_layers,
                dropout=model_params.dropout,
                num_poses=model_params.num_poses,
            )
        else:
            self.map_encoder = RewardMapEncoder(
                input_dim=model_params.input_dim,
                hidden_dim=model_params.hidden_dim,
                num_freq_bands=model_params.num_freq_bands,
                num_layers=model_params.num_layers,
                num_heads=model_params.num_heads,
                head_dim=model_params.head_dim,
                dropout=model_params.dropout,
                num_poses=model_params.num_poses,
                gated_attention=model_params.gated_attention,
                gate_has_dropout=model_params.gate_has_dropout,
            )
            self.agent_encoder = RewardAgentEncoder(
                input_dim=model_params.input_dim,
                hidden_dim=model_params.hidden_dim,
                num_freq_bands=model_params.num_freq_bands,
                num_layers=model_params.num_layers,
                num_heads=model_params.num_heads,
                head_dim=model_params.head_dim,
                dropout=model_params.dropout,
                num_poses=model_params.num_poses,
                only_ego_attends_map=model_params.only_ego_attends_map,
                gated_attention=model_params.gated_attention,
                gate_has_dropout=model_params.gate_has_dropout,
            )
            self.reward_encoder = RewardEncoder(
                hidden_dim=model_params.hidden_dim,
                num_layers=model_params.num_layers,
                dropout=model_params.dropout,
                num_poses=model_params.num_poses,
            )
        self.model_name = model_name
        self.dropout = model_params.dropout
        self.sigma_prior = model_params.sigma_prior
        self.num_training_scenarios = model_params.num_training_scenarios
        self.batch_size = model_params.batch_size
        self.estimate_u_r_stats = model_params.estimate_u_r_stats
        self.validate = model_params.validate
        self.simulate = model_params.simulate
        self.num_samples = model_params.num_samples
        self.lambda_u_r = model_params.lambda_u_r
        self.precision_tau = model_params.precision_tau
        self.model_dir = model_params.model_dir
        self.u_r_stats_dir = model_params.u_r_stats_dir
        self.gpus = model_params.gpus
        try:
            self.parallel_device_ids = device_parser.parse_gpu_ids(self.gpus)
        except:
            self.parallel_device_ids = None
        if self.parallel_device_ids is not None:
            self.map_location = f'cuda:{self.parallel_device_ids[0]}'
        else:
            self.map_location = 'cpu'
        self.device = torch.device(self.map_location)
        self.agent_types = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT']
        if self.estimate_u_r_stats:
            self.reward = []
            self.u_reward = []
            self.u_r_stats = {
                'reward_mean': None,
                'reward_std': None,
                'u_reward_mean': None,
                'u_reward_std': None
            }
        if self.validate or self.simulate:
            self._reward_mean = torch.tensor(1., device=torch.device(self.map_location))
            self._reward_std = torch.tensor(1., device=torch.device(self.map_location))
            self._u_reward_mean = torch.tensor(1., device=torch.device(self.map_location))
            self._u_reward_std = torch.tensor(1., device=torch.device(self.map_location))

            if self.u_r_stats_dir is not None:
                try:
                    u_r_stats = np.load(self.u_r_stats_dir, allow_pickle=True).item()
                    self._reward_mean = u_r_stats['reward_mean'].to(torch.device(self.map_location))
                    self._reward_std = u_r_stats['reward_std'].to(torch.device(self.map_location))
                    self._u_reward_mean = u_r_stats['u_reward_mean'].to(torch.device(self.map_location))
                    self._u_reward_std = u_r_stats['u_reward_std'].to(torch.device(self.map_location))
                    logger.info(f'u_r_stats loaded: {self.u_r_stats_dir}')

                except:
                    logger.info(f'failed to load u_r_stats: {self.u_r_stats_dir}')

        if self.model_dir is not None:
            trained_model_dict = torch.load(self.model_dir, map_location=self.map_location)
            map_encoder_dict = self.map_encoder.state_dict()
            trained_map_model_dict = {}
            for k, v in trained_model_dict['state_dict'].items():
                if k.replace('model.map_encoder.', '', 1) in list(map_encoder_dict.keys()):
                    trained_map_model_dict[k.replace('model.map_encoder.', '', 1)] = v
            assert len(trained_map_model_dict) > 0
            map_encoder_dict.update(trained_map_model_dict)
            self.map_encoder.load_state_dict(map_encoder_dict)

            agent_encoder_dict = self.agent_encoder.state_dict()
            trained_agent_model_dict = {}
            for k, v in trained_model_dict['state_dict'].items():
                if k.replace('model.agent_encoder.', '', 1) in list(agent_encoder_dict.keys()):
                    trained_agent_model_dict[k.replace('model.agent_encoder.', '', 1)] = v
            assert len(trained_agent_model_dict) > 0
            agent_encoder_dict.update(trained_agent_model_dict)
            self.agent_encoder.load_state_dict(agent_encoder_dict)

            reward_encoder_dict = self.reward_encoder.state_dict()
            trained_reward_model_dict = {}
            for k, v in trained_model_dict['state_dict'].items():
                if k.replace('model.reward_encoder.', '', 1) in list(reward_encoder_dict.keys()):
                    trained_reward_model_dict[k.replace('model.reward_encoder.', '', 1)] = v
            assert len(trained_reward_model_dict) > 0
            reward_encoder_dict.update(trained_reward_model_dict)
            self.reward_encoder.load_state_dict(reward_encoder_dict)

            logger.info(f'\nRewardFormer loaded: {self.model_dir}\n')

    def forward(self, features: FeaturesType) -> Dict[str, Any]:
        """
        Predict rewards.
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                        }
        :return: reward: rewards from network
                        {
                            "reward": Reward,
                            "trajectory": Trajectory,
                        }
        """
        if self.estimate_u_r_stats:
            with (torch.no_grad()):
                self.eval()  # turn off dropout
                map_enc = self.map_encoder(features)
                agent_enc = self.agent_encoder(features, map_enc)
                r_mu = self.reward_encoder(agent_enc)
                r_mu = r_mu['reward_enc'].mean(dim=0).mean(dim=0)
                self.train()  # turn on dropout
                r_samples = []
                for i_sample in range(self.num_samples):
                    map_enc = self.map_encoder(features)
                    agent_enc = self.agent_encoder(features, map_enc)
                    r = self.reward_encoder(agent_enc)
                    r = r['reward_enc']
                    r_samples.append(r)
                r_samples = torch.cat(r_samples, dim=0)
                variance = 1. / self.precision_tau * self.sigma_prior ** 2 + (r_samples ** 2).mean(dim=0) - r_samples.mean(dim=0) ** 2
                r_variance = variance.mean(dim=0)
                self.reward.append(r_mu.cpu())
                self.u_reward.append(r_variance.cpu())

                reward = torch.stack(self.reward)
                u_reward = torch.stack(self.u_reward)
                self.u_r_stats['reward_mean'] = reward.mean(dim=0)
                self.u_r_stats['reward_std'] = reward.std(dim=0)
                self.u_r_stats['u_reward_mean'] = u_reward.mean(dim=0)
                self.u_r_stats['u_reward_std'] = u_reward.std(dim=0)
                np.save(self.u_r_stats_dir + 'u_r_stats.npy', self.u_r_stats)

            return None

        elif self.validate:
            with (torch.no_grad()):
                self.eval()  # turn off dropout
                map_enc = self.map_encoder(features)
                agent_enc = self.agent_encoder(features, map_enc)
                r_mu = self.reward_encoder(agent_enc)['reward_enc'][:, 1:]
                if self.lambda_u_r > 0.:
                    self.train()  # turn on dropout
                    r_samples = []
                    for i_sample in range(self.num_samples):
                        map_enc = self.map_encoder(features)
                        agent_enc = self.agent_encoder(features, map_enc)
                        r = self.reward_encoder(agent_enc)['reward_enc'][:, 1:]
                        r_samples.append(r)
                    r_samples = torch.stack(r_samples, dim=1)  # [batch_size, num_samples, num_trajs, num_poses]
                    variance = 1. / self.precision_tau * self.sigma_prior ** 2 + (r_samples ** 2).mean(dim=1) - r_samples.mean(dim=1) ** 2
                    if self._u_reward_mean.device != variance.device:
                        self._u_reward_mean = self._u_reward_mean.to(device=variance.device)
                        self._u_reward_std = self._u_reward_std.to(device=variance.device)
                    r_variance_scaled = torch.relu(
                        (variance - self._u_reward_mean)
                        / self._u_reward_std
                    )
                else:
                    r_variance_scaled = r_mu.new_zeros(r_mu.shape)
                reward = r_mu.mean(dim=-1) - self.lambda_u_r * r_variance_scaled.mean(dim=-1)

                best_mode = torch.argmax(reward, dim=-1)
                trajectory_samples = features['vector_set_map'].trajectory_samples
                for sample in trajectory_samples:
                    for traj in sample.values():
                        assert traj.shape[0] == reward.shape[1]
                best_trajectory = []
                for sample_idx in range(len(trajectory_samples)):
                    best_traj = trajectory_samples[sample_idx]['pose_cartesian'][best_mode[sample_idx]]
                    best_traj = best_traj.cpu()
                    current_pose = best_traj[:, 0:1]
                    current_pose = StateSE2(current_pose[0, 0], current_pose[1, 0], current_pose[2, 0])
                    future_poses = best_traj[:, 1:]
                    future_poses = [StateSE2(future_poses[0, i], future_poses[1, i], future_poses[2, i]) for i in range(future_poses.shape[1])]
                    # Get all future poses relative to the ego coordinate system
                    trajectory_relative_poses = convert_absolute_to_relative_poses(current_pose, future_poses)
                    best_trajectory.append(torch.tensor(trajectory_relative_poses, dtype=torch.float32, device=reward.device))

            return {"reward": Reward(data=reward),
                    "trajectory": Trajectory(data=torch.stack(best_trajectory))}

        elif self.simulate:
            list_features = self.process_features(features)

            with (torch.no_grad()):
                reward = []
                for i, features in enumerate(list_features):
                    self.eval()  # turn off dropout
                    map_enc = self.map_encoder(features)
                    agent_enc = self.agent_encoder(features, map_enc)
                    assert len(agent_enc['x_a']) == 1, "batch_size is not 1 yet assumed to be 1 in self.reward_encoder"
                    r_mu = self.reward_encoder(agent_enc)['reward_enc'].squeeze()
                    if self.lambda_u_r > 0.:
                        self.train()  # turn on dropout
                        r_samples = []
                        for i_sample in range(self.num_samples):
                            map_enc = self.map_encoder(features)
                            agent_enc = self.agent_encoder(features, map_enc)
                            r = self.reward_encoder(agent_enc)['reward_enc'].squeeze()
                            r_samples.append(r)
                        r_samples = torch.stack(r_samples, dim=0)  # [num_samples, num_poses]
                        variance = 1. / self.precision_tau * self.sigma_prior ** 2 + (r_samples ** 2).mean(dim=0) - r_samples.mean(dim=0) ** 2
                        if self._u_reward_mean.device != variance.device:
                            self._u_reward_mean = self._u_reward_mean.to(device=variance.device)
                            self._u_reward_std = self._u_reward_std.to(device=variance.device)
                        r_variance_scaled = torch.relu(
                            (variance - self._u_reward_mean)
                            / self._u_reward_std
                        )
                    else:
                        r_variance_scaled = r_mu.new_zeros(r_mu.shape)
                    reward.append(r_mu.mean() - self.lambda_u_r * r_variance_scaled.mean())
                reward = torch.stack(reward)

                best_mode = torch.argmax(reward, dim=-1)
                trajectory_samples = features['vector_set_map'].trajectory_samples_cartesian[0]
                for data in trajectory_samples.values():
                    assert data.shape[0] == reward.shape[0]
                best_trajectory = []
                best_traj = trajectory_samples['pose_cartesian'][best_mode]
                best_traj = torch.from_numpy(best_traj)
                current_pose = best_traj[:, 0:1]
                current_pose = StateSE2(current_pose[0, 0], current_pose[1, 0], current_pose[2, 0])
                future_poses = best_traj[:, 1:]
                future_poses = [StateSE2(future_poses[0, i], future_poses[1, i], future_poses[2, i]) for i in range(future_poses.shape[1])]
                # Get all future poses relative to the ego coordinate system
                trajectory_relative_poses = convert_absolute_to_relative_poses(current_pose, future_poses)
                best_trajectory.append(torch.tensor(trajectory_relative_poses, dtype=torch.float32, device=reward.device))

            return {"reward": Reward(data=reward[best_mode].unsqueeze(0).unsqueeze(0)),
                    "trajectory": Trajectory(data=torch.stack(best_trajectory))}

        else:
            map_enc = self.map_encoder(features)
            # map_enc = None
            agent_enc = self.agent_encoder(features, map_enc)
            reward_enc = self.reward_encoder(agent_enc)
            reward_enc = reward_enc['reward_enc']
            rewards_actor = reward_enc[:, 1:]
            r_actor = rewards_actor.sum(-1)
            best_mode = torch.argmax(r_actor, dim=-1)
            trajectory_samples = features['vector_set_map'].trajectory_samples
            best_trajectory = []
            for sample_idx in range(len(trajectory_samples)):
                best_traj = trajectory_samples[sample_idx]['pose_cartesian'][best_mode[sample_idx]]
                best_traj = best_traj.cpu()
                current_pose = best_traj[:, 0:1]
                current_pose = StateSE2(current_pose[0, 0], current_pose[1, 0], current_pose[2, 0])
                future_poses = best_traj[:, 1:]
                future_poses = [StateSE2(future_poses[0, i], future_poses[1, i], future_poses[2, i]) for i in range(future_poses.shape[1])]
                # Get all future poses relative to the ego coordinate system
                trajectory_relative_poses = convert_absolute_to_relative_poses(current_pose, future_poses)
                best_trajectory.append(torch.tensor(trajectory_relative_poses, dtype=torch.float32, device=reward_enc.device))

            return {"reward": Reward(data=reward_enc),
                    "trajectory": Trajectory(data=torch.stack(best_trajectory))}

    def process_features(self, features: FeaturesType) -> List[FeaturesType]:
        """extract features within radius of interest"""
        batch_size = features['vector_set_map'].batch_size
        device = features['vector_set_map'].map_data['map_polygon']['position'][0].device
        trajectory_samples_cartesian = features['vector_set_map'].trajectory_samples_cartesian
        num_trajs = trajectory_samples_cartesian[0]['geo_center_pose_cartesian'].shape[0]
        num_poses = trajectory_samples_cartesian[0]['geo_center_pose_cartesian'].shape[2] - 1
        radius = 20
        max_GENERIC_OBJECT = 10
        list_features = []

        for i_traj in range(num_trajs):
            # process map data
            list_anchor_position, list_pt_position = [], []
            map_data = copy.deepcopy(features['vector_set_map'].map_data)
            pt_position = map_data['map_point']['position']
            num_pt = map_data['map_point']['num_nodes']
            for sample_idx in range(batch_size):
                anchor_position = trajectory_samples_cartesian[sample_idx]['geo_center_pose_cartesian'][i_traj, 0:2, 1:].transpose()
                anchor_position = torch.from_numpy(anchor_position).float().to(device)
                list_anchor_position.append(
                    # repeat_interleave: (num_poses, 2) -> (num_poses * num_pt[sample_idx], 2)
                    anchor_position.unsqueeze(1).expand(num_poses, num_pt[sample_idx], 2).reshape(-1, 2)
                )
                list_pt_position.append(
                    # repeat: (num_pt[sample_idx], 2) -> (num_pt[sample_idx] * num_poses, 2)
                    pt_position[sample_idx].unsqueeze(0).expand(num_poses, num_pt[sample_idx], 2).reshape(-1, 2)
                )
            anchor_position = torch.cat(list_anchor_position)
            pt_position = torch.cat(list_pt_position)
            pt_within_radius = torch.norm(pt_position - anchor_position, p=2, dim=1) < radius

            # concatenate edge index
            pt2pl_edge_index = [edge_index.long() for edge_index in map_data[('map_point', 'to', 'map_polygon')]['edge_index']]
            pl2pl_edge_index = [edge_index.long() for edge_index in map_data[('map_polygon', 'to', 'map_polygon')]['edge_index']]
            for sample_idx in range(1, batch_size):
                pt2pl_edge_index[sample_idx][0] += \
                    sum(map_data['map_point']['num_nodes'][0:sample_idx])
                pt2pl_edge_index[sample_idx][1] += \
                    sum(map_data['map_polygon']['num_nodes'][0:sample_idx])
                pl2pl_edge_index[sample_idx] += \
                    sum(map_data['map_polygon']['num_nodes'][0:sample_idx])
            split_pt2pl_edge_index = pt2pl_edge_index
            pt2pl_edge_index = torch.cat(pt2pl_edge_index, dim=1)

            unique_polygons = torch.unique(pt2pl_edge_index[1, :], return_counts=True)
            num_pl = map_data['map_polygon']['num_nodes']
            unique_polygons_0 = torch.split(unique_polygons[0], num_pl)
            unique_polygons_1 = torch.split(unique_polygons[1], num_pl)
            repeated_unique_polygons_0, repeated_unique_polygons_1 = [], []
            for sample_idx in range(batch_size):
                repeated_unique_polygons_0.append(
                    # repeat: (num_pt[sample_idx],) -> (num_pt[sample_idx] * num_poses,)
                    unique_polygons_0[sample_idx].unsqueeze(0).expand(num_poses, num_pl[sample_idx]).reshape(-1,)
                )
                repeated_unique_polygons_1.append(
                    # repeat: (num_pt[sample_idx],) -> (num_pt[sample_idx] * num_poses,)
                    unique_polygons_1[sample_idx].unsqueeze(0).expand(num_poses, num_pl[sample_idx]).reshape(-1,)
                )
            repeated_unique_polygons_0 = torch.cat(repeated_unique_polygons_0)
            repeated_unique_polygons_1 = torch.cat(repeated_unique_polygons_1)
            pl_within_radius = torch.split(pt_within_radius, repeated_unique_polygons_1.cpu().tolist())
            padded_pl_within_radius = pad_sequence(pl_within_radius, batch_first=True, padding_value=0)
            pl_within_radius = padded_pl_within_radius.any(dim=1)

            repeated_pt2pl_edge_index, repeated_pl2pl_edge_index, repeated_pl2pl_type = [], [], []
            pl2pl_type = map_data[('map_polygon', 'to', 'map_polygon')]['type']
            for sample_idx in range(batch_size):
                repeated_pt2pl_edge_index.append(
                    split_pt2pl_edge_index[sample_idx].unsqueeze(0)
                    .expand(num_poses, 2, split_pt2pl_edge_index[sample_idx].shape[1])
                    .transpose(1, 2)
                    .reshape(-1, 2)
                    .transpose(0, 1)
                )
                repeated_pl2pl_edge_index.append(
                    pl2pl_edge_index[sample_idx].unsqueeze(0)
                    .expand(num_poses, 2, pl2pl_edge_index[sample_idx].shape[1])
                    .transpose(1, 2)
                    .reshape(-1, 2)
                    .transpose(0, 1)
                )
                repeated_pl2pl_type.append(
                    pl2pl_type[sample_idx].unsqueeze(0)
                    .expand(num_poses, pl2pl_type[sample_idx].shape[0])
                    .reshape(1, -1)
                    .squeeze()
                )
            repeated_pt2pl_edge_index = torch.cat(repeated_pt2pl_edge_index, dim=1)
            repeated_pl2pl_edge_index = torch.cat(repeated_pl2pl_edge_index, dim=1)
            repeated_pl2pl_type = torch.cat(repeated_pl2pl_type)
            bool1 = torch.any(repeated_pl2pl_edge_index[0:1] - repeated_unique_polygons_0[pl_within_radius].unsqueeze(-1) == 0., dim=0)
            bool2 = torch.any(repeated_pl2pl_edge_index[1:] - repeated_unique_polygons_0[pl_within_radius].unsqueeze(-1) == 0., dim=0)
            pl2pl_edge_index_tmp1 = repeated_pl2pl_edge_index[:, bool1 & bool2]
            pl2pl_edge_index_tmp2 = repeated_pl2pl_edge_index[:, bool2]
            if pl2pl_edge_index_tmp1.shape[1] > 0:
                pl2pl_edge_index = pl2pl_edge_index_tmp1
                pl2pl_type = repeated_pl2pl_type[bool1 & bool2]
            else:
                pl2pl_edge_index = pl2pl_edge_index_tmp2
                pl2pl_type = repeated_pl2pl_type[bool2]
                pl_within_radius[pl2pl_edge_index_tmp2[0].long()] = True

            repeated_whether_within_radius = pl_within_radius.repeat_interleave(repeated_unique_polygons_1)
            map_data[('map_point', 'to', 'map_polygon')]['edge_index'] = \
                repeated_pt2pl_edge_index[:, repeated_whether_within_radius]

            # repeat and concatenate other elements
            list_pl_position, list_pl_orientation, list_pl_type, list_pl_is_intersection, \
                list_pl_tl_statuses = [], [], [], [], []
            pl_position = map_data['map_polygon']['position']
            pl_orientation = map_data['map_polygon']['orientation']
            pl_type = map_data['map_polygon']['type']
            pl_is_intersection = map_data['map_polygon']['is_intersection']
            pl_tl_statuses = map_data['map_polygon']['tl_statuses']
            list_pt_orientation, list_pt_magnitude, list_pt_type, list_pt_side, \
                list_pt_tl_statuses = [], [], [], [], []
            pt_orientation = map_data['map_point']['orientation']
            pt_magnitude = map_data['map_point']['magnitude']
            pt_type = map_data['map_point']['type']
            pt_side = map_data['map_point']['side']
            pt_tl_statuses = map_data['map_point']['tl_statuses']
            for sample_idx in range(batch_size):
                list_pl_position.append(
                    # repeat: (num_pl[sample_idx], 2) -> (num_pl[sample_idx] * num_poses, 2)
                    pl_position[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pl[sample_idx], 2)
                    .reshape(-1, 2)
                )
                list_pl_orientation.append(
                    pl_orientation[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pl[sample_idx])
                    .reshape(-1)
                )
                list_pl_type.append(
                    pl_type[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pl[sample_idx])
                    .reshape(-1)
                )
                list_pl_is_intersection.append(
                    pl_is_intersection[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pl[sample_idx])
                    .reshape(-1)
                )
                list_pl_tl_statuses.append(
                    pl_tl_statuses[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pl[sample_idx])
                    .reshape(-1)
                )
                list_pt_orientation.append(
                    # repeat: (num_pt[sample_idx],) -> (num_pt[sample_idx] * num_poses,)
                    pt_orientation[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pt[sample_idx],)
                    .reshape(-1)
                )
                list_pt_magnitude.append(
                    pt_magnitude[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pt[sample_idx],)
                    .reshape(-1)
                )
                list_pt_type.append(
                    pt_type[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pt[sample_idx],)
                    .reshape(-1)
                )
                list_pt_side.append(
                    pt_side[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pt[sample_idx],)
                    .reshape(-1)
                )
                list_pt_tl_statuses.append(
                    pt_tl_statuses[sample_idx].unsqueeze(0)
                    .expand(num_poses, num_pt[sample_idx],)
                    .reshape(-1)
                )
            pl_position = torch.cat(list_pl_position)
            pl_orientation = torch.cat(list_pl_orientation)
            pl_type = torch.cat(list_pl_type)
            pl_is_intersection = torch.cat(list_pl_is_intersection)
            pl_tl_statuses = torch.cat(list_pl_tl_statuses)
            pt_orientation = torch.cat(list_pt_orientation)
            pt_magnitude = torch.cat(list_pt_magnitude)
            pt_type = torch.cat(list_pt_type)
            pt_side = torch.cat(list_pt_side)
            pt_tl_statuses = torch.cat(list_pt_tl_statuses)

            map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'] = [pl2pl_edge_index]
            map_data[('map_polygon', 'to', 'map_polygon')]['type'] = [pl2pl_type]
            map_data[('map_point', 'to', 'map_polygon')]['edge_index'] = [map_data[('map_point', 'to', 'map_polygon')]['edge_index']]
            map_data['map_polygon']['position'] = [pl_position]
            map_data['map_polygon']['orientation'] = [pl_orientation]
            map_data['map_polygon']['type'] = [pl_type]
            map_data['map_polygon']['is_intersection'] = [pl_is_intersection]
            map_data['map_polygon']['tl_statuses'] = [pl_tl_statuses]
            map_data['map_point']['position'] = [pt_position]
            map_data['map_point']['orientation'] = [pt_orientation]
            map_data['map_point']['magnitude'] = [pt_magnitude]
            map_data['map_point']['type'] = [pt_type]
            map_data['map_point']['side'] = [pt_side]
            map_data['map_point']['tl_statuses'] = [pt_tl_statuses]

            # process agent data
            agent_data = copy.deepcopy(features['generic_agents'])  # In fact, this is target
            trajectories_global = agent_data.trajectories_global
            velocity_global = agent_data.velocity_global
            objects_types = agent_data.objects_types
            predict_mask = agent_data.predict_mask
            num_agents = [len(types) for types in objects_types]
            av_index = [0,]
            [av_index.append(av_index[-1] + num_a) for num_a in num_agents[:batch_size - 1]]

            list_anchor_pose, list_trajectories_global, \
                list_velocity_global, list_objects_types, list_predict_mask= [], [], [], [], []
            for sample_idx in range(batch_size):
                anchor_pose = trajectory_samples_cartesian[sample_idx]['geo_center_pose_cartesian'][i_traj, :, 1:].transpose()
                heading = anchor_pose[:, 2]
                anchor_pose = torch.from_numpy(anchor_pose).float().to(device)
                ego_velocity_local_x = trajectory_samples_cartesian[sample_idx]['rear_axle_velocity_x'][i_traj, 1:].transpose()
                ego_velocity_local_y = trajectory_samples_cartesian[sample_idx]['rear_axle_velocity_y'][i_traj, 1:].transpose()
                ego_velocity_global_x = np.cos(heading) * ego_velocity_local_x - np.sin(heading) * ego_velocity_local_y
                ego_velocity_global_y = np.sin(heading) * ego_velocity_local_x + np.cos(heading) * ego_velocity_local_y
                ego_velocity_global = np.stack((ego_velocity_global_x, ego_velocity_global_y), axis=1)
                ego_velocity_global = torch.from_numpy(ego_velocity_global).float().to(device)
                list_anchor_pose.append(
                    # repeat_interleave: (num_poses, 3) -> (num_poses * num_agents[sample_idx], 3)
                    anchor_pose.unsqueeze(1).expand(num_poses, num_agents[sample_idx], 3).reshape(-1, 3)
                )
                traj = [anchor_pose, ] + list(trajectories_global[sample_idx].values())[1:]
                list_trajectories_global.append(
                    torch.stack(traj).transpose(0, 1).reshape(-1, 3)
                )
                vel = [ego_velocity_global, ] + list(velocity_global[sample_idx].values())[1:]
                list_velocity_global.append(
                    torch.stack(vel).transpose(0, 1).reshape(-1, 2)
                )
                type = [self.agent_types.index(type) if type != 'AV' else 0 for type in objects_types[sample_idx]]
                list_objects_types.append(
                    torch.tensor(type, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .expand(num_poses, num_agents[sample_idx])
                    .reshape(-1)
                )
                list_predict_mask.append(
                    predict_mask[sample_idx].transpose(0, 1).reshape(-1)
                )

            anchor_pose = torch.cat(list_anchor_pose)
            trajectories_global = torch.cat(list_trajectories_global)
            velocity_global = torch.cat(list_velocity_global)
            objects_types = torch.cat(list_objects_types)
            predict_mask = torch.cat(list_predict_mask)

            distance_to_agents = torch.norm(trajectories_global[:, 0:2] - anchor_pose[:, 0:2], p=2, dim=1)
            agents_within_radius = (distance_to_agents < radius) & predict_mask
            generic_object_mask = objects_types == self.agent_types.index('GENERIC_OBJECT')
            argsort = torch.argsort(distance_to_agents, descending=False)
            sorted_generic_object_mask = generic_object_mask[argsort]
            sorted_generic_object_indices = argsort[sorted_generic_object_mask]
            num_agents2 = [[num] * num_poses for num in num_agents]
            num_agents3 = sum(num_agents2, [])
            all_indices = torch.arange(len(agents_within_radius), device=device, dtype=torch.long)
            all_indices = torch.split(all_indices, num_agents3)
            for i in range(len(all_indices)):
                start = all_indices[i][0]
                end = all_indices[i][-1] + 1
                batch_mask = (start <= sorted_generic_object_indices) & (sorted_generic_object_indices < end)
                ignore_index = sorted_generic_object_indices[batch_mask][max_GENERIC_OBJECT:]
                agents_within_radius.index_fill_(dim=0, index=ignore_index, value=False)

            split_agents_within_radius = torch.split(agents_within_radius, num_agents3)
            padded_agents_within_radius = pad_sequence(split_agents_within_radius, batch_first=True, padding_value=0)
            valid_num_agents = padded_agents_within_radius.sum(1)
            valid_av_index = [0]
            [valid_av_index.append(valid_av_index[-1] + num.item()) for num in valid_num_agents[:-1]]

            agent_data_output = {
                'num_nodes': [valid_num_agents.cpu().tolist()],
                'av_index': [valid_av_index],
                'position': [trajectories_global[agents_within_radius][:, 0:2]],
                'heading': [trajectories_global[agents_within_radius][:, 2]],
                'velocity': [velocity_global[agents_within_radius]],
                'type': [objects_types[agents_within_radius]],
            }

            list_features.append({
                'vector_set_map': VectorSetMap(map_data=map_data,
                                               trajectory_samples_cartesian=trajectory_samples_cartesian),
                'generic_agents': GenericAgents(agent_data=agent_data_output)
            })

        return list_features