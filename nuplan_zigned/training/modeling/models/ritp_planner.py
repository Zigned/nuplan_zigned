import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, cast, Any, Optional, Union
import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import math
from pytorch_lightning.utilities import device_parser
from copy import deepcopy

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType, move_features_type_to_device
from nuplan.common.actor_state.state_representation import StateSE2

from nuplan_zigned.training.preprocessing.target_builders.ritp_agents_trajectories_target_builder import AgentTrajectoryTargetBuilder
from nuplan_zigned.training.modeling.modules.ritp_actor import Actor
from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories
from nuplan_zigned.training.preprocessing.features.qcmae_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.features.qcmae_generic_agents import GenericAgents
from nuplan_zigned.training.modeling.modules.ritp_critic import Critic
from nuplan_zigned.training.modeling.modules.ritp_rewardformer import RewardFormer
from nuplan_zigned.training.preprocessing.feature_builders.ritp_vector_set_map_feature_builder import VectorSetMapFeatureBuilder
from nuplan_zigned.training.preprocessing.feature_builders.ritp_generic_agents_feature_builder import GenericAgentsFeatureBuilder
from nuplan_zigned.training.preprocessing.features.ritp_trajectory_utils import convert_absolute_to_relative_poses, convert_relative_to_absolute_poses
from nuplan_zigned.utils.replay_buffer import ReplayBuffer
from nuplan_zigned.utils.env import Env
from nuplan_zigned.utils.utils import (
    efficient_relative_to_absolute_poses,
    efficient_absolute_to_relative_poses,
)

logger = logging.getLogger(__name__)


@dataclass
class RITPPlannerFeatureParams:
    """
    Parameters for RITP planner features.
    """
    radius: float
    map_features: List[str]
    max_elements: Dict[str, int]
    max_points: Dict[str, int]
    max_agents: Dict[str, int]
    agent_features: List[str]
    max_agents: Dict[str, int]
    interpolation_method: str
    num_past_steps: int
    num_future_steps: int
    past_trajectory_sampling: TrajectorySampling
    future_trajectory_sampling: TrajectorySampling


@dataclass
class RITPPlannerModelParams:
    """
    Parameters for RITP planner
    :param num_imagined_experiences: number of imagined experiences (assert finetune_mode == 'pseudo_ground_truth'), default None.
    :param imagine_batch_size: batch size of imagined experiences (assert num_imagined_experiences is not None), default None.
    :param imagine_max_elements: maximum number of elements to extract per map feature layer of imagined states.
    :param imagine_max_agents: maximum number of agents to extract per agent feature layer of imagined states.
    :param finetune_mode: 'pseudo_ground_truth': finetune actor with pseudo ground truth (selected
        from dense action samples according to critic), 'critic': directly finetune actor with critic.
    :param finetune_range: 'full', 'decoder', 'refine_module', 'additional_refine_module', 'mlp'.
    :param only_evaluate_av: whether to only evaluate predicted ego trajectory while computing metrics.
    :param future_trajectory_sampling: used for uncertainty-aware policy training.
    :param discount: TD3 parameter.
    :param tau: TD3 parameter.
    :param expl_noise: TD3 parameter.
    :param policy_noise: TD3 parameter.
    :param noise_clip: TD3 parameter.
    :param policy_freq: TD3 parameter.
    :param start_timesteps: TD3 parameter.
    :param batch_size: TD3 parameter.
    :param latest_rl_step: used when resuming training.
    """
    num_imagined_experiences: int
    imagine_batch_size: int
    imagine_max_elements: int
    imagine_max_agents: Dict[str, int]
    finetune_mode: str
    finetune_range: str
    only_evaluate_av: bool
    num_rl_workers: int
    discount: float
    tau: float
    expl_noise: float
    policy_noise: float
    noise_clip: float
    policy_freq: int
    start_timesteps: int
    batch_size: int
    latest_step: Optional[int]
    latest_rl_step: Optional[int]


@dataclass
class QCMAEFeatureParams:
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
class QCMAETargetParams:
    """
    Parameters for QCMAE targets.
    :param agent_featrues: agents of interested types to be predicted.
    :param future_trajectory_sampling: Sampling parameters for future trajectory.
    """
    agent_featrues: List[str]
    num_past_steps: int
    future_trajectory_sampling: TrajectorySampling


@dataclass
class QCMAEModelParams:
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
    pretrained_model_dir: str


@dataclass
class RewardFormerFeatureParams:
    """
    Parameters for AVRL vector RewardFormer features.
    :param feature_types: List of feature types (agent and map) supported by model. Used in type embedding layer.
    :param total_max_points: maximum number of points per element, to maintain fixed sized features.
    :param feature_dimension: feature size, to maintain fixed sized features.
    :param agent_features: Agent features to request from agent feature builder.
    :param ego_dimension: Feature dimensionality to keep from ego features.
    :param agent_dimension: Feature dimensionality to keep from agent features.
    :param max_agents: Maximum number of agents, to maintain fixed sized features.
    :param map_features: Map features to request from vector set map feature builder.
    :param max_elements: Maximum number of elements to extract per map feature layer.
    :param max_points: Maximum number of points per feature to extract per map feature layer.
    :param vector_set_map_feature_radius: The query radius scope relative to the current ego-pose.
    :param interpolation_method: Interpolation method to apply when interpolating to maintain fixed size map elements.
    :param disable_map: whether to ignore map.
    :param disable_agents: whether to ignore agents.
    :param num_poses: number of poses in trajectory in addition to initial state. used for trajectory sampling.
    :param time_horizon: [s] time horizon of all poses. used for trajectory sampling.
    :param frenet_radius: [m] The minimum query radius scope relative to the current ego-pose. Will be adjusted according to speed. used for trajectory sampling.
    """

    feature_types: Dict[str, int]
    total_max_points: int
    feature_dimension: int
    agent_features: List[str]
    ego_dimension: int
    agent_dimension: int
    max_agents: Dict[str, int]
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


@dataclass
class RewardFormerModelParams:
    """
    Parameters for AVRL vector RewardFormer
    """
    dropout: float
    sigma_prior: float
    precision_tau: float
    input_dim: int
    hidden_dim: int
    num_freq_bands: int
    num_layers: int
    num_heads: int
    head_dim: int
    gated_attention: bool
    gate_has_dropout: bool
    only_ego_attends_map: bool
    num_samples: int
    lambda_u_r: float
    model_dir: str
    u_r_stats_dir: str


class RITPPlanner(TorchModuleWrapper):
    """
    Wrapper around transformer-based RITP planner.
    """
    def __init__(
        self,
        model_name: str,
        ritp_planner_feature_params: RITPPlannerFeatureParams,
        ritp_planner_model_params: RITPPlannerModelParams,
        qcmae_feature_params: QCMAEFeatureParams,
        qcmae_target_params: QCMAETargetParams,
        qcmae_model_params: QCMAEModelParams,
        rewardformer_feature_params: RewardFormerFeatureParams,
        rewardformer_model_params: RewardFormerModelParams
    ):
        """
        Initialize model.
        """
        super().__init__(
            feature_builders=[
                VectorSetMapFeatureBuilder(
                    radius=ritp_planner_feature_params.radius,
                    map_features=ritp_planner_feature_params.map_features,
                    max_elements=ritp_planner_feature_params.max_elements,
                    max_points=ritp_planner_feature_params.max_points,
                    interpolation_method=ritp_planner_feature_params.interpolation_method,
                ),
                GenericAgentsFeatureBuilder(
                    agent_features=ritp_planner_feature_params.agent_features,
                    trajectory_sampling=ritp_planner_feature_params.past_trajectory_sampling,
                    num_future_steps=ritp_planner_feature_params.num_future_steps,
                    a2ego_radius=ritp_planner_feature_params.radius,
                    max_agents=ritp_planner_feature_params.max_agents,
                ),
            ],
            target_builders=[AgentTrajectoryTargetBuilder(agent_featrues=qcmae_target_params.agent_featrues,
                                                          num_past_steps=qcmae_target_params.num_past_steps,
                                                          future_trajectory_sampling=qcmae_target_params.future_trajectory_sampling)],
            future_trajectory_sampling=qcmae_target_params.future_trajectory_sampling,
        )
        self.model_name = model_name
        self.num_imagined_experiences = ritp_planner_model_params.num_imagined_experiences
        self.imagine_batch_size = ritp_planner_model_params.imagine_batch_size
        self.imagine_max_elements = ritp_planner_model_params.imagine_max_elements
        self.imagine_max_agents = ritp_planner_model_params.imagine_max_agents
        self.finetune_mode = ritp_planner_model_params.finetune_mode
        self.finetune_range = ritp_planner_model_params.finetune_range
        self.num_rl_workers = ritp_planner_model_params.num_rl_workers
        self.map_location = 'cpu'
        self.device = torch.device(self.map_location)  # proper device will be assigned to it in forward
        self.num_modes = qcmae_model_params.num_modes

        # TD3 params
        self.discount = ritp_planner_model_params.discount
        self.tau = ritp_planner_model_params.tau
        self.expl_noise = ritp_planner_model_params.expl_noise
        self.policy_noise = ritp_planner_model_params.policy_noise
        self.noise_clip = ritp_planner_model_params.noise_clip
        self.policy_freq = ritp_planner_model_params.policy_freq
        self.start_timesteps = ritp_planner_model_params.start_timesteps
        self.batch_size = ritp_planner_model_params.batch_size
        if ritp_planner_model_params.latest_rl_step is None:
            self.latest_step = 0
            self.t = 0
            self.total_it = 0
        else:
            self.latest_step = ritp_planner_model_params.latest_step
            self.t = ritp_planner_model_params.latest_rl_step
            self.total_it = (self.t - self.start_timesteps) // self.num_rl_workers
        self.env: Optional[Env] = None  # environment for RL
        self.replay_buffer: Optional[ReplayBuffer] = None  # replay buffer for RL

        logger.info(f'batch size of RL: {self.batch_size}')
        logger.info(f'number of imagined experiences in RL: {self.num_imagined_experiences}')
        logger.info(f'batch size of imagined experiences in RL: {self.imagine_batch_size}')
        logger.info(f'RL finetune mode for actor: {self.finetune_mode}')
        logger.info(f'RL finetune range for actor: {self.finetune_range}')

        if self.num_imagined_experiences is not None:
            assert self.finetune_mode == 'pseudo_ground_truth', \
                f'finetune mode has to be "pseudo_ground_truth" when number of imagined experiences is not None'

        # -------------------------------------
        #             Actor Network
        # -------------------------------------
        self.actor = Actor(
            input_dim=qcmae_model_params.input_dim,
            hidden_dim=qcmae_model_params.hidden_dim,
            num_historical_steps=qcmae_model_params.num_past_steps + 1,
            pl2pl_radius=qcmae_model_params.pl2pl_radius,
            time_span=qcmae_model_params.time_span,
            pl2a_radius=qcmae_model_params.pl2a_radius,
            a2a_radius=qcmae_model_params.a2a_radius,
            num_freq_bands=qcmae_model_params.num_freq_bands,
            num_map_layers=qcmae_model_params.num_map_layers,
            num_agent_layers=qcmae_model_params.num_agent_layers,
            num_heads=qcmae_model_params.num_heads,
            head_dim=qcmae_model_params.head_dim,
            dropout=qcmae_model_params.dropout,
            map_features=qcmae_feature_params.map_features,
            agent_features=qcmae_feature_params.agent_features,
            output_dim=qcmae_model_params.output_dim,
            output_head=qcmae_model_params.output_head,
            num_future_steps=qcmae_model_params.num_future_steps,
            num_modes=qcmae_model_params.num_modes,
            num_recurrent_steps=qcmae_model_params.num_recurrent_steps,
            num_t2m_steps=qcmae_model_params.num_t2m_steps,
            pl2m_radius=qcmae_model_params.pl2m_radius,
            a2m_radius=qcmae_model_params.a2m_radius,
            num_decoder_layers=qcmae_model_params.num_dec_layers,
            radius=qcmae_feature_params.radius,
            max_agents=qcmae_feature_params.max_agents,
            pretrained_model_dir=qcmae_model_params.pretrained_model_dir,
            map_location=self.map_location,
            finetune_range=self.finetune_range,
        )
        self.actor_target = deepcopy(self.actor).to(self.device)

        # -------------------------------------
        #             Critic Network
        # -------------------------------------
        self.critic = Critic(
            input_dim=qcmae_model_params.input_dim,
            hidden_dim=qcmae_model_params.hidden_dim,
            num_historical_steps=qcmae_model_params.num_past_steps + 1,
            num_future_steps=qcmae_model_params.num_future_steps,
            pl2pl_radius=qcmae_model_params.pl2pl_radius,
            pl2a_radius=qcmae_model_params.pl2a_radius,
            a2a_radius=qcmae_model_params.a2a_radius,
            num_t2action_steps=qcmae_model_params.num_t2m_steps,
            pl2action_radius=qcmae_model_params.pl2m_radius,
            a2action_radius=qcmae_model_params.a2m_radius,
            num_freq_bands=qcmae_model_params.num_freq_bands,
            num_map_layers=qcmae_model_params.num_map_layers,
            num_agent_layers=qcmae_model_params.num_agent_layers,
            num_action_layers=qcmae_model_params.num_dec_layers,
            num_heads=qcmae_model_params.num_heads,
            head_dim=qcmae_model_params.head_dim,
            dropout=qcmae_model_params.dropout,
            time_span=qcmae_model_params.time_span,
            map_features=qcmae_feature_params.map_features,
            agent_features=qcmae_feature_params.agent_features,
            pretrained_model_dir=qcmae_model_params.pretrained_model_dir,
            map_location=self.map_location,
        )
        self.critic_target = deepcopy(self.critic).to(self.device)

        # -------------------------------------
        #             Reward Network
        # -------------------------------------
        self.rewardformer = RewardFormer(
            dropout=rewardformer_model_params.dropout,
            sigma_prior=rewardformer_model_params.sigma_prior,
            precision_tau=rewardformer_model_params.precision_tau,
            input_dim=rewardformer_model_params.input_dim,
            hidden_dim=rewardformer_model_params.hidden_dim,
            num_freq_bands=rewardformer_model_params.num_freq_bands,
            num_layers=rewardformer_model_params.num_layers,
            num_heads=rewardformer_model_params.num_heads,
            head_dim=rewardformer_model_params.head_dim,
            num_poses=rewardformer_feature_params.num_poses,
            time_horizon=rewardformer_feature_params.time_horizon,
            vector_set_map_feature_radius=rewardformer_feature_params.vector_set_map_feature_radius,
            max_agents=rewardformer_feature_params.max_agents,
            gated_attention=rewardformer_model_params.gated_attention,
            gate_has_dropout=rewardformer_model_params.gate_has_dropout,
            only_ego_attends_map=rewardformer_model_params.only_ego_attends_map,
            num_samples=rewardformer_model_params.num_samples,
            lambda_u_r=rewardformer_model_params.lambda_u_r,
            model_dir=rewardformer_model_params.model_dir,
            u_r_stats_dir=rewardformer_model_params.u_r_stats_dir,
            map_location=self.map_location,
        )

    def forward(self,
                features: FeaturesType,
                targets: TargetsType,
                opt_idx: int,
                evaluation: bool) -> Dict[str, Any]:
        """
        Predict rewards.
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                        }
        :param targets: supervisory signal in global frame
        :param opt_idx: optimizer's index
        :param evaluation: whether in evaluation loop or not
        :return: pred: prediction from network
        """
        self.device = features['vector_set_map'].map_data['map_polygon']['position'][0].device
        if opt_idx == 0 or evaluation:  # optimizer for critic network
            with torch.no_grad():
                current_ego_state = [ego_controller.get_state() for ego_controller in self.env.ego_controller]
                actor_features = self.actor.process_features(
                    features,
                    anchor_ego_state=current_ego_state
                )
                if self.env.imagined_ego_controller is not None:
                    ego_states = [[ego_controller.get_state() for ego_controller in ego_controllers]
                                  for ego_controllers in self.env.imagined_ego_controller]
                    imagined_actor_features = [
                        self.actor.process_features(
                            features,
                            anchor_ego_state=ego_state
                        )
                        for ego_state in ego_states
                    ]
                else:
                    imagined_actor_features = None
                actions = []
                actions_global_cartesian = []
                ego_velocities_global_cartesian = []
                ego_accelerations_global_cartesian = []

                if self.t < self.start_timesteps and not evaluation:
                    # Select action randomly
                    for sample_idx in range(len(current_ego_state)):
                        reference_line_coords = self.env.get_centerline_coords(
                            scenario=self.env.scenario[sample_idx],
                            anchor_ego_state=current_ego_state[sample_idx]
                        )
                        frenet_frame = self.env.build_frenet_frame(reference_line_coords)
                        trajectory_sample_frenet = self.env.sample_action(frenet_frame, current_ego_state[sample_idx])
                        trajectory_sample_cartesian = frenet_frame.frenet_to_cartesian(
                            trajectory_sample_frenet['poses_frenet'],
                            trajectory_sample_frenet['t'],
                            trajectory_sample_frenet['vs_frenet'],
                            trajectory_sample_frenet['vl_frenet'],
                            trajectory_sample_frenet['as_frenet'],
                            trajectory_sample_frenet['al_frenet'],
                        )
                        geo_center_pose_cartesian = self.env.get_geo_center_pose_cartesian(
                            trajectory_sample_cartesian,
                            self.env.ego_controller[sample_idx]
                        )
                        origin_state = StateSE2(geo_center_pose_cartesian[0, 0, 0],
                                                geo_center_pose_cartesian[0, 1, 0],
                                                geo_center_pose_cartesian[0, 2, 0])
                        absolute_states = [StateSE2(pose[0], pose[1], pose[2]) for pose in geo_center_pose_cartesian[0].transpose(1, 0)]
                        geo_center_pose_local_cartesian = convert_absolute_to_relative_poses(origin_state, absolute_states)
                        action = torch.as_tensor(geo_center_pose_local_cartesian, device=self.device)
                        action = action[1:]
                        geo_center_pose_cartesian = geo_center_pose_cartesian[0].transpose(1, 0)
                        action_global_cartesian = torch.as_tensor(geo_center_pose_cartesian,
                                                                  dtype=torch.float32, device=self.device)
                        ego_velocity_global_cartesian = (geo_center_pose_cartesian[1:, 0:2] -
                                                         geo_center_pose_cartesian[:-1, 0:2]) / self.env.dt
                        ego_velocity_global_cartesian = np.vstack([ego_velocity_global_cartesian,
                                                                   ego_velocity_global_cartesian[-1:, :]])
                        ego_acceleration_global_cartesian = (ego_velocity_global_cartesian[1:, 0:2] -
                                                             ego_velocity_global_cartesian[:-1, 0:2]) / self.env.dt
                        ego_acceleration_global_cartesian = np.vstack([ego_acceleration_global_cartesian,
                                                                       ego_acceleration_global_cartesian[-1:, :]])
                        action_global_cartesian = action_global_cartesian[1:, :]
                        ego_velocity_global_cartesian = torch.tensor(ego_velocity_global_cartesian[1:, :], dtype=torch.float32, device=self.device)
                        ego_acceleration_global_cartesian = torch.tensor(ego_acceleration_global_cartesian[1:, :], dtype=torch.float32, device=self.device)
                        pred = None
                        actions.append(action)
                        actions_global_cartesian.append(action_global_cartesian)
                        ego_velocities_global_cartesian.append(ego_velocity_global_cartesian)
                        ego_accelerations_global_cartesian.append(ego_acceleration_global_cartesian)
                        if sample_idx == len(current_ego_state) - 1:
                            actions = [torch.stack(actions)]
                            actions_global_cartesian = [torch.stack(actions_global_cartesian).to(device=self.device)]
                            ego_velocities_global_cartesian = [torch.stack(ego_velocities_global_cartesian).to(device=self.device)]
                            ego_accelerations_global_cartesian = [torch.stack(ego_accelerations_global_cartesian).to(device=self.device)]

                else:
                    # Select action according to policy
                    pred = self.actor(actor_features)
                    ego_index = actor_features['generic_agents'].agent_data['av_index']
                    pi = pred['pi']
                    for sample_idx in range(len(pi)):  # all samples were collated into one batch, so len(pi) should be 1
                        prob = F.softmax(pi[sample_idx][ego_index], dim=-1)
                        ego_most_likely_mode = prob.argmax(dim=-1)
                        ego_most_likely_traj = pred['loc_refine_pos'][sample_idx][ego_index, ego_most_likely_mode]
                        noise_magnitude = torch.randn((len(ego_index),), device=self.device)\
                                          * self.expl_noise\
                                          * torch.norm(ego_most_likely_traj[:, -1] - ego_most_likely_traj[:, 0], dim=1)
                        noise_orientation = torch.rand((len(ego_index),), device=self.device) * math.pi * 2
                        noise_ratio = torch.linspace(1. / ego_most_likely_traj.shape[1],
                                                     1.,
                                                     ego_most_likely_traj.shape[1],
                                                     device=noise_magnitude.device)
                        noise = torch.stack([noise_magnitude * noise_orientation.cos(),
                                             noise_magnitude * noise_orientation.sin()]).unsqueeze(-1) * noise_ratio
                        if not evaluation:
                            ego_traj_with_noise = ego_most_likely_traj + noise.permute(1, 2, 0)
                            position = ego_traj_with_noise
                        else:
                            position = ego_most_likely_traj
                        heading = torch.atan2(position[:, 1:, 1] - position[:, 0:-1, 1],
                                              position[:, 1:, 0] - position[:, 0:-1, 0])
                        heading = torch.cat([heading, heading[:, -1:]], dim=1)
                        # fix heading when speed is low
                        slow_mask = torch.norm(position[:, 1:] - position[:, :-1], p=2, dim=-1) < 0.2
                        slow_mask = torch.cat([slow_mask, slow_mask[:, -1:]], dim=1)

                        false_to_true = slow_mask[:, 1:] & ~slow_mask[:, :-1]
                        for i in range(position.shape[0]):
                            heading[i][slow_mask[i]] = 0.
                            f2t = torch.where(false_to_true[i])[0]
                            for idx in f2t:
                                heading[i, idx+1:][slow_mask[i, idx+1:]] = heading[i, idx]

                        action = torch.cat([position, heading.unsqueeze(-1)], dim=-1)
                        actions.append(action)
                        origin_state = torch.tensor(
                            [[state.center.x, state.center.y, state.center.heading] for state in current_ego_state],
                            dtype=position.dtype,
                            device=position.device
                        )
                        relative_states = torch.cat([position, heading.unsqueeze(-1)], dim=-1).unsqueeze(1)
                        origin_state = origin_state.to(dtype=torch.float64)
                        relative_states = relative_states.to(dtype=torch.float64)
                        action_global_cartesian = efficient_relative_to_absolute_poses(origin_state, relative_states).squeeze()
                        actions_global_cartesian.append(action_global_cartesian.to(dtype=torch.float32, device=self.device))
                        ego_velocity_global_cartesian = (action_global_cartesian[:, 1:, 0:2] -
                                                         action_global_cartesian[:, :-1, 0:2]) / self.env.dt
                        ego_velocity_global_cartesian = torch.cat([ego_velocity_global_cartesian,
                                                                   ego_velocity_global_cartesian[:, -1:, :]], dim=1)
                        ego_acceleration_global_cartesian = (ego_velocity_global_cartesian[:, 1:, 0:2] -
                                                             ego_velocity_global_cartesian[:, :-1, 0:2]) / self.env.dt
                        ego_acceleration_global_cartesian = torch.cat([ego_acceleration_global_cartesian,
                                                                       ego_acceleration_global_cartesian[:, -1:, :]], dim=1)
                        ego_velocities_global_cartesian.append(ego_velocity_global_cartesian.to(dtype=torch.float32, device=self.device))
                        ego_accelerations_global_cartesian.append(ego_acceleration_global_cartesian.to(dtype=torch.float32, device=self.device))

                    if evaluation:
                        num_agents = actor_features['generic_agents'].agent_data['num_nodes']
                        pred = {
                            key: list(torch.split(value[0], num_agents))
                            for key, value in pred.items()
                        }
                        index = 0
                        track_token_ids, objects_types, predict_mask = [], [], []
                        for i in range(len(ego_index)):
                            track_token_ids.append(
                                actor_features['generic_agents'].agent_data['id'][0][index: index+num_agents[i]]
                            )
                            objects_types.append(
                                actor_features['generic_agents'].agent_data['type'][0][index: index+num_agents[i]]
                            )
                            predict_mask.append(
                                actor_features['generic_agents'].agent_data['predict_mask'][0][index: index+num_agents[i]]
                            )
                            index += num_agents[i]

                if self.finetune_mode == 'pseudo_ground_truth':
                    # sample actions densely
                    deduplicated_samples_frenet = []
                    imagined_actions_global_cartesian = []
                    imagined_vels_global_cartesian = []
                    imagined_accs_global_cartesian = []
                    for sample_idx in range(len(current_ego_state)):
                        reference_line_coords, reference_lanes = self.env.get_centerline_coords(
                            scenario=self.env.scenario[sample_idx],
                            anchor_ego_state=current_ego_state[sample_idx],
                            return_reference_lanes=True,
                        )
                        frenet_frame = self.env.build_frenet_frame(reference_line_coords)
                        samples = self.env.sample_action_densely(
                                frenet_frame,
                                reference_lanes,
                                current_ego_state[sample_idx]
                            )

                        if self.num_imagined_experiences is not None:
                            img_action, img_vel, img_acc = self._get_imagined_actions(samples)
                            imagined_actions_global_cartesian.append(img_action)
                            imagined_vels_global_cartesian.append(img_vel)
                            imagined_accs_global_cartesian.append(img_acc)
                        else:
                            samples['imagined_action_idx'] = None
                            imagined_actions_global_cartesian = None
                            imagined_vels_global_cartesian = None
                            imagined_accs_global_cartesian = None

                        deduplicated_samples_frenet.append(samples)

                else:
                    deduplicated_samples_frenet = [None for _ in range(len(current_ego_state))]

                # unpack actor features
                actor_features = self.actor.unpack_features(actor_features)
                if imagined_actor_features is not None:
                    imagined_actor_features = [self.actor.unpack_features(f) for f in imagined_actor_features]

                # Store reward, history, and done in env
                if self.env.state is not None:
                    self.env.previous_state = {
                        'vector_set_map': self.env.state['vector_set_map'],
                        'generic_agents': self.env.state['generic_agents']
                    }
                    self.env.previous_action = self.env.action
                    self.env.previous_action_samples = self.env.action_samples
                    self.env.previous_target = self.env.target
                    self.env.previous_reward = self.env.reward
                    self.env.previous_done = self.env.done
                self.env.state = {
                    key: value.to_device(torch.device('cpu')).unpack()
                    for key, value in actor_features.items()
                }
                if imagined_actor_features is not None:
                    self.env.state['imagined_state'] = [
                        {
                            key: value.to_device(torch.device('cpu')).unpack()
                            for key, value in features.items()
                        }
                        for features in imagined_actor_features
                    ]
                else:
                    self.env.state['imagined_state'] = None
                filtered_targets = self._filter_targets(actor_features, deepcopy(targets))
                self.convert_absolute_to_relative_targets(filtered_targets)
                self.env.target = {
                    'agents_trajectories': filtered_targets['agents_trajectories'].to_device(torch.device('cpu')).unpack()
                }
                self.env.action = [action.cpu() for action in actions[0]]
                self.env.action_samples = deduplicated_samples_frenet

                # # TODO debug only
                # # self.env.visualize(features, actions_global_cartesian, show=True, save=False, tag='plan')
                # self.env.visualize(features, actions_global_cartesian, show=False, save=True, tag='plan')

                # Perform action and compute reward and done
                self.env.step(
                    features,
                    targets,
                    self.rewardformer,
                    actions_global_cartesian,
                    ego_velocities_global_cartesian,
                    ego_accelerations_global_cartesian,
                    imagined_actions_global_cartesian,
                    imagined_vels_global_cartesian,
                    imagined_accs_global_cartesian,
                    self.imagine_batch_size,
                )

            target_Q, current_Q1, current_Q2 = None, None, None
            next_target = None
            if self.t >= self.start_timesteps and not evaluation:
                if self.total_it % self.policy_freq == 0:
                    # Update the frozen target models
                    self._update_target_networks()

                self.total_it += 1

                # Sample replay buffer
                state, action, action_samples, target, reward, \
                    next_state, not_done, next_target = self.replay_buffer.sample(self.device)

                with torch.no_grad():
                    # Select action according to policy and add clipped noise
                    unpacked_state = state
                    list_vector_set_map = [s['vector_set_map'] for s in state]
                    list_generic_agents = [s['generic_agents'] for s in state]
                    state = {
                        'vector_set_map': VectorSetMap.collate(list_vector_set_map),
                        'generic_agents': GenericAgents.collate(list_generic_agents)
                    }
                    list_vector_set_map = [next_s['vector_set_map'] for next_s in next_state]
                    list_generic_agents = [next_s['generic_agents'] for next_s in next_state]
                    if self.finetune_mode == 'pseudo_ground_truth' and self.num_imagined_experiences is not None:
                        list_imagined_vector_set_map = [next_s['imagined_state']['vector_set_map'] for next_s in next_state]
                        list_imagined_generic_agents = [next_s['imagined_state']['generic_agents'] for next_s in next_state]
                        imagined_next_state = [
                            {
                                'vector_set_map': VectorSetMap.collate(list1),
                                'generic_agents': GenericAgents.collate(list2)
                            }
                            for list1, list2 in zip(list_imagined_vector_set_map, list_imagined_generic_agents)
                        ]
                    next_state = {
                        'vector_set_map': VectorSetMap.collate(list_vector_set_map),
                        'generic_agents': GenericAgents.collate(list_generic_agents)
                    }
                    pred = self.actor_target(next_state)
                    ego_index = next_state['generic_agents'].agent_data['av_index']
                    next_action = self._get_noised_next_action(pred, ego_index)

                    if self.finetune_mode == 'pseudo_ground_truth' and self.num_imagined_experiences is not None:
                        imagined_pred = [self.actor_target(img_next_s) for img_next_s in imagined_next_state]
                        imagined_next_action = [self._get_noised_next_action(img_pred, [ego_idx] * self.num_imagined_experiences)
                                                for img_pred, ego_idx in zip(imagined_pred, ego_index)]

                    # Compute the target Q value
                    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                    target_Q1 = torch.stack(target_Q1)
                    target_Q2 = torch.stack(target_Q2)
                    target_Q = torch.min(target_Q1, target_Q2)
                    if self.finetune_mode == 'pseudo_ground_truth' and self.num_imagined_experiences is not None:
                        imagined_target_Q1Q2 = [self.critic_target(next_s, next_a)
                                                for next_s, next_a in zip(imagined_next_state, imagined_next_action)]
                        imagined_target_Q1 = [torch.stack(Q[0]) for Q in imagined_target_Q1Q2]
                        imagined_target_Q2 = [torch.stack(Q[1]) for Q in imagined_target_Q1Q2]
                        imagined_target_Q = [torch.min(Q1, Q2) for Q1, Q2 in zip(imagined_target_Q1, imagined_target_Q2)]
                        target_Q = [torch.cat([tg_Q.unsqueeze(0), img_tg_Q]) for tg_Q, img_tg_Q in zip(target_Q, imagined_target_Q)]
                        target_Q = [r.squeeze(-1) + not_d * self.discount * Q
                                    for r, not_d, Q in zip(reward, not_done, target_Q)]
                        target_Q = torch.cat(target_Q)
                    else:
                        target_Q = torch.cat([r.squeeze(-1) for r in reward]) + torch.cat(not_done) * self.discount * target_Q

                self.replay_buffer.sampled_state = state
                self.replay_buffer.sampled_unpacked_state = unpacked_state
                self.replay_buffer.sampled_action = action
                self.replay_buffer.sampled_target = target
                self.replay_buffer.sampled_action_samples = action_samples

                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(state, action)
                if self.finetune_mode == 'pseudo_ground_truth' and self.num_imagined_experiences is not None:
                    imagined_actions = [self.env.convert_to_local_cartesian_samples(**samples,
                                                                                    only_return_imagined_action_samples=True)
                                        for samples in action_samples]
                    imagined_actions = [torch.as_tensor(action, device=self.device, dtype=torch.float32) for action in imagined_actions]
                    unpacked_state = self._convert_feature_to_limited_size(unpacked_state)
                    imagined_current_Q1Q2 = [self.critic(s, [action]) for s, action in zip(unpacked_state, imagined_actions)]
                    imagined_current_Q1 = [Q[0][0] for Q in imagined_current_Q1Q2]
                    imagined_current_Q2 = [Q[1][0] for Q in imagined_current_Q1Q2]
                    current_Q1 = torch.cat([torch.cat([Q.unsqueeze(0), img_Q]) for Q, img_Q in zip(current_Q1, imagined_current_Q1)])
                    current_Q2 = torch.cat([torch.cat([Q.unsqueeze(0), img_Q]) for Q, img_Q in zip(current_Q2, imagined_current_Q2)])
                else:
                    current_Q1, current_Q2 = torch.stack(current_Q1), torch.stack(current_Q2)

            if not evaluation:
                if pred is not None:
                    track_token_ids = next_state['generic_agents'].agent_data['id']
                    objects_types = next_state['generic_agents'].agent_data['type']
                    predict_mask = next_state['generic_agents'].agent_data['predict_mask']
                else:
                    track_token_ids = None
                    objects_types = None
                    predict_mask = None

            return {
                'td3': {
                    't': self.t,
                    'start_timesteps': self.start_timesteps,
                    'target_Q': target_Q,
                    'current_Q1': current_Q1,
                    'current_Q2': current_Q2,
                    'targets': next_target,
                },
                'agents_trajectories': AgentsTrajectories(trajectories=pred,
                                                          track_token_ids=track_token_ids,
                                                          objects_types=objects_types,
                                                          predict_mask=predict_mask),
                'evaluation': evaluation,
            }

        elif opt_idx == 1:  # optimizer for actor network
            actor_loss = None
            agents_trajectories = None
            target = None
            pseudo_targets = None
            if self.t >= self.start_timesteps:
                if self.total_it % self.policy_freq == 0:
                    # Compute actor loss
                    state = self.replay_buffer.sampled_state
                    unpacked_state = self.replay_buffer.sampled_unpacked_state
                    action = self.replay_buffer.sampled_action
                    target = self.replay_buffer.sampled_target
                    action_samples = self.replay_buffer.sampled_action_samples
                    pred = self.actor(state)
                    ego_index = state['generic_agents'].agent_data['av_index']
                    if self.finetune_mode == 'pseudo_ground_truth':
                        with torch.no_grad():
                            action_samples = [self.env.convert_to_local_cartesian_samples(**samples) for samples in action_samples]
                            action_samples = [torch.as_tensor(action, device=self.device, dtype=torch.float32) for action in action_samples]
                            action_samples = [torch.cat([a1.unsqueeze(0), a2], dim=0) for a1, a2 in zip(action, action_samples)]
                            try:
                                critics = [self.critic.Q1(s, [a]) for s, a in zip(unpacked_state, action_samples)]
                            except Exception as e:
                                logger.info(e)
                                logger.info(f'action_samples[0].shape[0]={action_samples[0].shape[0]}')
                                raise RuntimeError
                            best_idx = [critic[0].argmax() for critic in critics]
                            pseudo_ground_truth = [action_sample[idx] for action_sample, idx in zip(action_samples, best_idx)]
                            for tg, pseudo_gt in zip(target, pseudo_ground_truth):
                                tg['agents_trajectories'].trajectories[0]['AV'] = pseudo_gt
                                tg['agents_trajectories'].trajectories_global[0]['AV'] = None
                                tg['agents_trajectories'].velocity_global[0]['AV'] = None
                            pseudo_targets = target
                            actor_loss = 'to_be_computed_using_pseudo_ground_truth'
                    else:
                        # # use the most likely mode to compute actor loss
                        # pi = pred['pi']
                        # prob = [F.softmax(p[idx], dim=-1) for p, idx in zip(pi, ego_index)]
                        # ego_most_likely_mode = [pb.argmax() for pb in prob]
                        # ego_most_likely_traj = [traj[idx][mode] for traj, idx, mode in zip(pred['loc_refine_pos'], ego_index, ego_most_likely_mode)]
                        # position = [traj for traj in ego_most_likely_traj]
                        # heading = [torch.atan2(position[1:, 1] - position[0:-1, 1],
                        #                        position[1:, 0] - position[0:-1, 0]) for position in position]
                        # heading = [torch.cat([heading, heading[-1:]], dim=0) for heading in heading]
                        # action = [torch.cat([position, heading.unsqueeze(-1)], dim=-1)
                        #           for position, heading in zip(position, heading)]
                        # actor_loss = -torch.stack(self.critic.Q1(state, action)).mean()

                        # use the best mode (according to critic) to compute actor loss
                        position = [traj[idx] for traj, idx in zip(pred['loc_refine_pos'], ego_index)]
                        heading = [torch.atan2(position[:, 1:, 1] - position[:, 0:-1, 1],
                                               position[:, 1:, 0] - position[:, 0:-1, 0]) for position in position]
                        heading = [torch.cat([heading, heading[:, -1:]], dim=-1) for heading in heading]
                        action = [torch.cat([position, heading.unsqueeze(-1)], dim=-1)
                                  for position, heading in zip(position, heading)]
                        critics = [self.critic.Q1(state, [a[i] for a in action]) for i in range(self.num_modes)]
                        critics = torch.stack([torch.stack(c) for c in critics])
                        critics = critics.transpose(0, 1)
                        critics = critics.max(dim=-1).values
                        actor_loss = -critics.mean()

                    track_token_ids = state['generic_agents'].agent_data['id']
                    objects_types = state['generic_agents'].agent_data['type']
                    predict_mask = state['generic_agents'].agent_data['predict_mask']
                    agents_trajectories = AgentsTrajectories(trajectories=pred,
                                                             track_token_ids=track_token_ids,
                                                             objects_types=objects_types,
                                                             predict_mask=predict_mask)

            return {
                'td3': {
                    't': self.t,
                    'start_timesteps': self.start_timesteps,
                    'actor_loss': actor_loss,
                },
                'agents_trajectories': agents_trajectories,
                'targets': target,
                'pseudo_targets': pseudo_targets,
            }

        elif opt_idx is None:  # for visualization_callback
            with torch.no_grad():
                pred = self.actor(features)

            return {
                'agents_trajectories': AgentsTrajectories(trajectories=pred,
                                                          track_token_ids=features['generic_agents'].agent_data['id'],
                                                          objects_types=features['generic_agents'].agent_data['type'],
                                                          predict_mask=features['generic_agents'].agent_data['predict_mask']),
            }

    def _update_target_networks(self):

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def _filter_targets(self, actor_features: FeaturesType, targets: TargetsType) -> TargetsType:
        batch_size = actor_features['generic_agents'].batch_size
        agent_data = actor_features['generic_agents'].agent_data
        trajectories = [{} for _ in range(batch_size)]
        trajectories_global = [{} for _ in range(batch_size)]
        velocity_global = [{} for _ in range(batch_size)]
        track_token_ids = agent_data['id']
        objects_types = agent_data['type']
        objects_types = [
            [self.rewardformer.agent_types[int(encoding)] for encoding in objects_types[sample_idx]]
            for sample_idx in range(batch_size)
        ]
        objects_types = [['AV'] + types[1:] for types in objects_types]
        predict_mask = [mask.bool() for mask in agent_data['predict_mask']]

        for sample_idx in range(batch_size):
            for id in track_token_ids[sample_idx]:
                trajectories[sample_idx][id] = targets['agents_trajectories'].trajectories[sample_idx][id]
                trajectories_global[sample_idx][id] = targets['agents_trajectories'].trajectories_global[sample_idx][id]
                velocity_global[sample_idx][id] = targets['agents_trajectories'].velocity_global[sample_idx][id]

        return {
            'agents_trajectories': AgentsTrajectories(
                trajectories=trajectories,
                trajectories_global=trajectories_global,
                velocity_global=velocity_global,
                track_token_ids=track_token_ids,
                objects_types=objects_types,
                predict_mask=predict_mask
            )
        }

    def _get_imagined_actions(self, samples):
        num_trajs = 0
        num_left = [shape[0] for shape in samples['traj_left_ori_shape']]
        num_keep = [shape[0] for shape in samples['traj_keep_ori_shape']]
        num_right = [shape[0] for shape in samples['traj_right_ori_shape']]
        idx_range_left, idx_range_keep, idx_range_right = [], [], []
        for shape in samples['traj_left_ori_shape']:
            idx_range_left.append((0, shape[0]))
            num_trajs += shape[0]
        for shape in samples['traj_keep_ori_shape']:
            idx_range_keep.append((0, shape[0]))
            num_trajs += shape[0]
        for shape in samples['traj_right_ori_shape']:
            idx_range_right.append((0, shape[0]))
            num_trajs += shape[0]

        if self.num_imagined_experiences >= 20:
            num_left = [int(np.ceil(num / num_trajs * self.num_imagined_experiences)) for num in num_left]
            num_keep = [int(np.ceil(num / num_trajs * self.num_imagined_experiences)) for num in num_keep]
            num_right = [int(np.ceil(num / num_trajs * self.num_imagined_experiences)) for num in num_right]
            num_imagined_experiences = sum([sum(num_left), sum(num_keep), sum(num_right)])
            for i in range(num_imagined_experiences - self.num_imagined_experiences):
                found = False
                while not found:
                    idx = np.random.randint(0, len(num_keep))
                    if num_keep[idx] > 0:
                        num_keep[idx] -= 1
                        found = True
        else:
            idx_range = [idx_range_left] + [idx_range_keep] + [idx_range_right]
            num_left_keep_right = [[0 for _ in range(len(num_left))]] + \
                                  [[0 for _ in range(len(num_keep))]] + \
                                  [[0 for _ in range(len(num_right))]]
            selected_behavior = np.random.randint(0, len(num_left_keep_right), size=self.num_imagined_experiences)
            for i in selected_behavior:
                if sum([num_left, num_keep, num_right][i]) == 0:
                    continue
                found = False
                while not found:
                    j = np.random.randint(0, len(num_left_keep_right[i]))
                    if (idx_range[i][j][1] > num_left_keep_right[i][j]
                            or all([idx_range[i][k][1] <= num_left_keep_right[i][k] for k in range(len(num_left_keep_right[i]))])):
                        num_left_keep_right[i][j] += 1
                        found = True
            num_left, num_keep, num_right = num_left_keep_right

        idx_left = [
            np.random.randint(idx_range[0], idx_range[1], size=num)
            for num, idx_range in zip(num_left, idx_range_left)
        ]
        idx_keep = [
            np.random.randint(idx_range[0], idx_range[1], size=num)
            for num, idx_range in zip(num_keep, idx_range_keep)
        ]
        idx_right = [
            np.random.randint(idx_range[0], idx_range[1], size=num)
            for num, idx_range in zip(num_right, idx_range_right)
        ]
        samples['imagined_action_idx'] = {
            'left': idx_left,
            'keep': idx_keep,
            'right': idx_right,
        }
        imagined_action_global_cartesian = self.env.convert_to_local_cartesian_samples(
            **samples, only_return_imagined_action_samples=True, return_global=True).transpose(0, 2, 1)
        vel = (imagined_action_global_cartesian[:, 1:, 0:2] - imagined_action_global_cartesian[:, :-1, 0:2]) / self.env.dt
        vel = np.concatenate([vel, vel[:, -1:, :]], axis=1)
        acc = (vel[:, 1:, 0:2] - vel[:, :-1, 0:2]) / self.env.dt
        acc = np.concatenate([acc, acc[:, -1:, :]], axis=1)
        imagined_action_global_cartesian = torch.as_tensor(imagined_action_global_cartesian, dtype=torch.float32, device=self.device)
        imagined_vel_global_cartesian = torch.as_tensor(vel, dtype=torch.float32, device=self.device)
        imagined_acc_global_cartesian = torch.as_tensor(acc, dtype=torch.float32, device=self.device)

        return imagined_action_global_cartesian, imagined_vel_global_cartesian, imagined_acc_global_cartesian

    def _get_policy_noise(self, actions: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Return policy noises
        :param actions: num_trajs * (num_poses, 3) or num_batches * (num_trajs, num_poses, 3)
        :return:
        """
        actions = torch.stack(actions)
        displacements = torch.norm(actions[..., -1, :] - actions[..., 0, :], p=2, dim=-1)
        noise_magnitude = (torch.randn_like(displacements, device=self.device) * self.policy_noise
                           ).clamp(-self.noise_clip, self.noise_clip) * displacements
        noise_orientation = torch.rand_like(displacements, device=self.device) * math.pi * 2
        noise_ratio = torch.linspace(1. / actions.shape[-2], 1., actions.shape[-2], device=self.device)
        for _ in range(len(actions.shape) - 2):
            noise_ratio = noise_ratio.unsqueeze(0)
        noises = torch.stack([(noise_magnitude * noise_orientation.cos()).unsqueeze(-1) * noise_ratio,
                              (noise_magnitude * noise_orientation.sin()).unsqueeze(-1) * noise_ratio], dim=-1)
        noises = [noise for noise in noises]

        return noises

    def _get_noised_next_action(self,
                                pred: Dict[str, Any],
                                ego_index: List[int],
                                noises: Optional[Union[List[torch.Tensor], torch.Tensor]]=None) -> List[torch.Tensor]:
        pi = pred['pi']
        prob = [F.softmax(p[idx], dim=-1) for p, idx in zip(pi, ego_index)]
        ego_most_likely_mode = [pb.argmax() for pb in prob]
        ego_most_likely_traj = [traj[idx][mode] for traj, idx, mode in zip(pred['loc_refine_pos'],
                                                                           ego_index,
                                                                           ego_most_likely_mode)]
        if noises is None:
            noises = self._get_policy_noise(ego_most_likely_traj)
        next_position = [traj + noise for traj, noise in zip(ego_most_likely_traj, noises)]
        next_heading = [torch.atan2(position[1:, 1] - position[0:-1, 1],
                                    position[1:, 0] - position[0:-1, 0]) for position in next_position]
        next_heading = [torch.cat([heading, heading[-1:]], dim=0) for heading in next_heading]
        next_action = [torch.cat([position, heading.unsqueeze(-1)], dim=-1)
                       for position, heading in zip(next_position, next_heading)]

        return next_action

    def _convert_feature_to_limited_size(
            self, list_features: List[FeaturesType]
    ) -> List[FeaturesType]:
        output_list_features = []
        for features in list_features:
            map_data = features['vector_set_map'].map_data
            agent_data = features['generic_agents'].agent_data
            output_agent_data = {
                'num_nodes': [],
                'av_index': [],
                'valid_mask': [],
                'predict_mask': [],
                'id': [],
                'type': [],
                'position': [],
                'heading': [],
                'velocity': [],
                'length': [],
                'width': [],
            }
            output_map_data = {
                'map_polygon': map_data['map_polygon'],
                'map_point': map_data['map_point'],
                ('map_point', 'to', 'map_polygon'): {'edge_index': []},
                ('map_polygon', 'to', 'map_polygon'): {'edge_index': [], 'type': []},
                'num_pl_detail': map_data['num_pl_detail'],
                'num_pl_to_pl_edge_index_detail': map_data['num_pl_to_pl_edge_index_detail']
            }

            for sample_idx in range(features['vector_set_map'].batch_size):
                ego_idx = agent_data['av_index'][sample_idx]
                # map_data
                pt2pl_edge_index = map_data[('map_point', 'to', 'map_polygon')]['edge_index'][sample_idx]
                unique_polygons = torch.unique(pt2pl_edge_index[1, :], return_counts=True)
                num_map_elements = unique_polygons[0].shape[0]
                if num_map_elements > self.imagine_max_elements:
                    positions = map_data['map_polygon']['position'][sample_idx]
                    distances_to_ego = torch.norm(agent_data['position'][sample_idx][ego_idx:ego_idx + 1, -1, :] - positions, p=2, dim=1)
                    distance_threshold = distances_to_ego.sort().values[self.imagine_max_elements]
                    near_mask = distances_to_ego < distance_threshold
                    near_mask = near_mask[unique_polygons[0]]

                    pl2pl_edge_index = map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'][sample_idx]
                    pl2pl_type = map_data[('map_polygon', 'to', 'map_polygon')]['type'][sample_idx]
                    bool1 = torch.any(pl2pl_edge_index[0:1] - unique_polygons[0][near_mask].unsqueeze(-1) == 0., dim=0)
                    bool2 = torch.any(pl2pl_edge_index[1:] - unique_polygons[0][near_mask].unsqueeze(-1) == 0., dim=0)
                    pl2pl_edge_index_tmp1 = pl2pl_edge_index[:, bool1 & bool2]
                    pl2pl_edge_index_tmp2 = pl2pl_edge_index[:, bool2]
                    if pl2pl_edge_index_tmp1.shape[1] > 0:
                        pl2pl_edge_index = pl2pl_edge_index_tmp1
                        pl2pl_type = pl2pl_type[bool1 & bool2]
                    else:
                        pl2pl_edge_index = pl2pl_edge_index_tmp2
                        pl2pl_type = pl2pl_type[bool2]
                        near_mask[pl2pl_edge_index_tmp2[0]] = True

                    repeated_near_mask = near_mask.repeat_interleave(unique_polygons[1])
                    output_map_data[('map_point', 'to', 'map_polygon')]['edge_index'].append(
                        pt2pl_edge_index[:, repeated_near_mask]
                    )
                    output_map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'].append(pl2pl_edge_index)
                    output_map_data[('map_polygon', 'to', 'map_polygon')]['type'].append(pl2pl_type)
                else:
                    output_map_data[('map_point', 'to', 'map_polygon')]['edge_index'].append(
                        pt2pl_edge_index
                    )
                    output_map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'].append(
                        map_data[('map_polygon', 'to', 'map_polygon')]['edge_index'][sample_idx]
                    )
                    output_map_data[('map_polygon', 'to', 'map_polygon')]['type'].append(
                        map_data[('map_polygon', 'to', 'map_polygon')]['type'][sample_idx]
                    )

                # agent_data
                num_vehicles = (agent_data['type'][sample_idx] == self.actor._agent_types.index('VEHICLE')).sum().item()
                if num_vehicles > self.imagine_max_agents['VEHICLE']:
                    positions = agent_data['position'][sample_idx][:, -1, :]
                    distances_to_ego = torch.norm(positions[ego_idx:ego_idx + 1] - positions, p=2, dim=1)
                    distance_threshold = distances_to_ego.sort().values[self.imagine_max_agents['VEHICLE']]
                    near_mask = distances_to_ego <= distance_threshold
                    output_agent_data['num_nodes'].append(sum(near_mask).item())
                    output_agent_data['av_index'].append(ego_idx)
                    output_agent_data['valid_mask'].append(agent_data['valid_mask'][sample_idx][near_mask])
                    output_agent_data['predict_mask'].append(agent_data['predict_mask'][sample_idx][near_mask])
                    output_agent_data['id'].append([id for id, mask in zip(agent_data['id'][sample_idx], near_mask) if mask])
                    output_agent_data['type'].append(agent_data['type'][sample_idx][near_mask])
                    output_agent_data['position'].append(agent_data['position'][sample_idx][near_mask])
                    output_agent_data['heading'].append(agent_data['heading'][sample_idx][near_mask])
                    output_agent_data['velocity'].append(agent_data['velocity'][sample_idx][near_mask])
                    output_agent_data['length'].append(agent_data['length'][sample_idx][near_mask])
                    output_agent_data['width'].append(agent_data['width'][sample_idx][near_mask])
                else:
                    output_agent_data['num_nodes'].append(agent_data['num_nodes'][sample_idx])
                    output_agent_data['av_index'].append(agent_data['av_index'][sample_idx])
                    output_agent_data['valid_mask'].append(agent_data['valid_mask'][sample_idx])
                    output_agent_data['predict_mask'].append(agent_data['predict_mask'][sample_idx])
                    output_agent_data['id'].append(agent_data['id'][sample_idx])
                    output_agent_data['type'].append(agent_data['type'][sample_idx])
                    output_agent_data['position'].append(agent_data['position'][sample_idx])
                    output_agent_data['heading'].append(agent_data['heading'][sample_idx])
                    output_agent_data['velocity'].append(agent_data['velocity'][sample_idx])
                    output_agent_data['length'].append(agent_data['length'][sample_idx])
                    output_agent_data['width'].append(agent_data['width'][sample_idx])

            output_list_features.append(
                {
                    'vector_set_map': VectorSetMap(map_data=output_map_data),
                    'generic_agents': GenericAgents(agent_data=output_agent_data)
                }
            )

        return output_list_features

    def convert_absolute_to_relative_targets(self, targets: TargetsType) -> TargetsType:
        """
        old local frame to new local frame
        """
        trajectories_tg = targets['agents_trajectories'].trajectories
        for sample_idx in range(targets['agents_trajectories'].batch_size):
            if self.env.iteration[sample_idx] > 0:
                trajs = list(trajectories_tg[sample_idx].values())
                trajs = torch.stack(trajs, dim=0)
                anchor_poses = trajs[:, 0, :]
                poses_old_frame = trajs[:, 1:, :].unsqueeze(1)
                trajectories = efficient_absolute_to_relative_poses(anchor_poses, poses_old_frame)
                trajectories = trajectories.squeeze(1)
                for id, traj in zip(trajectories_tg[sample_idx].keys(), trajectories):
                    trajectories_tg[sample_idx][id] = traj