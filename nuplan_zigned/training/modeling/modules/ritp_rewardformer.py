import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, cast, Union, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.nn.utils.rnn import pad_sequence

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex

from nuplan_zigned.training.modeling.modules.ritp_avrl_vector_map_encoder import RewardMapEncoder
from nuplan_zigned.training.modeling.modules.ritp_avrl_vector_agent_encoder import RewardAgentEncoder
from nuplan_zigned.training.modeling.modules.ritp_avrl_vector_reward_encoder import RewardEncoder
from nuplan_zigned.utils.weight_init import weight_init
from nuplan_zigned.training.preprocessing.features.qcmae_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.features.qcmae_generic_agents import GenericAgents

logger = logging.getLogger(__name__)


class RewardFormer(nn.Module):

    def __init__(self,
                 dropout: float,
                 sigma_prior: float,
                 precision_tau: float,
                 num_poses: int,
                 time_horizon: float,
                 input_dim: int,
                 hidden_dim: int,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 vector_set_map_feature_radius: float,
                 max_agents: Dict[str, int],
                 only_ego_attends_map: bool,
                 gated_attention: bool,
                 gate_has_dropout: bool,
                 num_samples: int,
                 lambda_u_r: float,
                 model_dir: str,
                 u_r_stats_dir: str,
                 map_location: str,
                 ) -> None:
        super(RewardFormer, self).__init__()
        self.map_encoder = RewardMapEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            num_poses=num_poses,
            gated_attention=gated_attention,
            gate_has_dropout=gate_has_dropout,
        )
        self.agent_encoder = RewardAgentEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            num_poses=num_poses,
            only_ego_attends_map=only_ego_attends_map,
            gated_attention=gated_attention,
            gate_has_dropout=gate_has_dropout,
        )
        self.reward_encoder = RewardEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_poses=num_poses,
        )
        self.sigma = sigma_prior
        self.tau = precision_tau
        self.num_poses = num_poses
        self.time_horizon = time_horizon
        self.radius = vector_set_map_feature_radius
        self.max_agents = max_agents
        self.only_ego_attends_map = only_ego_attends_map
        self.num_samples = num_samples
        self.lambda_u_r = lambda_u_r

        self._reward_mean = torch.tensor(1., device=torch.device(map_location))
        self._reward_std = torch.tensor(1., device=torch.device(map_location))
        self._u_reward_mean = torch.tensor(1., device=torch.device(map_location))
        self._u_reward_std = torch.tensor(1., device=torch.device(map_location))

        if u_r_stats_dir is not None:
            try:
                u_r_stats = np.load(u_r_stats_dir, allow_pickle=True).item()
                self._reward_mean = u_r_stats['reward_mean'].to(torch.device(map_location))
                self._reward_std = u_r_stats['reward_std'].to(torch.device(map_location))
                self._u_reward_mean = u_r_stats['u_reward_mean'].to(torch.device(map_location))
                self._u_reward_std = u_r_stats['u_reward_std'].to(torch.device(map_location))
                logger.info(f'u_r_stats loaded: {u_r_stats_dir}')

            except:
                logger.info(f'failed to load u_r_stats: {u_r_stats_dir}')

        self.agent_types = ['VEHICLE', 'PEDESTRIAN', 'BICYCLE', 'TRAFFIC_CONE', 'BARRIER', 'CZONE_SIGN', 'GENERIC_OBJECT']

        self.apply(weight_init)

        self.map_encoder.to(device=torch.device(map_location))
        self.agent_encoder.to(device=torch.device(map_location))
        self.reward_encoder.to(device=torch.device(map_location))

        if model_dir is not None:
            trained_model_dict = torch.load(model_dir, map_location=map_location)
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

            logger.info(f'\nRewardFormer loaded: {model_dir}\n')

    def forward(self, features: FeaturesType) -> Dict[str, Any]:

        map_enc = self.map_encoder(features)
        scene_enc = self.agent_encoder(features, map_enc)
        reward = self.reward_encoder(scene_enc)

        return {'reward': reward['reward_enc']}

    def process_features(self,
                         features: FeaturesType,
                         targets: TargetsType,
                         anchor_ego_poses: List[torch.Tensor],
                         ego_velocity: List[torch.Tensor],
                         num_poses_for_eval: int) -> FeaturesType:
        """extract features within radius of interest"""
        batch_size = features['vector_set_map'].batch_size
        num_historical_steps = features['generic_agents'].agent_data['position'][0].shape[1]
        device = features['generic_agents'].agent_data['position'][0].device
        step = anchor_ego_poses[0].shape[1] // self.num_poses
        pose_range = range(step - 1,
                           anchor_ego_poses[0].shape[1],
                           step)
        pose_range = pose_range[0:num_poses_for_eval]
        num_poses = len(pose_range)

        # initialize
        map_data = copy.deepcopy(features['vector_set_map'].map_data)
        agent_data = copy.deepcopy(targets['agents_trajectories'])

        # process map data
        pt_position = map_data['map_point']['position']
        num_pt = map_data['map_point']['num_nodes']
        anchor_position = anchor_ego_poses[0][:, pose_range, 0:2]
        list_anchor_position, list_pt_position = [], []
        for sample_idx in range(batch_size):
            list_anchor_position.append(
                # repeat_interleave: (num_poses, 2) -> (num_poses * num_pt[sample_idx], 2)
                anchor_position[sample_idx].unsqueeze(1).expand(num_poses, num_pt[sample_idx], 2).reshape(-1, 2)
            )
            list_pt_position.append(
                # repeat: (num_pt[sample_idx], 2) -> (num_pt[sample_idx] * num_poses, 2)
                pt_position[sample_idx].unsqueeze(0).expand(num_poses, num_pt[sample_idx], 2).reshape(-1, 2)
            )
        anchor_position = torch.cat(list_anchor_position)
        pt_position = torch.cat(list_pt_position)
        pt_within_radius = torch.norm(pt_position - anchor_position, p=2, dim=1) < self.radius

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
        anchor_pose = anchor_ego_poses[0][:, pose_range]
        trajectories_global = agent_data.trajectories_global
        velocity_global = agent_data.velocity_global
        objects_types = agent_data.objects_types
        predict_mask = agent_data.predict_mask
        num_agents = [len(types) for types in objects_types]
        av_index = [0,]
        [av_index.append(av_index[-1] + num_a) for num_a in num_agents[:batch_size - 1]]
        pose_range2 = range(num_historical_steps + step - 1,
                            predict_mask[0].shape[1],
                            step)
        pose_range2 = pose_range2[0:num_poses_for_eval]

        list_anchor_pose, list_trajectories_global, \
            list_velocity_global, list_objects_types, list_predict_mask= [], [], [], [], []
        for sample_idx in range(batch_size):
            list_anchor_pose.append(
                # repeat_interleave: (num_poses, 3) -> (num_poses * num_agents[sample_idx], 3)
                anchor_pose[sample_idx].unsqueeze(1).expand(num_poses, num_agents[sample_idx], 3).reshape(-1, 3)
            )
            traj = [
                traj[pose_range] if id != 'AV' else anchor_pose[sample_idx]
                for id, traj in trajectories_global[sample_idx].items()
            ]
            list_trajectories_global.append(
                torch.stack(traj).transpose(0, 1).reshape(-1, 3)
            )
            vel = [
                vel[pose_range] if id != 'AV' else ego_velocity[0][sample_idx][pose_range]
                for id, vel in velocity_global[sample_idx].items()
            ]
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
            mask = [mask[pose_range2] for mask in predict_mask[sample_idx]]
            list_predict_mask.append(
                torch.stack(mask).transpose(0, 1).reshape(-1)
            )

        anchor_pose = torch.cat(list_anchor_pose)
        trajectories_global = torch.cat(list_trajectories_global)
        velocity_global = torch.cat(list_velocity_global)
        objects_types = torch.cat(list_objects_types)
        predict_mask = torch.cat(list_predict_mask)

        distance_to_agents = torch.norm(trajectories_global[:, 0:2] - anchor_pose[:, 0:2], p=2, dim=1)
        agents_within_radius = (distance_to_agents < self.radius) & predict_mask
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
            ignore_index = sorted_generic_object_indices[batch_mask][self.max_agents['GENERIC_OBJECT']:]
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

        return {
            'vector_set_map': VectorSetMap(map_data=map_data),
            'generic_agents': GenericAgents(agent_data=agent_data_output)
        }

    @property
    def reward_mean(self):
        return self._reward_mean

    @property
    def reward_std(self):
        return self._reward_std

    @property
    def u_reward_mean(self):
        return self._u_reward_mean

    @property
    def u_reward_std(self):
        return self._u_reward_std