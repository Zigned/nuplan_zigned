from typing import Optional, Dict, Union, Tuple, Any, List
import os
import logging
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from math import sin, cos
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from hydra.utils import instantiate
from scipy.interpolate import interp1d

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan_zigned.training.preprocessing.features.qcmae_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.features.qcmae_generic_agents import GenericAgents
from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.planning.simulation.controller.abstract_controller import AbstractEgoController
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from nuplan_zigned.training.modeling.modules.ritp_rewardformer import RewardFormer
from nuplan_zigned.utils.frenet_frame_object import FrenetFrame
from nuplan_zigned.utils.utils import (
	interpolate_polynomial,
	polynomial,
	polynomial_derivative,
	efficient_absolute_to_relative_poses,
)
from nuplan_zigned.training.preprocessing.feature_builders.avrl_vector_set_map_builder_utils import get_centerline_coords

logger = logging.getLogger(__name__)


class Env:
	def __init__(self,
				 v_min: float,
				 v_max: float,
				 a_min: float,
				 a_max: float,
				 lane_width: float,
				 mu: float,
				 g: float,
				 T: float,
				 num_poses: int,
				 ego_controller_config: Dict,
				 num_rl_workers: int,
				 # device=torch.device('cpu')
				 ):
		self.v_min = v_min
		self.v_max = v_max
		self.a_min = a_min
		self.a_max = a_max
		self.lane_width = lane_width
		self.mu = mu
		self.g = g
		self.T = T  # time horizon for planning
		self.num_poses = num_poses  # num of poses for planning
		self.ego_controller_config = ego_controller_config
		self.num_rl_workers = num_rl_workers

		# if isinstance(device, str):
		# 	self.parallel_device_ids = device_parser.parse_gpu_ids(device)
		# 	device = torch.device(f'cuda:{self.parallel_device_ids[0]}')
		# self.device = device

		self.dt = T / num_poses
		self.t = np.linspace(0., T, num=num_poses + 1)

		self.scenario: Optional[List[AbstractScenario]] = None  # current scenario
		self.ego_controller: Optional[List[AbstractEgoController]] = None
		self.imagined_ego_controller: Optional[List[List[AbstractEgoController]]] = None
		self.iteration: Optional[List[int]] = None  # current iteration
		self.num_of_iterations: Optional[List[int]] = None  # number of iterations (including initial iteration)
		self.time_point: Optional[List[TimePoint]] = None  # current time point
		self.time_points: Optional[List[List[TimePoint]]] = None  # all time points of a scenario

		# MDP variables to be added into replay buffer
		self.previous_state = None
		self.previous_action = None
		self.previous_action_samples = None
		self.previous_target = None
		self.previous_reward = None
		self.previous_done = None
		self.state = None
		self.action = None
		self.action_samples = None
		self.reward = None
		self.reward_log = None  # to be logged
		self.next_state = None
		self.done = [False] * num_rl_workers
		self.target = None
		self.episode_end = [False] * num_rl_workers

		# ego history in global frame
		self.ego_historical_position = [None] * num_rl_workers  # position of ego center
		self.ego_historical_heading = [None] * num_rl_workers
		self.ego_historical_velocity = [None] * num_rl_workers

		self.detect_radius = 20.
		self.offroad_distance = 3.

		self._traffic_light_statuses = ['GREEN', 'YELLOW', 'RED', 'UNKNOWN']

	def on_scenario_start(self, scenarios: List[AbstractScenario]) -> None:
		self.ego_controller = [
			instantiate(self.ego_controller_config, scenario=scenario)
			for scenario in scenarios
		]
		self.scenario = scenarios
		self.num_of_iterations = [
			scenario.get_number_of_iterations() for scenario in scenarios
		]
		self.time_points = [
			[scenario.start_time] + list(
				scenario.get_future_timestamps(
					iteration=0, time_horizon=scenario.duration_s.time_s, num_samples=num_of_iterations - 1)
			)
			for num_of_iterations, scenario in zip(self.num_of_iterations, scenarios)
		]
		self.time_point = [
			time_points[iteration]
			for time_points, iteration in zip(self.time_points, self.iteration)
		]

	def sample_action(self, frenet_frame: FrenetFrame, anchor_ego_state: EgoState) -> Dict[str, np.ndarray]:
		"""Sample action in Frenet frame"""
		action = (np.random.rand(2,) - 0.5) * 2  # [-1, 1]
		vs_target = 0.5 * (action[0] + 1.) * self.v_max  # [0, v_max]
		v_min = max(self.v_min, anchor_ego_state.dynamic_car_state.speed + self.a_min * self.T)
		v_max = min(self.v_max, anchor_ego_state.dynamic_car_state.speed + self.a_max * self.T)
		vs_target = np.reshape(max(v_min, vs_target), (-1,))
		vs_target = np.reshape(min(v_max, vs_target), (-1,))
		l_target = np.reshape((3 / 2 * self.lane_width) * action[1], (-1,))  # [- 3/2 * lane_width, + 3/2 * lane_width]

		current_ego_station = frenet_frame.get_nearest_station_from_position(anchor_ego_state.rear_axle.point)
		reference_line_pose = frenet_frame.get_nearest_pose_from_position(anchor_ego_state.rear_axle.point)
		current_ego_lateral = frenet_frame.get_lateral_from_position(anchor_ego_state.rear_axle.point)
		cos_delta_theta = np.cos(anchor_ego_state.rear_axle.heading - reference_line_pose.heading)  # cos(heading_in_frenet)
		sin_delta_theta = np.sin(anchor_ego_state.rear_axle.heading - reference_line_pose.heading)
		ego_velocity_local = anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d
		ego_acceleration_local = anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d
		if ego_velocity_local.x * cos_delta_theta + ego_velocity_local.y * sin_delta_theta < 0.5 \
				and ego_acceleration_local.x * cos_delta_theta + ego_acceleration_local.y * sin_delta_theta < 0.:
			coefficient = 0.
		else:
			coefficient = 1.

		# 4th-order Polynomial s(t)
		P_s = interpolate_polynomial(
			deg=4,
			x_1=self.T,
			y_0=current_ego_station * np.ones_like(vs_target),
			y_prime_0=(ego_velocity_local.x * cos_delta_theta + ego_velocity_local.y * sin_delta_theta) * np.ones_like(vs_target),
			y_pprime_0=coefficient * (ego_acceleration_local.x * cos_delta_theta + ego_acceleration_local.y * sin_delta_theta) * np.ones_like(vs_target),
			y_prime_1=vs_target,
			y_pprime_1=np.zeros_like(vs_target)
		)
		# 5th-order Polynomial l(t)
		P_l = interpolate_polynomial(
			deg=5,
			x_1=self.T,
			y_0=current_ego_lateral * np.ones_like(l_target),
			y_prime_0=(ego_velocity_local.x * sin_delta_theta + ego_velocity_local.y * cos_delta_theta) * np.ones_like(l_target),
			y_pprime_0=(ego_acceleration_local.x * sin_delta_theta + ego_acceleration_local.y * cos_delta_theta) * np.ones_like(l_target),
			y_1=l_target,
			y_prime_1=np.zeros_like(l_target),
			y_pprime_1=np.zeros_like(l_target)
		)

		s_t = polynomial(self.t, P_s.A)
		l_t = polynomial(self.t, P_l.A)
		vs_t = polynomial_derivative(self.t, P_s.A, order=1)
		vl_t = polynomial_derivative(self.t, P_l.A, order=1)
		as_t = polynomial_derivative(self.t, P_s.A, order=2)
		al_t = polynomial_derivative(self.t, P_l.A, order=2)

		trajs = np.zeros((s_t.shape[0] * l_t.shape[0], 7, self.num_poses + 1))
		for i in range(s_t.shape[0]):
			for j in range(l_t.shape[0]):
				theta_t = np.arctan2(
					polynomial_derivative(self.t, P_l[:, j].A, order=1),
					polynomial_derivative(self.t, P_s[:, i].A, order=1)
				)
				trajs[i * l_t.shape[0] + j, 0, :] = s_t[i]
				trajs[i * l_t.shape[0] + j, 1, :] = l_t[j]
				trajs[i * l_t.shape[0] + j, 2, :] = theta_t
				trajs[i * l_t.shape[0] + j, 3, :] = vs_t[i]
				trajs[i * l_t.shape[0] + j, 4, :] = vl_t[j]
				trajs[i * l_t.shape[0] + j, 5, :] = as_t[i]
				trajs[i * l_t.shape[0] + j, 6, :] = al_t[j]

		# fix initial heading
		trajs[:, 2, 0] = anchor_ego_state.rear_axle.heading - reference_line_pose.heading

		return {
			't': self.t,
			'poses_frenet': trajs[:, 0:3],
			'vs_frenet': trajs[:, 3],
			'vl_frenet': trajs[:, 4],
			'as_frenet': trajs[:, 5],
			'al_frenet': trajs[:, 6],
		}

	def sample_action_densely(
			self,
			frenet_frame: FrenetFrame,
			reference_lanes: List[Union[NuPlanLane, NuPlanLaneConnector]],
			anchor_ego_state: EgoState
	) -> Dict[str, Any]:
		"""
		Sample action densely. Used to generate pseudo ground truth.
		For lane keeping:
		0|--------------------|t1|--------------------|t2|--------------------|T
		t1∈[2, T-2], t2∈[2, T-2], t1<t2
		For lane changing:
		0|----------------------------------------|t1|------------------------|T
		t1∈[2, T]
		"""
		current_ego_station = frenet_frame.get_nearest_station_from_position(anchor_ego_state.center.point)
		reference_line_pose = frenet_frame.get_nearest_pose_from_position(anchor_ego_state.center.point)
		current_ego_lateral = frenet_frame.get_lateral_from_position(anchor_ego_state.center.point)
		current_ego_heading = anchor_ego_state.center.heading - reference_line_pose.heading  # heading_in_frenet
		ego_velocity_local = anchor_ego_state.dynamic_car_state.center_velocity_2d
		ego_acceleration_local = anchor_ego_state.dynamic_car_state.center_acceleration_2d

		v_min = max(self.v_min, anchor_ego_state.dynamic_car_state.speed + self.a_min * self.T)
		v_max = min(self.v_max, anchor_ego_state.dynamic_car_state.speed + self.a_max * self.T)
		v_max_law = max([lane.speed_limit_mps if lane.speed_limit_mps is not None else v_max for lane in reference_lanes])
		v_max = min(v_max, v_max_law)
		vs_target = np.arange(v_min, v_max, step=4)
		if v_max_law - vs_target[-1] > 1.:
			vs_target = np.hstack((vs_target, [v_max_law]))
		l_left = np.linspace(3 * self.lane_width / 2,  self.lane_width / 2, num=11)
		l_mid = np.linspace(self.lane_width / 2,  - self.lane_width / 2, num=11)
		l_right = np.linspace(- self.lane_width / 2,  - 3 * self.lane_width / 2, num=11)

		# LANE CHANGING: LEFT
		traj_left, traj_left_dedup = self.get_lane_changing_samples(
			current_ego_station,
			current_ego_lateral,
			current_ego_heading,
			anchor_ego_state.dynamic_car_state.speed,
			vs_target,
			l_left,
			ego_velocity_local,
			ego_acceleration_local,
		)
		# LANE KEEPING
		traj_keep, traj_keep_dedup = self.get_lane_keeping_samples(
			current_ego_station,
			current_ego_lateral,
			current_ego_heading,
			anchor_ego_state.dynamic_car_state.speed,
			vs_target,
			l_mid,
			ego_velocity_local,
			ego_acceleration_local,
		)
		# LANE CHANGING: RIGHT
		traj_right, traj_right_dedup = self.get_lane_changing_samples(
			current_ego_station,
			current_ego_lateral,
			current_ego_heading,
			anchor_ego_state.dynamic_car_state.speed,
			vs_target,
			l_right,
			ego_velocity_local,
			ego_acceleration_local,
		)

		traj_left_ori_shape = [s.shape for s in traj_left['s_t']]
		traj_keep_ori_shape = [s.shape for s in traj_keep['s_t']]
		traj_right_ori_shape = [s.shape for s in traj_right['s_t']]

		return {
			'anchor_ego_state': anchor_ego_state,
			'frenet_frame': frenet_frame,
			'traj_left_dedup': traj_left_dedup,
			'traj_keep_dedup': traj_keep_dedup,
			'traj_right_dedup': traj_right_dedup,
			'traj_left_ori_shape': traj_left_ori_shape,
			'traj_keep_ori_shape': traj_keep_ori_shape,
			'traj_right_ori_shape': traj_right_ori_shape,
		}

	def convert_to_local_cartesian_samples(
			self,
			anchor_ego_state: EgoState,
			frenet_frame: FrenetFrame,
			traj_left_dedup: Dict[str, Any],
			traj_keep_dedup: Dict[str, Any],
			traj_right_dedup: Dict[str, Any],
			traj_left_ori_shape: List[Tuple[int, int]],
			traj_keep_ori_shape: List[Tuple[int, int]],
			traj_right_ori_shape: List[Tuple[int, int]],
			imagined_action_idx: Optional[Dict[str, np.ndarray]],
			only_return_imagined_action_samples: bool=False,
			return_global: bool=False,
			device: torch.device=torch.device('cpu'),
	) -> np.ndarray:
		# interpolate points using stations
		vectorized_method_x = np.vectorize(lambda pt: pt.x)
		vectorized_method_y = np.vectorize(lambda pt: pt.y)
		traj_left_pt1 = {
			's_t_0': [frenet_frame.reference_line.interpolate(s) for s in traj_left_dedup['s_t_0']],
			's_t_1': [frenet_frame.reference_line.interpolate(s) for s in traj_left_dedup['s_t_1']],
		}
		traj_left_xy1 = {
			's_t_0': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_left_pt1['s_t_0']],
			's_t_1': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_left_pt1['s_t_1']],
		}
		traj_left_pt1, traj_left_xy1 = self.duplicate_points(traj_left_ori_shape, traj_left_dedup, traj_left_pt1, traj_left_xy1)
		traj_left_pt2 = {
			's_t_0': [frenet_frame.reference_line.interpolate(s + frenet_frame.distance_for_heading_estimation) for s in traj_left_dedup['s_t_0']],
			's_t_1': [frenet_frame.reference_line.interpolate(s + frenet_frame.distance_for_heading_estimation) for s in traj_left_dedup['s_t_1']],
		}
		traj_left_xy2 = {
			's_t_0': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_left_pt2['s_t_0']],
			's_t_1': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_left_pt2['s_t_1']],
		}
		traj_left_pt2, traj_left_xy2 = self.duplicate_points(traj_left_ori_shape, traj_left_dedup, traj_left_pt2, traj_left_xy2)

		traj_keep_pt1 = {
			's_t_0': [frenet_frame.reference_line.interpolate(s) for s in traj_keep_dedup['s_t_0']],
			's_t_1': [frenet_frame.reference_line.interpolate(s) for s in traj_keep_dedup['s_t_1']],
			's_t_2': [frenet_frame.reference_line.interpolate(s) for s in traj_keep_dedup['s_t_2']],
		}
		traj_keep_xy1 = {
			's_t_0': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_keep_pt1['s_t_0']],
			's_t_1': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_keep_pt1['s_t_1']],
			's_t_2': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_keep_pt1['s_t_2']],
		}
		traj_keep_pt1, traj_keep_xy1 = self.duplicate_points(traj_keep_ori_shape, traj_keep_dedup, traj_keep_pt1, traj_keep_xy1)
		traj_keep_pt2 = {
			's_t_0': [frenet_frame.reference_line.interpolate(s + frenet_frame.distance_for_heading_estimation) for s in traj_keep_dedup['s_t_0']],
			's_t_1': [frenet_frame.reference_line.interpolate(s + frenet_frame.distance_for_heading_estimation) for s in traj_keep_dedup['s_t_1']],
			's_t_2': [frenet_frame.reference_line.interpolate(s + frenet_frame.distance_for_heading_estimation) for s in traj_keep_dedup['s_t_2']],
		}
		traj_keep_xy2 = {
			's_t_0': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_keep_pt2['s_t_0']],
			's_t_1': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_keep_pt2['s_t_1']],
			's_t_2': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_keep_pt2['s_t_2']],
		}
		traj_keep_pt2, traj_keep_xy2 = self.duplicate_points(traj_keep_ori_shape, traj_keep_dedup, traj_keep_pt2, traj_keep_xy2)

		traj_right_pt1 = {
			's_t_0': [frenet_frame.reference_line.interpolate(s) for s in traj_right_dedup['s_t_0']],
			's_t_1': [frenet_frame.reference_line.interpolate(s) for s in traj_right_dedup['s_t_1']],
		}
		traj_right_xy1 = {
			's_t_0': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_right_pt1['s_t_0']],
			's_t_1': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_right_pt1['s_t_1']],
		}
		traj_right_pt1, traj_right_xy1 = self.duplicate_points(traj_right_ori_shape, traj_right_dedup, traj_right_pt1, traj_right_xy1)
		traj_right_pt2 = {
			's_t_0': [frenet_frame.reference_line.interpolate(s + frenet_frame.distance_for_heading_estimation) for s in traj_right_dedup['s_t_0']],
			's_t_1': [frenet_frame.reference_line.interpolate(s + frenet_frame.distance_for_heading_estimation) for s in traj_right_dedup['s_t_1']],
		}
		traj_right_xy2 = {
			's_t_0': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_right_pt2['s_t_0']],
			's_t_1': [[vectorized_method_x(point), vectorized_method_y(point)] if len(point) > 0 else [point, point]
					  for point in traj_right_pt2['s_t_1']],
		}
		traj_right_pt2, traj_right_xy2 = self.duplicate_points(traj_right_ori_shape, traj_right_dedup, traj_right_pt2, traj_right_xy2)

		# revert to original trajectories
		traj_left = self.duplicate_trajs(traj_left_ori_shape, traj_left_dedup)
		traj_keep = self.duplicate_trajs(traj_keep_ori_shape, traj_keep_dedup)
		traj_right = self.duplicate_trajs(traj_right_ori_shape, traj_right_dedup)

		# keep only imagined action samples
		if only_return_imagined_action_samples:
			assert imagined_action_idx is not None
			for i in range(len(imagined_action_idx['left'])):
				idx = imagined_action_idx['left'][i]
				traj_left_pt1[i] = traj_left_pt1[i][idx]
				traj_left_pt2[i] = traj_left_pt2[i][idx]
				traj_left_xy1[i] = traj_left_xy1[i][idx]
				traj_left_xy2[i] = traj_left_xy2[i][idx]
				traj_left['s_t'][i] = traj_left['s_t'][i][idx]
				traj_left['l_t'][i] = traj_left['l_t'][i][idx]
			for i in range(len(imagined_action_idx['keep'])):
				idx = imagined_action_idx['keep'][i]
				traj_keep_pt1[i] = traj_keep_pt1[i][idx]
				traj_keep_pt2[i] = traj_keep_pt2[i][idx]
				traj_keep_xy1[i] = traj_keep_xy1[i][idx]
				traj_keep_xy2[i] = traj_keep_xy2[i][idx]
				traj_keep['s_t'][i] = traj_keep['s_t'][i][idx]
				traj_keep['l_t'][i] = traj_keep['l_t'][i][idx]
			for i in range(len(imagined_action_idx['right'])):
				idx = imagined_action_idx['right'][i]
				traj_right_pt1[i] = traj_right_pt1[i][idx]
				traj_right_pt2[i] = traj_right_pt2[i][idx]
				traj_right_xy1[i] = traj_right_xy1[i][idx]
				traj_right_xy2[i] = traj_right_xy2[i][idx]
				traj_right['s_t'][i] = traj_right['s_t'][i][idx]
				traj_right['l_t'][i] = traj_right['l_t'][i][idx]

		# wrap deduplicated trajectories
		trajs = {
			key: np.vstack((np.vstack(traj_left[key]), np.vstack(traj_keep[key]), np.vstack(traj_right[key])))
			for key in traj_keep.keys()
		}
		poses = np.stack((trajs['s_t'], trajs['l_t'], np.zeros_like(trajs['l_t'])), axis=1)
		trajs_pt1 = np.vstack((np.vstack(traj_left_pt1), np.vstack(traj_keep_pt1), np.vstack(traj_right_pt1)))
		trajs_pt2 = np.vstack((np.vstack(traj_left_pt2), np.vstack(traj_keep_pt2), np.vstack(traj_right_pt2)))
		trajs_xy1 = np.vstack((np.vstack(traj_left_xy1), np.vstack(traj_keep_xy1), np.vstack(traj_right_xy1)))
		trajs_xy2 = np.vstack((np.vstack(traj_left_xy2), np.vstack(traj_keep_xy2), np.vstack(traj_right_xy2)))

		# convert to global frame (geometric center)
		trajs_cartesian = frenet_frame.frenet_to_cartesian(
			pose=poses,
			pt1=trajs_pt1,
			pt2=trajs_pt2,
			xy1=trajs_xy1,
			xy2=trajs_xy2,
			compute_heading=False
		)

		# estimate heading
		heading = np.arctan2(np.diff(trajs_cartesian['pose_cartesian'][:, 1, :]),
							 np.diff(trajs_cartesian['pose_cartesian'][:, 0, :]))
		heading[heading == 0.] = anchor_ego_state.center.heading
		trajs_cartesian['pose_cartesian'][:, 2, :-1] = heading
		trajs_cartesian['pose_cartesian'][:, 2, -1] = heading[:, -1]

		if return_global:
			return trajs_cartesian['pose_cartesian'][:, :, 1:]

		# global cartesian to local cartesian
		current_pose = np.array([anchor_ego_state.center.serialize()])
		absolute_poses = trajs_cartesian['pose_cartesian'].transpose(0, 2, 1)[:, np.newaxis, 1:, :]
		# relative_poses = efficient_absolute_to_relative_poses(current_pose, absolute_poses).squeeze()
		current_pose_tensor = torch.as_tensor(current_pose, dtype=torch.float64, device=device)
		absolute_poses_tensor = torch.as_tensor(absolute_poses, dtype=torch.float64, device=device)
		relative_poses_tensor = efficient_absolute_to_relative_poses(current_pose_tensor, absolute_poses_tensor).squeeze()

		samples = relative_poses_tensor.float()
		
		return samples

	def get_lane_keeping_samples(
			self,
			current_ego_station,
			current_ego_lateral,
			current_ego_heading,
			current_ego_speed,
			vs_target,
			l_target,
			ego_velocity_local,
			ego_acceleration_local,
	) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
		t1 = np.around(self.t[20:-20:10], decimals=1)
		t2 = np.around(self.t[20:-20:10], decimals=1)
		t1_meshgrid, t2_meshgrid = np.meshgrid(t1, t2)
		t1_idx_meshgrid, t2_idx_meshgrid = np.meshgrid(np.arange(0, t1.shape[0]), np.arange(0, t2.shape[0]))
		valid_mask = t2_meshgrid > t1_meshgrid
		t_seg_0 = [np.linspace(self.t[0], T, num=int(np.around(T / self.dt)) + 1) for T in t1]
		t_seg_1 = [np.linspace(start, end, num=int(np.around((end - start) / self.dt)) + 1)
				   for start, end in zip(t1_meshgrid[valid_mask], t2_meshgrid[valid_mask])]
		t_seg_2 = [np.linspace(start, self.T, num=int(np.around((self.T - start) / self.dt)) + 1)
				   for start in t2_meshgrid[valid_mask]]
		# 4th-order Polynomial s(t): segment 0
		cos_heading = np.cos(current_ego_heading)
		sin_heading = np.sin(current_ego_heading)
		if ego_velocity_local.x * cos_heading + ego_velocity_local.y * sin_heading < 0.5 \
				and ego_acceleration_local.x * cos_heading + ego_acceleration_local.y * sin_heading < 0.:
			coefficient = 0.
		else:
			coefficient = 1.
		P_s_0 = interpolate_polynomial(
			deg=4,
			x_0=np.zeros_like(t1),
			x_1=t1,
			y_0=current_ego_station * np.ones_like(vs_target),
			y_prime_0=(ego_velocity_local.x * cos_heading + ego_velocity_local.y * sin_heading) * np.ones_like(vs_target),
			y_pprime_0=coefficient * (ego_acceleration_local.x * cos_heading + ego_acceleration_local.y * sin_heading) * np.ones_like(vs_target),
			y_prime_1=vs_target,
			y_pprime_1=np.zeros_like(vs_target)
		)
		# 5th-order Polynomial l(t): segment 0
		P_l_0 = interpolate_polynomial(
			deg=5,
			x_0=np.zeros_like(t1),
			x_1=t1,
			y_0=current_ego_lateral * np.ones_like(l_target),
			y_prime_0=(ego_velocity_local.x * sin_heading + ego_velocity_local.y * cos_heading) * np.ones_like(l_target),
			y_pprime_0=(ego_acceleration_local.x * sin_heading + ego_acceleration_local.y * cos_heading) * np.ones_like(l_target),
			y_1=l_target,
			y_prime_1=np.zeros_like(l_target),
			y_pprime_1=np.zeros_like(l_target)
		)
		s_t_0 = [polynomial(t_seg, P_s.A) for t_seg, P_s in zip(t_seg_0, P_s_0)]
		vs_t_0 = [polynomial_derivative(t_seg, P_s.A, order=1) for t_seg, P_s in zip(t_seg_0, P_s_0)]
		as_t_0 = [polynomial_derivative(t_seg, P_s.A, order=2) for t_seg, P_s in zip(t_seg_0, P_s_0)]
		l_t_0 = [polynomial(t_seg, P_l.A) for t_seg, P_l in zip(t_seg_0, P_l_0)]
		vl_t_0 = [polynomial_derivative(t_seg, P_l.A, order=1) for t_seg, P_l in zip(t_seg_0, P_l_0)]
		al_t_0 = [polynomial_derivative(t_seg, P_l.A, order=2) for t_seg, P_l in zip(t_seg_0, P_l_0)]

		s_t_0 = [s_t.repeat(l_target.shape[0], axis=0) for s_t in s_t_0]
		vs_t_0 = [vs_t.repeat(l_target.shape[0], axis=0) for vs_t in vs_t_0]
		as_t_0 = [as_t.repeat(l_target.shape[0], axis=0) for as_t in as_t_0]
		l_t_0 = [np.tile(l_t, reps=(vs_target.shape[0], 1)) for l_t in l_t_0]
		vl_t_0 = [np.tile(vl_t, reps=(vs_target.shape[0], 1)) for vl_t in vl_t_0]
		al_t_0 = [np.tile(al_t, reps=(vs_target.shape[0], 1)) for al_t in al_t_0]

		feasible_mask = [
			np.all((np.sqrt(as_t ** 2 + al_t ** 2) <= self.a_max) & (np.abs(al_t) < self.a_max), axis=1)
			for as_t, al_t in zip(as_t_0, al_t_0)
		]

		# handle trajectories with small inital and target speeds
		current_speed_is_small = current_ego_speed < 0.5
		target_speed_is_small = vs_target < 0.1
		slow_mask = [(current_speed_is_small & target_speed_is_small).repeat(l_target.shape[0]) for _ in range(len(s_t_0))]
		not_slow_mask = [~mask for mask in slow_mask]
		for mask in slow_mask:
			mask[:-1] = mask[:-1] & (~mask[1:])
		for i in range(len(s_t_0)):
			if any(slow_mask[i]):
				s_t_0[i][slow_mask[i]] = s_t_0[i][slow_mask[i]][:, 0]
				vs_t_0[i][slow_mask[i]] = 0.
				as_t_0[i][slow_mask[i]] = 0.
				l_t_0[i][slow_mask[i]] = l_t_0[i][slow_mask[i]][:, 0]
				vl_t_0[i][slow_mask[i]] = 0.
				al_t_0[i][slow_mask[i]] = 0.
		# remove trajectories with invalid initial heading
		initial_heading = [np.arctan2(np.diff(l_t[:, [0, 5]]), np.diff(s_t[:, [0, 5]])).squeeze() for s_t, l_t in zip(s_t_0, l_t_0)]
		valid_heading_mask = [np.abs(h - current_ego_heading) < 0.2 for h in initial_heading]
		feasible_mask = [fmask & (nsmask | smask) & hmask
						 for fmask, nsmask, smask, hmask in zip(feasible_mask,
																not_slow_mask,
																slow_mask,
																valid_heading_mask)]

		# assume stopping near lane centerline (to reduce computation)
		idx_center = len(l_target) // 2
		l_target_ub = np.round(l_target[idx_center-1], decimals=3) + 0.01
		l_target_lb = np.round(l_target[idx_center+1], decimals=3) - 0.01
		mask = [((l_t[:, -1] > l_target_ub) | (l_t[:, -1] < l_target_lb)) & (vs_t[:, -1] < 0.1)
				for l_t, vs_t in zip(l_t_0, vs_t_0)]
		feasible_mask = [fmask & ~m for fmask, m in zip(feasible_mask, mask)]

		s_t_0 = [s_t[mask] for s_t, mask in zip(s_t_0, feasible_mask)]
		vs_t_0 = [vs_t[mask] for vs_t, mask in zip(vs_t_0, feasible_mask)]
		as_t_0 = [as_t[mask] for as_t, mask in zip(as_t_0, feasible_mask)]
		l_t_0 = [l_t[mask] for l_t, mask in zip(l_t_0, feasible_mask)]
		vl_t_0 = [vl_t[mask] for vl_t, mask in zip(vl_t_0, feasible_mask)]
		al_t_0 = [al_t[mask] for al_t, mask in zip(al_t_0, feasible_mask)]

		# 4th-order Polynomial s(t): segment 1
		P_s_1 = interpolate_polynomial(
			deg=4,
			x_0=t1_meshgrid[valid_mask],
			x_1=t2_meshgrid[valid_mask],
			y_0=[s_t_0[idx][:, -1] for idx in t1_idx_meshgrid[valid_mask]],
			y_prime_0=[vs_t_0[idx][:, -1] for idx in t1_idx_meshgrid[valid_mask]],
			y_pprime_0=[as_t_0[idx][:, -1] for idx in t1_idx_meshgrid[valid_mask]],
			y_prime_1=vs_target,
			y_pprime_1=np.zeros_like(vs_target)
		)
		# 5th-order Polynomial l(t): segment 1
		P_l_1 = interpolate_polynomial(
			deg=5,
			x_0=t1_meshgrid[valid_mask],
			x_1=t2_meshgrid[valid_mask],
			y_0=[l_t_0[idx][:, -1] for idx in t1_idx_meshgrid[valid_mask]],
			y_prime_0=[vl_t_0[idx][:, -1] for idx in t1_idx_meshgrid[valid_mask]],
			y_pprime_0=[al_t_0[idx][:, -1] for idx in t1_idx_meshgrid[valid_mask]],
			y_1=l_target,
			y_prime_1=np.zeros_like(l_target),
			y_pprime_1=np.zeros_like(l_target)
		)
		s_t_1 = [polynomial(t_seg, P_s.A) for t_seg, P_s in zip(t_seg_1, P_s_1)]
		vs_t_1 = [polynomial_derivative(t_seg, P_s.A, order=1) for t_seg, P_s in zip(t_seg_1, P_s_1)]
		as_t_1 = [polynomial_derivative(t_seg, P_s.A, order=2) for t_seg, P_s in zip(t_seg_1, P_s_1)]
		l_t_1 = [polynomial(t_seg, P_l.A) for t_seg, P_l in zip(t_seg_1, P_l_1)]
		vl_t_1 = [polynomial_derivative(t_seg, P_l.A, order=1) for t_seg, P_l in zip(t_seg_1, P_l_1)]
		al_t_1 = [polynomial_derivative(t_seg, P_l.A, order=2) for t_seg, P_l in zip(t_seg_1, P_l_1)]

		s_t_0 = [s_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for s_t in s_t_0]
		vs_t_0 = [vs_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for vs_t in vs_t_0]
		as_t_0 = [as_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for as_t in as_t_0]
		l_t_0 = [l_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for l_t in l_t_0]
		vl_t_0 = [vl_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for vl_t in vl_t_0]
		al_t_0 = [al_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for al_t in al_t_0]

		s_t_0 = [s_t_0[idx] for idx in t1_idx_meshgrid[valid_mask]]
		vs_t_0 = [vs_t_0[idx] for idx in t1_idx_meshgrid[valid_mask]]
		as_t_0 = [as_t_0[idx] for idx in t1_idx_meshgrid[valid_mask]]
		l_t_0 = [l_t_0[idx] for idx in t1_idx_meshgrid[valid_mask]]
		vl_t_0 = [vl_t_0[idx] for idx in t1_idx_meshgrid[valid_mask]]
		al_t_0 = [al_t_0[idx] for idx in t1_idx_meshgrid[valid_mask]]

		s_t_1 = [s_t.repeat(l_target.shape[0], axis=0) for s_t in s_t_1]
		vs_t_1 = [vs_t.repeat(l_target.shape[0], axis=0) for vs_t in vs_t_1]
		as_t_1 = [as_t.repeat(l_target.shape[0], axis=0) for as_t in as_t_1]
		l_t_1 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(l_t, l_t.shape[0] / l_target.shape[0])
			]) if len(l_t) > 0 else l_t
			for l_t in l_t_1
		]
		vl_t_1 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(vl_t, vl_t.shape[0] / l_target.shape[0])
			]) if len(vl_t) > 0 else vl_t
			for vl_t in vl_t_1
		]
		al_t_1 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(al_t, al_t.shape[0] / l_target.shape[0])
			]) if len(al_t) > 0 else al_t
			for al_t in al_t_1
		]

		# handle trajectories with small inital and target speeds or with invalid acceleration
		initial_speed_is_small = [vs[:, -1] < 0.5 for vs in vs_t_0]
		target_speed_is_small = [vs[:, -1] < 0.1 for vs in vs_t_1]
		slow_mask = [v0_is_small & v1_is_small for v0_is_small, v1_is_small in zip(initial_speed_is_small, target_speed_is_small)]
		not_slow_mask = [~mask for mask in slow_mask]
		for mask in slow_mask:
			mask[:-1] = mask[:-1] & (~mask[1:])
		for i in range(len(s_t_0)):
			s_t_1[i][slow_mask[i]] = s_t_1[i][slow_mask[i]][:, 0].reshape(-1, 1)
			vs_t_1[i][slow_mask[i]] = 0.
			as_t_1[i][slow_mask[i]] = 0.
			l_t_1[i][slow_mask[i]] = l_t_1[i][slow_mask[i]][:, 0].reshape(-1, 1)
			vl_t_1[i][slow_mask[i]] = 0.
			al_t_1[i][slow_mask[i]] = 0.
		# remove trajectories with invalid initial heading
		initial_heading = [np.arctan2(np.diff(l_t[:, [0, 5]]), np.diff(s_t[:, [0, 5]])).squeeze() for s_t, l_t in zip(s_t_1, l_t_1)]
		valid_heading_mask = [np.abs(h) < 0.2 for h in initial_heading]
		feasible_mask = [(nsmask | smask) & hmask
						 for nsmask, smask, hmask in zip(not_slow_mask,
														 slow_mask,
														 valid_heading_mask)]
		feasible_mask = [
			fmask & np.all((np.sqrt(as_t ** 2 + al_t ** 2) <= self.a_max) & (np.abs(al_t) < self.a_max), axis=1)
			for fmask, as_t, al_t in zip(feasible_mask, as_t_1, al_t_1)
		]

		# assume stopping near lane centerline (to reduce computation)
		mask = [((l_t[:, -1] > l_target_ub) | (l_t[:, -1] < l_target_lb)) & (vs_t[:, -1] < 0.1)
				for l_t, vs_t in zip(l_t_1, vs_t_1)]
		feasible_mask = [fmask & ~m for fmask, m in zip(feasible_mask, mask)]

		s_t_0 = [s_t[mask] for s_t, mask in zip(s_t_0, feasible_mask)]
		vs_t_0 = [vs_t[mask] for vs_t, mask in zip(vs_t_0, feasible_mask)]
		as_t_0 = [as_t[mask] for as_t, mask in zip(as_t_0, feasible_mask)]
		l_t_0 = [l_t[mask] for l_t, mask in zip(l_t_0, feasible_mask)]
		vl_t_0 = [vl_t[mask] for vl_t, mask in zip(vl_t_0, feasible_mask)]
		al_t_0 = [al_t[mask] for al_t, mask in zip(al_t_0, feasible_mask)]
		s_t_1 = [s_t[mask] for s_t, mask in zip(s_t_1, feasible_mask)]
		vs_t_1 = [vs_t[mask] for vs_t, mask in zip(vs_t_1, feasible_mask)]
		as_t_1 = [as_t[mask] for as_t, mask in zip(as_t_1, feasible_mask)]
		l_t_1 = [l_t[mask] for l_t, mask in zip(l_t_1, feasible_mask)]
		vl_t_1 = [vl_t[mask] for vl_t, mask in zip(vl_t_1, feasible_mask)]
		al_t_1 = [al_t[mask] for al_t, mask in zip(al_t_1, feasible_mask)]

		# 4th-order Polynomial s(t): segment 2
		P_s_2 = interpolate_polynomial(
			deg=4,
			x_0=t2_meshgrid[valid_mask],
			x_1=np.ones_like(t2_meshgrid[valid_mask]) * self.T,
			y_0=[s_t[:, -1] for s_t in s_t_1],
			y_prime_0=[vs_t[:, -1] for vs_t in vs_t_1],
			y_pprime_0=[as_t[:, -1] for as_t in as_t_1],
			y_prime_1=vs_target,
			y_pprime_1=np.zeros_like(vs_target)
		)
		# 5th-order Polynomial l(t): segment 2
		P_l_2 = interpolate_polynomial(
			deg=5,
			x_0=t2_meshgrid[valid_mask],
			x_1=np.ones_like(t2_meshgrid[valid_mask]) * self.T,
			y_0=[l_t[:, -1] for l_t in l_t_1],
			y_prime_0=[vl_t[:, -1] for vl_t in vl_t_1],
			y_pprime_0=[al_t[:, -1] for al_t in al_t_1],
			y_1=l_target,
			y_prime_1=np.zeros_like(l_target),
			y_pprime_1=np.zeros_like(l_target)
		)
		s_t_2 = [polynomial(t_seg, P_s.A) for t_seg, P_s in zip(t_seg_2, P_s_2)]
		vs_t_2 = [polynomial_derivative(t_seg, P_s.A, order=1) for t_seg, P_s in zip(t_seg_2, P_s_2)]
		as_t_2 = [polynomial_derivative(t_seg, P_s.A, order=2) for t_seg, P_s in zip(t_seg_2, P_s_2)]
		l_t_2 = [polynomial(t_seg, P_l.A) for t_seg, P_l in zip(t_seg_2, P_l_2)]
		vl_t_2 = [polynomial_derivative(t_seg, P_l.A, order=1) for t_seg, P_l in zip(t_seg_2, P_l_2)]
		al_t_2 = [polynomial_derivative(t_seg, P_l.A, order=2) for t_seg, P_l in zip(t_seg_2, P_l_2)]

		s_t_0 = [s_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for s_t in s_t_0]
		vs_t_0 = [vs_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for vs_t in vs_t_0]
		as_t_0 = [as_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for as_t in as_t_0]
		l_t_0 = [l_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for l_t in l_t_0]
		vl_t_0 = [vl_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for vl_t in vl_t_0]
		al_t_0 = [al_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for al_t in al_t_0]

		s_t_1 = [s_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for s_t in s_t_1]
		vs_t_1 = [vs_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for vs_t in vs_t_1]
		as_t_1 = [as_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for as_t in as_t_1]
		l_t_1 = [l_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for l_t in l_t_1]
		vl_t_1 = [vl_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for vl_t in vl_t_1]
		al_t_1 = [al_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for al_t in al_t_1]

		s_t_2 = [s_t.repeat(l_target.shape[0], axis=0) for s_t in s_t_2]
		vs_t_2 = [vs_t.repeat(l_target.shape[0], axis=0) for vs_t in vs_t_2]
		as_t_2 = [as_t.repeat(l_target.shape[0], axis=0) for as_t in as_t_2]
		l_t_2 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(l_t, l_t.shape[0] / l_target.shape[0])
			]) if len(l_t) > 0 else l_t
			for l_t in l_t_2
		]
		vl_t_2 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(vl_t, vl_t.shape[0] / l_target.shape[0])
			]) if len(vl_t) > 0 else vl_t
			for vl_t in vl_t_2
		]
		al_t_2 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(al_t, al_t.shape[0] / l_target.shape[0])
			]) if len(al_t) > 0 else al_t
			for al_t in al_t_2
		]

		# handle trajectories with small inital and target speeds or with invalid acceleration
		initial_speed_is_small = [vs[:, -1] < 0.5 for vs in vs_t_1]
		target_speed_is_small = [vs[:, -1] < 0.1 for vs in vs_t_2]
		slow_mask = [v0_is_small & v1_is_small for v0_is_small, v1_is_small in zip(initial_speed_is_small, target_speed_is_small)]
		not_slow_mask = [~mask for mask in slow_mask]
		for mask in slow_mask:
			mask[:-1] = mask[:-1] & (~mask[1:])
		for i in range(len(s_t_0)):
			s_t_2[i][slow_mask[i]] = s_t_2[i][slow_mask[i]][:, 0].reshape(-1, 1)
			vs_t_2[i][slow_mask[i]] = 0.
			as_t_2[i][slow_mask[i]] = 0.
			l_t_2[i][slow_mask[i]] = l_t_2[i][slow_mask[i]][:, 0].reshape(-1, 1)
			vl_t_2[i][slow_mask[i]] = 0.
			al_t_2[i][slow_mask[i]] = 0.
		# remove trajectories with invalid initial heading
		initial_heading = [np.arctan2(np.diff(l_t[:, [0, 5]]), np.diff(s_t[:, [0, 5]])).squeeze() for s_t, l_t in zip(s_t_2, l_t_2)]
		valid_heading_mask = [np.abs(h) < 0.2 for h in initial_heading]
		feasible_mask = [(nsmask | smask) & hmask
						 for nsmask, smask, hmask in zip(not_slow_mask,
														 slow_mask,
														 valid_heading_mask)]
		feasible_mask = [
			fmask & np.all((np.sqrt(as_t ** 2 + al_t ** 2) <= self.a_max) & (np.abs(al_t) < self.a_max), axis=1)
			for fmask, as_t, al_t in zip(feasible_mask, as_t_2, al_t_2)
		]

		# assume stopping near lane centerline (to reduce computation)
		mask = [((l_t[:, -1] > l_target_ub) | (l_t[:, -1] < l_target_lb)) & (vs_t[:, -1] < 0.1)
				for l_t, vs_t in zip(l_t_2, vs_t_2)]
		feasible_mask = [fmask & ~m for fmask, m in zip(feasible_mask, mask)]

		s_t_0 = [s_t[mask] for s_t, mask in zip(s_t_0, feasible_mask)]
		vs_t_0 = [vs_t[mask] for vs_t, mask in zip(vs_t_0, feasible_mask)]
		as_t_0 = [as_t[mask] for as_t, mask in zip(as_t_0, feasible_mask)]
		l_t_0 = [l_t[mask] for l_t, mask in zip(l_t_0, feasible_mask)]
		vl_t_0 = [vl_t[mask] for vl_t, mask in zip(vl_t_0, feasible_mask)]
		al_t_0 = [al_t[mask] for al_t, mask in zip(al_t_0, feasible_mask)]
		s_t_1 = [s_t[mask] for s_t, mask in zip(s_t_1, feasible_mask)]
		vs_t_1 = [vs_t[mask] for vs_t, mask in zip(vs_t_1, feasible_mask)]
		as_t_1 = [as_t[mask] for as_t, mask in zip(as_t_1, feasible_mask)]
		l_t_1 = [l_t[mask] for l_t, mask in zip(l_t_1, feasible_mask)]
		vl_t_1 = [vl_t[mask] for vl_t, mask in zip(vl_t_1, feasible_mask)]
		al_t_1 = [al_t[mask] for al_t, mask in zip(al_t_1, feasible_mask)]
		s_t_2 = [s_t[mask] for s_t, mask in zip(s_t_2, feasible_mask)]
		vs_t_2 = [vs_t[mask] for vs_t, mask in zip(vs_t_2, feasible_mask)]
		as_t_2 = [as_t[mask] for as_t, mask in zip(as_t_2, feasible_mask)]
		l_t_2 = [l_t[mask] for l_t, mask in zip(l_t_2, feasible_mask)]
		vl_t_2 = [vl_t[mask] for vl_t, mask in zip(vl_t_2, feasible_mask)]
		al_t_2 = [al_t[mask] for al_t, mask in zip(al_t_2, feasible_mask)]

		s_t = [np.hstack([st0[:, 0:-1], st1[:, 0:-1], st2]) for st0, st1, st2 in zip(s_t_0, s_t_1, s_t_2)]
		vs_t = [np.hstack([vst0[:, 0:-1], vst1[:, 0:-1], vst2]) for vst0, vst1, vst2 in zip(vs_t_0, vs_t_1, vs_t_2)]
		as_t = [np.hstack([ast0[:, 0:-1], ast1[:, 0:-1], ast2]) for ast0, ast1, ast2 in zip(as_t_0, as_t_1, as_t_2)]
		l_t = [np.hstack([lt0[:, 0:-1], lt1[:, 0:-1], lt2]) for lt0, lt1, lt2 in zip(l_t_0, l_t_1, l_t_2)]
		vl_t = [np.hstack([vlt0[:, 0:-1], vlt1[:, 0:-1], vlt2]) for vlt0, vlt1, vlt2 in zip(vl_t_0, vl_t_1, vl_t_2)]
		al_t = [np.hstack([alt0[:, 0:-1], alt1[:, 0:-1], alt2]) for alt0, alt1, alt2 in zip(al_t_0, al_t_1, al_t_2)]

		# record deduplicated trajs
		traj_dedup = {}
		# 0
		unique = [np.unique(s_t[:, -1], return_index=True) for s_t in s_t_0]
		unique_s = [u[0] for u in unique]
		unique_s_idx = [u[1] for u in unique]
		traj_dedup['s_t_0'] = [s_t[idx].astype(np.float32) for s_t, idx in zip(s_t_0, unique_s_idx)]
		traj_dedup['s_t_0_idx'] = [[np.where(s_t[:, -1] == u)[0].astype(np.int32) for u in u_s]
								   for u_s, s_t in zip(unique_s, s_t_0)]
		l_t_0 = [np.round(l_t, decimals=4) for l_t in l_t_0]
		indicator = [l_t[:, 0] + l_t[:, -1] * 1000. for l_t in l_t_0]
		unique = [np.unique(indic, return_index=True) for indic in indicator]
		unique_indic = [u[0] for u in unique]
		unique_l_idx = [u[1] for u in unique]
		traj_dedup['l_t_0'] = [l_t[idx].astype(np.float32) for l_t, idx in zip(l_t_0, unique_l_idx)]
		traj_dedup['l_t_0_idx'] = [[np.where(indic == ind)[0].astype(np.int32) for ind in u_indic]
								   for u_indic, indic in zip(unique_indic, indicator)]
		# 1
		unique = [np.unique(s_t[:, -1], return_index=True) for s_t in s_t_1]
		unique_s = [u[0] for u in unique]
		unique_s_idx = [u[1] for u in unique]
		traj_dedup['s_t_1'] = [s_t[idx].astype(np.float32) for s_t, idx in zip(s_t_1, unique_s_idx)]
		traj_dedup['s_t_1_idx'] = [[np.where(s_t[:, -1] == u)[0].astype(np.int32) for u in u_s]
								   for u_s, s_t in zip(unique_s, s_t_1)]
		l_t_1 = [np.round(l_t, decimals=4) for l_t in l_t_1]
		indicator = [l_t[:, 0] + l_t[:, -1] * 1000. for l_t in l_t_1]
		unique = [np.unique(indic, return_index=True) for indic in indicator]
		unique_indic = [u[0] for u in unique]
		unique_l_idx = [u[1] for u in unique]
		traj_dedup['l_t_1'] = [l_t[idx].astype(np.float32) for l_t, idx in zip(l_t_1, unique_l_idx)]
		traj_dedup['l_t_1_idx'] = [[np.where(indic == ind)[0].astype(np.int32) for ind in u_indic]
								   for u_indic, indic in zip(unique_indic, indicator)]
		# 2
		unique = [np.unique(s_t[:, -1], return_index=True) for s_t in s_t_2]
		unique_s = [u[0] for u in unique]
		unique_s_idx = [u[1] for u in unique]
		traj_dedup['s_t_2'] = [s_t[idx].astype(np.float32) for s_t, idx in zip(s_t_2, unique_s_idx)]
		traj_dedup['s_t_2_idx'] = [[np.where(s_t[:, -1] == u)[0].astype(np.int32) for u in u_s]
								   for u_s, s_t in zip(unique_s, s_t_2)]
		l_t_2 = [np.round(l_t, decimals=4) for l_t in l_t_2]
		indicator = [l_t[:, 0] + l_t[:, -1] * 1000. for l_t in l_t_2]
		unique = [np.unique(indic, return_index=True) for indic in indicator]
		unique_indic = [u[0] for u in unique]
		unique_l_idx = [u[1] for u in unique]
		traj_dedup['l_t_2'] = [l_t[idx].astype(np.float32) for l_t, idx in zip(l_t_2, unique_l_idx)]
		traj_dedup['l_t_2_idx'] = [[np.where(indic == ind)[0].astype(np.int32) for ind in u_indic]
								   for u_indic, indic in zip(unique_indic, indicator)]

		return {
			's_t': s_t,
			'vs_t': vs_t,
			'as_t': as_t,
			'l_t': l_t,
			'vl_t': vl_t,
			'al_t': al_t
		}, traj_dedup

	def get_lane_changing_samples(
			self,
			current_ego_station,
			current_ego_lateral,
			current_ego_heading,
			current_ego_speed,
			vs_target,
			l_target,
			ego_velocity_local,
			ego_acceleration_local,
	) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
		t1 = np.around(self.t[20:-20:10], decimals=1)
		t1_idx = np.arange(0, t1.shape[0])
		t_seg_0 = [np.linspace(self.t[0], T, num=int(np.around(T / self.dt)) + 1) for T in t1]
		t_seg_1 = [np.linspace(start, self.T, num=int(np.around((self.T - start) / self.dt)) + 1)
				   for start in t1]
		# 4th-order Polynomial s(t): segment 0
		cos_heading = np.cos(current_ego_heading)
		sin_heading = np.sin(current_ego_heading)
		if ego_velocity_local.x * cos_heading + ego_velocity_local.y * sin_heading < 0.5 \
				and ego_acceleration_local.x * cos_heading + ego_acceleration_local.y * sin_heading < 0.:
			coefficient = 0.
		else:
			coefficient = 1.
		P_s_0 = interpolate_polynomial(
			deg=4,
			x_0=np.zeros_like(t1),
			x_1=t1,
			y_0=current_ego_station * np.ones_like(vs_target),
			y_prime_0=(ego_velocity_local.x * cos_heading + ego_velocity_local.y * sin_heading) * np.ones_like(vs_target),
			y_pprime_0=coefficient * (ego_acceleration_local.x * cos_heading + ego_acceleration_local.y * sin_heading) * np.ones_like(vs_target),
			y_prime_1=vs_target,
			y_pprime_1=np.zeros_like(vs_target)
		)
		# 5th-order Polynomial l(t): segment 0
		P_l_0 = interpolate_polynomial(
			deg=5,
			x_0=np.zeros_like(t1),
			x_1=t1,
			y_0=current_ego_lateral * np.ones_like(l_target),
			y_prime_0=(ego_velocity_local.x * sin_heading + ego_velocity_local.y * cos_heading) * np.ones_like(l_target),
			y_pprime_0=(ego_acceleration_local.x * sin_heading + ego_acceleration_local.y * cos_heading) * np.ones_like(l_target),
			y_1=l_target,
			y_prime_1=np.zeros_like(l_target),
			y_pprime_1=np.zeros_like(l_target)
		)
		s_t_0 = [polynomial(t_seg, P_s.A) for t_seg, P_s in zip(t_seg_0, P_s_0)]
		vs_t_0 = [polynomial_derivative(t_seg, P_s.A, order=1) for t_seg, P_s in zip(t_seg_0, P_s_0)]
		as_t_0 = [polynomial_derivative(t_seg, P_s.A, order=2) for t_seg, P_s in zip(t_seg_0, P_s_0)]
		l_t_0 = [polynomial(t_seg, P_l.A) for t_seg, P_l in zip(t_seg_0, P_l_0)]
		vl_t_0 = [polynomial_derivative(t_seg, P_l.A, order=1) for t_seg, P_l in zip(t_seg_0, P_l_0)]
		al_t_0 = [polynomial_derivative(t_seg, P_l.A, order=2) for t_seg, P_l in zip(t_seg_0, P_l_0)]

		s_t_0 = [s_t.repeat(l_target.shape[0], axis=0) for s_t in s_t_0]
		vs_t_0 = [vs_t.repeat(l_target.shape[0], axis=0) for vs_t in vs_t_0]
		as_t_0 = [as_t.repeat(l_target.shape[0], axis=0) for as_t in as_t_0]
		l_t_0 = [np.tile(l_t, reps=(vs_target.shape[0], 1)) for l_t in l_t_0]
		vl_t_0 = [np.tile(vl_t, reps=(vs_target.shape[0], 1)) for vl_t in vl_t_0]
		al_t_0 = [np.tile(al_t, reps=(vs_target.shape[0], 1)) for al_t in al_t_0]

		feasible_mask = [
			np.all((np.sqrt(as_t ** 2 + al_t ** 2) <= self.a_max) & (np.abs(al_t) < self.a_max), axis=1)
			for as_t, al_t in zip(as_t_0, al_t_0)
		]

		# handle trajectories with small inital and target speeds
		current_speed_is_small = current_ego_speed < 0.5
		target_speed_is_small = vs_target < 0.1
		slow_mask = [(current_speed_is_small & target_speed_is_small).repeat(l_target.shape[0]) for _ in range(len(s_t_0))]
		not_slow_mask = [~mask for mask in slow_mask]
		for mask in slow_mask:
			mask[:-1] = mask[:-1] & (~mask[1:])
		for i in range(len(s_t_0)):
			if any(slow_mask[i]):
				s_t_0[i][slow_mask[i]] = s_t_0[i][slow_mask[i]][:, 0]
				vs_t_0[i][slow_mask[i]] = 0.
				as_t_0[i][slow_mask[i]] = 0.
				l_t_0[i][slow_mask[i]] = l_t_0[i][slow_mask[i]][:, 0]
				vl_t_0[i][slow_mask[i]] = 0.
				al_t_0[i][slow_mask[i]] = 0.
		# remove trajectories with invalid initial heading
		initial_heading = [np.arctan2(np.diff(l_t[:, [0, 5]]), np.diff(s_t[:, [0, 5]])).squeeze() for s_t, l_t in zip(s_t_0, l_t_0)]
		valid_heading_mask = [np.abs(h - current_ego_heading) < 0.2 for h in initial_heading]
		feasible_mask = [fmask & (nsmask | smask) & hmask
						 for fmask, nsmask, smask, hmask in zip(feasible_mask,
																not_slow_mask,
																slow_mask,
																valid_heading_mask)]

		s_t_0 = [s_t[mask] for s_t, mask in zip(s_t_0, feasible_mask)]
		vs_t_0 = [vs_t[mask] for vs_t, mask in zip(vs_t_0, feasible_mask)]
		as_t_0 = [as_t[mask] for as_t, mask in zip(as_t_0, feasible_mask)]
		l_t_0 = [l_t[mask] for l_t, mask in zip(l_t_0, feasible_mask)]
		vl_t_0 = [vl_t[mask] for vl_t, mask in zip(vl_t_0, feasible_mask)]
		al_t_0 = [al_t[mask] for al_t, mask in zip(al_t_0, feasible_mask)]

		# 4th-order Polynomial s(t): segment 1
		P_s_1 = interpolate_polynomial(
			deg=4,
			x_0=t1,
			x_1=np.ones_like(t1) * self.T,
			y_0=[s_t_0[idx][:, -1] for idx in t1_idx],
			y_prime_0=[vs_t_0[idx][:, -1] for idx in t1_idx],
			y_pprime_0=[as_t_0[idx][:, -1] for idx in t1_idx],
			y_prime_1=vs_target,
			y_pprime_1=np.zeros_like(vs_target)
		)
		# 5th-order Polynomial l(t): segment 1
		P_l_1 = interpolate_polynomial(
			deg=5,
			x_0=t1,
			x_1=np.ones_like(t1) * self.T,
			y_0=[l_t_0[idx][:, -1] for idx in t1_idx],
			y_prime_0=[vl_t_0[idx][:, -1] for idx in t1_idx],
			y_pprime_0=[al_t_0[idx][:, -1] for idx in t1_idx],
			y_1=l_target,
			y_prime_1=np.zeros_like(l_target),
			y_pprime_1=np.zeros_like(l_target)
		)
		s_t_1 = [polynomial(t_seg, P_s.A) for t_seg, P_s in zip(t_seg_1, P_s_1)]
		vs_t_1 = [polynomial_derivative(t_seg, P_s.A, order=1) for t_seg, P_s in zip(t_seg_1, P_s_1)]
		as_t_1 = [polynomial_derivative(t_seg, P_s.A, order=2) for t_seg, P_s in zip(t_seg_1, P_s_1)]
		l_t_1 = [polynomial(t_seg, P_l.A) for t_seg, P_l in zip(t_seg_1, P_l_1)]
		vl_t_1 = [polynomial_derivative(t_seg, P_l.A, order=1) for t_seg, P_l in zip(t_seg_1, P_l_1)]
		al_t_1 = [polynomial_derivative(t_seg, P_l.A, order=2) for t_seg, P_l in zip(t_seg_1, P_l_1)]

		s_t_0 = [s_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for s_t in s_t_0]
		vs_t_0 = [vs_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for vs_t in vs_t_0]
		as_t_0 = [as_t.repeat(vs_target.shape[0] * l_target.shape[0], axis=0) for as_t in as_t_0]
		l_t_0 = [l_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for l_t in l_t_0]
		vl_t_0 = [vl_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for vl_t in vl_t_0]
		al_t_0 = [al_t.repeat(l_target.shape[0] * vs_target.shape[0], axis=0) for al_t in al_t_0]

		s_t_1 = [s_t.repeat(l_target.shape[0], axis=0) for s_t in s_t_1]
		vs_t_1 = [vs_t.repeat(l_target.shape[0], axis=0) for vs_t in vs_t_1]
		as_t_1 = [as_t.repeat(l_target.shape[0], axis=0) for as_t in as_t_1]
		l_t_1 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(l_t, l_t.shape[0] / l_target.shape[0])
			]) if len(l_t) > 0 else l_t
			for l_t in l_t_1
		]
		vl_t_1 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(vl_t, vl_t.shape[0] / l_target.shape[0])
			]) if len(vl_t) > 0 else vl_t
			for vl_t in vl_t_1
		]
		al_t_1 = [
			np.vstack([
				np.tile(split, reps=(vs_target.shape[0], 1))
				for split in np.array_split(al_t, al_t.shape[0] / l_target.shape[0])
			]) if len(al_t) > 0 else al_t
			for al_t in al_t_1
		]

		# handle trajectories with small inital and target speeds or with invalid acceleration
		initial_speed_is_small = [vs[:, -1] < 0.5 for vs in vs_t_0]
		target_speed_is_small = [vs[:, -1] < 0.1 for vs in vs_t_1]
		slow_mask = [v0_is_small & v1_is_small for v0_is_small, v1_is_small in zip(initial_speed_is_small, target_speed_is_small)]
		not_slow_mask = [~mask for mask in slow_mask]
		for mask in slow_mask:
			mask[:-1] = mask[:-1] & (~mask[1:])
		for i in range(len(s_t_0)):
			s_t_1[i][slow_mask[i]] = s_t_1[i][slow_mask[i]][:, 0].reshape(-1, 1)
			vs_t_1[i][slow_mask[i]] = 0.
			as_t_1[i][slow_mask[i]] = 0.
			l_t_1[i][slow_mask[i]] = l_t_1[i][slow_mask[i]][:, 0].reshape(-1, 1)
			vl_t_1[i][slow_mask[i]] = 0.
			al_t_1[i][slow_mask[i]] = 0.
		# remove trajectories with invalid initial heading
		initial_heading = [np.arctan2(np.diff(l_t[:, [0, 5]]), np.diff(s_t[:, [0, 5]])).squeeze() for s_t, l_t in zip(s_t_1, l_t_1)]
		valid_heading_mask = [np.abs(h) < 0.2 for h in initial_heading]
		feasible_mask = [(nsmask | smask) & hmask
						 for nsmask, smask, hmask in zip(not_slow_mask,
														 slow_mask,
														 valid_heading_mask)]
		feasible_mask = [
			fmask & np.all((np.sqrt(as_t ** 2 + al_t ** 2) <= self.a_max) & (np.abs(al_t) < self.a_max), axis=1)
			for fmask, as_t, al_t in zip(feasible_mask, as_t_1, al_t_1)
		]

		s_t_0 = [s_t[mask] for s_t, mask in zip(s_t_0, feasible_mask)]
		vs_t_0 = [vs_t[mask] for vs_t, mask in zip(vs_t_0, feasible_mask)]
		as_t_0 = [as_t[mask] for as_t, mask in zip(as_t_0, feasible_mask)]
		l_t_0 = [l_t[mask] for l_t, mask in zip(l_t_0, feasible_mask)]
		vl_t_0 = [vl_t[mask] for vl_t, mask in zip(vl_t_0, feasible_mask)]
		al_t_0 = [al_t[mask] for al_t, mask in zip(al_t_0, feasible_mask)]
		s_t_1 = [s_t[mask] for s_t, mask in zip(s_t_1, feasible_mask)]
		vs_t_1 = [vs_t[mask] for vs_t, mask in zip(vs_t_1, feasible_mask)]
		as_t_1 = [as_t[mask] for as_t, mask in zip(as_t_1, feasible_mask)]
		l_t_1 = [l_t[mask] for l_t, mask in zip(l_t_1, feasible_mask)]
		vl_t_1 = [vl_t[mask] for vl_t, mask in zip(vl_t_1, feasible_mask)]
		al_t_1 = [al_t[mask] for al_t, mask in zip(al_t_1, feasible_mask)]

		s_t = [np.hstack([st0[:, 0:-1], st1]) for st0, st1 in zip(s_t_0, s_t_1)]
		vs_t = [np.hstack([vst0[:, 0:-1], vst1]) for vst0, vst1 in zip(vs_t_0, vs_t_1)]
		as_t = [np.hstack([ast0[:, 0:-1], ast1]) for ast0, ast1 in zip(as_t_0, as_t_1)]
		l_t = [np.hstack([lt0[:, 0:-1], lt1]) for lt0, lt1 in zip(l_t_0, l_t_1)]
		vl_t = [np.hstack([vlt0[:, 0:-1], vlt1]) for vlt0, vlt1 in zip(vl_t_0, vl_t_1)]
		al_t = [np.hstack([alt0[:, 0:-1], alt1]) for alt0, alt1 in zip(al_t_0, al_t_1)]

		# record deduplicated trajs
		traj_dedup = {}
		# 0
		unique = [np.unique(s_t[:, -1], return_index=True) for s_t in s_t_0]
		unique_s = [u[0] for u in unique]
		unique_s_idx = [u[1] for u in unique]
		traj_dedup['s_t_0'] = [s_t[idx].astype(np.float32) for s_t, idx in zip(s_t_0, unique_s_idx)]
		traj_dedup['s_t_0_idx'] = [[np.where(s_t[:, -1] == u)[0].astype(np.int32) for u in u_s]
								   for u_s, s_t in zip(unique_s, s_t_0)]
		l_t_0 = [np.round(l_t, decimals=4) for l_t in l_t_0]
		indicator = [l_t[:, 0] + l_t[:, -1] * 1000. for l_t in l_t_0]
		unique = [np.unique(indic, return_index=True) for indic in indicator]
		unique_indic = [u[0] for u in unique]
		unique_l_idx = [u[1] for u in unique]
		traj_dedup['l_t_0'] = [l_t[idx].astype(np.float32) for l_t, idx in zip(l_t_0, unique_l_idx)]
		traj_dedup['l_t_0_idx'] = [[np.where(indic == ind)[0].astype(np.int32) for ind in u_indic]
								   for u_indic, indic in zip(unique_indic, indicator)]
		# 1
		unique = [np.unique(s_t[:, -1], return_index=True) for s_t in s_t_1]
		unique_s = [u[0] for u in unique]
		unique_s_idx = [u[1] for u in unique]
		traj_dedup['s_t_1'] = [s_t[idx].astype(np.float32) for s_t, idx in zip(s_t_1, unique_s_idx)]
		traj_dedup['s_t_1_idx'] = [[np.where(s_t[:, -1] == u)[0].astype(np.int32) for u in u_s]
								   for u_s, s_t in zip(unique_s, s_t_1)]
		l_t_1 = [np.round(l_t, decimals=4) for l_t in l_t_1]
		indicator = [l_t[:, 0] + l_t[:, -1] * 1000. for l_t in l_t_1]
		unique = [np.unique(indic, return_index=True) for indic in indicator]
		unique_indic = [u[0] for u in unique]
		unique_l_idx = [u[1] for u in unique]
		traj_dedup['l_t_1'] = [l_t[idx].astype(np.float32) for l_t, idx in zip(l_t_1, unique_l_idx)]
		traj_dedup['l_t_1_idx'] = [[np.where(indic == ind)[0].astype(np.int32) for ind in u_indic]
								   for u_indic, indic in zip(unique_indic, indicator)]

		return {
			's_t': s_t,
			'vs_t': vs_t,
			'as_t': as_t,
			'l_t': l_t,
			'vl_t': vl_t,
			'al_t': al_t
		}, traj_dedup

	def duplicate_points(self, shape, traj_dedup, traj_pt, traj_xy):
		if len(traj_pt.keys()) == 3:
			s_t_pt = [np.zeros(s, dtype='O') for s in shape]
			s_t_xy = [np.zeros((s[0], 2, s[1])) for s in shape]
			ptr = [np.zeros((s.shape[0],), dtype=int) for s in s_t_pt]
			for s_t, p, pt, idx, s_t_xy_i, xy in zip(s_t_pt, ptr, traj_pt['s_t_0'], traj_dedup['s_t_0_idx'], s_t_xy, traj_xy['s_t_0']):
				xy = np.stack(xy, axis=1)
				for i in range(len(idx)):
					s_t[idx[i], :pt[i, :-1].shape[0]] = pt[i, :-1]
					s_t_xy_i[idx[i], :, :pt[i, :-1].shape[0]] = xy[i, :, :-1]
					p[idx[i]] += pt[i, :-1].shape[0]
			for s_t, p, pt, idx, s_t_xy_i, xy in zip(s_t_pt, ptr, traj_pt['s_t_1'], traj_dedup['s_t_1_idx'], s_t_xy, traj_xy['s_t_1']):
				xy = np.stack(xy, axis=1)
				for i in range(len(idx)):
					s_t[idx[i], p[idx[i]][0]:p[idx[i]][0] + pt[i, :-1].shape[0]] = pt[i, :-1]
					s_t_xy_i[idx[i], :, p[idx[i]][0]:p[idx[i]][0] + pt[i, :-1].shape[0]] = xy[i, :, :-1]
					p[idx[i]] += pt[i, :-1].shape[0]
			for s_t, p, pt, idx, s_t_xy_i, xy in zip(s_t_pt, ptr, traj_pt['s_t_2'], traj_dedup['s_t_2_idx'], s_t_xy, traj_xy['s_t_2']):
				xy = np.stack(xy, axis=1)
				for i in range(len(idx)):
					s_t[idx[i], p[idx[i]][0]:p[idx[i]][0] + pt[i].shape[0]] = pt[i]
					s_t_xy_i[idx[i], :, p[idx[i]][0]:p[idx[i]][0] + pt[i].shape[0]] = xy[i]
					p[idx[i]] += pt[i].shape[0]

		else:  # len(traj_pt.keys()) == 2
			s_t_pt = [np.zeros(s, dtype='O') for s in shape]
			s_t_xy = [np.zeros((s[0], 2, s[1])) for s in shape]
			ptr = [np.zeros((s.shape[0],), dtype=int) for s in s_t_pt]
			for s_t, p, pt, idx, s_t_xy_i, xy in zip(s_t_pt, ptr, traj_pt['s_t_0'], traj_dedup['s_t_0_idx'], s_t_xy, traj_xy['s_t_0']):
				xy = np.stack(xy, axis=1)
				for i in range(len(idx)):
					s_t[idx[i], :pt[i, :-1].shape[0]] = pt[i, :-1]
					s_t_xy_i[idx[i], :, :pt[i, :-1].shape[0]] = xy[i, :, :-1]
					p[idx[i]] += pt[i, :-1].shape[0]
			for s_t, p, pt, idx, s_t_xy_i, xy in zip(s_t_pt, ptr, traj_pt['s_t_1'], traj_dedup['s_t_1_idx'], s_t_xy, traj_xy['s_t_1']):
				xy = np.stack(xy, axis=1)
				for i in range(len(idx)):
					s_t[idx[i], p[idx[i]][0]:p[idx[i]][0] + pt[i].shape[0]] = pt[i]
					s_t_xy_i[idx[i], :, p[idx[i]][0]:p[idx[i]][0] + pt[i].shape[0]] = xy[i]
					p[idx[i]] += pt[i].shape[0]

		return s_t_pt, s_t_xy

	def duplicate_trajs(self, shapes, traj_dedup):
		traj = {
			's_t': [np.zeros(shape) for shape in shapes],
			'l_t': [np.zeros(shape) for shape in shapes]
		}
		s_ptr = [np.zeros((shape[0],), dtype=int) for shape in shapes]
		l_ptr = [np.zeros((shape[0],), dtype=int) for shape in shapes]
		for s_p, l_p, s_idx, l_idx, s_t_i, l_t_i, s_t, l_t in zip(s_ptr,
																  l_ptr,
																  traj_dedup['s_t_0_idx'],
																  traj_dedup['l_t_0_idx'],
																  traj_dedup['s_t_0'],
																  traj_dedup['l_t_0'],
																  traj['s_t'],
																  traj['l_t']):
			for i in range(len(s_idx)):
				s_t[s_idx[i], :s_t_i[i, :-1].shape[0]] = s_t_i[i, :-1]
				s_p[s_idx[i]] += s_t_i[i, :-1].shape[0]
			for i in range(len(l_idx)):
				l_t[l_idx[i], :l_t_i[i, :-1].shape[0]] = l_t_i[i, :-1]
				l_p[l_idx[i]] += l_t_i[i, :-1].shape[0]
		if 's_t_2' not in traj_dedup.keys():
			# left or right
			for s_p, l_p, s_idx, l_idx, s_t_i, l_t_i, s_t, l_t in zip(s_ptr,
																	  l_ptr,
																	  traj_dedup['s_t_1_idx'],
																	  traj_dedup['l_t_1_idx'],
																	  traj_dedup['s_t_1'],
																	  traj_dedup['l_t_1'],
																	  traj['s_t'],
																	  traj['l_t']):
				for i in range(len(s_idx)):
					s_t[s_idx[i], s_p[s_idx[i]][0]:s_p[s_idx[i]][0] + s_t_i[i].shape[0]] = s_t_i[i]
					s_p[s_idx[i]] += s_t_i[i].shape[0]
				for i in range(len(l_idx)):
					l_t[l_idx[i], l_p[l_idx[i]][0]:l_p[l_idx[i]][0] + l_t_i[i].shape[0]] = l_t_i[i]
					l_p[l_idx[i]] += l_t_i[i].shape[0]
		else:
			# keep
			# left or right
			for s_p, l_p, s_idx, l_idx, s_t_i, l_t_i, s_t, l_t in zip(s_ptr,
																	  l_ptr,
																	  traj_dedup['s_t_1_idx'],
																	  traj_dedup['l_t_1_idx'],
																	  traj_dedup['s_t_1'],
																	  traj_dedup['l_t_1'],
																	  traj['s_t'],
																	  traj['l_t']):
				for i in range(len(s_idx)):
					s_t[s_idx[i], s_p[s_idx[i]][0]:s_p[s_idx[i]][0] + s_t_i[i, :-1].shape[0]] = s_t_i[i, :-1]
					s_p[s_idx[i]] += s_t_i[i, :-1].shape[0]
				for i in range(len(l_idx)):
					l_t[l_idx[i], l_p[l_idx[i]][0]:l_p[l_idx[i]][0] + l_t_i[i, :-1].shape[0]] = l_t_i[i, :-1]
					l_p[l_idx[i]] += l_t_i[i, :-1].shape[0]
			for s_p, l_p, s_idx, l_idx, s_t_i, l_t_i, s_t, l_t in zip(s_ptr,
																	  l_ptr,
																	  traj_dedup['s_t_2_idx'],
																	  traj_dedup['l_t_2_idx'],
																	  traj_dedup['s_t_2'],
																	  traj_dedup['l_t_2'],
																	  traj['s_t'],
																	  traj['l_t']):
				for i in range(len(s_idx)):
					s_t[s_idx[i], s_p[s_idx[i]][0]:s_p[s_idx[i]][0] + s_t_i[i].shape[0]] = s_t_i[i]
					s_p[s_idx[i]] += s_t_i[i].shape[0]
				for i in range(len(l_idx)):
					l_t[l_idx[i], l_p[l_idx[i]][0]:l_p[l_idx[i]][0] + l_t_i[i].shape[0]] = l_t_i[i]
					l_p[l_idx[i]] += l_t_i[i].shape[0]

		return traj

	def step(self,
			 features: FeaturesType,
			 targets: TargetsType,
			 rewardformer: RewardFormer,
			 action_global_cartesian: List[torch.Tensor],
			 ego_velocity_global_cartesian: List[torch.Tensor],
			 ego_acceleration_global_cartesian: List[torch.Tensor],
			 imagined_action_global_cartesian: Optional[List[torch.Tensor]]=None,
			 imagined_ego_velocity_global_cartesian: Optional[List[torch.Tensor]]=None,
			 imagined_ego_acceleration_global_cartesian: Optional[List[torch.Tensor]]=None,
			 imagine_batch_size: Optional[int]=None
			 ) -> None:
		with torch.no_grad():
			if imagined_action_global_cartesian is not None:
				imagine = True
				num_imagine = [a.shape[0] for a in imagined_action_global_cartesian]
				assert (imagined_ego_velocity_global_cartesian is not None
						and imagined_ego_acceleration_global_cartesian is not None
						and imagine_batch_size is not None)

				valid_imagine_mask = [num==max(num_imagine) and num != 0 for num in num_imagine]
				num_imagine = max(num_imagine)
			else:
				num_imagine = 0
				imagine_batch_size = 1
				imagine = False

			num_poses_for_eval = len(rewardformer.reward_std)
			batch_size = features['vector_set_map'].batch_size
			valid_batch_size = sum(valid_imagine_mask) if imagine else batch_size
			if imagine:
				action_global_cartesian = action_global_cartesian + [
					torch.stack([a[i]
								 for a, mask in zip(imagined_action_global_cartesian, valid_imagine_mask)
								 if mask])
					for i in range(num_imagine)
				]
				ego_velocity_global_cartesian = ego_velocity_global_cartesian + [
					torch.stack([vel[i]
								 for vel, mask in zip(imagined_ego_velocity_global_cartesian, valid_imagine_mask)
								 if mask])
					for i in range(num_imagine)
				]
				ego_acceleration_global_cartesian = ego_acceleration_global_cartesian + [
					torch.stack([acc[i]
								 for acc, mask in zip(imagined_ego_acceleration_global_cartesian, valid_imagine_mask)
								 if mask])
					for i in range(num_imagine)
				]

				duplicated_features = {k: [value for value, mask in zip(v.unpack(), valid_imagine_mask) if mask]
									   for k, v in copy.deepcopy(features).items()}
				duplicated_features = {
					k: v * imagine_batch_size
					for k, v in duplicated_features.items()
				}
				duplicated_features = {
					'vector_set_map': VectorSetMap.collate(duplicated_features['vector_set_map']),
					'generic_agents': GenericAgents.collate(duplicated_features['generic_agents'])
				}
				duplicated_targets = {'agents_trajectories': [tg
															  for tg, mask in zip(copy.deepcopy(targets['agents_trajectories']).unpack(),
																				  valid_imagine_mask)
															  if mask]}
				duplicated_targets['agents_trajectories'] = duplicated_targets['agents_trajectories'] * imagine_batch_size
				duplicated_targets['agents_trajectories'] = AgentsTrajectories.collate(duplicated_targets['agents_trajectories'])

			list_reward = []
			for i in [0] + list(range(1, num_imagine + 1, imagine_batch_size)):
				# compute reward using rewardformer
				if i == 0:
					rewardformer_features = rewardformer.process_features(
						features,
						targets,
						action_global_cartesian[i:i + 1],
						ego_velocity_global_cartesian[i:i + 1],
						num_poses_for_eval
					)
				else:
					rewardformer_features = rewardformer.process_features(
						duplicated_features,
						duplicated_targets,
						[torch.cat(action_global_cartesian[i:i + imagine_batch_size])],
						[torch.cat(ego_velocity_global_cartesian[i:i + imagine_batch_size])],
						num_poses_for_eval
					)

				rewardformer.eval()
				reward_mean = rewardformer(rewardformer_features)['reward']
				reward_mean = reward_mean.reshape(-1, num_poses_for_eval)
				if rewardformer.lambda_u_r > 0.:
					rewardformer.train()
					reward_samples = [rewardformer(rewardformer_features)['reward'].reshape(-1, num_poses_for_eval) for _ in range(rewardformer.num_samples)]
					reward_samples = torch.stack(reward_samples, dim=0)
					reward_variance = (1 / rewardformer.tau * rewardformer.sigma ** 2
									   + (reward_samples ** 2).mean(dim=0) - reward_samples.mean(dim=0) ** 2)
					if rewardformer.u_reward_mean.device != reward_variance.device:
						rewardformer._u_reward_mean = rewardformer._u_reward_mean.to(device=reward_variance.device)
						rewardformer._u_reward_std = rewardformer._u_reward_std.to(device=reward_variance.device)
					reward_variance_scaled = torch.relu(
						(reward_variance - rewardformer.u_reward_mean)
						/ rewardformer.u_reward_std
					)
				else:
					reward_variance_scaled = reward_mean.new_zeros(reward_mean.shape)
				reward = reward_mean.mean(dim=-1) - rewardformer.lambda_u_r * reward_variance_scaled.mean(dim=-1)
				reward = reward.reshape(-1, batch_size) if i == 0 else reward.reshape(-1, valid_batch_size)
				# make reward > 0
				reward = 1.1 ** reward  # Exponential transformation
				list_reward += [r for r in reward]
				if i == 0:
					reward_mean_log = reward_mean.mean(dim=-1).squeeze().cpu()
					reward_variance_scaled_log = reward_variance_scaled.mean(dim=-1).squeeze().cpu()

			for i in range(num_imagine + 1):
				# return negative reward if done or running a red light in planning horizon
				neighbor_features = self.get_predicted_neighbor_features(
					features,
					targets,
					[pose for pose in action_global_cartesian[i]],
					num_poses_for_eval,
					step=action_global_cartesian[i].shape[1] // rewardformer.num_poses,
					mask=valid_imagine_mask if i > 0 else None
				)
				predicted_done = self.get_predicted_done(neighbor_features)
				list_reward[i] += -10 * list_reward[i].new_tensor(predicted_done)
				list_reward[i] = list_reward[i].unsqueeze(-1).cpu()
				if i == 0:
					reward_log = list_reward[i].squeeze()

				ego_controller = copy.deepcopy(self.ego_controller)

				valid_sample_idx = 0
				for sample_idx in range(batch_size):
					if i == 0 or valid_imagine_mask[sample_idx]:
						# update ego state with ego_controller
						if self.iteration[sample_idx] < self.num_of_iterations[sample_idx]:
							current_iteration = SimulationIteration(
								time_point=self.time_point[sample_idx],
								index=self.iteration[sample_idx],
							)
							next_iteration = SimulationIteration(
								time_point=self.time_points[sample_idx][self.iteration[sample_idx] + 1],
								index=self.iteration[sample_idx] + 1,
							)

							if imagine and i != 0:
								action = action_global_cartesian[i][valid_sample_idx].cpu()
								velocity = ego_velocity_global_cartesian[i][valid_sample_idx].cpu()
								acceleration = ego_acceleration_global_cartesian[i][valid_sample_idx].cpu()
								valid_sample_idx += 1
							else:
								action = action_global_cartesian[i][sample_idx].cpu()
								velocity = ego_velocity_global_cartesian[i][sample_idx].cpu()
								acceleration = ego_acceleration_global_cartesian[i][sample_idx].cpu()

							# single_step_planner: [0.5, 1.0] -> [0.1, 0.2, ..., 1.0]
							if action.size(0) == 2:
								current_state = np.array([
									ego_controller[sample_idx].get_state().center.x,
									ego_controller[sample_idx].get_state().center.y,
									ego_controller[sample_idx].get_state().center.heading
								])
								current_v = ego_controller[sample_idx].get_state().dynamic_car_state.center_velocity_2d.array
								current_a = ego_controller[sample_idx].get_state().dynamic_car_state.center_acceleration_2d.array
								t = np.array([0., 0.5, 1.])
								poses = np.vstack((current_state, action.cpu().numpy()))
								current_v = np.array([
									current_v[0] * np.cos(current_state[-1]) - current_v[1] * np.sin(current_state[-1]),
									current_v[0] * np.sin(current_state[-1]) + current_v[1] * np.cos(current_state[-1]),
								])
								current_a = np.array([
									current_a[0] * np.cos(current_state[-1]) - current_a[1] * np.sin(current_state[-1]),
									current_a[0] * np.sin(current_state[-1]) + current_a[1] * np.cos(current_state[-1]),
								])
								v = np.vstack((current_v, velocity.cpu().numpy()))
								a = np.vstack((current_a, acceleration.cpu().numpy()))
								f_poses = interp1d(t, poses.transpose())
								f_v = interp1d(t, v.transpose())
								f_a = interp1d(t, a.transpose())
								t_new = np.linspace(0, 1, 11)
								poses_new = f_poses(t_new).transpose()
								v_new = f_v(t_new).transpose()
								a_new = f_a(t_new).transpose()
								action = action.new_tensor(poses_new[1:])
								velocity = velocity.new_tensor(v_new[1:])
								acceleration = acceleration.new_tensor(a_new[1:])

							center_states = [StateSE2(x.item(), y.item(), heading.item()) for x, y, heading in action]
							center_velocity_2d = [StateVector2D((vx * heading.cos() + vy * heading.sin()).item(),
																(vy * heading.cos() - vx * heading.sin()).item())
												  for (vx, vy), heading in zip(velocity, action[:, -1])]  # in local frame
							center_acceleration_2d = [StateVector2D((ax * heading.cos() + ay * heading.sin()).item(),
																	(ay * heading.cos() - ax * heading.sin()).item())
													  for (ax, ay), heading in zip(acceleration, action[:, -1])]  # in local frame

							# add current state
							center_states = [StateSE2(ego_controller[sample_idx].get_state().center.x,
													  ego_controller[sample_idx].get_state().center.y,
													  ego_controller[sample_idx].get_state().center.heading)] + center_states
							center_velocity_2d = [StateVector2D(ego_controller[sample_idx].get_state().dynamic_car_state.center_velocity_2d.x,
																ego_controller[sample_idx].get_state().dynamic_car_state.center_velocity_2d.y)] + center_velocity_2d
							center_acceleration_2d = [StateVector2D(ego_controller[sample_idx].get_state().dynamic_car_state.center_acceleration_2d.x,
																	ego_controller[sample_idx].get_state().dynamic_car_state.center_acceleration_2d.y)] + center_acceleration_2d

							planned_trajectory = InterpolatedTrajectory(
								[EgoState.build_from_center(
									center=center_state,
									center_velocity_2d=center_velocity_2d,
									center_acceleration_2d=center_acceleration_2d,
									tire_steering_angle=0.0,
									time_point=time_point,
									vehicle_parameters=self.scenario[sample_idx].ego_vehicle_parameters
								) for center_state, center_velocity_2d, center_acceleration_2d, time_point
									in zip(center_states, center_velocity_2d, center_acceleration_2d,
										   self.time_points[sample_idx][self.iteration[sample_idx]: self.iteration[sample_idx] + self.num_poses + 1])]
							)
							ego_controller[sample_idx].update_state(
								current_iteration=current_iteration,
								next_iteration=next_iteration,
								ego_state=ego_controller[sample_idx].get_state(),
								trajectory=planned_trajectory,
							)

							next_state = ego_controller[sample_idx].get_state()
							if i == 0:
								self.ego_historical_position[sample_idx] = torch.cat(
									[self.ego_historical_position[sample_idx][1:, :],
									 action_global_cartesian[i][sample_idx].new_tensor([[next_state.center.x, next_state.center.y]])],
									dim=0)
								self.ego_historical_heading[sample_idx] = torch.cat(
									[self.ego_historical_heading[sample_idx][1:],
									 action_global_cartesian[i][sample_idx].new_tensor([next_state.center.heading])],
									dim=0)
								self.ego_historical_velocity[sample_idx] = torch.cat(
									[self.ego_historical_velocity[sample_idx][1:, :],
									 action_global_cartesian[i][sample_idx].new_tensor([[next_state.dynamic_car_state.center_velocity_2d.x * cos(next_state.center.heading)
																						 - next_state.dynamic_car_state.center_velocity_2d.y * sin(next_state.center.heading),
																						 next_state.dynamic_car_state.center_velocity_2d.y * cos(next_state.center.heading)
																						 + next_state.dynamic_car_state.center_velocity_2d.x * sin(next_state.center.heading)]])],
									dim=0)

				if i == 0:
					ego_controller_0 = ego_controller
					self.imagined_ego_controller = []
				if i > 0:
					self.imagined_ego_controller.append(ego_controller)
				if i == num_imagine:
					self.ego_controller = ego_controller_0

			self.reward = []
			valid_sample_idx = 0
			for sample_idx in range(batch_size):
				if imagine:
					if valid_imagine_mask[sample_idx]:
						self.reward.append(
							torch.stack(
								[list_reward[0][sample_idx]] + [r[valid_sample_idx] for r in list_reward[1:]]
							)
						)
						valid_sample_idx += 1
					else:
						self.reward.append(None)
				else:
					self.reward.append(
						torch.stack(
							[list_reward[0][sample_idx]]
						)
					)

			# done
			current_state = [ctrl.get_state() for ctrl in self.ego_controller]
			neighbor_features = self.get_neighbor_features(features, current_state)
			self.done = self.get_done(neighbor_features, targets)
			if self.previous_done is not None:
				# once done is true, done for the following step are true
				self.done = [done or previous_done for done, previous_done in zip(self.done, self.previous_done)]

			self.reward_log = {
				'done': self.done,
				'episode_end': self.episode_end,
				'reward_mean': reward_mean_log,
				'reward_variance': reward_variance_scaled_log,
				'exponential_penalized_reward': reward_log
			}

	def get_neighbor_features(self,
							  features: FeaturesType,
							  anchor_ego_state: List[EgoState]
							  ) -> Dict[str, Dict[str, Any]]:
		batch_size = features['vector_set_map'].batch_size
		# process map data
		map_data = copy.deepcopy(features['vector_set_map'].map_data)
		# map_data = features['vector_set_map'].map_data
		pt_within_radius = [torch.norm(position - position.new_tensor(state.center.array), p=2, dim=1) < self.detect_radius
							for position, state in zip(map_data['map_point']['position'], anchor_ego_state)]
		unique_polygons = [torch.unique(edge_index[1, :], return_counts=True) for edge_index in map_data[('map_point', 'to', 'map_polygon')]['edge_index']]
		pl_within_radius = [torch.split(bools, [split_size for split_size in polygons[1]])
							for bools, polygons in zip(pt_within_radius, unique_polygons)]
		# pl_within_radius = [torch.stack([torch.any(whether) for whether in bools]) for bools in pl_within_radius]
		padded_pl_within_radius = [pad_sequence(bools, batch_first=True, padding_value=0) for bools in pl_within_radius]
		pl_within_radius = [bools.any(dim=1) for bools in padded_pl_within_radius]

		repeated_whether_within_radius = [pl_within_radius[i].repeat_interleave(unique_polygons[i][1]) for i in range(batch_size)]
		map_data['map_point']['position'] = [map_data['map_point']['position'][i][repeated_whether_within_radius[i]] for i in range(batch_size)]
		map_data['map_point']['orientation'] = [map_data['map_point']['orientation'][i][repeated_whether_within_radius[i]] for i in range(batch_size)]
		map_data['map_point']['tl_statuses'] = [map_data['map_point']['tl_statuses'][i][repeated_whether_within_radius[i]] for i in range(batch_size)]

		map_data_output = {
			'map_polygon': {'position': [position.cpu().numpy() for position in map_data['map_polygon']['position']],
							'orientation': [orientation.cpu().numpy() for orientation in map_data['map_polygon']['orientation']]},
			'map_point': {'position': [position.cpu().numpy() for position in map_data['map_point']['position']],
						  'orientation': [orientation.cpu().numpy() for orientation in map_data['map_point']['orientation']],
						  'tl_statuses': [tl_statuses.int().cpu().numpy() for tl_statuses in map_data['map_point']['tl_statuses']]},
		}

		# process agent data
		agent_data = copy.deepcopy(features['generic_agents'].agent_data)
		# agent_data = features['generic_agents'].agent_data
		whether_within_radius = [torch.norm(position[:, -1, :] - position.new_tensor(state.center.array), p=2, dim=1) < self.detect_radius
								 for position, state in zip(agent_data['position'], anchor_ego_state)]
		agent_data['position'] = [agent_data['position'][i][whether_within_radius[i]] for i in range(batch_size)]
		agent_data['heading'] = [agent_data['heading'][i][whether_within_radius[i]] for i in range(batch_size)]
		agent_data['length'] = [agent_data['length'][i][whether_within_radius[i]] for i in range(batch_size)]
		agent_data['width'] = [agent_data['width'][i][whether_within_radius[i]] for i in range(batch_size)]

		agent_data_output = {
			'av_index': agent_data['av_index'],
			'position': [position.cpu().numpy() for position in agent_data['position']],
			'heading': [heading.cpu().numpy() for heading in agent_data['heading']],
			'length': [length.cpu().numpy() for length in agent_data['length']],
			'width': [width.cpu().numpy() for width in agent_data['width']],
		}

		return {
			'batch_size': batch_size,
			'map_data': map_data_output,
			'agent_data': agent_data_output
		}

	def get_predicted_neighbor_features(self,
										features: FeaturesType,
										targets: TargetsType,
										anchor_poses: List[torch.Tensor],
										num_poses_for_eval: int,
										step: int,
										mask: Optional[List[bool]]=None
										) -> Dict[str, Dict[str, Any]]:
		if mask is None:
			valid_batch_size = features['vector_set_map'].batch_size
		else:
			valid_batch_size = sum(mask)
		anchor_positions = [poses[:, :2] for poses in anchor_poses]
		anchor_headings = [poses[:, 2] for poses in anchor_poses]
		pose_range = range(step - 1,
						   anchor_positions[0].shape[0],
						   step)
		pose_range = pose_range[0:num_poses_for_eval]
		anchor_positions = [anchor_pos[pose_range] for anchor_pos in anchor_positions]
		anchor_headings = [anchor_h[pose_range] for anchor_h in anchor_headings]
		# process map data
		map_data = copy.deepcopy(features['vector_set_map'].map_data)
		if mask is not None:
			map_data = {
				k: {kk: [vvv for vvv, m in zip(vv, mask) if m] for kk, vv in v.items()}
				if isinstance(v, dict)
				else [vv for vv, m in zip(v, mask) if m]
				for k, v in map_data.items()
			}
		pt_within_radius = [torch.norm(position.unsqueeze(0) - anchor_pos.unsqueeze(1), p=2, dim=-1) < self.detect_radius
							for position, anchor_pos in zip(map_data['map_point']['position'], anchor_positions)]
		unique_polygons = [torch.unique(edge_index[1, :], return_counts=True)
						   for edge_index in map_data[('map_point', 'to', 'map_polygon')]['edge_index']]
		pl_within_radius = [torch.split(bools.transpose(0, 1), [split_size for split_size in polygons[1]], dim=0)
							for bools, polygons in zip(pt_within_radius, unique_polygons)]
		padded_pl_within_radius = [pad_sequence(bools, batch_first=True, padding_value=0)
								   for bools in pl_within_radius]  # (num_pl, num_pt, num_poses)
		pl_within_radius = [bools.any(dim=1) for bools in padded_pl_within_radius]

		repeated_whether_within_radius = [pl_within_radius[i].repeat_interleave(unique_polygons[i][1], dim=0)
										  for i in range(valid_batch_size)]
		map_data['map_point']['position'] = [
			[
				map_data['map_point']['position'][i][mask].cpu().numpy()
				for mask in repeated_whether_within_radius[i].transpose(0, 1)
			]
			for i in range(valid_batch_size)
		]
		map_data['map_point']['orientation'] = [
			[
				map_data['map_point']['orientation'][i][mask].cpu().numpy()
				for mask in repeated_whether_within_radius[i].transpose(0, 1)
			]
			for i in range(valid_batch_size)
		]
		map_data['map_point']['tl_statuses'] = [
			[
				map_data['map_point']['tl_statuses'][i][mask].int().cpu().numpy()
				for mask in repeated_whether_within_radius[i].transpose(0, 1)
			]
			for i in range(valid_batch_size)
		]

		map_data_output = {
			'map_point': {'position': map_data['map_point']['position'],
						  'orientation': map_data['map_point']['orientation'],
						  'tl_statuses': map_data['map_point']['tl_statuses']},
		}

		# process agent data
		agent_data = copy.deepcopy(targets['agents_trajectories'])
		if mask is not None:
			agent_data.objects_types = [data for data, m in zip(agent_data.objects_types, mask) if m]
			agent_data.predict_mask = [data for data, m in zip(agent_data.predict_mask, mask) if m]
			agent_data.track_token_ids = [data for data, m in zip(agent_data.track_token_ids, mask) if m]
			agent_data.trajectories = [data for data, m in zip(agent_data.trajectories, mask) if m]
			agent_data.trajectories_global = [data for data, m in zip(agent_data.trajectories_global, mask) if m]
			agent_data.velocity_global = [data for data, m in zip(agent_data.velocity_global, mask) if m]
			av_index = [data for data, m in zip(features['generic_agents'].agent_data['av_index'], mask) if m]
			feature_id = [data for data, m in zip(features['generic_agents'].agent_data['id'], mask) if m]
			feature_length = [data for data, m in zip(features['generic_agents'].agent_data['length'], mask) if m]
			feature_width = [data for data, m in zip(features['generic_agents'].agent_data['width'], mask) if m]
		else:
			av_index = features['generic_agents'].agent_data['av_index']
			feature_id = features['generic_agents'].agent_data['id']
			feature_length = features['generic_agents'].agent_data['length']
			feature_width = features['generic_agents'].agent_data['width']

		ids = [
			[id for id in agent_data.track_token_ids[i] if id in feature_id[i]]
			for i in range(valid_batch_size)
		]
		positions = [
			torch.stack([agent_data.trajectories_global[i][id][pose_range, :2] for id in ids[i]])
			for i in range(valid_batch_size)
		]
		headings = [
			torch.stack([agent_data.trajectories_global[i][id][pose_range, 2] for id in ids[i]])
			for i in range(valid_batch_size)
		]
		index = [[feature_id[i].index(id) for id in ids[i]] for i in range(valid_batch_size)]
		lengths = [feature_length[i][index[i]]for i in range(valid_batch_size)]
		widths = [feature_width[i][index[i]]for i in range(valid_batch_size)]

		whether_within_radius = [torch.norm(position - anchor_pos.unsqueeze(0), p=2, dim=-1) < self.detect_radius
								 for position, anchor_pos in zip(positions, anchor_positions)]
		positions = [positions[i][whether_within_radius[i].any(dim=1)].cpu().numpy() for i in range(valid_batch_size)]
		headings = [headings[i][whether_within_radius[i].any(dim=1)].cpu().numpy() for i in range(valid_batch_size)]
		lengths = [lengths[i][whether_within_radius[i].any(dim=1)].cpu().numpy() for i in range(valid_batch_size)]
		widths = [widths[i][whether_within_radius[i].any(dim=1)].cpu().numpy() for i in range(valid_batch_size)]

		for sample_idx in range(valid_batch_size):
			if positions[sample_idx].shape[0] > 0:
				positions[sample_idx][av_index[sample_idx]] = anchor_positions[sample_idx].cpu().numpy()
				headings[sample_idx][av_index[sample_idx]] = anchor_headings[sample_idx].cpu().numpy()
			else:
				positions[sample_idx] =  anchor_positions[sample_idx].unsqueeze(0).cpu().numpy()
				headings[sample_idx] =  anchor_headings[sample_idx].unsqueeze(0).cpu().numpy()

		agent_data_output = {
			'av_index': av_index,
			'position': positions,
			'heading': headings,
			'length': lengths,
			'width': widths,
		}

		return {
			'batch_size': valid_batch_size,
			'map_data': map_data_output,
			'agent_data': agent_data_output
		}

	def get_done(self,
				 neighbor_features: Dict,
				 targets: TargetsType,
				 max_deviation: float=20.) -> List[bool]:
		"""
		Done is True if ego vehicle crashes into other agents, drives off-road,
		runs a red light, or deviates too far from the real position in the dataset.
		"""
		overlap_agents, offroad, run_red_light, too_far = [], [], [], []
		for sample_idx in range(neighbor_features['batch_size']):
			# crashes into other agents
			av_index = neighbor_features['agent_data']['av_index'][sample_idx]
			ego_pose = np.hstack([neighbor_features['agent_data']['position'][sample_idx][av_index, -1, :],
								  neighbor_features['agent_data']['heading'][sample_idx][av_index, -1:]])
			if neighbor_features['agent_data']['position'][sample_idx].shape[0] > 1:
				ego_length = neighbor_features['agent_data']['length'][sample_idx][0]
				ego_width = neighbor_features['agent_data']['width'][sample_idx][0]

				obs_poses = np.hstack([neighbor_features['agent_data']['position'][sample_idx][av_index + 1:, -1, :],
									   neighbor_features['agent_data']['heading'][sample_idx][av_index + 1:, -1:]])
				obs_length = neighbor_features['agent_data']['length'][sample_idx][av_index + 1:]
				obs_width = neighbor_features['agent_data']['width'][sample_idx][av_index + 1:]

				xA, yA, xB, yB, xC, yC, xD, yD = self.get_vertices_from_center(ego_pose[0],
																			   ego_pose[1],
																			   ego_pose[2],
																			   ego_length,
																			   ego_width,
																			   rad=True)
				for i_obs in range(obs_poses.shape[0]):
					(xA_obs_i, yA_obs_i, xB_obs_i, yB_obs_i,
					 xC_obs_i, yC_obs_i, xD_obs_i, yD_obs_i) = self.get_vertices_from_center(obs_poses[i_obs, 0],
																							 obs_poses[i_obs, 1],
																							 obs_poses[i_obs, 2],
																							 obs_length[i_obs],
																							 obs_width[i_obs],
																							 rad=True)

					overlap_i = self.overlap(xA, yA, xB, yB, xC, yC, xD, yD,
											 xA_obs_i, yA_obs_i, xB_obs_i, yB_obs_i,
											 xC_obs_i, yC_obs_i, xD_obs_i, yD_obs_i)
					overlap_i = overlap_i * (1 + overlap_i) * (2 - overlap_i) * (1 - overlap_i) / 24  # 1: overlap, 0: safe
					f_overlap_tmp = np.vstack((f_overlap_tmp, overlap_i.reshape(1, -1))) \
						if i_obs != 0 else overlap_i.reshape(1, -1)
				f_overlap = np.max(f_overlap_tmp, axis=0)
				overlap_agents.append(f_overlap[0] == 1)
			else:
				overlap_agents.append(False)

			# drives off-road
			map_pt_position = neighbor_features['map_data']['map_point']['position'][sample_idx]
			if map_pt_position.shape[0] > 0:
				distance = np.linalg.norm(ego_pose[0:2] - map_pt_position, axis=1)
				argmin = distance.argmin()
				min_distance_to_map_pt = distance[argmin]
				if min_distance_to_map_pt > self.offroad_distance:
					offroad.append(True)
				else:
					offroad.append(False)
			else:
				offroad.append(True)

			# run a red light
			map_pt_tl_status = neighbor_features['map_data']['map_point']['tl_statuses'][sample_idx]
			if map_pt_position.shape[0] > 0:
				tl_status = map_pt_tl_status[argmin]
				tl_status = self._traffic_light_statuses[tl_status]
				run_red_light.append(tl_status == 'RED')
			else:
				run_red_light.append(False)

			# deviates too far from the real position in the dataset
			current_state = self.ego_controller[sample_idx].get_state()
			current_position = current_state.center.array
			gt_position = targets['agents_trajectories'].trajectories_global[sample_idx]['AV'][0, :2].cpu().numpy()
			deviation =	np.linalg.norm((current_position - gt_position))
			too_far.append(deviation > max_deviation)

		return [
			bool(bool1 or bool2 or bool3 or bool4)
			for bool1, bool2, bool3, bool4
			in zip(overlap_agents, offroad, run_red_light, too_far)
		]

	def get_predicted_done(self,
						   neighbor_features: Dict,) -> List[bool]:
		"""
		Done is True if ego vehicle crashes into other agents or drives off-road.
		"""
		overlap_agents, offroad, run_red_light = [], [], []
		for sample_idx in range(neighbor_features['batch_size']):
			# crashes into other agents
			av_index = neighbor_features['agent_data']['av_index'][sample_idx]
			ego_pose = np.hstack([neighbor_features['agent_data']['position'][sample_idx][av_index],
								  neighbor_features['agent_data']['heading'][sample_idx][av_index][:, np.newaxis]])
			if neighbor_features['agent_data']['position'][sample_idx].shape[0] > 1:
				ego_length = neighbor_features['agent_data']['length'][sample_idx][0]
				ego_width = neighbor_features['agent_data']['width'][sample_idx][0]

				obs_poses = np.concatenate(
					[neighbor_features['agent_data']['position'][sample_idx][av_index + 1:],
					 neighbor_features['agent_data']['heading'][sample_idx][av_index + 1:, :, np.newaxis]],
					axis=-1
				)
				obs_length = neighbor_features['agent_data']['length'][sample_idx][av_index + 1:]
				obs_width = neighbor_features['agent_data']['width'][sample_idx][av_index + 1:]

				xA, yA, xB, yB, xC, yC, xD, yD = self.get_vertices_from_center(ego_pose[:, 0],
																			   ego_pose[:, 1],
																			   ego_pose[:, 2],
																			   ego_length,
																			   ego_width,
																			   rad=True)
				for i_obs in range(obs_poses.shape[0]):
					(xA_obs_i, yA_obs_i, xB_obs_i, yB_obs_i,
					 xC_obs_i, yC_obs_i, xD_obs_i, yD_obs_i) = self.get_vertices_from_center(obs_poses[i_obs, :, 0],
																							 obs_poses[i_obs, :, 1],
																							 obs_poses[i_obs, :, 2],
																							 obs_length[i_obs],
																							 obs_width[i_obs],
																							 rad=True)

					overlap_i = self.overlap(xA, yA, xB, yB, xC, yC, xD, yD,
											 xA_obs_i, yA_obs_i, xB_obs_i, yB_obs_i,
											 xC_obs_i, yC_obs_i, xD_obs_i, yD_obs_i)
					overlap_i = overlap_i * (1 + overlap_i) * (2 - overlap_i) * (1 - overlap_i) / 24  # 1: overlap, 0: safe
					f_overlap_tmp = np.vstack((f_overlap_tmp, overlap_i.reshape(1, -1))) \
						if i_obs != 0 else overlap_i.reshape(1, -1)
					if overlap_i == 1:
						break
				f_overlap = np.max(f_overlap_tmp, axis=0)
				overlap_agents.append(f_overlap[0] == 1)
			else:
				overlap_agents.append(False)

			# drives off-road or run a red light
			map_pt_position = neighbor_features['map_data']['map_point']['position'][sample_idx]
			map_pt_tl_status = neighbor_features['map_data']['map_point']['tl_statuses'][sample_idx]
			off = []
			run_red = []
			for i_pose in range(len(map_pt_position)):
				if map_pt_position[i_pose].shape[0] > 0:
					distance = np.linalg.norm(ego_pose[i_pose, 0:2] - map_pt_position[i_pose], axis=1)
					argmin = distance.argmin()
					min_distance_to_map_pt = distance[argmin]
					if min_distance_to_map_pt > self.offroad_distance:
						off.append(True)
					else:
						off.append(False)
				else:
					off.append(True)

				if map_pt_position[i_pose].shape[0] > 0:
					tl_status = map_pt_tl_status[i_pose][argmin]
					tl_status = self._traffic_light_statuses[tl_status]
					run_red.append(tl_status == 'RED')
				else:
					run_red.append(False)

			offroad.append(any(off))
			run_red_light.append(any(run_red))

		return [bool1 or bool2 or bool3 for bool1, bool2, bool3 in zip(overlap_agents, offroad, run_red_light)]

	@staticmethod
	def get_vertices_from_center(x, y, heading, L, W, rad=False):
		"""
		x, y, heading: pose of center with shape (num_time_steps, num_trajs)
        rear          front
         D              A
         		E
         C              B
        """
		if not rad:
			heading = heading / 180. * np.pi
		AE = BE = CE = DE = ((W / 2) ** 2 + (L / 2) ** 2) ** 0.5
		angle = np.arctan(W / L)
		# A
		xA = x + AE * np.cos(heading + angle)
		yA = y + AE * np.sin(heading + angle)
		# B
		xB = x + BE * np.cos(heading - angle)
		yB = y + BE * np.sin(heading - angle)
		# C
		xC = x - CE * np.cos(heading + angle)
		yC = y - CE * np.sin(heading + angle)
		# D
		xD = x - DE * np.cos(heading - angle)
		yD = y - DE * np.sin(heading - angle)
		return xA, yA, xB, yB, xC, yC, xD, yD

	@staticmethod
	def get_vector(point1, point2):
		return point2[0] - point1[0], point2[1] - point1[1]

	@staticmethod
	def inner_product(v1, v2):
		return np.array(v1[0]) * np.array(v2[0]) + np.array(v1[1]) * np.array(v2[1])

	@staticmethod
	def outer_product(v1, v2):
		return np.array(v1[0]) * np.array(v2[1]) - np.array(v1[1]) * np.array(v2[0])

	def overlap(self, xA1, yA1, xB1, yB1, xC1, yC1, xD1, yD1, xA2, yA2, xB2, yB2, xC2, yC2, xD2, yD2):
		if not isinstance(xA1, float):
			xA2 = np.kron(xA2, np.ones_like(xA1[0]))
			yA2 = np.kron(yA2, np.ones_like(xA1[0]))
			xB2 = np.kron(xB2, np.ones_like(xA1[0]))
			yB2 = np.kron(yB2, np.ones_like(xA1[0]))
			xC2 = np.kron(xC2, np.ones_like(xA1[0]))
			yC2 = np.kron(yC2, np.ones_like(xA1[0]))
			xD2 = np.kron(xD2, np.ones_like(xA1[0]))
			yD2 = np.kron(yD2, np.ones_like(xA1[0]))
		v_A1B1 = self.get_vector((xA1, yA1), (xB1, yB1))
		v_B1C1 = self.get_vector((xB1, yB1), (xC1, yC1))
		v_C1D1 = self.get_vector((xC1, yC1), (xD1, yD1))
		v_D1A1 = self.get_vector((xD1, yD1), (xA1, yA1))
		v_A2B2 = self.get_vector((xA2, yA2), (xB2, yB2))
		v_B2C2 = self.get_vector((xB2, yB2), (xC2, yC2))
		v_C2D2 = self.get_vector((xC2, yC2), (xD2, yD2))
		v_D2A2 = self.get_vector((xD2, yD2), (xA2, yA2))

		# Does any edge of rectangle 2 has an interaction with any edge of rectangle 1?
		# A2B2 - A1B1/B1C1/C1D1/D1A1
		v_A2A1 = self.get_vector((xA2, yA2), (xA1, yA1))
		v_A2B1 = self.get_vector((xA2, yA2), (xB1, yB1))
		outer1 = self.outer_product(v_A2B2, v_A2A1)
		outer11 = self.outer_product(v_A2B2, v_A2B1)
		v_A1A2 = self.get_vector((xA1, yA1), (xA2, yA2))
		v_A1B2 = self.get_vector((xA1, yA1), (xB2, yB2))
		outer111 = self.outer_product(v_A1B1, v_A1A2)
		outer1111 = self.outer_product(v_A1B1, v_A1B2)
		clear1 = np.sign(outer1 * outer11) + np.sign(outer111 * outer1111)  # -2: overlap, 2/1/0/-1: clear
		# clear1 = False if np.sign(outer1) != np.sign(outer11) and np.sign(outer111) != np.sign(outer1111) else True
		v_A2C1 = self.get_vector((xA2, yA2), (xC1, yC1))
		outer2 = self.outer_product(v_A2B2, v_A2C1)
		v_B1A2 = self.get_vector((xB1, yB1), (xA2, yA2))
		v_B1B2 = self.get_vector((xB1, yB1), (xB2, yB2))
		outer22 = self.outer_product(v_B1C1, v_B1A2)
		outer222 = self.outer_product(v_B1C1, v_B1B2)
		clear2 = np.sign(outer11 * outer2) + np.sign(outer22 * outer222)
		# clear2 = False if np.sign(outer11) != np.sign(outer2) and np.sign(outer22) != np.sign(outer222) else True
		v_A2D1 = self.get_vector((xA2, yA2), (xD1, yD1))
		outer3 = self.outer_product(v_A2B2, v_A2D1)
		v_C1A2 = self.get_vector((xC1, yC1), (xA2, yA2))
		v_C1B2 = self.get_vector((xC1, yC1), (xB2, yB2))
		outer33 = self.outer_product(v_C1D1, v_C1A2)
		outer333 = self.outer_product(v_C1D1, v_C1B2)
		clear3 = np.sign(outer2 * outer3) + np.sign(outer33 * outer333)
		# clear3 = False if np.sign(outer2) != np.sign(outer3) and np.sign(outer33) != np.sign(outer333) else True
		v_D1A2 = self.get_vector((xD1, yD1), (xA2, yA2))
		v_D1B2 = self.get_vector((xD1, yD1), (xB2, yB2))
		outer4 = self.outer_product(v_D1A1, v_D1A2)
		outer44 = self.outer_product(v_D1A1, v_D1B2)
		clear4 = np.sign(outer3 * outer1) + np.sign(outer4 * outer44)
		# clear4 = False if np.sign(outer4) != np.sign(outer1) and np.sign(outer44) != np.sign(outer444) else True
		A2B2_clear = np.minimum(np.minimum(clear1, clear2), np.minimum(clear3, clear4))
		# A2B2_clear = True if clear1 and clear2 and clear3 and clear4 is True else False
		# B2C2 - A1B1/B1C1/C1D1/D1A1
		v_B2A1 = self.get_vector((xB2, yB2), (xA1, yA1))
		v_B2B1 = self.get_vector((xB2, yB2), (xB1, yB1))
		outer1 = self.outer_product(v_B2C2, v_B2A1)
		outer11 = self.outer_product(v_B2C2, v_B2B1)
		v_A1C2 = self.get_vector((xA1, yA1), (xC2, yC2))
		outer111 = self.outer_product(v_A1B1, v_A1B2)
		outer1111 = self.outer_product(v_A1B1, v_A1C2)
		clear1 = np.sign(outer1 * outer11) + np.sign(outer111 * outer1111)
		# clear1 = False if np.sign(outer1) != np.sign(outer11) and np.sign(outer111) != np.sign(outer1111) else True
		v_B2C1 = self.get_vector((xB2, yB2), (xC1, yC1))
		outer2 = self.outer_product(v_B2C2, v_B2C1)
		v_B1C2 = self.get_vector((xB1, yB1), (xC2, yC2))
		outer22 = self.outer_product(v_B1C1, v_B1B2)
		outer222 = self.outer_product(v_B1C1, v_B1C2)
		clear2 = np.sign(outer11 * outer2) + np.sign(outer22 * outer222)
		# clear2 = False if np.sign(outer11) != np.sign(outer2) and np.sign(outer22) != np.sign(outer222) else True
		v_B2D1 = self.get_vector((xB2, yB2), (xD1, yD1))
		outer3 = self.outer_product(v_B2C2, v_B2D1)
		v_C1C2 = self.get_vector((xC1, yC1), (xC2, yC2))
		outer33 = self.outer_product(v_C1D1, v_C1B2)
		outer333 = self.outer_product(v_C1D1, v_C1C2)
		clear3 = np.sign(outer2 * outer3) + np.sign(outer33 * outer333)
		# clear3 = False if np.sign(outer2) != np.sign(outer3) and np.sign(outer33) != np.sign(outer333) else True
		v_D1C2 = self.get_vector((xD1, yD1), (xC2, yC2))
		outer4 = self.outer_product(v_D1A1, v_D1B2)
		outer44 = self.outer_product(v_D1A1, v_D1C2)
		clear4 = np.sign(outer3 * outer1) + np.sign(outer4 * outer44)
		# clear4 = False if np.sign(outer3) != np.sign(outer1) and np.sign(outer4) != np.sign(outer44) else True
		B2C2_clear = np.minimum(np.minimum(clear1, clear2), np.minimum(clear3, clear4))
		# B2C2_clear = True if clear1 and clear2 and clear3 and clear4 is True else False
		# C2D2 - A1B1/B1C1/C1D1/D1A1
		v_C2A1 = self.get_vector((xC2, yC2), (xA1, yA1))
		v_C2B1 = self.get_vector((xC2, yC2), (xB1, yB1))
		outer1 = self.outer_product(v_C2D2, v_C2A1)
		outer11 = self.outer_product(v_C2D2, v_C2B1)
		v_A1D2 = self.get_vector((xA1, yA1), (xD2, yD2))
		outer111 = self.outer_product(v_A1B1, v_A1C2)
		outer1111 = self.outer_product(v_A1B1, v_A1D2)
		clear1 = np.sign(outer1 * outer11) + np.sign(outer111 * outer1111)
		# clear1 = False if np.sign(outer1) != np.sign(outer11) and np.sign(outer111) != np.sign(outer1111) else True
		v_C2C1 = self.get_vector((xC2, yC2), (xC1, yC1))
		outer2 = self.outer_product(v_C2D2, v_C2C1)
		v_B1D2 = self.get_vector((xB1, yB1), (xD2, yD2))
		outer22 = self.outer_product(v_B1C1, v_B1C2)
		outer222 = self.outer_product(v_B1C1, v_B1D2)
		clear2 = np.sign(outer11 * outer2) + np.sign(outer22 * outer222)
		# clear2 = False if np.sign(outer11) != np.sign(outer2) and np.sign(outer22) != np.sign(outer222) else True
		v_C2D1 = self.get_vector((xC2, yC2), (xD1, yD1))
		outer3 = self.outer_product(v_C2D2, v_C2D1)
		v_C1D2 = self.get_vector((xC1, yC1), (xD2, yD2))
		outer33 = self.outer_product(v_C1D1, v_C1C2)
		outer333 = self.outer_product(v_C1D1, v_C1D2)
		clear3 = np.sign(outer2 * outer3) + np.sign(outer33 * outer333)
		# clear3 = False if np.sign(outer2) != np.sign(outer3) and np.sign(outer33) != np.sign(outer333) else True
		v_D1D2 = self.get_vector((xD1, yD1), (xD2, yD2))
		outer4 = self.outer_product(v_D1A1, v_D1C2)
		outer44 = self.outer_product(v_D1A1, v_D1D2)
		clear4 = np.sign(outer3 * outer1) + np.sign(outer4 * outer44)
		# clear4 = False if np.sign(outer3) != np.sign(outer1) and np.sign(outer4) != np.sign(outer44) else True
		C2D2_clear = np.minimum(np.minimum(clear1, clear2), np.minimum(clear3, clear4))
		# C2D2_clear = True if clear1 and clear2 and clear3 and clear4 is True else False
		# D2A2 - A1B1/B1C1/C1D1/D1A1
		v_D2A1 = self.get_vector((xD2, yD2), (xA1, yA1))
		v_D2B1 = self.get_vector((xD2, yD2), (xB1, yB1))
		outer1 = self.outer_product(v_D2A2, v_D2A1)
		outer11 = self.outer_product(v_D2A2, v_D2B1)
		outer111 = self.outer_product(v_A1B1, v_A1D2)
		outer1111 = self.outer_product(v_A1B1, v_A1A2)
		clear1 = np.sign(outer1 * outer11) + np.sign(outer111 * outer1111)
		# clear1 = False if np.sign(outer1) != np.sign(outer11) and np.sign(outer111) != np.sign(outer1111) else True
		v_D2C1 = self.get_vector((xD2, yD2), (xC1, yC1))
		outer2 = self.outer_product(v_D2A2, v_D2C1)
		outer22 = self.outer_product(v_B1C1, v_B1D2)
		outer222 = self.outer_product(v_B1C1, v_B1A2)
		clear2 = np.sign(outer11 * outer2) + np.sign(outer22 * outer222)
		# clear2 = False if np.sign(outer11) != np.sign(outer2) and np.sign(outer22) != np.sign(outer222) else True
		v_D2D1 = self.get_vector((xD2, yD2), (xD1, yD1))
		outer3 = self.outer_product(v_D2A2, v_D2D1)
		outer33 = self.outer_product(v_C1D1, v_C1D2)
		outer333 = self.outer_product(v_C1D1, v_C1A2)
		clear3 = np.sign(outer2 * outer3) + np.sign(outer33 * outer333)
		# clear3 = False if np.sign(outer2) != np.sign(outer3) and np.sign(outer33) != np.sign(outer333) else True
		outer4 = self.outer_product(v_D1A1, v_D1D2)
		outer44 = self.outer_product(v_D1A1, v_D1A2)
		clear4 = np.sign(outer3 * outer1) + np.sign(outer4 * outer44)
		# clear4 = False if np.sign(outer3) != np.sign(outer1) and np.sign(outer4) != np.sign(outer44) else True
		D2A2_clear = np.minimum(np.minimum(clear1, clear2), np.minimum(clear3, clear4))
		# D2A2_clear = True if clear1 and clear2 and clear3 and clear4 is True else False
		edges_clear = np.minimum(np.minimum(A2B2_clear, B2C2_clear), np.minimum(C2D2_clear, D2A2_clear))  # -2: overlap, 2/1/0/-1: clear
		# edges_clear = True if A2B2_clear and B2C2_clear and C2D2_clear and D2A2_clear is True else False

		# Are all vertices of rectangle 1 out of rectangle 2
		# A1
		inner1 = self.outer_product(v_A2B2, v_A2A1) * self.outer_product(v_C2D2, v_C2A1)  # +: in
		inner2 = self.outer_product(v_B2C2, v_B2A1) * self.outer_product(v_D2A2, v_D2A1)
		# A1_clear = False if inner1 >= 0. and inner2 >= 0. else True
		A1_clear = - np.sign(inner1) - np.sign(inner2)  # -2: in
		# B1
		inner1 = self.outer_product(v_A2B2, v_A2B1) * self.outer_product(v_C2D2, v_C2B1)
		inner2 = self.outer_product(v_B2C2, v_B2B1) * self.outer_product(v_D2A2, v_D2B1)
		# B1_clear = False if inner1 >= 0. and inner2 >= 0. else True
		B1_clear = - np.sign(inner1) - np.sign(inner2)  # -2: in
		# C1
		inner1 = self.outer_product(v_A2B2, v_A2C1) * self.outer_product(v_C2D2, v_C2C1)
		inner2 = self.outer_product(v_B2C2, v_B2C1) * self.outer_product(v_D2A2, v_D2C1)
		# C1_clear = False if inner1 >= 0. and inner2 >= 0. else True
		C1_clear = - np.sign(inner1) - np.sign(inner2)  # -2: in
		# D1
		inner1 = self.outer_product(v_A2B2, v_A2D1) * self.outer_product(v_C2D2, v_C2D1)
		inner2 = self.outer_product(v_B2C2, v_B2D1) * self.outer_product(v_D2A2, v_D2D1)
		# D1_clear = False if inner1 >= 0. and inner2 >= 0. else True
		D1_clear = - np.sign(inner1) - np.sign(inner2)  # -2: in
		# Are all vertices of rectangle 2 out of rectangle 1
		# A2
		v_A1A2 = self.get_vector((xA1, yA1), (xA2, yA2))
		v_C1A2 = self.get_vector((xC1, yC1), (xA2, yA2))
		v_B1A2 = self.get_vector((xB1, yB1), (xA2, yA2))
		v_D1A2 = self.get_vector((xD1, yD1), (xA2, yA2))
		inner1 = self.outer_product(v_A1B1, v_A1A2) * self.outer_product(v_C1D1, v_C1A2)
		inner2 = self.outer_product(v_B1C1, v_B1A2) * self.outer_product(v_D1A1, v_D1A2)
		# A2_clear = False if inner1 >= 0. and inner2 >= 0. else True
		A2_clear = - np.sign(inner1) - np.sign(inner2)  # -2: in
		# B2
		v_A1B2 = self.get_vector((xA1, yA1), (xB2, yB2))
		v_C1B2 = self.get_vector((xC1, yC1), (xB2, yB2))
		v_B1B2 = self.get_vector((xB1, yB1), (xB2, yB2))
		v_D1B2 = self.get_vector((xD1, yD1), (xB2, yB2))
		inner1 = self.outer_product(v_A1B1, v_A1B2) * self.outer_product(v_C1D1, v_C1B2)
		inner2 = self.outer_product(v_B1C1, v_B1B2) * self.outer_product(v_D1A1, v_D1B2)
		# B2_clear = False if inner1 >= 0. and inner2 >= 0. else True
		B2_clear = - np.sign(inner1) - np.sign(inner2)  # -2: in
		# C2
		v_A1C2 = self.get_vector((xA1, yA1), (xC2, yC2))
		v_C1C2 = self.get_vector((xC1, yC1), (xC2, yC2))
		v_B1C2 = self.get_vector((xB1, yB1), (xC2, yC2))
		v_D1C2 = self.get_vector((xD1, yD1), (xC2, yC2))
		inner1 = self.outer_product(v_A1B1, v_A1C2) * self.outer_product(v_C1D1, v_C1C2)
		inner2 = self.outer_product(v_B1C1, v_B1C2) * self.outer_product(v_D1A1, v_D1C2)
		# C2_clear = False if inner1 >= 0. and inner2 >= 0. else True
		C2_clear = - np.sign(inner1) - np.sign(inner2)  # -2: in
		# D2
		v_A1D2 = self.get_vector((xA1, yA1), (xD2, yD2))
		v_C1D2 = self.get_vector((xC1, yC1), (xD2, yD2))
		v_B1D2 = self.get_vector((xB1, yB1), (xD2, yD2))
		v_D1D2 = self.get_vector((xD1, yD1), (xD2, yD2))
		inner1 = self.outer_product(v_A1B1, v_A1D2) * self.outer_product(v_C1D1, v_C1D2)
		inner2 = self.outer_product(v_B1C1, v_B1D2) * self.outer_product(v_D1A1, v_D1D2)
		# D2_clear = False if inner1 >= 0. and inner2 >= 0. else True
		D2_clear = - np.sign(inner1) - np.sign(inner2)  # -2: in
		# vertices_clear = True if A1_clear and B1_clear and C1_clear and D1_clear and A2_clear and B2_clear and C2_clear and D2_clear else False
		vertices_clear = np.minimum(np.minimum(np.minimum(A1_clear, B1_clear), np.minimum(C1_clear, D1_clear)),
									np.minimum(np.minimum(A2_clear, B2_clear), np.minimum(C2_clear, D2_clear)))
		# All_clear = True if edges_clear and vertices_clear else False
		All_clear = np.minimum(edges_clear, vertices_clear)  # -2: overlap
		return All_clear.min(0)  # minimum of each column

	def get_centerline_coords(self,
							  scenario: AbstractScenario,
							  anchor_ego_state: EgoState=None,
							  return_reference_lanes: bool=False) \
			-> Tuple[np.ndarray, List[Union[NuPlanLane, NuPlanLaneConnector]]]:
		"""get coordinates of centerline that is the closest to ego vehicle"""
		output = get_centerline_coords(
			scenario,
			100.,
			self.v_max,
			self.a_max,
			self.T,
			anchor_ego_state,
			return_reference_lanes
		)
		if return_reference_lanes:
			reference_line_coords, _, reference_lanes = output
			return reference_line_coords, reference_lanes
		else:
			reference_line_coords, _ = output
			return reference_line_coords

	def build_frenet_frame(self, reference_line_coords: np.ndarray) -> FrenetFrame:
		"""build Frenet frame given coordinates of reference line"""
		return FrenetFrame(reference_line_coords)

	# def build_current_ego_state(self) -> EgoState:
	# 	car_footprint = CarFootprint(StateSE2(self.ego['geo_center']['x'],
	# 										  self.ego['geo_center']['y'],
	# 										  self.ego['geo_center']['heading']),
	# 								 self.scenario.ego_vehicle_parameters)
	# 	dynamic_car_state = DynamicCarState(
	# 		self.scenario.ego_vehicle_parameters.rear_axle_to_center,
	# 		StateVector2D(self.ego['rear_axle']['vx'],
	# 					  self.ego['rear_axle']['vy']),
	# 		StateVector2D(self.ego['rear_axle']['ax'],
	# 					  self.ego['rear_axle']['ay']),
	# 		angular_velocity=self.ego['angular_velocity']
	# 	)
	# 	ego_state = EgoState(car_footprint,
	# 						 dynamic_car_state,
	# 						 tire_steering_angle=0.,
	# 						 is_in_auto_mode=True,
	# 						 time_point=self.time_point)
	#
	# 	return ego_state

	def get_geo_center_pose_cartesian(self, trajectory_sample_cartesian, ego_controller):
		"""Return pose of geometric center in Cartesian frame given pose of rear axle center in Cartesian frame"""
		geo_center_pose_cartesian = np.array([
			trajectory_sample_cartesian['pose_cartesian'][:, 0, :] + ego_controller.get_state().car_footprint.vehicle_parameters.rear_axle_to_center * np.cos(
				trajectory_sample_cartesian['pose_cartesian'][:, 2, :]),
			trajectory_sample_cartesian['pose_cartesian'][:, 1, :] + ego_controller.get_state().car_footprint.vehicle_parameters.rear_axle_to_center * np.sin(
				trajectory_sample_cartesian['pose_cartesian'][:, 2, :]),
			trajectory_sample_cartesian['pose_cartesian'][:, 2, :]
		]).transpose(1, 0, 2)
		return geo_center_pose_cartesian

	def log(self, information: str):
		logger.info(information)

	def visualize(self,
				  features,
				  planned_trajectory_global,
				  show=True,
				  save=False,
				  tag=None,):
		features = {
			key: value.to_device(torch.device('cpu'))
			for key, value in features.items()
		}
		planned_trajectory_global = [traj.cpu() if isinstance(traj, torch.Tensor) else traj for traj in planned_trajectory_global]
		batch_size = features['vector_set_map'].batch_size
		map_data = features['vector_set_map'].map_data
		agent_data = features['generic_agents'].agent_data
		for sample_idx in range(batch_size):
			fig, ax = plt.subplots(1, 1, figsize=(20, 20))

			# plot map
			pt2pl_edge_index = map_data['map_point', 'to', 'map_polygon']['edge_index'][sample_idx].long().numpy()
			unique_pl = np.unique(pt2pl_edge_index[1])
			for pl_index in unique_pl:
				mask = pt2pl_edge_index[1] == pl_index
				plt.plot(map_data['map_point']['position'][sample_idx][mask][:, 0],
						 map_data['map_point']['position'][sample_idx][mask][:, 1])
				plt.scatter(map_data['map_point']['position'][sample_idx][mask][:, 0],
						 map_data['map_point']['position'][sample_idx][mask][:, 1], s=1)

			# plot agents
			num_agents = agent_data['position'][sample_idx].shape[0]
			anchorx = None
			anchory = None
			for i_a in range(num_agents):
				_, _, _, _, xC, yC, _, _ = self.get_vertices_from_center(agent_data['position'][sample_idx][i_a, -1, 0],
																		 agent_data['position'][sample_idx][i_a, -1, 1],
																		 agent_data['heading'][sample_idx][i_a, -1],
																		 agent_data['length'][sample_idx][i_a],
																		 agent_data['width'][sample_idx][i_a],
																		 rad=True)
				heading = agent_data['heading'][sample_idx][i_a, -1]
				length = agent_data['length'][sample_idx][i_a]
				width = agent_data['width'][sample_idx][i_a]
				xy = (xC, yC)
				if i_a == 0:
					color = [0.15, 0.53, 0.79]
				else:
					color = [0.52, 0.52, 0.52]
				rect = patches.Rectangle(xy, length, width, angle=np.rad2deg(heading.numpy()), color=color)
				ax.add_patch(rect)
				plt.scatter(agent_data['position'][sample_idx][i_a, -1, 0],
							agent_data['position'][sample_idx][i_a, -1, 1],
							color='k',
							marker='*')
				if anchorx is None:
					anchorx = xC
					anchory = yC

			# plot planned trajectory
			if len(planned_trajectory_global) == 1:
				plt.plot(planned_trajectory_global[0][sample_idx, :, 0],
						 planned_trajectory_global[0][sample_idx, :, 1],
						 marker='>')
				plt.scatter(planned_trajectory_global[0][sample_idx, 0::10, 0],
							planned_trajectory_global[0][sample_idx, 0::10, 1],
							color='tab:red')
			elif len(planned_trajectory_global) == batch_size:
				if isinstance(planned_trajectory_global[sample_idx], torch.Tensor):
					plt.plot(planned_trajectory_global[sample_idx][:, :, 0].transpose(0, 1),
							 planned_trajectory_global[sample_idx][:, :, 1].transpose(0, 1),
							 marker='>')
					plt.scatter(planned_trajectory_global[sample_idx][:, 0::10, 0].transpose(0, 1),
								planned_trajectory_global[sample_idx][:, 0::10, 1].transpose(0, 1),
								color='tab:red')
				else:
					plt.plot(planned_trajectory_global[sample_idx][:, :, 0].transpose(1, 0),
							 planned_trajectory_global[sample_idx][:, :, 1].transpose(1, 0),
							 marker='>')
					plt.scatter(planned_trajectory_global[sample_idx][:, 0::10, 0].transpose(1, 0),
								planned_trajectory_global[sample_idx][:, 0::10, 1].transpose(1, 0),
								color='tab:red')

			window = 100
			plt.xlim(anchorx - window, anchorx + window)
			plt.ylim(anchory - window, anchory + window)
			# plt.axis('equal')
			if save:
				path = './debug_figs'
				if not os.path.exists(path):
					os.makedirs(path)
				if tag is None:
					plt.savefig(f'{path}/scenario_token: {self.scenario[sample_idx].token} - iteration: {self.iteration[sample_idx]}.png', dpi=600)
				else:
					plt.savefig(f'{path}/scenario_token: {self.scenario[sample_idx].token} - iteration: {self.iteration[sample_idx]} - {tag}.png', dpi=600)

			if show:
				plt.show()
			plt.close()