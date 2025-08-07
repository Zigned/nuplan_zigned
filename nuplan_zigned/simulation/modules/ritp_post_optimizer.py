import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple, cast, Union, Any, Optional

import os
import contextlib
import torch
import torch.nn.functional as F
import numpy as np
import scipy
from cvxopt import solvers, matrix
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('Agg')  # use non-interactive backend
import shapely.geometry as geom
from shapely.geometry.point import Point
from scipy.spatial.distance import cdist
from scipy.special import softmax
import shapely

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput, PlannerInitialization
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.nuplan_map.roadblock import NuPlanRoadBlock
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    get_neighbor_vector_map,
)
from nuplan.common.maps.nuplan_map.utils import (
    extract_roadblock_objects,
    compute_curvature,
)
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    _generate_profile_from_initial_condition_and_derivatives,
)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from nuplan_zigned.utils.env import Env
from nuplan_zigned.utils.frenet_frame_object import FrenetFrame
from nuplan_zigned.training.preprocessing.feature_builders.avrl_feature_builder_utils import (
    get_on_route_indices,
    get_roadblock_successors_given_route,
    extract_proximal_roadblock_objects,
)
from nuplan_zigned.utils.utils import (
    interpolate_polynomial,
    point_to_point_distance,
    efficient_relative_to_absolute_poses,
    efficient_absolute_to_relative_poses,
    polynomial,
    polynomial_derivative,
    smooth_headings,
)
from nuplan_zigned.training.preprocessing.feature_builders.avrl_vector_set_map_builder_utils import (
    visualize,
)
from nuplan_zigned.utils.scenario_manager.scenario_manager import ScenarioManager
from nuplan_zigned.simulation.modules.trajectory_evaluator import TrajectoryEvaluator

class PostOptimizer:
    """
    Model-Driven Post Optimizer.
    """
    def __init__(self,
                 map_features: List[str],
                 num_future_steps: int,
                 frenet_radius: float,
                 num_plans: int,
                 num_modes_for_eval: int,
                 step_interval_for_eval: int,
                 acc_limit: float,
                 acc_exponent: float,
                 dec_limit: float,
                 time_headway: float,
                 safety_margin: float,
                 use_rule_based_refine: bool=False,
                 ) -> None:
        """
        :param map_features: name of map features
        :param num_future_steps: number of future steps
        :param frenet_radius: radius for searching surrounding elements
        :param num_plans: number of trajectories to be planned
        :param num_modes_for_eval: number of modes of surrounding agents' trajectories used for trajectory evaluation
        :param step_interval_for_eval: step interval for trajectory evaluation
        :param acc_limit: maximum acceleration of ego vehicle in IDM
        :param acc_exponent: accleration exponent for ego vehicle in IDM
        :param dec_limit: maximum deceleration of ego vehicle in IDM (positive means deceleration)
        :param time_headway: desired time headway to leading vehicle in IDM
        :param safety_margin: minimum distance to leading vehicle in IDM
        :param use_rule_based_refine: whether to use rules to refine trajectories planned by actor
        """
        self.num_future_steps = num_future_steps
        self.frenet_radius = frenet_radius
        self.num_plans = num_plans
        self.num_modes_for_eval = num_modes_for_eval
        self.step_interval_for_eval = step_interval_for_eval
        self.num_points_for_eval = num_future_steps // step_interval_for_eval + 1
        self.use_rule_based_refine = use_rule_based_refine
        self._polygon_types = map_features
        self._point_types = map_features
        self._traffic_light_statuses = ['GREEN', 'YELLOW', 'RED', 'UNKNOWN']
        self._historical_roadblocks_ids = []

        # params for IDM
        self.acc_limit = acc_limit
        self.acc_exponent = acc_exponent
        self.dec_limit = dec_limit
        self.time_headway = time_headway
        self.safety_margin = safety_margin
        self._previous_v_max = None
        self._v_alpha = 0.9

        # for rule-based traj refine
        self.scenario_manager: ScenarioManager = None
        self._trajectory_evaluator = TrajectoryEvaluator(dt=0.1, num_frames=40) if use_rule_based_refine else None
        self.interested_objects_types = [
            TrackedObjectType.EGO,
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        ]

    def forward(self,
                features: FeaturesType,
                pred: Dict[str, List[torch.Tensor]],
                current_input: PlannerInput,
                initialization: PlannerInitialization,
                scenario: Optional[AbstractScenario]) -> Dict[str, Any]:
        map_api = initialization.map_api
        mission_goal = initialization.mission_goal
        route_roadblock_ids = initialization.route_roadblock_ids
        sample_idx = 0  # batch size is alway 1 in each simulation
        agent_data = features['generic_agents'].agent_data
        av_index = agent_data['av_index'][sample_idx]
        num_agents = agent_data['num_nodes'][sample_idx]

        # build Frenet frame
        frenet_frame, reference_line_lanes, sth_to_plot = self.build_frenet_frame(
            current_input,
            map_api,
            mission_goal,
            route_roadblock_ids
        )

        # transform agents' trajectories into those in global frame
        current_ego_state = current_input.history.ego_states[-1]
        current_observations = current_input.history.observations[-1]
        current_observations_in_pred = []
        current_observations_states_in_pred = []
        tracked_objects_ids = [obj.track_token for obj in current_observations.tracked_objects.tracked_objects]
        for agent_id in agent_data['id'][sample_idx][av_index + 1:]:
            index = tracked_objects_ids.index(agent_id)
            current_observations_in_pred.append(current_observations.tracked_objects.tracked_objects[index])
            current_observations_states_in_pred.append(current_observations.tracked_objects.tracked_objects[index].center)
        # current_states = [current_ego_state.rear_axle] + current_observations_states_in_pred
        current_states = [current_ego_state.center] + current_observations_states_in_pred
        current_poses = np.array([current_state.serialize() for current_state in current_states])
        pi = pred['pi'][sample_idx]
        prob = F.softmax(pi, dim=-1)
        # sort according to predicted probabilities
        argsort = torch.argsort(prob, dim=-1, descending=True)
        agent_indices = torch.arange(num_agents).unsqueeze(-1).expand(-1, prob.shape[1])
        pred_positions = pred['loc_refine_pos'][sample_idx]
        prob = prob[agent_indices, argsort][:, :self.num_modes_for_eval]
        pred_positions = pred_positions[agent_indices, argsort][:, :self.num_modes_for_eval]
        num_modes = pred_positions.shape[1]
        pred_heading = torch.atan2(pred_positions[:, :, 1:, 1] - pred_positions[:, :, 0:-1, 1],
                                   pred_positions[:, :, 1:, 0] - pred_positions[:, :, 0:-1, 0])
        pred_heading = torch.cat([pred_heading, pred_heading[:, :, -1:]], dim=-1)
        pred_poses = torch.cat([pred_positions, pred_heading.unsqueeze(-1)], dim=-1).cpu().numpy()
        # smooth heading
        num_poses = pred_poses.shape[2]
        current_plus_pred_poses = smooth_headings(
            np.concatenate(
                (np.broadcast_to(np.zeros((1, 1, 1)), (num_agents * num_modes, 1, 3)),
                 pred_poses.reshape(-1, num_poses, 3)), axis=1
            )
        ).reshape(num_agents, num_modes, num_poses + 1, 3)
        pred_poses = current_plus_pred_poses[:, :, 1:, :]
        # local to global
        pred_absolute_poses = efficient_relative_to_absolute_poses(current_poses, pred_poses)
        agents_absolute_poses = np.concatenate(
            [np.broadcast_to(np.expand_dims(current_poses, axis=(1, 2)), shape=(num_agents, num_modes, 1, 3)),
             pred_absolute_poses],
            axis=2
        )

        # transform agents' trajectories into those in Frenet frame
        agents_points = [Point2D(pose[0], pose[1]) for pose in agents_absolute_poses.reshape((-1, 3))]
        agents_stations = frenet_frame.get_nearest_station_from_position(agents_points)
        reference_line_poses = frenet_frame.get_nearest_pose_from_position(agents_points, agents_stations)
        reference_line_headings = np.array([ref_line_pose.heading for ref_line_pose in reference_line_poses])
        agents_laterals = frenet_frame.get_lateral_from_position(agents_points, agents_stations)
        agents_heading_frenet = agents_absolute_poses[:, :, :, 2:] - reference_line_headings.reshape((num_agents, num_modes, self.num_future_steps + 1, 1))
        agents_poses_frenet = np.concatenate([
            agents_stations.reshape((num_agents, num_modes, self.num_future_steps + 1, 1)),
            agents_laterals.reshape((num_agents, num_modes, self.num_future_steps + 1, 1)),
            agents_heading_frenet], axis=-1)
        ego_poses_frenet = agents_poses_frenet[av_index, :self.num_plans]
        # fix station jump
        stations_diff = ego_poses_frenet[:, 1:, 0] - ego_poses_frenet[:, :-1, 0]
        stations_jump = stations_diff / 0.1 > 30.
        stations_jump_idx = np.where(stations_jump)
        for i, idx in zip(*stations_jump_idx):
            ego_poses_frenet[i, idx + 1:] = ego_poses_frenet[i, idx:idx + 1]

        # identify overlap region
        ego_dim = {
            'length': current_input.history.current_state[0].agent.box.length,
            'width': current_input.history.current_state[0].agent.box.width,
            'rear_axle_to_center_dist': current_input.history.current_state[0].car_footprint.rear_axle_to_center_dist,
            'rear_axle_to_front_dist': current_input.history.current_state[0].agent.box.length / 2 +
                                       current_input.history.current_state[0].car_footprint.rear_axle_to_center_dist,
        }
        obs_dim = {
            'length': [],
            'width': [],
        }
        tracked_objects = {
            obj.track_token: obj for obj in current_input.history.current_state[1].tracked_objects.tracked_objects
        }
        for pred_object_id in agent_data['id'][sample_idx][av_index + 1:]:
            assert pred_object_id in tracked_objects
            obs_dim['length'].append(tracked_objects[pred_object_id].box.length)
            obs_dim['width'].append(tracked_objects[pred_object_id].box.width)
        obs_dim['length'] = np.array(obs_dim['length'])
        obs_dim['width'] = np.array(obs_dim['width'])
        overlap_region = self.get_overlap_region(
            agents_poses_frenet[av_index, :self.num_plans],
            agents_poses_frenet[av_index + 1:],
            ego_dim,
            obs_dim,
        )

        # ego poses at center -> ego poses at rear axle
        ego_poses_frenet = self.center_to_rear_axle(ego_poses_frenet, ego_dim['rear_axle_to_center_dist'])

        # rule-based trajectory refine
        if self.use_rule_based_refine:
            # obs info
            obs_track_tokens = agent_data['id'][sample_idx][av_index + 1:]
            obs_info = self._get_obs_info(obs_track_tokens, tracked_objects, obs_dim, agents_absolute_poses)
            # update scenario manager (borrowed from PLUTO)
            ego_state = current_input.history.ego_states[-1]
            self.scenario_manager.update_ego_state(ego_state)
            self.scenario_manager.update_drivable_area_map()
            route_roadblocks_ids = self.scenario_manager.get_route_roadblock_ids()  # initialize
            reference_lines = self.scenario_manager.get_reference_lines(length=100)  # cache reference_lines
            # rule-based refine
            ego_poses_frenet = self.rule_based_traj_refine(current_input,
                                                           ego_poses_frenet,
                                                           reference_line_lanes,
                                                           frenet_frame,
                                                           obs_info)

        # QP path optimization
        qp_paths, qp_paths_succeeded, qp_paths_costs = self.qp_path(ego_dim,
                                                                    ego_poses_frenet,
                                                                    overlap_region,
                                                                    frenet_frame,
                                                                    reference_line_lanes)

        # QP speed optimization
        qp_speeds, qp_speeds_succeeded, qp_speeds_costs = self.qp_speed(current_ego_state,
                                                                        ego_dim,
                                                                        obs_dim,
                                                                        ego_poses_frenet,
                                                                        agents_poses_frenet[av_index + 1:, :self.num_modes_for_eval],
                                                                        frenet_frame,
                                                                        reference_line_lanes,
                                                                        features,
                                                                        current_input.traffic_light_data,
                                                                        qp_paths)

        # QP solutions
        qp_solutions, qp_costs = [], []
        qp_path_solutions, qp_path_costs = [], []
        qp_path_solution = None
        qp_succeeded = [path and speed for path, speed in zip(qp_paths_succeeded, qp_speeds_succeeded)]
        for i_plan in range(len(qp_succeeded)):
            if qp_succeeded[i_plan]:
                f_l_s = interp1d(qp_paths[i_plan][:, 0], qp_paths[i_plan][:, 1], fill_value='extrapolate')
                stations = qp_speeds[i_plan][:, 1]
                laterals = f_l_s(stations)
                headings = np.arctan2(np.diff(laterals[1:]), np.diff(stations[1:]))
                headings = np.hstack((ego_poses_frenet[i_plan, 0, 2], headings, headings[-1]))
                qp_solutions.append(np.vstack((stations, laterals, headings)))
                qp_costs.append(qp_paths_costs[i_plan] + qp_speeds_costs[i_plan])
            if qp_paths_succeeded[i_plan]:
                qp_path_solutions.append(qp_paths[i_plan])
                qp_path_costs.append(qp_paths_costs[i_plan])
        if len(qp_path_costs) > 0:
            opt_idx = np.argmin(qp_path_costs)
            qp_path_solution = self.rear_axle_to_center(qp_path_solutions[opt_idx], ego_dim['rear_axle_to_center_dist'])
        if len(qp_costs) > 0:
            opt_idx = np.argmin(qp_costs)
            qp_solution = qp_solutions[opt_idx].T
            qp_solution = np.expand_dims(qp_solution, axis=0)

            # rear axle to center
            qp_solution = self.rear_axle_to_center(qp_solution, ego_dim['rear_axle_to_center_dist'])

            # Frenet frame to global frame
            qp_solution_global = frenet_frame.frenet_to_cartesian(qp_solution.transpose(0, 2, 1))['pose_cartesian'][0].T

            # global frame to local frame
            qp_solution_relative = efficient_absolute_to_relative_poses(current_poses[av_index:av_index + 1],
                                                                        qp_solution_global[1:][np.newaxis, np.newaxis, :])
            qp_solution_relative = np.squeeze(qp_solution_relative)
            planned_traj = Trajectory(data=qp_solution_relative[np.newaxis,])
        else:
            # IDM as alternative
            idm_solution_global, idm_solution = self.idm(
                current_ego_state,
                current_observations_in_pred,
                agents_poses_frenet[:, 0, 0, :],
                ego_dim,
                obs_dim,
                frenet_frame,
                reference_line_lanes,
                current_input.traffic_light_data,
                qp_path_solution
            )
            # planned_traj = Trajectory(data=pred_poses[av_index, 0:1])
            planned_traj = Trajectory(data=idm_solution)

        # TODO debug only
        # if current_input.iteration.index % 5 == 0:
        if False:
            qp_path_solution_global = None
            if len(qp_path_costs) > 0:
                opt_idx = np.argmin(qp_path_costs)
                qp_path_solution = qp_path_solutions[opt_idx]
                qp_path_solution = np.expand_dims(qp_path_solution, axis=0)

                # rear axle to center
                qp_path_solution = self.rear_axle_to_center(qp_path_solution, ego_dim['rear_axle_to_center_dist'])

                # Frenet frame to global frame
                qp_path_solution_global = frenet_frame.frenet_to_cartesian(qp_path_solution.transpose(0, 2, 1))['pose_cartesian'][0].T
            ego_poses_refined = None
            if self.use_rule_based_refine:
                ego_poses_refined = frenet_frame.frenet_to_cartesian(ego_poses_frenet.transpose(0, 2, 1))['pose_cartesian'][0].T
            try:
                trajs = {
                    'ego_traj_pred': agents_absolute_poses[av_index],
                    'refined_pred': ego_poses_refined,
                    'qp_path': qp_path_solution_global,
                    'ego_traj_optimized': qp_solution_global,
                    'obs_trajs': agents_absolute_poses[av_index + 1:],
                }
            except:
                try:
                    trajs = {
                        'ego_traj_pred': agents_absolute_poses[av_index],
                        'refined_pred': ego_poses_refined,
                        'qp_path': qp_path_solution_global,
                        'ego_traj_optimized': idm_solution_global,
                        'obs_trajs': agents_absolute_poses[av_index + 1:],
                    }
                except:
                    trajs = {
                        'ego_traj_pred': agents_absolute_poses[av_index],
                        'refined_pred': ego_poses_refined,
                        'qp_path': qp_path_solution_global,
                        'ego_traj_optimized': None,
                        'obs_trajs': agents_absolute_poses[av_index + 1:],
                    }
            self.visualize(
                current_input,
                scenario,
                features,
                frenet_frame,
                overlap_region,
                trajs,
                ego_dim,
                obs_dim,
                show=False,
                save=True
            )

            # TODO debug only: visualization
            sth_to_plot['scenario'] = scenario
            sth_to_plot['anchor_ego_state'] = current_ego_state
            sth_to_plot['iteration'] = current_input.iteration.index
            visualize(sth_to_plot)

        return {'trajectory': planned_traj}

    def build_frenet_frame(self, current_input, map_api, mission_goal, route_roadblock_ids):
        dt = 0.1
        T = self.num_future_steps * dt  # time_horizon
        num_points = int(T / dt + 1)
        v_max = 90 / 3.6
        a_max = 4.
        # find neighbor map objects within neighbor_radius
        current_ego_state = current_input.history.current_state[0]
        current_ego_coords = Point2D(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)
        v_max = min(v_max, current_ego_state.dynamic_car_state.speed + a_max * T)
        s_max = 0.5 * (current_ego_state.dynamic_car_state.speed + v_max) * T
        neighbor_radius = max(s_max, self.frenet_radius)
        (
            lane_seg_coords,  # centerlines of lanes
            lane_seg_conns,
            lane_seg_groupings,
            lane_seg_lane_ids,  # a lane consists of many lane segments
            lane_seg_roadblock_ids,  # a roadblock consists of many lanes
        ) = get_neighbor_vector_map(map_api, current_ego_coords, neighbor_radius)

        # find proximal route roadblock
        route_roadblocks: List[NuPlanRoadBlock] = [map_api.get_map_object(id, SemanticMapLayer.ROADBLOCK) for id in route_roadblock_ids]
        route_roadblock_centroids: List[Point] = [rb.polygon.centroid for rb in route_roadblocks]
        current_ego_point = Point(current_ego_state.rear_axle.x, current_ego_state.rear_axle.y)
        distance_to_route_roadblocks = [current_ego_point.distance(centroid) for centroid in route_roadblock_centroids]
        proximal_route_roadblock_index = np.argmin(distance_to_route_roadblocks)
        proximal_route_roadblock: NuPlanRoadBlock = route_roadblocks[proximal_route_roadblock_index]
        predecessors_of_proximal_route_rb = proximal_route_roadblock.incoming_edges
        successors_of_proximal_route_rb = proximal_route_roadblock.outgoing_edges
        ancestors_of_proximal_route_rb = [*predecessors_of_proximal_route_rb]
        descendants_of_proximal_route_rb = [*successors_of_proximal_route_rb]
        for rb in predecessors_of_proximal_route_rb:
            ancestors_of_proximal_route_rb += rb.incoming_edges
            for rb2 in rb.incoming_edges:
                ancestors_of_proximal_route_rb += rb2.incoming_edges
                for rb3 in rb2.incoming_edges:
                    ancestors_of_proximal_route_rb += rb3.incoming_edges
                    for rb4 in rb3.incoming_edges:
                        ancestors_of_proximal_route_rb += rb4.incoming_edges
        for rb in successors_of_proximal_route_rb:
            descendants_of_proximal_route_rb += rb.outgoing_edges
            for rb2 in rb.outgoing_edges:
                descendants_of_proximal_route_rb += rb2.outgoing_edges
                for rb3 in rb2.outgoing_edges:
                    descendants_of_proximal_route_rb += rb3.outgoing_edges
                    for rb4 in rb3.outgoing_edges:
                        descendants_of_proximal_route_rb += rb4.outgoing_edges
        ancestors_rb_ids = [rb.id for rb in ancestors_of_proximal_route_rb]
        descendants_rb_ids = [rb.id for rb in descendants_of_proximal_route_rb]

        # find on route lane seg indices
        on_point_roadblocks = extract_roadblock_objects(map_api, current_ego_state.rear_axle.point)
        if len(on_point_roadblocks) == 0:
            on_point_roadblocks = extract_proximal_roadblock_objects(map_api, current_ego_state.rear_axle.point, 3.)
        on_point_roadblocks_back_up = on_point_roadblocks

        proximal_route_roadblock_as_on_point_roadblocks = False
        if len(on_point_roadblocks) == 0:
            proximal_route_roadblock_as_on_point_roadblocks = True
        elif (np.all([rb.id not in route_roadblock_ids for rb in on_point_roadblocks])
              and np.all([rb.id not in ancestors_rb_ids for rb in on_point_roadblocks])
              and np.all([rb.id not in descendants_rb_ids for rb in on_point_roadblocks])):
            # none on_point_roadblocks is in route_roadblock_ids or ancestors_rb_ids or descendants_rb_ids
            proximal_route_roadblock_as_on_point_roadblocks = True
        if proximal_route_roadblock_as_on_point_roadblocks:
            on_point_roadblocks = [proximal_route_roadblock]

        if len(route_roadblock_ids) > 0:
            roadblock_successors_given_route = get_roadblock_successors_given_route(
                route_roadblock_ids,
                on_point_roadblocks,
                current_ego_state=current_ego_state,
                historical_roadblocks_ids=self._historical_roadblocks_ids)
            on_route_roadblocks = roadblock_successors_given_route['on_route_roadblocks']
            routes = roadblock_successors_given_route['routes']
        else:
            # on_route_roadblocks = []
            # raise ValueError("on_route_roadblocks is empty")
            on_route_roadblocks = on_point_roadblocks
            routes = on_point_roadblocks
        on_route_roadblock_ids = [roadblock.id for roadblock in on_route_roadblocks]
        routes_roadblock_ids = [[roadblock.id for roadblock in route] for route in routes]
        on_route_indices = get_on_route_indices(routes_roadblock_ids, lane_seg_roadblock_ids, inputting_routes=True)

        if len(on_route_indices) == 0:
            # handle when on_route_indices is empty (probably because on_route_roadblocks are too far away)
            if len(route_roadblock_ids) > 0:
                roadblock_successors_given_route = get_roadblock_successors_given_route(
                    route_roadblock_ids,
                    on_point_roadblocks_back_up,
                    current_ego_state=current_ego_state,
                    historical_roadblocks_ids=self._historical_roadblocks_ids)
                on_route_roadblocks = roadblock_successors_given_route['on_route_roadblocks']
                routes = roadblock_successors_given_route['routes']
            else:
                on_route_roadblocks = on_point_roadblocks_back_up
                routes = on_point_roadblocks
            on_route_roadblock_ids = [roadblock.id for roadblock in on_route_roadblocks]
            routes_roadblock_ids = [[roadblock.id for roadblock in route] for route in routes]
            on_route_indices = get_on_route_indices(routes_roadblock_ids, lane_seg_roadblock_ids, inputting_routes=True)

        # extract a baseline path that starts at the current ego position
        on_route_lane_seg_groupings = []
        on_route_lane_seg_grouping_start_indices = []
        on_route_lane_seg_grouping_end_indices = []
        for lane_seg_grouping in lane_seg_groupings.groupings:
            if lane_seg_grouping[0] in on_route_indices or lane_seg_grouping[-1] in on_route_indices:
                on_route_lane_seg_groupings.append(lane_seg_grouping)
                on_route_lane_seg_grouping_start_indices.append(lane_seg_grouping[0])
                on_route_lane_seg_grouping_end_indices.append(lane_seg_grouping[-1])
        on_route_lane_seg_grouping_start_coord_array = np.array(
            [[lane_seg_coords.coords[index][0].array, lane_seg_coords.coords[index][1].array]
             for index in on_route_lane_seg_grouping_start_indices])
        on_route_lane_seg_grouping_end_coord_array = np.array(
            [[lane_seg_coords.coords[index][0].array, lane_seg_coords.coords[index][1].array]
             for index in on_route_lane_seg_grouping_end_indices])
        current_ego_coord_array = np.array([current_ego_state.rear_axle.x, current_ego_state.rear_axle.y])

        # find the closest four lane seg groups (current lane seg groups)
        on_route_lane_seg_grouping_coord_array = []
        on_route_lane_seg_groupings_indices = []
        on_route_lane_ids = []
        for on_route_lane_seg_grouping in on_route_lane_seg_groupings:
            coord_tmp = []
            lane_id_tmp = None
            for index in on_route_lane_seg_grouping:
                coord_tmp.append([lane_seg_coords.coords[index][0].array, lane_seg_coords.coords[index][1].array])
                if lane_id_tmp is None:
                    lane_id_tmp = lane_seg_lane_ids.lane_ids[index]
            on_route_lane_seg_grouping_coord_array = on_route_lane_seg_grouping_coord_array + coord_tmp
            if len(on_route_lane_seg_groupings_indices) == 0:
                on_route_lane_seg_groupings_indices.append(np.arange(len(coord_tmp)))
            else:
                on_route_lane_seg_groupings_indices.append(np.arange(on_route_lane_seg_groupings_indices[-1][-1] + 1,
                                                                     on_route_lane_seg_groupings_indices[-1][
                                                                         -1] + 1 + len(coord_tmp)))
            on_route_lane_ids.append(lane_id_tmp)
        on_route_lane_seg_grouping_coord_array = np.array(on_route_lane_seg_grouping_coord_array)
        distance_to_ego = point_to_point_distance(current_ego_coord_array,
                                                  on_route_lane_seg_grouping_coord_array[:, 0, :])
        grouping_wise_min_distance = []
        for indices in on_route_lane_seg_groupings_indices:
            grouping_wise_min_distance.append(min(distance_to_ego[indices]))
        minimum_distance = min(grouping_wise_min_distance)
        closest_four_lane_seg_grouping_indices = np.where([dist <= 3. for dist in grouping_wise_min_distance])[0][:4]
        if len(closest_four_lane_seg_grouping_indices) == 0:
            closest_four_lane_seg_grouping_indices = np.where(grouping_wise_min_distance == minimum_distance)[0]
        closest_four_lane_ids = [on_route_lane_ids[index] for index in closest_four_lane_seg_grouping_indices]
        closest_four_lanes = []
        for lane_id in closest_four_lane_ids:
            lane_or_lane_connector = map_api._get_lane(lane_id)
            if lane_or_lane_connector is None:
                lane_or_lane_connector = map_api._get_lane_connector(lane_id)
            closest_four_lanes.append(lane_or_lane_connector)
        # search for the longest lane sequence
        reference_line_candidates = []
        for lane in closest_four_lanes:
            stack = [(lane, [lane.id], 1)]  # (current node, current path, current depth)
            max_depth = 0
            result = []
            while stack:
                node, path, depth = stack.pop()
                if depth > max_depth:
                    max_depth = depth
                    result = [path]
                elif depth == max_depth:
                    result.append(path)
                if depth >= 15:
                    continue
                on_route_outgoing_edge = [edge for edge in node.outgoing_edges if edge.id in on_route_lane_ids]
                # for edge in on_route_outgoing_edge:
                #     if edge.id not in path:  # in case of loop
                #         stack.append((edge, path + [edge.id], depth + 1))
                for edge in on_route_outgoing_edge:
                    outgoing_edges_not_in_path = True
                    if len(edge.outgoing_edges) > 0:
                        outgoing_edges_not_in_path = [e.id not in path for e in edge.outgoing_edges]
                        outgoing_edges_not_in_path = bool(np.all(outgoing_edges_not_in_path))
                    if edge.id not in path and outgoing_edges_not_in_path:  # in case of loop
                        stack.append((edge, path + [edge.id], depth + 1))

            # lane_id to lane or lane_connector object
            lanes = []
            for lane_id in result[0]:  # take the first when multiple equal-length routes
                lane_or_lane_connector = map_api._get_lane(lane_id)
                if lane_or_lane_connector is None:
                    lane_or_lane_connector = map_api._get_lane_connector(lane_id)
                lanes.append(lane_or_lane_connector)
            reference_line_candidates.append(lanes)
        lengths = [len(candidate) for candidate in reference_line_candidates]
        max_length = max(lengths)
        arg_longest = np.where(np.array(lengths) == max_length)[0]
        distance_to_ego = []
        for index in arg_longest:
            coords = np.array([s.array for s in reference_line_candidates[index][0].baseline_path.discrete_path])
            distance_to_ego.append(np.min(np.linalg.norm(current_ego_coord_array - coords, axis=1)))
        arg_closet = arg_longest[np.argmin(distance_to_ego)]
        reference_line_lanes = reference_line_candidates[arg_closet]

        # manually set adjacent edges of lane connectors
        for edge in reference_line_lanes:
            if isinstance(edge, NuPlanLaneConnector):
                parent_roadblock_id = edge.parent.id
                incoming_edges_ids = [e.id for e in edge.incoming_edges]
                outgoing_edges_ids = [e.id for e in edge.outgoing_edges]
                excluded_ids = incoming_edges_ids + outgoing_edges_ids + [edge.id]
                layer_names = [SemanticMapLayer.LANE_CONNECTOR]
                middle_index = len(edge.baseline_path.discrete_path) // 2
                point1 = edge.baseline_path.discrete_path[middle_index].point
                heading1 = edge.baseline_path.discrete_path[middle_index].heading
                nearest_lane_connectors = map_api.get_proximal_map_objects(point=point1, radius=4, layers=layer_names)
                nearest_lane_connectors = nearest_lane_connectors[SemanticMapLayer.LANE_CONNECTOR]
                nearest_lane_connectors = [l for l in nearest_lane_connectors if l.id not in excluded_ids and l.parent.id == parent_roadblock_id]
                len_vector1_2 = 2.
                point2 = Point2D(x=point1.x + len_vector1_2 * np.cos(heading1), y=point1.y + len_vector1_2 * np.sin(heading1))
                vector1_2 = (point2.x - point1.x, point2.y - point1.y)
                edge.custom_adjacent_edges = [None, None]
                distances = [None, None]
                for l in nearest_lane_connectors:
                    middle_index = len(l.baseline_path.discrete_path) // 2
                    point3 = l.baseline_path.discrete_path[middle_index].point
                    vector1_3 = (point3.x - point1.x, point3.y - point1.y)
                    outter_product = vector1_2[0] * vector1_3[1] - vector1_2[1] * vector1_3[0]
                    dist = (vector1_2[0] * vector1_3[1] + vector1_2[1] * vector1_3[1]) / len_vector1_2  # |b|∙cos(𝜃) = a∙b / |a|
                    if outter_product >= 0:
                        if edge.custom_adjacent_edges[0] is None or dist < distances[0]:
                            edge.custom_adjacent_edges[0] = l
                            distances[0] = dist
                    else:
                        if edge.custom_adjacent_edges[1] is None or dist < distances[1]:
                            edge.custom_adjacent_edges[1] = l
                            distances[1] = dist

        # extract reference_line_coords
        reference_line_coords = []
        reference_line_lane_ids = []
        for lane in reference_line_lanes:
            reference_line_coords = reference_line_coords + [statese2.array for statese2 in lane.baseline_path.discrete_path]
            reference_line_lane_ids = reference_line_lane_ids + [lane.id]
        reference_line_coords = np.array(reference_line_coords)
        diff = reference_line_coords[1:] - reference_line_coords[:-1]
        duplicated_indices = np.where((diff[:, 0] == 0.) & (diff[:, 1] == 0.))[0]
        reference_line_coords = np.delete(reference_line_coords, duplicated_indices, axis=0)

        # extend reference line to make sure it is longer than left and right boundaries (useful for identifying Feasible Tunnel)
        extend_length = 10.
        start_heading = np.arctan2(reference_line_coords[1, 1] - reference_line_coords[0, 1],
                                   reference_line_coords[1, 0] - reference_line_coords[0, 0])
        end_heading = np.arctan2(reference_line_coords[-1, 1] - reference_line_coords[-2, 1],
                                   reference_line_coords[-1, 0] - reference_line_coords[-2, 0])
        prefix = copy.deepcopy(reference_line_coords[0:1, :])
        suffix = copy.deepcopy(reference_line_coords[-1:, :])
        prefix[0, 0] -= extend_length * np.cos(start_heading)
        prefix[0, 1] -= extend_length * np.sin(start_heading)
        suffix[0, 0] += extend_length * np.cos(end_heading)
        suffix[0, 1] += extend_length * np.sin(end_heading)
        reference_line_coords = np.concatenate([prefix, reference_line_coords, suffix], axis=0)

        # build Frenet frame
        frenet_frame = FrenetFrame(reference_line_coords)

        # record on_point_roadblocks
        self._historical_roadblocks_ids += [rb.id for rb in on_point_roadblocks if rb.id not in self._historical_roadblocks_ids]

        sth_to_plot = None
        # TODO debug only: sth to plot
        # if True:
        if False:
            sth_to_plot = {
                'lane_seg_coords': lane_seg_coords,
                'on_route_roadblocks': on_route_roadblocks,
                'roadblock_successors_given_route': roadblock_successors_given_route,
                'lane_seg_roadblock_ids': lane_seg_roadblock_ids,
                'route_roadblocks': route_roadblocks,
                'route_roadblock_ids': route_roadblock_ids,
                'reference_line_lanes': reference_line_lanes,
                'on_route_indices': on_route_indices,
                'on_route_lane_seg_grouping_start_coord_array': on_route_lane_seg_grouping_start_coord_array,
                'on_route_lane_seg_grouping_end_coord_array': on_route_lane_seg_grouping_end_coord_array,
                'reference_line_coords': reference_line_coords,
            }

        return frenet_frame, reference_line_lanes, sth_to_plot

    def get_overlap_region(self, ego_trajs, obs_trajs, ego_dim, obs_dim):
        num_plans = ego_trajs.shape[0]
        num_poses = ego_trajs.shape[1]
        num_modes = obs_trajs.shape[1]
        obs_trajs = obs_trajs.reshape(-1, num_poses, 3)
        num_obs = obs_trajs.shape[0]
        obs_dim = copy.deepcopy(obs_dim)
        obs_dim['length'] = np.expand_dims(np.repeat(obs_dim['length'], num_modes), axis=-1)
        obs_dim['width'] = np.expand_dims(np.repeat(obs_dim['width'], num_modes), axis=-1)

        # ignore obs following ego
        ego_obs_lateral_dist = np.abs(ego_trajs[0, 0, 1] - obs_trajs[:, 0, 1])
        same_lane_mask = ego_obs_lateral_dist <= (ego_dim['width'] / 2 + obs_dim['width'] / 2).reshape(-1)
        behind_mask = ego_trajs[0, 0, 0] > obs_trajs[:, 0, 0]
        valid_mask = ~(same_lane_mask & behind_mask)

        list_overlap_smin = []
        list_overlap_smax = []
        list_overlap_lmin = []
        list_overlap_lmax = []
        list_overlaps = []

        for ego_traj in ego_trajs:
            sA_ego, lA_ego, sB_ego, lB_ego, \
                sC_ego, lC_ego, sD_ego, lD_ego = Env.get_vertices_from_center(
                ego_traj[:, 0], ego_traj[:, 1], ego_traj[:, 2], ego_dim['length'], ego_dim['width'], rad=True
            )
            sA_obs, lA_obs, sB_obs, lB_obs, \
                sC_obs, lC_obs, sD_obs, lD_obs = Env.get_vertices_from_center(
                obs_trajs[valid_mask][:, :, 0],
                obs_trajs[valid_mask][:, :, 1],
                obs_trajs[valid_mask][:, :, 2],
                obs_dim['length'][valid_mask],
                obs_dim['width'][valid_mask],
                rad=True
            )
            overlap_s, overlap_l = self.get_overlap_sl(sA_ego, sB_ego, sC_ego, sD_ego,
                                                       sA_obs, sB_obs, sC_obs, sD_obs,
                                                       lA_obs, lB_obs, lC_obs, lD_obs)

            overlap_smin = np.amin(overlap_s, axis=1)
            overlap_smax = np.amax(overlap_s, axis=1)
            overlap_lmin = np.amin(overlap_l, axis=1)
            overlap_lmax = np.amax(overlap_l, axis=1)
            overlaps = overlap_smax != 0  # bool

            list_overlap_smin.append(overlap_smin)
            list_overlap_smax.append(overlap_smax)
            list_overlap_lmin.append(overlap_lmin)
            list_overlap_lmax.append(overlap_lmax)
            list_overlaps.append(overlaps)

        return {
            'smin': list_overlap_smin,
            'smax': list_overlap_smax,
            'lmin': list_overlap_lmin,
            'lmax': list_overlap_lmax,
            'overlaps': list_overlaps
        }

    def rule_based_traj_refine(self,
                               current_input,
                               ego_poses_frenet,
                               reference_line_lanes,
                               frenet_frame,
                               obs_info):
        refined_trajs = []

        # generate candidate trajs based on piece-wise polynomials
        num_l_tgt_per_lane = 11
        candidate_trajs = self._get_candidate_trajs(current_input, ego_poses_frenet, reference_line_lanes, num_l_tgt_per_lane)

        # trim trajs for rule-based evaluation: trajs[0:num_eval_frames]
        for i_plan in range(len(candidate_trajs)):
            trajs = candidate_trajs[i_plan]
            # keep
            s_keep_seg0 = trajs['s_keep']['seg0']
            l_s_keep_seg0 = trajs['l_s_keep']['seg0'][0]
            segment_count_keep = 1
            if s_keep_seg0.shape[0] < self._trajectory_evaluator.num_frames + 1:
                s_keep_seg1 = trajs['s_keep']['seg1']
                l_s_keep_seg1 = np.concatenate(trajs['l_s_keep']['seg1'], axis=0)
                l_s_keep_seg0 = np.repeat(l_s_keep_seg0, num_l_tgt_per_lane, axis=0)
                s_keep = np.concatenate((s_keep_seg0, s_keep_seg1), axis=0)
                l_s_keep = np.concatenate((l_s_keep_seg0, l_s_keep_seg1), axis=1)
                segment_count_keep += 1
                s_keep = np.broadcast_to(s_keep, l_s_keep.shape)
                if s_keep.shape[1] < self._trajectory_evaluator.num_frames + 1:
                    # if still not enough, extend
                    s_keep_ = np.concatenate([seg for seg in trajs['s_keep'].values() if seg is not None], axis=0)
                    l_s_keep_extend = np.broadcast_to(l_s_keep[:, -1:], (
                        l_s_keep.shape[0], self._trajectory_evaluator.num_frames + 1 - l_s_keep.shape[1]))
                    l_s_keep_ = np.concatenate((l_s_keep, l_s_keep_extend), axis=1)
                    s_keep_ = s_keep_[:self._trajectory_evaluator.num_frames + 1]
                    s_keep_trim = np.broadcast_to(s_keep_[np.newaxis], l_s_keep_.shape)
                    l_s_keep_trim = l_s_keep_[:, :self._trajectory_evaluator.num_frames + 1]
                else:
                    s_keep_trim = s_keep[:, :self._trajectory_evaluator.num_frames + 1]
                    l_s_keep_trim = l_s_keep[:, :self._trajectory_evaluator.num_frames + 1]
            else:
                s_keep = s_keep_seg0
                l_s_keep = l_s_keep_seg0
                s_keep = np.broadcast_to(s_keep, l_s_keep.shape)
                s_keep_trim = s_keep[:, :self._trajectory_evaluator.num_frames + 1]
                l_s_keep_trim = l_s_keep[:, :self._trajectory_evaluator.num_frames + 1]
            segment_count_keep_eval = segment_count_keep
            # left
            s_left_seg0 = trajs['s_left']['seg0']
            l_s_left_seg0 = trajs['l_s_left']['seg0'][0]
            segment_count_left = 1
            if s_left_seg0.shape[0] < self._trajectory_evaluator.num_frames + 1:
                s_left_seg1 = trajs['s_left']['seg1']
                l_s_left_seg1 = np.concatenate(trajs['l_s_left']['seg1'], axis=0)
                l_s_left_seg0 = np.repeat(l_s_left_seg0, num_l_tgt_per_lane, axis=0)
                s_left = np.concatenate((s_left_seg0, s_left_seg1), axis=0)
                l_s_left = np.concatenate((l_s_left_seg0, l_s_left_seg1), axis=1)
                segment_count_left += 1
            else:
                s_left = s_left_seg0
                l_s_left = l_s_left_seg0
            s_left = np.broadcast_to(s_left, l_s_left.shape)
            s_left_trim = s_left[:, :self._trajectory_evaluator.num_frames + 1]
            l_s_left_trim = l_s_left[:, :self._trajectory_evaluator.num_frames + 1]
            segment_count_left_eval = segment_count_left
            # right
            s_right_seg0 = trajs['s_right']['seg0']
            l_s_right_seg0 = trajs['l_s_right']['seg0'][0]
            segment_count_right = 1
            if s_right_seg0.shape[0] < self._trajectory_evaluator.num_frames + 1:
                s_right_seg1 = trajs['s_right']['seg1']
                l_s_right_seg1 = np.concatenate(trajs['l_s_right']['seg1'], axis=0)
                l_s_right_seg0 = np.repeat(l_s_right_seg0, num_l_tgt_per_lane, axis=0)
                s_right = np.concatenate((s_right_seg0, s_right_seg1), axis=0)
                l_s_right = np.concatenate((l_s_right_seg0, l_s_right_seg1), axis=1)
                segment_count_right += 1
            else:
                s_right = s_right_seg0
                l_s_right = l_s_right_seg0
            s_right = np.broadcast_to(s_right, l_s_right.shape)
            s_right_trim = s_right[:, :self._trajectory_evaluator.num_frames + 1]
            l_s_right_trim = l_s_right[:, :self._trajectory_evaluator.num_frames + 1]
            segment_count_right_eval = segment_count_right
            # concatenate
            s_trim = np.concatenate((s_right_trim, s_keep_trim, s_left_trim), axis=0)
            l_s_trim = np.concatenate((l_s_right_trim, l_s_keep_trim, l_s_left_trim), axis=0)
            heading_trim = np.arctan2(l_s_trim[:, 1:] - l_s_trim[:, :-1],
                                      s_trim[:, 1:] - s_trim[:, :-1])
            heading_trim = np.concatenate((np.broadcast_to(ego_poses_frenet[i_plan, 0, 2:], (heading_trim.shape[0], 1)), heading_trim), axis=1)
            trajs_eval_frenet = np.stack((np.broadcast_to(s_trim, l_s_trim.shape), l_s_trim, heading_trim), axis=-1)

            # frenet -> global
            trajs_eval_global = frenet_frame.frenet_to_cartesian(
                trajs_eval_frenet.transpose((0, 2, 1))
            )['pose_cartesian'].transpose((0, 2, 1))

            # rule-based evaluate (borrowed from PLUTO)
            current_ego_state = current_input.history.ego_states[-1]
            if not self.scenario_manager._route_manager.reference_lines:
                baseline_path = frenet_frame.reference_line
            else:
                baseline_path = self._get_ego_baseline_path(
                    self.scenario_manager.get_cached_reference_lines(), current_ego_state
                )
            rule_based_scores = self._trajectory_evaluator.evaluate(
                candidate_trajectories=trajs_eval_global,
                init_ego_state=current_ego_state,
                detections=current_input.history.observations[-1],
                traffic_light_data=current_input.traffic_light_data,
                agents_info=obs_info,
                route_lane_dict=self.scenario_manager.get_route_lane_dicts(),
                drivable_area_map=self.scenario_manager.drivable_area_map,
                baseline_path=baseline_path,
            )

            # recover all candidate trajs
            # keep
            if segment_count_keep == 1 and l_s_keep.shape[1] < self.num_future_steps + 1:
                l_s_keep_seg1 = np.concatenate(trajs['l_s_keep']['seg1'], axis=0)
                l_s_keep = np.repeat(l_s_keep, num_l_tgt_per_lane, axis=0)
                l_s_keep = np.concatenate((l_s_keep, l_s_keep_seg1), axis=1)
                segment_count_keep += 1
            if segment_count_keep == 2 and l_s_keep.shape[1] < self.num_future_steps + 1:
                l_s_keep_seg2 = np.concatenate(trajs['l_s_keep']['seg2'], axis=0)
                l_s_keep = np.repeat(l_s_keep, num_l_tgt_per_lane, axis=0)
                l_s_keep_seg2 = np.tile(l_s_keep_seg2, (num_l_tgt_per_lane, 1))
                l_s_keep = np.concatenate((l_s_keep, l_s_keep_seg2), axis=1)
                segment_count_keep += 1
            s_keep = np.concatenate([seg for seg in trajs['s_keep'].values() if seg is not None], axis=0)
            s_keep = np.broadcast_to(s_keep[np.newaxis], l_s_keep.shape)
            trajs_keep_frenet = np.stack((s_keep, l_s_keep), axis=-1)
            # left
            if segment_count_left == 1 and l_s_left.shape[1] < self.num_future_steps + 1:
                l_s_left_seg1 = np.concatenate(trajs['l_s_left']['seg1'], axis=0)
                l_s_left = np.repeat(l_s_left, num_l_tgt_per_lane, axis=0)
                l_s_left = np.concatenate((l_s_left, l_s_left_seg1), axis=1)
                segment_count_left += 1
            s_left = np.concatenate([seg for seg in trajs['s_left'].values() if seg is not None], axis=0)
            s_left = np.broadcast_to(s_left[np.newaxis], l_s_left.shape)
            trajs_left_frenet = np.stack((s_left, l_s_left), axis=-1)
            # right
            if segment_count_right == 1 and l_s_right.shape[1] < self.num_future_steps + 1:
                l_s_right_seg1 = np.concatenate(trajs['l_s_right']['seg1'], axis=0)
                l_s_right = np.repeat(l_s_right, num_l_tgt_per_lane, axis=0)
                l_s_right = np.concatenate((l_s_right, l_s_right_seg1), axis=1)
                segment_count_right += 1
            s_right = np.concatenate([seg for seg in trajs['s_right'].values() if seg is not None], axis=0)
            s_right = np.broadcast_to(s_right[np.newaxis], l_s_right.shape)
            trajs_right_frenet = np.stack((s_right, l_s_right), axis=-1)
            # concatenate
            trajs_frenet = np.concatenate((trajs_right_frenet, trajs_keep_frenet, trajs_left_frenet), axis=0)

            # calculate deviation between piece-wise trajectories and predicted trajectory
            neg_deviations_frenet = -np.abs(trajs_frenet[:, :, 1] - ego_poses_frenet[:, :, 1]).mean(axis=-1)
            neg_deviations_max = neg_deviations_frenet.max()
            neg_deviations_min = neg_deviations_frenet.min()
            preference_scores = (neg_deviations_frenet - neg_deviations_min) / (neg_deviations_max - neg_deviations_min)

            # repeat rule-based scores if needed
            if segment_count_right_eval == 1 and segment_count_left_eval == 1:
                rule_based_scores_right = rule_based_scores[:num_l_tgt_per_lane]
                rule_based_scores_keep = rule_based_scores[num_l_tgt_per_lane:-num_l_tgt_per_lane]
                rule_based_scores_left = rule_based_scores[-num_l_tgt_per_lane:]
            elif segment_count_right_eval == 2 and segment_count_left_eval == 2:
                rule_based_scores_right = rule_based_scores[:num_l_tgt_per_lane ** 2]
                rule_based_scores_keep = rule_based_scores[num_l_tgt_per_lane ** 2:-num_l_tgt_per_lane ** 2]
                rule_based_scores_left = rule_based_scores[-num_l_tgt_per_lane ** 2:]
            else:
                raise RuntimeError('Error encountered when splitting rule_based_scores')
            if segment_count_right == 2 and segment_count_right_eval == 1:
                rule_based_scores_right = np.repeat(rule_based_scores_right, num_l_tgt_per_lane, axis=0)
            if segment_count_left == 2 and segment_count_left_eval == 1:
                rule_based_scores_left = np.repeat(rule_based_scores_left, num_l_tgt_per_lane, axis=0)
            if segment_count_keep_eval == 1:
                if segment_count_keep == 2:
                    rule_based_scores_keep = np.repeat(rule_based_scores_keep, num_l_tgt_per_lane, axis=0)
                if segment_count_keep == 3:
                    rule_based_scores_keep = np.repeat(rule_based_scores_keep, num_l_tgt_per_lane, axis=0)
                    rule_based_scores_keep = np.repeat(rule_based_scores_keep, num_l_tgt_per_lane, axis=0)
            elif segment_count_keep_eval == 2:
                if segment_count_keep == 3:
                    rule_based_scores_keep = np.repeat(rule_based_scores_keep, num_l_tgt_per_lane, axis=0)
            rule_based_scores = np.concatenate((rule_based_scores_right, rule_based_scores_keep, rule_based_scores_left), axis=0)
            rule_in_mask = rule_based_scores.astype(bool)

            # final score
            final_scores = rule_based_scores + preference_scores
            if np.any(rule_in_mask):
                best_index = final_scores[rule_in_mask].argmax()
                best_traj = trajs_frenet[rule_in_mask][best_index]
            else:
                best_index = final_scores.argmax()
                best_traj = trajs_frenet[best_index]

            # estimate heading
            heading = np.arctan2(best_traj[1:, 1] - best_traj[0:-1, 1],
                                 best_traj[1:, 0] - best_traj[0:-1, 0])
            heading = np.concatenate([ego_poses_frenet[i_plan, 0, 2:], heading], axis=0)
            best_traj = np.concatenate([best_traj, heading[:, np.newaxis]], axis=-1)
            best_traj = smooth_headings(best_traj[np.newaxis], use_savgol_filter=False)[0]

            refined_trajs.append(best_traj)

        return np.stack(refined_trajs)

    def qp_path(self,
                ego_dim,
                ego_poses_frenet,
                overlap_region,
                frenet_frame,
                reference_line_lanes):
        r"""
        QP path planning.
        Objective:
        Minimize J(f) = Σ_i=0^3 w_i * J_i(f)
        where:
            J_0(f) = ∫(f(s) - h(s))² ds
            J_1(f) = ∫(f'(s))² ds
            J_2(f) = ∫(f''(s))² ds
            J_3(f) = ∫(f'''(s))² ds
        :param ego_dim: ego dimensions
        :param ego_poses_frenet: shape (num_plans, num_future_steps + 1, 3), ego poses at rear axle in Frenet frame
        :param overlap_region: overlap regions
        :param frenet_frame: Frenet frame object
        :param reference_line_lanes: reference line lanes
        :return: QP paths with reference point at rear axle, QP paths succeeded, QP paths costs.
        """
        num_points = self.num_points_for_eval
        step_eval = self.step_interval_for_eval
        current_ego_pose = ego_poses_frenet[0, 0, :]
        min_qp_planning_length = 80.
        self.s = np.linspace(
            ego_poses_frenet[:, 0, 0],
            max(ego_poses_frenet[:, -1, 0], ego_poses_frenet[:, 0, 0] + min_qp_planning_length),
            num=self.num_future_steps + 1
        ).transpose(1, 0)
        s_step = self.s[:, 1] - self.s[:, 0]
        f_l_s = [interp1d(poses[:, 0], poses[:, 1], bounds_error=False, fill_value=(poses[0, 1], poses[-1, 1])) for poses in ego_poses_frenet]
        ego_l_s = [fun(s) for fun, s in zip(f_l_s, self.s)]  # l(s)
        f_heading_s = [interp1d(poses[:, 0], poses[:, 2], bounds_error=False, fill_value=(poses[0, 2], poses[-1, 2])) for poses in ego_poses_frenet]
        ego_heading_s = [fun(s) for fun, s in zip(f_heading_s, self.s)]  # heading(s)
        safe_dist_to_obj = 0.3
        safe_dist_to_bd = 0.2
        qp_path_w0 = 4e-1
        qp_path_w1 = 1e2
        qp_path_w2 = 1e3
        qp_path_w3 = 1e5
        qp_paths = []
        qp_paths_succeeded = []
        qp_paths_costs = []

        # QP Path ➤ Feasible Tunnel
        """ 
        Assume heading 𝜃 < pi/12
        """
        # QP Path ➤ Feasible Tunnel ➤ consider road boundaries
        left_lanes = [lane.adjacent_edges[0] if isinstance(lane, NuPlanLane) else lane.custom_adjacent_edges[0] for lane in reference_line_lanes]  # note that lane connectors don't have adjacent_edges
        right_lanes = [lane.adjacent_edges[1] if isinstance(lane, NuPlanLane) else lane.custom_adjacent_edges[1] for lane in reference_line_lanes]
        # reference lane's boundaries
        reference_lane_left_boundaries = [lane.left_boundary for lane in reference_line_lanes]
        reference_lane_right_boundaries = [lane.right_boundary for lane in reference_line_lanes]
        reference_lane_left_boundaries = [line.discrete_path for line in reference_lane_left_boundaries]
        reference_lane_right_boundaries = [line.discrete_path for line in reference_lane_right_boundaries]
        ref_l_bd = sum(reference_lane_left_boundaries, [])
        ref_r_bd = sum(reference_lane_right_boundaries, [])
        ref_l_bd_points = [state.point for state in ref_l_bd]
        ref_r_bd_points = [state.point for state in ref_r_bd]
        ref_l_bd_stations = frenet_frame.get_nearest_station_from_position(ref_l_bd_points)
        ref_r_bd_stations = frenet_frame.get_nearest_station_from_position(ref_r_bd_points)
        ref_l_bd_laterals = frenet_frame.get_lateral_from_position(ref_l_bd_points, ref_l_bd_stations)
        ref_r_bd_laterals = frenet_frame.get_lateral_from_position(ref_r_bd_points, ref_r_bd_stations)
        # f_ref_l_bd = interp1d(ref_l_bd_stations, ref_l_bd_laterals, fill_value='extrapolate')
        # f_ref_r_bd = interp1d(ref_r_bd_stations, ref_r_bd_laterals, fill_value='extrapolate')
        f_ref_l_bd = interp1d(ref_l_bd_stations, ref_l_bd_laterals, bounds_error=False, fill_value=(ref_l_bd_laterals[0], ref_l_bd_laterals[-1]))
        f_ref_r_bd = interp1d(ref_r_bd_stations, ref_r_bd_laterals, bounds_error=False, fill_value=(ref_r_bd_laterals[0], ref_r_bd_laterals[-1]))
        lateral_ref_lb = f_ref_l_bd(self.s)
        lateral_ref_rb = f_ref_r_bd(self.s)
        # initialize boundaries
        current_ego_lateral = current_ego_pose[1]
        # lateral_max = ampl_factor + np.amax(ego_poses_frenet[:, :, 1], axis=1, keepdims=True)
        # lateral_min = - ampl_factor + np.amin(ego_poses_frenet[:, :, 1], axis=1, keepdims=True)
        # self.lateral_lb = (lateral_max * np.ones_like(self.s))
        # self.lateral_rb = (lateral_min * np.ones_like(self.s))
        lane_width = ref_l_bd_laterals[0] - ref_r_bd_laterals[0]
        lateral_max = current_ego_lateral + 2 * lane_width
        lateral_min = current_ego_lateral - 2 * lane_width
        self.lateral_lb = (lateral_max * np.ones_like(self.s))
        self.lateral_rb = (lateral_min * np.ones_like(self.s))
        # left and right lanes' boundaries
        left_lane_left_boundaries = [lane.left_boundary if lane is not None else None for lane in left_lanes]
        left_lane_left_boundaries = [line.discrete_path if line is not None else [] for line in left_lane_left_boundaries]
        right_lane_right_boundaries = [lane.right_boundary if lane is not None else None for lane in right_lanes]
        right_lane_right_boundaries = [line.discrete_path if line is not None else [] for line in right_lane_right_boundaries]
        left_lane_l_bd = sum(left_lane_left_boundaries, [])
        right_lane_r_bd = sum(right_lane_right_boundaries, [])
        left_lane_l_bd_points = [state.point for state in left_lane_l_bd]
        right_lane_r_bd_points = [state.point for state in right_lane_r_bd]
        if len(left_lane_l_bd_points) > 0:
            left_lane_l_bd_stations = frenet_frame.get_nearest_station_from_position(left_lane_l_bd_points)
            left_lane_l_bd_laterals = frenet_frame.get_lateral_from_position(left_lane_l_bd_points, left_lane_l_bd_stations)
            left_lane_masks = []  # whether left lane boundary points exists
            idx = 0
            for splits in left_lane_left_boundaries:
                length = len(splits)
                if length > 0:
                    left_lane_masks.append((left_lane_l_bd_stations[idx] < self.s)
                                           & (self.s < left_lane_l_bd_stations[idx + length - 1]))
                    idx = idx + length
            # f_left_lane_l_bd = interp1d(left_lane_l_bd_stations, left_lane_l_bd_laterals, fill_value='extrapolate')
            f_left_lane_l_bd = interp1d(left_lane_l_bd_stations, left_lane_l_bd_laterals, bounds_error=False, fill_value=(left_lane_l_bd_laterals[0], left_lane_l_bd_laterals[-1]))
            lateral_left_lane_lb = f_left_lane_l_bd(self.s)
            lateral_left_lane_lb -= safe_dist_to_bd
        else:
            left_lane_masks = []
            lateral_ref_lb -= safe_dist_to_bd
        if len(right_lane_r_bd_points) > 0:
            right_lane_r_bd_stations = frenet_frame.get_nearest_station_from_position(right_lane_r_bd_points)
            right_lane_r_bd_laterals = frenet_frame.get_lateral_from_position(right_lane_r_bd_points, right_lane_r_bd_stations)
            right_lane_masks = []  # whether right lane boundary points exists
            idx = 0
            for splits in right_lane_right_boundaries:
                length = len(splits)
                if length > 0:
                    right_lane_masks.append((right_lane_r_bd_stations[idx] < self.s)
                                            & (self.s < right_lane_r_bd_stations[idx + length - 1]))
                    idx = idx + length
            # f_right_lane_r_bd = interp1d(right_lane_r_bd_stations, right_lane_r_bd_laterals, fill_value='extrapolate')
            f_right_lane_r_bd = interp1d(right_lane_r_bd_stations, right_lane_r_bd_laterals, bounds_error=False, fill_value=(right_lane_r_bd_laterals[0], right_lane_r_bd_laterals[-1]))
            lateral_right_lane_rb = f_right_lane_r_bd(self.s)
            lateral_right_lane_rb += safe_dist_to_bd
        else:
            right_lane_masks = []
            lateral_ref_rb += safe_dist_to_bd
        # merge boundaries
        lateral_lb = copy.deepcopy(lateral_ref_lb)
        lateral_rb = copy.deepcopy(lateral_ref_rb)
        narrow_bd_of_lane_connector = True
        """
        Since lane connectors don't have adjacent lanes, 
        True: _____    _____        False: ______________
                   ----     
                   ----
              -----    -----               --------------
        """
        if len(left_lane_masks) > 0:
            if narrow_bd_of_lane_connector:
                for left_lane_mask in left_lane_masks:
                    lateral_lb[left_lane_mask] = np.maximum(lateral_ref_lb[left_lane_mask], lateral_left_lane_lb[left_lane_mask])
            else:
                lateral_lb = np.maximum(lateral_ref_lb, lateral_left_lane_lb)
        if len(right_lane_masks) > 0:
            if narrow_bd_of_lane_connector:
                for right_lane_mask in right_lane_masks:
                    lateral_rb[right_lane_mask] = np.minimum(lateral_ref_rb[right_lane_mask], lateral_right_lane_rb[right_lane_mask])
            else:
                lateral_rb = np.minimum(lateral_ref_rb, lateral_right_lane_rb)
        self.lateral_lb = np.minimum(self.lateral_lb, lateral_lb)
        self.lateral_rb = np.maximum(self.lateral_rb, lateral_rb)

        for i_plan in range(self.num_plans):
            smin, smax, lmin, lmax = (overlap_region['smin'][i_plan], overlap_region['smax'][i_plan],
                                      overlap_region['lmin'][i_plan], overlap_region['lmax'][i_plan])
            overlaps = overlap_region['overlaps'][i_plan]

            # an overlap region (rectangle) is left if it's closer to the left boundary
            # left_mask = (lmin + lmax) / 2 >= current_ego_lateral
            # right_mask = ~left_mask
            rec_center_s = (smin + smax) / 2
            rec_center_l = (lmin + lmax) / 2
            # f_lateral_lb = interp1d(self.s[i_plan], self.lateral_lb[i_plan], fill_value='extrapolate')
            # f_lateral_rb = interp1d(self.s[i_plan], self.lateral_rb[i_plan], fill_value='extrapolate')
            f_lateral_lb = interp1d(self.s[i_plan], self.lateral_lb[i_plan], bounds_error=False, fill_value=(self.lateral_lb[i_plan][0], self.lateral_lb[i_plan][-1]))
            f_lateral_rb = interp1d(self.s[i_plan], self.lateral_rb[i_plan], bounds_error=False, fill_value=(self.lateral_rb[i_plan][0], self.lateral_rb[i_plan][-1]))
            lb = f_lateral_lb(rec_center_s)
            rb = f_lateral_rb(rec_center_s)
            left_mask = lb - rec_center_l <= rec_center_l - rb
            right_mask = ~left_mask
            left_mask_stats = [mask[overlap].sum() for mask, overlap in zip(left_mask, overlaps)]
            right_mask_stats = [mask[overlap].sum() for mask, overlap in zip(right_mask, overlaps)]
            for i in range(left_mask.shape[0]):
                left_mask[i, :] = left_mask_stats[i] > right_mask_stats[i]
            right_mask = ~left_mask

            left_mask = left_mask[overlaps]
            right_mask = right_mask[overlaps]
            smin = smin[overlaps]
            smax = smax[overlaps]
            lmin = lmin[overlaps]
            lmax = lmax[overlaps]
            num_recs = smin.shape[0]
            overlap_station_mask1 = smin.reshape((-1, 1)) < np.broadcast_to(self.s[i_plan] + s_step[i_plan], shape=(num_recs, self.num_future_steps + 1))
            overlap_station_mask2 = smax.reshape((-1, 1)) > np.broadcast_to(self.s[i_plan] - s_step[i_plan], shape=(num_recs, self.num_future_steps + 1))
            overlap_station_mask3 = smin.reshape((-1, 1)) < np.broadcast_to(self.s[i_plan] - s_step[i_plan], shape=(num_recs, self.num_future_steps + 1))
            overlap_station_mask4 = smax.reshape((-1, 1)) > np.broadcast_to(self.s[i_plan] + s_step[i_plan], shape=(num_recs, self.num_future_steps + 1))
            overlap_station_mask = (overlap_station_mask1 & overlap_station_mask2) | (overlap_station_mask3 & overlap_station_mask4)
            lateral_lb = np.repeat(self.lateral_lb[i_plan].reshape(1, -1), repeats=num_recs, axis=0)
            lateral_rb = np.repeat(self.lateral_rb[i_plan].reshape(1, -1), repeats=num_recs, axis=0)
            left_overlap_mask = left_mask.reshape((-1, 1)) & overlap_station_mask
            right_overlap_mask = right_mask.reshape((-1, 1)) & overlap_station_mask
            lmin = np.broadcast_to(lmin.reshape((-1, 1)), shape=(num_recs, self.num_future_steps + 1))
            lmax = np.broadcast_to(lmax.reshape((-1, 1)), shape=(num_recs, self.num_future_steps + 1))
            lateral_lb[left_overlap_mask] = np.minimum(lateral_lb[left_overlap_mask], lmin[left_overlap_mask] - safe_dist_to_obj)
            lateral_rb[right_overlap_mask] = np.maximum(lateral_rb[right_overlap_mask], lmax[right_overlap_mask] + safe_dist_to_obj)
            if lateral_lb.shape[0] > 0:
                self.lateral_lb[i_plan] = np.amin(lateral_lb, axis=0)
            if lateral_rb.shape[0] > 0:
                self.lateral_rb[i_plan] = np.amax(lateral_rb, axis=0)

            # cut feasible tunnel
            # f_lateral_lb = interp1d(self.s[i_plan], self.lateral_lb[i_plan], fill_value='extrapolate')
            # f_lateral_rb = interp1d(self.s[i_plan], self.lateral_rb[i_plan], fill_value='extrapolate')
            f_lateral_lb = interp1d(self.s[i_plan], self.lateral_lb[i_plan], bounds_error=False, fill_value=(self.lateral_lb[i_plan][0], self.lateral_lb[i_plan][-1]))
            f_lateral_rb = interp1d(self.s[i_plan], self.lateral_rb[i_plan], bounds_error=False, fill_value=(self.lateral_rb[i_plan][0], self.lateral_rb[i_plan][-1]))
            # feasible_tunnel_width = self.lateral_lb[i_plan] - self.lateral_rb[i_plan]
            curve1_points = np.column_stack((self.s[i_plan], self.lateral_lb[i_plan]))
            curve2_points = np.column_stack((self.s[i_plan], self.lateral_rb[i_plan]))
            distance_matrix = cdist(curve1_points, curve2_points, metric='euclidean')
            feasible_tunnel_width = np.min(distance_matrix, axis=1)
            neck_index1 = np.where(feasible_tunnel_width < ego_dim['width'] + safe_dist_to_obj)[0]
            lb_is_smaller_than_rb = self.lateral_lb[i_plan] < self.lateral_rb[i_plan]
            neck_index2 = np.where(lb_is_smaller_than_rb)[0]
            neck_index = None
            if len(neck_index1) > 0:
                if len(neck_index2) > 0:
                    neck_index = min(neck_index1[0], neck_index2[0])
                else:
                    neck_index = neck_index1[0]
            elif len(neck_index2) > 0:
                neck_index = neck_index2[0]
            if neck_index is not None:  # tunnel is narrow
                if len(neck_index1) > 0:
                    if len(neck_index2) > 0:
                        neck_index = min(neck_index1[0], neck_index2[0])
                    else:
                        neck_index = neck_index1[0]
                elif len(neck_index2) > 0:
                        neck_index = neck_index2[0]
                self.s[i_plan] = np.linspace(self.s[i_plan, 0], self.s[i_plan, neck_index - 1], num=self.num_future_steps + 1)
                s_step[i_plan] = self.s[i_plan, 1] - self.s[i_plan, 0]
                fun = f_l_s[i_plan]
                ego_l_s[i_plan] = fun(self.s[i_plan])  # l(s)
                fun = f_heading_s[i_plan]
                ego_heading_s[i_plan] = fun(self.s[i_plan])  # heading(s)
                self.lateral_lb[i_plan] = f_lateral_lb(self.s[i_plan])
                self.lateral_rb[i_plan] = f_lateral_rb(self.s[i_plan])

            # QP Path ➤ Constraints
            # QP Path ➤ Constraints ➤ Equality Constraints
            num_points_each_piece = self.num_future_steps // 5  # number of points of each polynomial (does not include end point, except for the last piece)
            qp_x0_idx = 0
            qp_x1_idx = num_points_each_piece
            qp_x2_idx = num_points_each_piece * 2
            qp_x3_idx = num_points_each_piece * 3
            qp_x4_idx = num_points_each_piece * 4
            qp_x5_idx = self.num_future_steps
            qp_x0 = self.s[i_plan, qp_x0_idx]
            qp_x1 = self.s[i_plan, qp_x1_idx]
            qp_x2 = self.s[i_plan, qp_x2_idx]
            qp_x3 = self.s[i_plan, qp_x3_idx]
            qp_x4 = self.s[i_plan, qp_x4_idx]
            qp_x5 = self.s[i_plan, qp_x5_idx]
            qp_x1_x0 = qp_x1 - qp_x0
            qp_x2_x1 = qp_x2 - qp_x1
            qp_x3_x2 = qp_x3 - qp_x2
            qp_x4_x3 = qp_x4 - qp_x3
            qp_x5_x4 = qp_x5 - qp_x4
            qp_A = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 0
                             [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 1
                             [0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 2
                             [1., qp_x1_x0, qp_x1_x0 ** 2, qp_x1_x0 ** 3, -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 3
                             [0., 1., 2 * qp_x1_x0, 3 * qp_x1_x0 ** 2, 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 4
                             [0., 0., 2., 6 * qp_x1_x0, 0., 0., -2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 5
                             [0., 0., 0., 0., 1., qp_x2_x1, qp_x2_x1 ** 2, qp_x2_x1 ** 3, -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 6
                             [0., 0., 0., 0., 0., 1., 2 * qp_x2_x1, 3 * qp_x2_x1 ** 2, 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 7
                             [0., 0., 0., 0., 0., 0., 2., 6 * qp_x2_x1, 0., 0., -2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 8
                             [0., 0., 0., 0., 0., 0., 0., 0., 1., qp_x3_x2, qp_x3_x2 ** 2, qp_x3_x2 ** 3, -1., 0., 0., 0., 0., 0., 0., 0.],  # 9
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2 * qp_x3_x2, 3 * qp_x3_x2 ** 2, 0., -1., 0., 0., 0., 0., 0., 0.],  # 10
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 6 * qp_x3_x2, 0., 0., -2., 0., 0., 0., 0., 0.],  # 11
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., qp_x4_x3, qp_x4_x3 ** 2, qp_x4_x3 ** 3, -1., 0., 0., 0.],  # 12
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2 * qp_x4_x3, 3 * qp_x4_x3 ** 2, 0., -1., 0., 0.],  # 13
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 6 * qp_x4_x3, 0, 0., -2., 0.],  # 14
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 1., 2 * qp_x5_x4, 3 * qp_x5_x4 ** 2],  # 15
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 2., 6 * qp_x5_x4],  # 16
                             ])
            dl_ds_0 = np.tan(current_ego_pose[2])  # estimated by tan(heading)
            ddl_dds_0 = ((np.tan(ego_poses_frenet[i_plan, step_eval, 2]) - np.tan(current_ego_pose[2])) /
                         (ego_poses_frenet[i_plan, step_eval, 0] - ego_poses_frenet[i_plan, 0, 0] + 1e-8))  # estimated by tan(heading) divided by delta s
            qp_b = np.array([[current_ego_lateral],  # 0
                             [dl_ds_0],  # 1
                             [ddl_dds_0],  # 2
                             [0.],  # 3
                             [0.],  # 4
                             [0.],  # 5
                             [0.],  # 6
                             [0.],  # 7
                             [0.],  # 8
                             [0.],  # 9
                             [0.],  # 10
                             [0.],  # 11
                             [0.],  # 12
                             [0.],  # 13
                             [0.],  # 14
                             [0.],  # 15
                             [0.]])  # 16
            # QP Path ➤ Constraints ➤ Inequality Constraints
            # QP Path ➤ Constraints ➤ Inequality Constraints ➤ Front Left corner
            qp_IA_f0 = lambda s: np.array([[-1., -s - ego_dim['rear_axle_to_front_dist'], -s ** 2 - 2 * s * ego_dim['rear_axle_to_front_dist'], -s ** 3 - 3 * s ** 2 * ego_dim['rear_axle_to_front_dist'],
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.]])
            qp_IA_0 = np.vstack([qp_IA_f0(self.s[i_plan][i] - qp_x0) for i in range(qp_x0_idx, qp_x1_idx, step_eval)])  # [17, 17 + num_points_each_piece / step)
            qp_IA_f1 = lambda s: np.array([[0., 0., 0., 0.,
                                            -1., -s - ego_dim['rear_axle_to_front_dist'], -s ** 2 - 2 * s * ego_dim['rear_axle_to_front_dist'], -s ** 3 - 3 * s ** 2 * ego_dim['rear_axle_to_front_dist'],
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.]])
            qp_IA_1 = np.vstack([qp_IA_f1(self.s[i_plan][i] - qp_x1) for i in range(qp_x1_idx, qp_x2_idx, step_eval)])  # [17 + num_points_each_piece / step, 17 + num_points_each_piece / step * 2)
            qp_IA_f2 = lambda s: np.array([[0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            -1., -s - ego_dim['rear_axle_to_front_dist'], -s ** 2 - 2 * s * ego_dim['rear_axle_to_front_dist'], -s ** 3 - 3 * s ** 2 * ego_dim['rear_axle_to_front_dist'],
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.]])
            qp_IA_2 = np.vstack([qp_IA_f2(self.s[i_plan][i] - qp_x2) for i in range(qp_x2_idx, qp_x3_idx, step_eval)])  # [17 + num_points_each_piece / step * 2, 17 + num_points_each_piece / step * 3)
            qp_IA_f3 = lambda s: np.array([[0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            -1., -s - ego_dim['rear_axle_to_front_dist'], -s ** 2 - 2 * s * ego_dim['rear_axle_to_front_dist'], -s ** 3 - 3 * s ** 2 * ego_dim['rear_axle_to_front_dist'],
                                            0., 0., 0., 0.]])
            qp_IA_3 = np.vstack([qp_IA_f3(self.s[i_plan][i] - qp_x3) for i in range(qp_x3_idx, qp_x4_idx, step_eval)])  # [17 + num_points_each_piece / step * 3, 17 + num_points_each_piece / step * 4)
            qp_IA_f4 = lambda s: np.array([[0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            -1., -s - ego_dim['rear_axle_to_front_dist'], -s ** 2 - 2 * s * ego_dim['rear_axle_to_front_dist'], -s ** 3 - 3 * s ** 2 * ego_dim['rear_axle_to_front_dist']]])
            qp_IA_4 = np.vstack([qp_IA_f4(self.s[i_plan][i] - qp_x4) for i in range(qp_x4_idx, qp_x5_idx + 1, step_eval)])  # [17 + num_points_each_piece / step * 4, 17 + num_points_each_piece / step * 5 + 1)
            # f_l_interp1d = interp1d(self.s[i_plan], self.lateral_lb[i_plan], fill_value='extrapolate')
            qp_Ib_0_4 = ego_dim['width'] / 2 - f_lateral_lb(self.s[i_plan][::step_eval] + ego_dim['rear_axle_to_front_dist']).reshape(num_points, 1)
            # QP Path ➤ Constraints ➤ Inequality Constraints ➤ Front Right corner
            qp_IA_f5 = lambda s: np.array([[1., s + ego_dim['rear_axle_to_front_dist'], s ** 2 + 2 * s * ego_dim['rear_axle_to_front_dist'], s ** 3 + 3 * s ** 2 * ego_dim['rear_axle_to_front_dist'],
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.]])
            qp_IA_5 = np.vstack([qp_IA_f5(self.s[i_plan][i] - qp_x0) for i in range(qp_x0_idx, qp_x1_idx, step_eval)])  # [17 + num_points_each_piece / step * 5 + 1, 17 + num_points_each_piece / step * 6 + 1)
            qp_IA_f6 = lambda s: np.array([[0., 0., 0., 0.,
                                            1., s + ego_dim['rear_axle_to_front_dist'], s ** 2 + 2 * s * ego_dim['rear_axle_to_front_dist'], s ** 3 + 3 * s ** 2 * ego_dim['rear_axle_to_front_dist'],
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.]])
            qp_IA_6 = np.vstack([qp_IA_f6(self.s[i_plan][i] - qp_x1) for i in range(qp_x1_idx, qp_x2_idx, step_eval)])  # [17 + num_points_each_piece / step * 6 + 1, 17 + num_points_each_piece / step * 7 + 1)
            qp_IA_f7 = lambda s: np.array([[0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            1., s + ego_dim['rear_axle_to_front_dist'], s ** 2 + 2 * s * ego_dim['rear_axle_to_front_dist'], s ** 3 + 3 * s ** 2 * ego_dim['rear_axle_to_front_dist'],
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.]])
            qp_IA_7 = np.vstack([qp_IA_f7(self.s[i_plan][i] - qp_x2) for i in range(qp_x2_idx, qp_x3_idx, step_eval)])  # [17 + num_points_each_piece / step * 7 + 1, 17 + num_points_each_piece / step * 8 + 1)
            qp_IA_f8 = lambda s: np.array([[0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            1., s + ego_dim['rear_axle_to_front_dist'], s ** 2 + 2 * s * ego_dim['rear_axle_to_front_dist'], s ** 3 + 3 * s ** 2 * ego_dim['rear_axle_to_front_dist'],
                                            0., 0., 0., 0.]])
            qp_IA_8 = np.vstack([qp_IA_f8(self.s[i_plan][i] - qp_x3) for i in range(qp_x3_idx, qp_x4_idx, step_eval)])  # [17 + num_points_each_piece / step * 8 + 1, 17 + num_points_each_piece / step * 9 + 1)
            qp_IA_f9 = lambda s: np.array([[0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            0., 0., 0., 0.,
                                            1., s + ego_dim['rear_axle_to_front_dist'], s ** 2 + 2 * s * ego_dim['rear_axle_to_front_dist'], s ** 3 + 3 * s ** 2 * ego_dim['rear_axle_to_front_dist']]])
            qp_IA_9 = np.vstack([qp_IA_f9(self.s[i_plan][i] - qp_x4) for i in range(qp_x4_idx, qp_x5_idx + 1, step_eval)])  # [17 + num_points_each_piece / step * 9 + 1, 17 + num_points_each_piece / step * 10 + 2)
            # f_r_interp1d = interp1d(self.s[i_plan], self.lateral_rb[i_plan], fill_value='extrapolate')
            qp_Ib_5_9 = ego_dim['width'] / 2 + f_lateral_rb(self.s[i_plan][::step_eval] + ego_dim['rear_axle_to_front_dist']).reshape(num_points, 1)
            # QP Path ➤ Objective Functional
            # QP Path ➤ Objective Functional ➤ Objective Functional 0
            G_0 = np.array([[qp_x1_x0, 1 / 2 * qp_x1_x0 ** 2, 1 / 3 * qp_x1_x0 ** 3, 1 / 4 * qp_x1_x0 ** 4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0.],
                            [1 / 2 * qp_x1_x0 ** 2, 1 / 3 * qp_x1_x0 ** 3, 1 / 4 * qp_x1_x0 ** 4, 1 / 5 * qp_x1_x0 ** 5, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [1 / 3 * qp_x1_x0 ** 3, 1 / 4 * qp_x1_x0 ** 4, 1 / 5 * qp_x1_x0 ** 5, 1 / 6 * qp_x1_x0 ** 6, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [1 / 4 * qp_x1_x0 ** 4, 1 / 5 * qp_x1_x0 ** 5, 1 / 6 * qp_x1_x0 ** 6, 1 / 7 * qp_x1_x0 ** 7, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., qp_x2_x1, 1 / 2 * qp_x2_x1 ** 2, 1 / 3 * qp_x2_x1 ** 3, 1 / 4 * qp_x2_x1 ** 4, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0.],
                            [0., 0., 0., 0., 1 / 2 * qp_x2_x1 ** 2, 1 / 3 * qp_x2_x1 ** 3, 1 / 4 * qp_x2_x1 ** 4, 1 / 5 * qp_x2_x1 ** 5, 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 1 / 3 * qp_x2_x1 ** 3, 1 / 4 * qp_x2_x1 ** 4, 1 / 5 * qp_x2_x1 ** 5, 1 / 6 * qp_x2_x1 ** 6, 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 1 / 4 * qp_x2_x1 ** 4, 1 / 5 * qp_x2_x1 ** 5, 1 / 6 * qp_x2_x1 ** 6, 1 / 7 * qp_x2_x1 ** 7, 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., qp_x3_x2, 1 / 2 * qp_x3_x2 ** 2, 1 / 3 * qp_x3_x2 ** 3, 1 / 4 * qp_x3_x2 ** 4, 0., 0., 0., 0., 0.,
                             0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 1 / 2 * qp_x3_x2 ** 2, 1 / 3 * qp_x3_x2 ** 3, 1 / 4 * qp_x3_x2 ** 4, 1 / 5 * qp_x3_x2 ** 5, 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 1 / 3 * qp_x3_x2 ** 3, 1 / 4 * qp_x3_x2 ** 4, 1 / 5 * qp_x3_x2 ** 5, 1 / 6 * qp_x3_x2 ** 6, 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 1 / 4 * qp_x3_x2 ** 4, 1 / 5 * qp_x3_x2 ** 5, 1 / 6 * qp_x3_x2 ** 6, 1 / 7 * qp_x3_x2 ** 7, 0.,
                             0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x4_x3, 1 / 2 * qp_x4_x3 ** 2, 1 / 3 * qp_x4_x3 ** 3, 1 / 4 * qp_x4_x3 ** 4, 0.,
                             0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 2 * qp_x4_x3 ** 2, 1 / 3 * qp_x4_x3 ** 3, 1 / 4 * qp_x4_x3 ** 4,
                             1 / 5 * qp_x4_x3 ** 5, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 3 * qp_x4_x3 ** 3, 1 / 4 * qp_x4_x3 ** 4, 1 / 5 * qp_x4_x3 ** 5,
                             1 / 6 * qp_x4_x3 ** 6, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 4 * qp_x4_x3 ** 4, 1 / 5 * qp_x4_x3 ** 5, 1 / 6 * qp_x4_x3 ** 6,
                             1 / 7 * qp_x4_x3 ** 7, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x5_x4, 1 / 2 * qp_x5_x4 ** 2, 1 / 3 * qp_x5_x4 ** 3,
                             1 / 4 * qp_x5_x4 ** 4],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 2 * qp_x5_x4 ** 2, 1 / 3 * qp_x5_x4 ** 3, 1 / 4 * qp_x5_x4 ** 4,
                             1 / 5 * qp_x5_x4 ** 5],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 3 * qp_x5_x4 ** 3, 1 / 4 * qp_x5_x4 ** 4, 1 / 5 * qp_x5_x4 ** 5,
                             1 / 6 * qp_x5_x4 ** 6],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 4 * qp_x5_x4 ** 4, 1 / 5 * qp_x5_x4 ** 5, 1 / 6 * qp_x5_x4 ** 6,
                             1 / 7 * qp_x5_x4 ** 7]
                            ])
            c_0_0 = np.matrix([self.trapz(s_step[i_plan], ego_l_s[i_plan][qp_x0_idx:qp_x1_idx + 1])])
            c_0_1 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x0_idx:qp_x1_idx + 1]
                                          * (self.s[i_plan, qp_x0_idx:qp_x1_idx + 1] - self.s[i_plan, qp_x0_idx]))])
            c_0_2 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x0_idx:qp_x1_idx + 1]
                                          * (self.s[i_plan, qp_x0_idx:qp_x1_idx + 1] - self.s[i_plan, qp_x0_idx]) ** 2)])
            c_0_3 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x0_idx:qp_x1_idx + 1]
                                          * (self.s[i_plan, qp_x0_idx:qp_x1_idx + 1] - self.s[i_plan, qp_x0_idx]) ** 3)])
            c_1_0 = np.matrix([self.trapz(s_step[i_plan], ego_l_s[i_plan][qp_x1_idx:qp_x2_idx + 1])])
            c_1_1 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x1_idx:qp_x2_idx + 1]
                                          * (self.s[i_plan, qp_x1_idx:qp_x2_idx + 1] - self.s[i_plan, qp_x1_idx]))])
            c_1_2 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x1_idx:qp_x2_idx + 1]
                                          * (self.s[i_plan, qp_x1_idx:qp_x2_idx + 1] - self.s[i_plan, qp_x1_idx]) ** 2)])
            c_1_3 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x1_idx:qp_x2_idx + 1]
                                          * (self.s[i_plan, qp_x1_idx:qp_x2_idx + 1] - self.s[i_plan, qp_x1_idx]) ** 3)])
            c_2_0 = np.matrix([self.trapz(s_step[i_plan], ego_l_s[i_plan][qp_x2_idx:qp_x3_idx + 1])])
            c_2_1 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x2_idx:qp_x3_idx + 1]
                                          * (self.s[i_plan, qp_x2_idx:qp_x3_idx + 1] - self.s[i_plan, qp_x2_idx]))])
            c_2_2 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x2_idx:qp_x3_idx + 1]
                                          * (self.s[i_plan, qp_x2_idx:qp_x3_idx + 1] - self.s[i_plan, qp_x2_idx]) ** 2)])
            c_2_3 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x2_idx:qp_x3_idx + 1]
                                          * (self.s[i_plan, qp_x2_idx:qp_x3_idx + 1] - self.s[i_plan, qp_x2_idx]) ** 3)])
            c_3_0 = np.matrix([self.trapz(s_step[i_plan], ego_l_s[i_plan][qp_x3_idx:qp_x4_idx + 1])])
            c_3_1 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x3_idx:qp_x4_idx + 1]
                                          * (self.s[i_plan, qp_x3_idx:qp_x4_idx + 1] - self.s[i_plan, qp_x3_idx]))])
            c_3_2 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x3_idx:qp_x4_idx + 1]
                                          * (self.s[i_plan, qp_x3_idx:qp_x4_idx + 1] - self.s[i_plan, qp_x3_idx]) ** 2)])
            c_3_3 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x3_idx:qp_x4_idx + 1]
                                          * (self.s[i_plan, qp_x3_idx:qp_x4_idx + 1] - self.s[i_plan, qp_x3_idx]) ** 3)])
            c_4_0 = np.matrix([self.trapz(s_step[i_plan], ego_l_s[i_plan][qp_x4_idx:qp_x5_idx + 1])])
            c_4_1 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x4_idx:qp_x5_idx + 1]
                                          * (self.s[i_plan, qp_x4_idx:qp_x5_idx + 1] - self.s[i_plan, qp_x4_idx]))])
            c_4_2 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x4_idx:qp_x5_idx + 1]
                                          * (self.s[i_plan, qp_x4_idx:qp_x5_idx + 1] - self.s[i_plan, qp_x4_idx]) ** 2)])
            c_4_3 = np.matrix([self.trapz(s_step[i_plan],
                                          ego_l_s[i_plan][qp_x4_idx:qp_x5_idx + 1]
                                          * (self.s[i_plan, qp_x4_idx:qp_x5_idx + 1] - self.s[i_plan, qp_x4_idx]) ** 3)])
            qp_c = np.vstack((c_0_0, c_0_1, c_0_2, c_0_3, c_1_0, c_1_1, c_1_2, c_1_3, c_2_0, c_2_1, c_2_2, c_2_3, c_3_0, c_3_1, c_3_2, c_3_3, c_4_0, c_4_1, c_4_2, c_4_3))
            # QP Path ➤ Objective Functional ➤ Objective Functional 1
            G_1 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., qp_x1_x0, qp_x1_x0 ** 2, qp_x1_x0 ** 3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., qp_x1_x0 ** 2, 4 / 3 * qp_x1_x0 ** 3, 3 / 2 * qp_x1_x0 ** 4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., qp_x1_x0 ** 3, 3 / 2 * qp_x1_x0 ** 4, 9 / 5 * qp_x1_x0 ** 5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., qp_x2_x1, qp_x2_x1 ** 2, qp_x2_x1 ** 3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., qp_x2_x1 ** 2, 4 / 3 * qp_x2_x1 ** 3, 3 / 2 * qp_x2_x1 ** 4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., qp_x2_x1 ** 3, 3 / 2 * qp_x2_x1 ** 4, 9 / 5 * qp_x2_x1 ** 5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x3_x2, qp_x3_x2 ** 2, qp_x3_x2 ** 3, 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x3_x2 ** 2, 4 / 3 * qp_x3_x2 ** 3, 3 / 2 * qp_x3_x2 ** 4, 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x3_x2 ** 3, 3 / 2 * qp_x3_x2 ** 4, 9 / 5 * qp_x3_x2 ** 5, 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x4_x3, qp_x4_x3 ** 2, qp_x4_x3 ** 3, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x4_x3 ** 2, 4 / 3 * qp_x4_x3 ** 3, 3 / 2 * qp_x4_x3 ** 4, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x4_x3 ** 3, 3 / 2 * qp_x4_x3 ** 4, 9 / 5 * qp_x4_x3 ** 5, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x5_x4, qp_x5_x4 ** 2, qp_x5_x4 ** 3],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x5_x4 ** 2, 4 / 3 * qp_x5_x4 ** 3, 3 / 2 * qp_x5_x4 ** 4],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_x5_x4 ** 3, 3 / 2 * qp_x5_x4 ** 4, 9 / 5 * qp_x5_x4 ** 5]])
            # QP Path ➤ Objective Functional ➤ Objective Functional 2
            G_2 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 4 * qp_x1_x0, 6 * qp_x1_x0 ** 2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 6 * qp_x1_x0 ** 2, 12 * qp_x1_x0 ** 3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 4 * qp_x2_x1, 6 * qp_x2_x1 ** 2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 6 * qp_x2_x1 ** 2, 12 * qp_x2_x1 ** 3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4 * qp_x3_x2, 6 * qp_x3_x2 ** 2, 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6 * qp_x3_x2 ** 2, 12 * qp_x3_x2 ** 3, 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4 * qp_x4_x3, 6 * qp_x4_x3 ** 2, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6 * qp_x4_x3 ** 2, 12 * qp_x4_x3 ** 3, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4 * qp_x5_x4, 6 * qp_x5_x4 ** 2],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6 * qp_x5_x4 ** 2, 12 * qp_x5_x4 ** 3]])
            # QP Path ➤ Objective Functional ➤ Objective Functional 3
            G_3 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 36 * qp_x1_x0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 36 * qp_x2_x1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 36 * qp_x3_x2, 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 36 * qp_x4_x3, 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 36 * qp_x5_x4]])
            # QP Path ➤ Objective Functional ➤ Objective Functional 0 + 1 + 2 + 3
            qp_G = qp_path_w0 * G_0 + qp_path_w1 * G_1 + qp_path_w2 * G_2 + qp_path_w3 * G_3
            qp_c = - qp_path_w0 * qp_c

            # CVXOPT solver

            """
            min 1/2 x^T * P * x + q^T * x
             x 
            subject to Gx <= h, Ax = b
            """
            P_ = matrix(qp_G)
            q_ = matrix(qp_c)
            G_ = matrix(-np.vstack((qp_IA_0, qp_IA_1, qp_IA_2, qp_IA_3, qp_IA_4, qp_IA_5, qp_IA_6, qp_IA_7, qp_IA_8, qp_IA_9)))
            h_ = matrix(-np.vstack((qp_Ib_0_4, qp_Ib_5_9)))
            idx = list(range(0, 15))
            idx.append(15)
            idx.append(16)
            A_ = matrix(qp_A[idx, :])
            b_ = matrix(qp_b[idx, :])

            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    try:
                        sol = solvers.qp(P_, q_, G_, h_, A_, b_)
                    except:
                        sol = {'x': None, 'status': 'fail'}
            qp_res_P = np.array(sol['x'])
            if sol['status'] != 'optimal':
                qp_path_succeeded = False
            else:
                qp_path_succeeded = True
            # print(f'QP path succeeded for plan {i_plan}: {qp_path_succeeded}')

            if qp_path_succeeded:
                qp_res_P = np.squeeze(qp_res_P)
                p_0 = qp_res_P[0:4]
                p_1 = qp_res_P[4:8]
                p_2 = qp_res_P[8:12]
                p_3 = qp_res_P[12:16]
                p_4 = qp_res_P[16:]
                l_s0 = polynomial(self.s[i_plan][0:num_points_each_piece] - current_ego_pose[0], p_0)
                l_s1 = polynomial(self.s[i_plan][0:num_points_each_piece] - current_ego_pose[0], p_1)
                l_s2 = polynomial(self.s[i_plan][0:num_points_each_piece] - current_ego_pose[0], p_2)
                l_s3 = polynomial(self.s[i_plan][0:num_points_each_piece] - current_ego_pose[0], p_3)
                l_s4 = polynomial(self.s[i_plan][0:num_points_each_piece + 1] - current_ego_pose[0], p_4)
                qp_res_l = np.hstack((l_s0, l_s1, l_s2, l_s3, l_s4))
                dl_ds0 = polynomial_derivative(self.s[i_plan][0:num_points_each_piece] - current_ego_pose[0], p_0, order=1)
                dl_ds1 = polynomial_derivative(self.s[i_plan][0:num_points_each_piece] - current_ego_pose[0], p_1, order=1)
                dl_ds2 = polynomial_derivative(self.s[i_plan][0:num_points_each_piece] - current_ego_pose[0], p_2, order=1)
                dl_ds3 = polynomial_derivative(self.s[i_plan][0:num_points_each_piece] - current_ego_pose[0], p_3, order=1)
                dl_ds4 = polynomial_derivative(self.s[i_plan][0:num_points_each_piece + 1] - current_ego_pose[0], p_4, order=1)
                qp_res_dl_ds = np.hstack((dl_ds0, dl_ds1, dl_ds2, dl_ds3, dl_ds4))
                qp_res_heading = np.arctan(qp_res_dl_ds)
                qp_res_poses = np.vstack((self.s[i_plan], qp_res_l, qp_res_heading))
                cost0 = 1 / 2 * qp_path_w0 * qp_res_P.T.dot(G_0.dot(qp_res_P)) + qp_c.T.dot(qp_res_P)
                cost1 = 1 / 2 * qp_path_w1 * qp_res_P.T.dot(G_1.dot(qp_res_P))
                cost2 = 1 / 2 * qp_path_w2 * qp_res_P.T.dot(G_2.dot(qp_res_P))
                cost3 = 1 / 2 * qp_path_w3 * qp_res_P.T.dot(G_3.dot(qp_res_P))
                qp_path_cost = float(cost0 + cost1 + cost2 + cost3)
            else:
                qp_res_poses = np.vstack((self.s[i_plan], ego_l_s[i_plan], ego_heading_s[i_plan]))
                qp_path_cost = None

            qp_paths.append(qp_res_poses.transpose(1, 0))
            qp_paths_succeeded.append(qp_path_succeeded)
            qp_paths_costs.append(qp_path_cost)

        return qp_paths, qp_paths_succeeded, qp_paths_costs

    def qp_speed(self,
                 current_ego_state,
                 ego_dim,
                 obs_dim,
                 ego_poses_frenet,
                 obs_poses_frenet,
                 frenet_frame,
                 reference_line_lanes,
                 features,
                 traffic_light_data,
                 qp_paths):
        """
        QP speed planning.
        :param current_ego_state: current ego state
        :param ego_dim: ego dimensions
        :param obs_dim: obstacles dimensions
        :param ego_poses_frenet: shape (num_plans, num_future_steps + 1, 3), ego poses at rear axle in Frenet frame
        :param obs_poses_frenet: shape (num_obs, num_modes_for_eval, num_future_steps + 1, 3), obstacles' poses at center in Frenet frame
        :param frenet_frame: frenet frame object
        :param reference_line_lanes: reference line lanes
        :param features: FeaturesType
        :param traffic_light_data: traffic light data
        :param qp_paths: QP paths with reference point at rear axle
        :return: QP speeds, QP speeds succeeded, QP speeds costs.
        """
        num_points = self.num_points_for_eval
        step_eval = self.step_interval_for_eval
        obs_dim = copy.deepcopy(obs_dim)
        num_obs = obs_dim['length'].shape[0]
        obs_dim['length'] = np.repeat(obs_dim['length'], self.num_modes_for_eval)
        obs_dim['width'] = np.repeat(obs_dim['width'], self.num_modes_for_eval)
        obs_poses_frenet = obs_poses_frenet.reshape(-1, self.num_future_steps + 1, 3)
        qp_paths = np.stack(qp_paths)
        self.T = self.num_future_steps * 0.1  # time_horizon
        self.dt = self.T / self.num_future_steps
        self.t = np.linspace(0., self.T, num=self.num_future_steps + 1)
        self.a_min = -4.
        self.a_max = 2.4  # 4.
        self.v_max = min([lane.speed_limit_mps if lane.speed_limit_mps is not None else 15. for lane in reference_line_lanes[:3]])
        f_s_t = [interp1d(self.t, poses[:, 0], fill_value='extrapolate') for poses in ego_poses_frenet]
        ego_s_t = [fun(self.t) for fun in f_s_t]  # s(t)
        ego_vs_t = [np.diff(s_t) / np.diff(self.t) for s_t in ego_s_t]  # vs(t)
        ego_vs_t = [np.hstack((vs_t, vs_t[-1])) for vs_t in ego_vs_t]
        safe_dist = 2.
        qp_speed_w0 = 1.
        qp_speed_w1 = 0.
        qp_speed_w2 = 20.
        qp_speed_w3 = 1.
        qp_speeds = []
        qp_speeds_succeeded = []
        qp_speeds_costs = []

        self.s_ub = np.zeros((self.num_plans, self.num_future_steps + 1))
        self.s_lb = np.zeros((self.num_plans, self.num_future_steps + 1))
        for i_plan in range(self.num_plans):
            # QP Speed ➤ Feasible Region
            ego_obs_lateral_dist = np.abs(qp_paths[i_plan, :, 1] - obs_poses_frenet[:, :, 1])
            same_lane_mask_for_ahead = ego_obs_lateral_dist <= (ego_dim['width'] / 2 + obs_dim['width'] / 2).reshape(-1, 1)
            same_lane_mask_for_behind = ego_obs_lateral_dist <= (ego_dim['width'] / 2 + obs_dim['width'] / 2 - 0.5).reshape(-1, 1)
            ahead_mask = ego_poses_frenet[i_plan, :, 0] < obs_poses_frenet[:, :, 0]
            behind_mask = ego_poses_frenet[i_plan, :, 0] > obs_poses_frenet[:, :, 0]
            s_ub = np.ones_like(self.s[i_plan]) * self.s[i_plan, 0] + self.v_max * self.T  # note that self.s is stations of center of rear axle
            s_lb = np.ones_like(self.s[i_plan]) * self.s[i_plan, 0]
            if num_obs > 0:
                s_ub = np.repeat(s_ub.reshape(1, -1), num_obs * self.num_modes_for_eval, axis=0)
                s_lb = np.repeat(s_lb.reshape(1, -1), num_obs * self.num_modes_for_eval, axis=0)
                obs_ahead_length = np.broadcast_to(
                    obs_dim['length'].reshape(-1, 1),
                    shape=(num_obs * self.num_modes_for_eval, self.num_future_steps + 1)
                )[same_lane_mask_for_ahead & ahead_mask]
                obs_behind_length = np.broadcast_to(
                    obs_dim['length'].reshape(-1, 1),
                    shape=(num_obs * self.num_modes_for_eval, self.num_future_steps + 1)
                )[same_lane_mask_for_behind & behind_mask]
                s_ub[same_lane_mask_for_ahead & ahead_mask] = obs_poses_frenet[same_lane_mask_for_ahead & ahead_mask, 0] \
                                                    - obs_ahead_length / 2 \
                                                    - ego_dim['rear_axle_to_front_dist'] \
                                                    - safe_dist
                s_lb[same_lane_mask_for_behind & behind_mask] = obs_poses_frenet[same_lane_mask_for_behind & behind_mask, 0] \
                                                     + obs_behind_length / 2 \
                                                     + (ego_dim['length'] - ego_dim['rear_axle_to_front_dist']) \
                                                     + safe_dist
                s_ub = np.amin(s_ub, axis=0)
                s_lb = np.amax(s_lb, axis=0)

            s_ub_jump_idx = np.where(s_ub[1:] < s_ub[:-1])[0]
            while len(s_ub_jump_idx) > 0:
                index = s_ub_jump_idx[-1] + 1
                s_ub[0: index] = np.where(s_ub[0: index] > s_ub[index], s_ub[index], s_ub[0: index])
                s_ub_jump_idx = np.where(s_ub[1:] < s_ub[:-1])[0]

            s_lb_jump_idx = np.where(s_lb[:-1] > s_lb[1:])[0]
            while len(s_lb_jump_idx) > 0:
                index = s_lb_jump_idx[0]
                s_lb[index:] = np.where(s_lb[index:] < s_lb[index], s_lb[index], s_lb[index:])
                s_lb_jump_idx = np.where(s_lb[:-1] > s_lb[1:])[0]

            red_lane_connector_ids = [str(tl_data.lane_connector_id) for tl_data in traffic_light_data if tl_data.status.name == 'RED']
            red_reference_lanes = [pl for pl in reference_line_lanes if pl.id in red_lane_connector_ids]
            if len(red_reference_lanes) > 0:
                red_pl_points = [pl.baseline_path.discrete_path[0].point for pl in red_reference_lanes]
                red_pl_stations = frenet_frame.get_nearest_station_from_position(red_pl_points)
                front_mask = red_pl_stations > ego_poses_frenet[0, 0, 0]
                if np.any(front_mask):
                    min_red_pl_station = min(red_pl_stations[front_mask])
                    s_ub = np.minimum(s_ub, min_red_pl_station - ego_dim['length'] / 2)

            self.s_ub[i_plan] = s_ub
            self.s_lb[i_plan] = s_lb

            # QP Speed
            num_points_each_piece = self.num_future_steps // 5
            qp_t0_idx = 0
            qp_t1_idx = num_points_each_piece
            qp_t2_idx = num_points_each_piece * 2
            qp_t3_idx = num_points_each_piece * 3
            qp_t4_idx = num_points_each_piece * 4
            qp_t5_idx = self.num_future_steps
            qp_t0 = self.t[qp_t0_idx]
            qp_t1 = self.t[qp_t1_idx]
            qp_t2 = self.t[qp_t2_idx]
            qp_t3 = self.t[qp_t3_idx]
            qp_t4 = self.t[qp_t4_idx]
            qp_t5 = self.t[qp_t5_idx]
            qp_t1_t0 = qp_t1 - qp_t0
            qp_t2_t1 = qp_t2 - qp_t1
            qp_t3_t2 = qp_t3 - qp_t2
            qp_t4_t3 = qp_t4 - qp_t3
            qp_t5_t4 = qp_t5 - qp_t4
            num_points_qp_speed = 11
            step = self.num_future_steps // (num_points_qp_speed - 1)
            t0 = 0.
            t1 = self.dt * step
            t2 = self.dt * step * 2
            t3 = self.dt * step * 3
            t4 = self.dt * step * 4
            t5 = self.dt * step * 5
            t6 = self.dt * step * 6
            t7 = self.dt * step * 7
            t8 = self.dt * step * 8
            t9 = self.dt * step * 9
            t10 = self.dt * step * 10

            # QP Speed ➤ Constraints
            # QP Speed ➤ Constraints ➤ Equality Constraints
            qp_vA = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 0
                              [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 1
                              [0., 0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 2
                              [1., qp_t1_t0, qp_t1_t0 ** 2, qp_t1_t0 ** 3, -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 3
                              [0., 1., 2 * qp_t1_t0, 3 * qp_t1_t0 ** 2, 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 4
                              [0., 0., 2., 6 * qp_t1_t0, 0., 0., -2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 5
                              [0., 0., 0., 0., 1., qp_t2_t1, qp_t2_t1 ** 2, qp_t2_t1 ** 3, -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 6
                              [0., 0., 0., 0., 0., 1., 2 * qp_t2_t1, 3 * qp_t2_t1 ** 2, 0., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 7
                              [0., 0., 0., 0., 0., 0., 2., 6 * qp_t2_t1, 0., 0., -2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 8
                              [0., 0., 0., 0., 0., 0., 0., 0., 1., qp_t3_t2, qp_t3_t2 ** 2, qp_t3_t2 ** 3, -1., 0., 0., 0., 0., 0., 0., 0.],  # 9
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2 * qp_t3_t2, 3 * qp_t3_t2 ** 2, 0., -1., 0., 0., 0., 0., 0., 0.],  # 10
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 6 * qp_t3_t2, 0., 0., -2., 0., 0., 0., 0., 0.],  # 11
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., qp_t4_t3, qp_t4_t3 ** 2, qp_t4_t3 ** 3, -1., 0., 0., 0.],  # 12
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2 * qp_t4_t3, 3 * qp_t4_t3 ** 2, 0., -1., 0., 0.],  # 13
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 2., 6 * qp_t4_t3, 0, 0., -2., 0.],  # 14
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1, qp_t5_t4, qp_t5_t4 ** 2, qp_t5_t4 ** 3],  # 15
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 1., 2 * qp_t5_t4, 3 * qp_t5_t4 ** 2],  # 16
                              [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0, 0., 2., 6 * qp_t5_t4],  # 17
                              ])
            vs_0 = (current_ego_state.dynamic_car_state.rear_axle_velocity_2d.x * np.cos(ego_poses_frenet[i_plan, 0, 2])
                    - current_ego_state.dynamic_car_state.rear_axle_velocity_2d.y * np.sin(ego_poses_frenet[i_plan, 0, 2]))
            as_0 = (current_ego_state.dynamic_car_state.rear_axle_acceleration_2d.x * np.cos(ego_poses_frenet[i_plan, 0, 2])
                    - current_ego_state.dynamic_car_state.rear_axle_acceleration_2d.y * np.sin(ego_poses_frenet[i_plan, 0, 2]))
            s_des = ego_poses_frenet[i_plan, -1, 0]
            vs_des = (ego_poses_frenet[i_plan, -1, 0] - ego_poses_frenet[i_plan, -1 - step_eval, 0]) / (self.dt * step_eval)
            qp_vb = np.array([[qp_paths[i_plan, 0, 0]],  # 0: s(t0)
                              [vs_0],  # 1: s'(t0)
                              [as_0],  # 2: s"(t0)
                              [0.],  # 3
                              [0.],  # 4
                              [0.],  # 5
                              [0.],  # 6
                              [0.],  # 7
                              [0.],  # 8
                              [0.],  # 9
                              [0.],  # 10
                              [0.],  # 11
                              [0.],  # 12
                              [0.],  # 13
                              [0.],  # 14
                              [s_des],  # 15: s(t5)
                              [max(vs_des, 0.)],  # 16: s'(t5)
                              [0.],  # 17: s"(t5)
                              ])
            # QP Speed ➤ Constraints ➤ Inequality Constraints
            # 0 s(t_i) <= s(t_i+1)
            qp_vIA0_0_1 = np.array([[0., t1, t1 ** 2, t1 ** 3,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.]])  # qp v I0
            qp_vIA0_1_2 = np.array([[0., t2 - t1, t2 ** 2 - t1 ** 2, t2 ** 3 - t1 ** 3,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.]])  # qp v I1
            qp_vIA0_2_3 = np.array([[0., 0., 0., 0.,
                                     0., t3 - t2, (t3 - qp_t1) ** 2 - (t2 - qp_t1) ** 2, (t3 - qp_t1) ** 3 - (t2 - qp_t1) ** 3,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.]])  # qp v I2
            qp_vIA0_3_4 = np.array([[0., 0., 0., 0.,
                                     0., t4 - t3, (t4 - qp_t1) ** 2 - (t3 - qp_t1) ** 2, (t4 - qp_t1) ** 3 - (t3 - qp_t1) ** 3,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.]])  # qp v I3
            qp_vIA0_4_5 = np.array([[0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., t5 - t4, (t5 - qp_t2) ** 2 - (t4 - qp_t2) ** 2, (t5 - qp_t2) ** 3 - (t4 - qp_t2) ** 3,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.]])  # qp v I4
            qp_vIA0_5_6 = np.array([[0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., t6 - t5, (t6 - qp_t2) ** 2 - (t5 - qp_t2) ** 2, (t6 - qp_t2) ** 3 - (t5 - qp_t2) ** 3,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.]])  # qp v I5
            qp_vIA0_6_7 = np.array([[0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., t7 - t6, (t7 - qp_t3) ** 2 - (t6 - qp_t3) ** 2, (t7 - qp_t3) ** 3 - (t6 - qp_t3) ** 3,
                                     0., 0., 0., 0.]])  # qp v I6
            qp_vIA0_7_8 = np.array([[0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., t8 - t7, (t8 - qp_t3) ** 2 - (t7 - qp_t3) ** 2, (t8 - qp_t3) ** 3 - (t7 - qp_t3) ** 3,
                                     0., 0., 0., 0.]])  # qp v I7
            qp_vIA0_8_9 = np.array([[0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., 0., 0., 0.,
                                     0., t9 - t8, (t9 - qp_t4) ** 2 - (t8 - qp_t4) ** 2, (t9 - qp_t4) ** 3 - (t8 - qp_t4) ** 3,
                                     ]])  # qp v I8
            qp_vIA0_9_10 = np.array([[0., 0., 0., 0.,
                                      0., 0., 0., 0.,
                                      0., 0., 0., 0.,
                                      0., 0., 0., 0.,
                                      0., t10 - t9, (t10 - qp_t4) ** 2 - (t9 - qp_t4) ** 2, (t10 - qp_t4) ** 3 - (t9 - qp_t4) ** 3,
                                      ]])  # qp v I9
            qp_vIb0_1_10 = np.zeros((num_points_qp_speed - 1, 1))
            # 1 s_lb <= s(t_i)
            qp_vIA1_1 = np.array([[1., t1, t1 ** 2, t1 ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I10
            qp_vIA1_2 = np.array([[1., t2, t2 ** 2, t2 ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I11
            qp_vIA1_3 = np.array([[0., 0., 0., 0.,
                                   1., t3 - qp_t1, (t3 - qp_t1) ** 2, (t3 - qp_t1) ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I12
            qp_vIA1_4 = np.array([[0., 0., 0., 0.,
                                   1., t4 - qp_t1, (t4 - qp_t1) ** 2, (t4 - qp_t1) ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I13
            qp_vIA1_5 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   1., t5 - qp_t2, (t5 - qp_t2) ** 2, (t5 - qp_t2) ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I14
            qp_vIA1_6 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   1., t6 - qp_t2, (t6 - qp_t2) ** 2, (t6 - qp_t2) ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I15
            qp_vIA1_7 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   1., t7 - qp_t3, (t7 - qp_t3) ** 2, (t7 - qp_t3) ** 3,
                                   0., 0., 0., 0.]])  # qp v I16
            qp_vIA1_8 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   1., t8 - qp_t3, (t8 - qp_t3) ** 2, (t8 - qp_t3) ** 3,
                                   0., 0., 0., 0.]])  # qp v I17
            qp_vIA1_9 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   1., t9 - qp_t4, (t9 - qp_t4) ** 2, (t9 - qp_t4) ** 3,
                                   ]])  # qp v I18
            qp_vIA1_10 = np.array([[0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    1., t10 - qp_t4, (t10 - qp_t4) ** 2, (t10 - qp_t4) ** 3
                                    ]])  # qp v I19
            qp_vIb1_1_10 = np.array([self.s_lb[i_plan, range(step, self.num_future_steps + 1, step)]]).T
            # 2 s(t_i) <= s_ub
            qp_vIA2_1 = np.array([[-1., -t1, -t1 ** 2, -t1 ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I20
            qp_vIA2_2 = np.array([[-1., -t2, -t2 ** 2, -t2 ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I21
            qp_vIA2_3 = np.array([[0., 0., 0., 0.,
                                   -1., -(t3 - qp_t1), -(t3 - qp_t1) ** 2, -(t3 - qp_t1) ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I22
            qp_vIA2_4 = np.array([[0., 0., 0., 0.,
                                   -1., -(t4 - qp_t1), -(t4 - qp_t1) ** 2, -(t4 - qp_t1) ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I23
            qp_vIA2_5 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   -1., -(t5 - qp_t2), -(t5 - qp_t2) ** 2, -(t5 - qp_t2) ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I24
            qp_vIA2_6 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   -1., -(t6 - qp_t2), -(t6 - qp_t2) ** 2, -(t6 - qp_t2) ** 3,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I25
            qp_vIA2_7 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   -1., -(t7 - qp_t3), -(t7 - qp_t3) ** 2, -(t7 - qp_t3) ** 3,
                                   0., 0., 0., 0.]])  # qp v I26
            qp_vIA2_8 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   -1., -(t8 - qp_t3), -(t8 - qp_t3) ** 2, -(t8 - qp_t3) ** 3,
                                   0., 0., 0., 0.]])  # qp v I27
            qp_vIA2_9 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   -1., -(t9 - qp_t4), -(t9 - qp_t4) ** 2, -(t9 - qp_t4) ** 3,
                                   ]])  # qp v I28
            qp_vIA2_10 = np.array([[0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    -1., -(t10 - qp_t4), -(t10 - qp_t4) ** 2, -(t10 - qp_t4) ** 3
                                    ]])  # qp v I29
            qp_vIb2_1_10 = np.array([-self.s_ub[i_plan, range(step, self.num_future_steps + 1, step)]]).T
            # 3 s'(t_i) <= v_max
            qp_vIA3_1 = np.array([[0., -1., -2 * t1, -3 * t1 ** 2,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I30
            qp_vIA3_2 = np.array([[0., -1., -2 * t2, -3 * t2 ** 2,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I31
            qp_vIA3_3 = np.array([[0., 0., 0., 0.,
                                   0., -1., -2 * (t3 - qp_t1), -3 * (t3 - qp_t1) ** 2,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I32
            qp_vIA3_4 = np.array([[0., 0., 0., 0.,
                                   0., -1., -2 * (t4 - qp_t1), -3 * (t4 - qp_t1) ** 2,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I33
            qp_vIA3_5 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., -1., -2 * (t5 - qp_t2), -3 * (t5 - qp_t2) ** 2,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I34
            qp_vIA3_6 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., -1., -2 * (t6 - qp_t2), -3 * (t6 - qp_t2) ** 2,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I35
            qp_vIA3_7 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., -1., -2 * (t7 - qp_t3), -3 * (t7 - qp_t3) ** 2,
                                   0., 0., 0., 0.]])  # qp v I36
            qp_vIA3_8 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., -1., -2 * (t8 - qp_t3), -3 * (t8 - qp_t3) ** 2,
                                   0., 0., 0., 0.]])  # qp v I37
            qp_vIA3_9 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., -1., -2 * (t9 - qp_t4), -3 * (t9 - qp_t4) ** 2,
                                   ]])  # qp v I38
            qp_vIA3_10 = np.array([[0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,  # qp v I39
                                    0., -1., -2 * (t10 - qp_t4), -3 * (t10 - qp_t4) ** 2,
                                    ]])
            qp_vIb3_1_10 = -np.ones((num_points_qp_speed - 1, 1)) * self.v_max
            # 4 s"(t_i) >= a_min
            qp_vIA4_1 = np.array([[0., 0., 2, 6 * t1,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I40
            qp_vIA4_2 = np.array([[0., 0., 2, 6 * t2,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I41
            qp_vIA4_3 = np.array([[0., 0., 0., 0.,
                                   0., 0., 2, 6 * (t3 - qp_t1),
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I42
            qp_vIA4_4 = np.array([[0., 0., 0., 0.,
                                   0., 0., 2, 6 * (t4 - qp_t1),
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I43
            qp_vIA4_5 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 2, 6 * (t5 - qp_t2),
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I44
            qp_vIA4_6 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 2, 6 * (t6 - qp_t2),
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I45
            qp_vIA4_7 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 2, 6 * (t7 - qp_t3),
                                   0., 0., 0., 0.]])  # qp v I46
            qp_vIA4_8 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 2, 6 * (t8 - qp_t3),
                                   0., 0., 0., 0.]])  # qp v I47
            qp_vIA4_9 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 2, 6 * (t9 - qp_t4),
                                   ]])  # qp v I48
            qp_vIA4_10 = np.array([[0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 2, 6 * (t10 - qp_t4)
                                    ]])  # qp v I49
            qp_vIb4_1_10 = np.ones((num_points_qp_speed - 1, 1)) * self.a_min
            # 5 s"(t_i) <= a_max
            qp_vIA5_1 = np.array([[0., 0., -2, -6 * t1,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I50
            qp_vIA5_2 = np.array([[0., 0., -2, -6 * t2,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I51
            qp_vIA5_3 = np.array([[0., 0., 0., 0.,
                                   0., 0., -2, -6 * (t3 - qp_t1),
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I52
            qp_vIA5_4 = np.array([[0., 0., 0., 0.,
                                   0., 0., -2, -6 * (t4 - qp_t1),
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I53
            qp_vIA5_5 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., -2, -6 * (t5 - qp_t2),
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I54
            qp_vIA5_6 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., -2, -6 * (t6 - qp_t2),
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.]])  # qp v I55
            qp_vIA5_7 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., -2, -6 * (t7 - qp_t3),
                                   0., 0., 0., 0.]])  # qp v I56
            qp_vIA5_8 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., -2, -6 * (t8 - qp_t3),
                                   0., 0., 0., 0.]])  # qp v I57
            qp_vIA5_9 = np.array([[0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., 0., 0.,
                                   0., 0., -2, -6 * (t9 - qp_t4),
                                   ]])  # qp v I58
            qp_vIA5_10 = np.array([[0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., 0., 0.,
                                    0., 0., -2, -6 * (t10 - qp_t4)
                                    ]])  # qp v I59
            qp_vIb5_1_10 = -np.ones((num_points_qp_speed - 1, 1)) * self.a_max
            # QP Speed ➤ Constraints ➤ vstack
            qp_vIA = np.vstack((
                qp_vIA0_0_1, qp_vIA0_1_2, qp_vIA0_2_3, qp_vIA0_3_4, qp_vIA0_4_5, qp_vIA0_5_6, qp_vIA0_6_7, qp_vIA0_7_8, qp_vIA0_8_9, qp_vIA0_9_10,
                qp_vIA1_1, qp_vIA1_2, qp_vIA1_3, qp_vIA1_4, qp_vIA1_5, qp_vIA1_6, qp_vIA1_7, qp_vIA1_8, qp_vIA1_9, qp_vIA1_10,
                qp_vIA2_1, qp_vIA2_2, qp_vIA2_3, qp_vIA2_4, qp_vIA2_5, qp_vIA2_6, qp_vIA2_7, qp_vIA2_8, qp_vIA2_9, qp_vIA2_10,
                qp_vIA3_1, qp_vIA3_2, qp_vIA3_3, qp_vIA3_4, qp_vIA3_5, qp_vIA3_6, qp_vIA3_7, qp_vIA3_8, qp_vIA3_9, qp_vIA3_10,
                qp_vIA4_1, qp_vIA4_2, qp_vIA4_3, qp_vIA4_4, qp_vIA4_5, qp_vIA4_6, qp_vIA4_7, qp_vIA4_8, qp_vIA4_9, qp_vIA4_10,
                qp_vIA5_1, qp_vIA5_2, qp_vIA5_3, qp_vIA5_4, qp_vIA5_5, qp_vIA5_6, qp_vIA5_7, qp_vIA5_8, qp_vIA5_9, qp_vIA5_10,
            ))
            qp_vIb = np.vstack((qp_vIb0_1_10, qp_vIb1_1_10, qp_vIb2_1_10, qp_vIb3_1_10, qp_vIb4_1_10, qp_vIb5_1_10))

            # QP Speed ➤ Objective Functional
            # QP Speed ➤ Objective Functional ➤ Objective Functional 0
            vG_0 = np.array([
                [qp_t1_t0, 1 / 2 * qp_t1_t0 ** 2, 1 / 3 * qp_t1_t0 ** 3, 1 / 4 * qp_t1_t0 ** 4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1 / 2 * qp_t1_t0 ** 2, 1 / 3 * qp_t1_t0 ** 3, 1 / 4 * qp_t1_t0 ** 4, 1 / 5 * qp_t1_t0 ** 5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [1 / 3 * qp_t1_t0 ** 3, 1 / 4 * qp_t1_t0 ** 4, 1 / 5 * qp_t1_t0 ** 5, 1 / 6 * qp_t1_t0 ** 6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [1 / 4 * qp_t1_t0 ** 4, 1 / 5 * qp_t1_t0 ** 5, 1 / 6 * qp_t1_t0 ** 6, 1 / 7 * qp_t1_t0 ** 7, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., qp_t2_t1, 1 / 2 * qp_t2_t1 ** 2, 1 / 3 * qp_t2_t1 ** 3, 1 / 4 * qp_t2_t1 ** 4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1 / 2 * qp_t2_t1 ** 2, 1 / 3 * qp_t2_t1 ** 3, 1 / 4 * qp_t2_t1 ** 4, 1 / 5 * qp_t2_t1 ** 5, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 1 / 3 * qp_t2_t1 ** 3, 1 / 4 * qp_t2_t1 ** 4, 1 / 5 * qp_t2_t1 ** 5, 1 / 6 * qp_t2_t1 ** 6, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 1 / 4 * qp_t2_t1 ** 4, 1 / 5 * qp_t2_t1 ** 5, 1 / 6 * qp_t2_t1 ** 6, 1 / 7 * qp_t2_t1 ** 7, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., qp_t3_t2, 1 / 2 * qp_t3_t2 ** 2, 1 / 3 * qp_t3_t2 ** 3, 1 / 4 * qp_t3_t2 ** 4, 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1 / 2 * qp_t3_t2 ** 2, 1 / 3 * qp_t3_t2 ** 3, 1 / 4 * qp_t3_t2 ** 4, 1 / 5 * qp_t3_t2 ** 5, 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1 / 3 * qp_t3_t2 ** 3, 1 / 4 * qp_t3_t2 ** 4, 1 / 5 * qp_t3_t2 ** 5, 1 / 6 * qp_t3_t2 ** 6, 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 1 / 4 * qp_t3_t2 ** 4, 1 / 5 * qp_t3_t2 ** 5, 1 / 6 * qp_t3_t2 ** 6, 1 / 7 * qp_t3_t2 ** 7, 0., 0., 0., 0., 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t4_t3, 1 / 2 * qp_t4_t3 ** 2, 1 / 3 * qp_t4_t3 ** 3, 1 / 4 * qp_t4_t3 ** 4, 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 2 * qp_t4_t3 ** 2, 1 / 3 * qp_t4_t3 ** 3, 1 / 4 * qp_t4_t3 ** 4, 1 / 5 * qp_t4_t3 ** 5, 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 3 * qp_t4_t3 ** 3, 1 / 4 * qp_t4_t3 ** 4, 1 / 5 * qp_t4_t3 ** 5, 1 / 6 * qp_t4_t3 ** 6, 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 4 * qp_t4_t3 ** 4, 1 / 5 * qp_t4_t3 ** 5, 1 / 6 * qp_t4_t3 ** 6, 1 / 7 * qp_t4_t3 ** 7, 0.,
                 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t5_t4, 1 / 2 * qp_t5_t4 ** 2, 1 / 3 * qp_t5_t4 ** 3, 1 / 4 * qp_t5_t4 ** 4],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 2 * qp_t5_t4 ** 2, 1 / 3 * qp_t5_t4 ** 3, 1 / 4 * qp_t5_t4 ** 4,
                 1 / 5 * qp_t5_t4 ** 5],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 3 * qp_t5_t4 ** 3, 1 / 4 * qp_t5_t4 ** 4, 1 / 5 * qp_t5_t4 ** 5,
                 1 / 6 * qp_t5_t4 ** 6],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1 / 4 * qp_t5_t4 ** 4, 1 / 5 * qp_t5_t4 ** 5, 1 / 6 * qp_t5_t4 ** 6,
                 1 / 7 * qp_t5_t4 ** 7]
            ])
            s = ego_poses_frenet[i_plan, :, 0]
            qp_s0 = s[qp_t0_idx]
            qp_s1 = s[qp_t1_idx]
            qp_s2 = s[qp_t2_idx]
            qp_s3 = s[qp_t3_idx]
            qp_s4 = s[qp_t4_idx]
            qp_s5 = s[qp_t5_idx]
            s0 = s[0]
            s1 = s[step]
            s2 = s[step * 2]
            s3 = s[step * 3]
            s4 = s[step * 4]
            s5 = s[step * 5]
            s6 = s[step * 6]
            s7 = s[step * 7]
            s8 = s[step * 8]
            s9 = s[step * 9]
            s10 = s[step * 10]
            vc0_0_0 = np.matrix([1 / 2 * (s1 + qp_s0) * (t1 - qp_t0) +
                                 1 / 2 * (s2 + s1) * (t2 - t1)])
            vc0_0_1 = np.matrix([(2 * s1 + qp_s0) / 6 * (t1 - qp_t0) ** 2 +
                                 (s2 - s1) / (t2 - t1) *
                                 ((1 / 3 * t2 ** 3 - 1 / 2 * (qp_t0 + t1) * t2 ** 2 + qp_t0 * t1 * t2) -
                                  (1 / 3 * t1 ** 3 - 1 / 2 * (qp_t0 + t1) * t1 ** 2 + qp_t0 * t1 * t1)) +
                                 (1 / 2 * s1 * ((t2 - qp_t0) ** 2 - (t1 - qp_t0) ** 2))])
            vc0_0_2 = np.matrix([1 / 12 * (3 * s1 + qp_s0) * (t1 - qp_t0) ** 3 +
                                 (s2 - s1) / (t2 - t1) *
                                 ((1 / 4 * t2 ** 4 - 1 / 3 * (2 * qp_t0 + t1) * t2 ** 3 + 1 / 2 * (qp_t0 ** 2 + 2 * qp_t0 * t1) * t2 ** 2 - qp_t0 ** 2 * t1 * t2) -
                                  (1 / 4 * t1 ** 4 - 1 / 3 * (2 * qp_t0 + t1) * t1 ** 3 + 1 / 2 * (qp_t0 ** 2 + 2 * qp_t0 * t1) * t1 ** 2 - qp_t0 ** 2 * t1 * t1)) +
                                 (1 / 3 * s1 * ((t2 - qp_t0) ** 3 - (t1 - qp_t0) ** 3))])
            vc0_0_3 = np.matrix([(4 * s1 + qp_s0) / 20 * (t1 - qp_t0) ** 4 +
                                 (s2 - s1) / (t2 - t1) *
                                 ((1 / 5 * t2 ** 5 - 1 / 4 * (3 * qp_t0 + t1) * t2 ** 4 + 1 / 3 * (3 * qp_t0 ** 2 + 3 * qp_t0 * t1) *
                                   t2 ** 3 - 1 / 2 * (qp_t0 ** 3 + 3 * qp_t0 ** 2 * t1) * t2 ** 2 + qp_t0 ** 3 * t1 * t2) -
                                  (1 / 5 * t1 ** 5 - 1 / 4 * (3 * qp_t0 + t1) * t1 ** 4 + 1 / 3 * (3 * qp_t0 ** 2 + 3 * qp_t0 * t1) *
                                   t1 ** 3 - 1 / 2 * (qp_t0 ** 3 + 3 * qp_t0 ** 2 * t1) * t1 ** 2 + qp_t0 ** 3 * t1 * t1)) +
                                 (1 / 4 * s1 * ((t2 - qp_t0) ** 4 - (t1 - qp_t0) ** 4))])
            vc0_1_0 = np.matrix([1 / 2 * (s3 + qp_s1) * (t3 - qp_t1) +
                                 1 / 2 * (s4 + s3) * (t4 - t3)])
            vc0_1_1 = np.matrix([(2 * s3 + qp_s1) / 6 * (t3 - qp_t1) ** 2 +
                                 (s4 - s3) / (t4 - t3) *
                                 ((1 / 3 * t4 ** 3 - 1 / 2 * (qp_t1 + t3) * t4 ** 2 + qp_t1 * t3 * t4) -
                                  (1 / 3 * t3 ** 3 - 1 / 2 * (qp_t1 + t3) * t3 ** 2 + qp_t1 * t3 * t3)) +
                                 (1 / 2 * s3 * ((t4 - qp_t1) ** 2 - (t3 - qp_t1) ** 2))])
            vc0_1_2 = np.matrix([1 / 12 * (3 * s3 + qp_s1) * (t3 - qp_t1) ** 3 +
                                 (s4 - s3) / (t4 - t3) *
                                 ((1 / 4 * t4 ** 4 - 1 / 3 * (2 * qp_t1 + t3) * t4 ** 3 + 1 / 2 * (qp_t1 ** 2 + 2 * qp_t1 * t3) *
                                   t4 ** 2 - qp_t1 ** 2 * t3 * t4) -
                                  (1 / 4 * t3 ** 4 - 1 / 3 * (2 * qp_t1 + t3) * t3 ** 3 + 1 / 2 * (qp_t1 ** 2 + 2 * qp_t1 * t3) *
                                   t3 ** 2 - qp_t1 ** 2 * t3 * t3)) +
                                 (1 / 3 * s3 * ((t4 - qp_t1) ** 3 - (t3 - qp_t1) ** 3))])
            vc0_1_3 = np.matrix([(4 * s3 + qp_s1) / 20 * (t3 - qp_t1) ** 4 +
                                 (s4 - s3) / (t4 - t3) *
                                 ((1 / 5 * t4 ** 5 - 1 / 4 * (3 * qp_t1 + t3) * t4 ** 4 + 1 / 3 * (
                                             3 * qp_t1 ** 2 + 3 * qp_t1 * t3) * t4 ** 3 - 1 / 2 * (qp_t1 ** 3 + 3 * qp_t1 ** 2 * t3) *
                                   t4 ** 2 + qp_t1 ** 3 * t3 * t4) -
                                  (1 / 5 * t3 ** 5 - 1 / 4 * (3 * qp_t1 + t3) * t3 ** 4 + 1 / 3 * (
                                              3 * qp_t1 ** 2 + 3 * qp_t1 * t3) * t3 ** 3 - 1 / 2 * (qp_t1 ** 3 + 3 * qp_t1 ** 2 * t3) *
                                   t3 ** 2 + qp_t1 ** 3 * t3 * t3)) +
                                 (1 / 4 * s3 * ((t4 - qp_t1) ** 4 - (t3 - qp_t1) ** 4))])
            vc0_2_0 = np.matrix([1 / 2 * (s5 + qp_s2) * (t5 - qp_t2) +
                                 1 / 2 * (s6 + s5) * (t6 - t5)])
            vc0_2_1 = np.matrix([(2 * s5 + qp_s2) / 6 * (t5 - qp_t2) ** 2 +
                                 (s6 - s5) / (t6 - t5) *
                                 ((1 / 3 * t6 ** 3 - 1 / 2 * (qp_t2 + t5) * t6 ** 2 + qp_t2 * t5 * t6) -
                                  (1 / 3 * t5 ** 3 - 1 / 2 * (qp_t2 + t5) * t5 ** 2 + qp_t2 * t5 * t5)) +
                                 (1 / 2 * s5 * ((t6 - qp_t2) ** 2 - (t5 - qp_t2) ** 2))])
            vc0_2_2 = np.matrix([1 / 12 * (3 * s5 + qp_s2) * (t5 - qp_t2) ** 3 +
                                 (s6 - s5) / (t6 - t5) *
                                 ((1 / 4 * t6 ** 4 - 1 / 3 * (2 * qp_t2 + t5) * t6 ** 3 + 1 / 2 * (qp_t2 ** 2 + 2 * qp_t2 * t5) *
                                   t6 ** 2 - qp_t2 ** 2 * t5 * t6) -
                                  (1 / 4 * t5 ** 4 - 1 / 3 * (2 * qp_t2 + t5) * t5 ** 3 + 1 / 2 * (qp_t2 ** 2 + 2 * qp_t2 * t5) *
                                   t5 ** 2 - qp_t2 ** 2 * t5 * t5)) +
                                 (1 / 3 * s5 * ((t6 - qp_t2) ** 3 - (t5 - qp_t2) ** 3))])
            vc0_2_3 = np.matrix([(4 * s5 + qp_s2) / 20 * (t5 - qp_t2) ** 4 +
                                 (s6 - s5) / (t6 - t5) *
                                 ((1 / 5 * t6 ** 5 - 1 / 4 * (3 * qp_t2 + t5) * t6 ** 4 + 1 / 3 * (
                                             3 * qp_t2 ** 2 + 3 * qp_t2 * t5) * t6 ** 3 - 1 / 2 * (qp_t2 ** 3 + 3 * qp_t2 ** 2 * t5) *
                                   t6 ** 2 + qp_t2 ** 3 * t5 * t6) -
                                  (1 / 5 * t5 ** 5 - 1 / 4 * (3 * qp_t2 + t5) * t5 ** 4 + 1 / 3 * (
                                              3 * qp_t2 ** 2 + 3 * qp_t2 * t5) * t5 ** 3 - 1 / 2 * (qp_t2 ** 3 + 3 * qp_t2 ** 2 * t5) *
                                   t5 ** 2 + qp_t2 ** 3 * t5 * t5)) +
                                 (1 / 4 * s5 * ((t6 - qp_t2) ** 4 - (t5 - qp_t2) ** 4))])
            vc0_3_0 = np.matrix([1 / 2 * (s7 + qp_s3) * (t7 - qp_t3) +
                                 1 / 2 * (s8 + s7) * (t8 - t7)])
            vc0_3_1 = np.matrix([(2 * s7 + qp_s3) / 6 * (t7 - qp_t3) ** 2 +
                                 (s8 - s7) / (t8 - t7) *
                                 ((1 / 3 * t8 ** 3 - 1 / 2 * (qp_t3 + t7) * t8 ** 2 + qp_t3 * t7 * t8) -
                                  (1 / 3 * t7 ** 3 - 1 / 2 * (qp_t3 + t7) * t7 ** 2 + qp_t3 * t7 * t7)) +
                                 (1 / 2 * s7 * ((t8 - qp_t3) ** 2 - (t7 - qp_t3) ** 2))])
            vc0_3_2 = np.matrix([1 / 12 * (3 * s7 + qp_s3) * (t7 - qp_t3) ** 3 +
                                 (s8 - s7) / (t8 - t7) *
                                 ((1 / 4 * t8 ** 4 - 1 / 3 * (2 * qp_t3 + t7) * t8 ** 3 + 1 / 2 * (qp_t3 ** 2 + 2 * qp_t3 * t7) *
                                   t8 ** 2 - qp_t3 ** 2 * t7 * t8) -
                                  (1 / 4 * t7 ** 4 - 1 / 3 * (2 * qp_t3 + t7) * t7 ** 3 + 1 / 2 * (qp_t3 ** 2 + 2 * qp_t3 * t7) *
                                   t7 ** 2 - qp_t3 ** 2 * t7 * t7)) +
                                 (1 / 3 * s7 * ((t8 - qp_t3) ** 3 - (t7 - qp_t3) ** 3))])
            vc0_3_3 = np.matrix([(4 * s7 + qp_s3) / 20 * (t7 - qp_t3) ** 4 +
                                 (s8 - s7) / (t8 - t7) *
                                 ((1 / 5 * t8 ** 5 - 1 / 4 * (3 * qp_t3 + t7) * t8 ** 4 + 1 / 3 * (
                                             3 * qp_t3 ** 2 + 3 * qp_t3 * t7) * t8 ** 3 - 1 / 2 * (qp_t3 ** 3 + 3 * qp_t3 ** 2 * t7) *
                                   t8 ** 2 + qp_t3 ** 3 * t7 * t8) -
                                  (1 / 5 * t7 ** 5 - 1 / 4 * (3 * qp_t3 + t7) * t7 ** 4 + 1 / 3 * (
                                              3 * qp_t3 ** 2 + 3 * qp_t3 * t7) * t7 ** 3 - 1 / 2 * (qp_t3 ** 3 + 3 * qp_t3 ** 2 * t7) *
                                   t7 ** 2 + qp_t3 ** 3 * t7 * t7)) +
                                 (1 / 4 * s7 * ((t8 - qp_t3) ** 4 - (t7 - qp_t3) ** 4))])
            vc0_4_0 = np.matrix([1 / 2 * (s9 + qp_s4) * (t9 - qp_t4) +
                                 1 / 2 * (s10 + s9) * (t10 - t9)])
            vc0_4_1 = np.matrix([(2 * s9 + qp_s4) / 6 * (t9 - qp_t4) ** 2 +
                                 (s10 - s9) / (t10 - t9) *
                                 ((1 / 3 * t10 ** 3 - 1 / 2 * (qp_t4 + t9) * t10 ** 2 + qp_t4 * t9 * t10) -
                                  (1 / 3 * t9 ** 3 - 1 / 2 * (qp_t4 + t9) * t9 ** 2 + qp_t4 * t9 * t9)) +
                                 (1 / 2 * s9 * ((t10 - qp_t4) ** 2 - (t9 - qp_t4) ** 2))])
            vc0_4_2 = np.matrix([1 / 12 * (3 * s9 + qp_s4) * (t9 - qp_t4) ** 3 +
                                 (s10 - s9) / (t10 - t9) *
                                 ((1 / 4 * t10 ** 4 - 1 / 3 * (2 * qp_t4 + t9) * t10 ** 3 + 1 / 2 * (qp_t4 ** 2 + 2 * qp_t4 * t9) *
                                   t10 ** 2 - qp_t4 ** 2 * t9 * t10) -
                                  (1 / 4 * t9 ** 4 - 1 / 3 * (2 * qp_t4 + t9) * t9 ** 3 + 1 / 2 * (qp_t4 ** 2 + 2 * qp_t4 * t9) *
                                   t9 ** 2 - qp_t4 ** 2 * t9 * t9)) +
                                 (1 / 3 * s9 * ((t10 - qp_t4) ** 3 - (t9 - qp_t4) ** 3))])
            vc0_4_3 = np.matrix([(4 * s9 + qp_s4) / 20 * (t9 - qp_t4) ** 4 +
                                 (s10 - s9) / (t10 - t9) *
                                 ((1 / 5 * t10 ** 5 - 1 / 4 * (3 * qp_t4 + t9) * t10 ** 4 + 1 / 3 * (
                                             3 * qp_t4 ** 2 + 3 * qp_t4 * t9) * t10 ** 3 - 1 / 2 * (qp_t4 ** 3 + 3 * qp_t4 ** 2 * t9) *
                                   t10 ** 2 + qp_t4 ** 3 * t9 * t10) -
                                  (1 / 5 * t9 ** 5 - 1 / 4 * (3 * qp_t4 + t9) * t9 ** 4 + 1 / 3 * (
                                              3 * qp_t4 ** 2 + 3 * qp_t4 * t9) * t9 ** 3 - 1 / 2 * (qp_t4 ** 3 + 3 * qp_t4 ** 2 * t9) *
                                   t9 ** 2 + qp_t4 ** 3 * t9 * t9)) +
                                 (1 / 4 * s9 * ((t10 - qp_t4) ** 4 - (t9 - qp_t4) ** 4))])
            qp_vc0 = np.vstack((vc0_0_0, vc0_0_1, vc0_0_2, vc0_0_3,
                                vc0_1_0, vc0_1_1, vc0_1_2, vc0_1_3,
                                vc0_2_0, vc0_2_1, vc0_2_2, vc0_2_3,
                                vc0_3_0, vc0_3_1, vc0_3_2, vc0_3_3,
                                vc0_4_0, vc0_4_1, vc0_4_2, vc0_4_3))
            # QP Speed ➤ Objective Functional ➤ Objective Functional 1
            vG_1 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., qp_t1_t0, qp_t1_t0 ** 2, qp_t1_t0 ** 3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., qp_t1_t0 ** 2, 4 / 3 * qp_t1_t0 ** 3, 3 / 2 * qp_t1_t0 ** 4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., qp_t1_t0 ** 3, 3 / 2 * qp_t1_t0 ** 4, 9 / 5 * qp_t1_t0 ** 5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., qp_t2_t1, qp_t2_t1 ** 2, qp_t2_t1 ** 3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., qp_t2_t1 ** 2, 4 / 3 * qp_t2_t1 ** 3, 3 / 2 * qp_t2_t1 ** 4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., qp_t2_t1 ** 3, 3 / 2 * qp_t2_t1 ** 4, 9 / 5 * qp_t2_t1 ** 5, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t3_t2, qp_t3_t2 ** 2, qp_t3_t2 ** 3, 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t3_t2 ** 2, 4 / 3 * qp_t3_t2 ** 3, 3 / 2 * qp_t3_t2 ** 4, 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t3_t2 ** 3, 3 / 2 * qp_t3_t2 ** 4, 9 / 5 * qp_t3_t2 ** 5, 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t4_t3, qp_t4_t3 ** 2, qp_t4_t3 ** 3, 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t4_t3 ** 2, 4 / 3 * qp_t4_t3 ** 3, 3 / 2 * qp_t4_t3 ** 4, 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t4_t3 ** 3, 3 / 2 * qp_t4_t3 ** 4, 9 / 5 * qp_t4_t3 ** 5, 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t5_t4, qp_t5_t4 ** 2, qp_t5_t4 ** 3],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t5_t4 ** 2, 4 / 3 * qp_t5_t4 ** 3, 3 / 2 * qp_t5_t4 ** 4],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., qp_t5_t4 ** 3, 3 / 2 * qp_t5_t4 ** 4, 9 / 5 * qp_t5_t4 ** 5]])
            vc1_0_0 = np.matrix([0.])
            vc1_0_1 = np.matrix([qp_vb[16] * (qp_t1 - qp_t0)])
            vc1_0_2 = np.matrix([qp_vb[16] * ((t1 - qp_t0) ** 2 + (t2 - t1) ** 2)])
            vc1_0_3 = np.matrix([qp_vb[16] * ((t1 - qp_t0) ** 3 + (t2 - t1) ** 3)])
            vc1_1_0 = np.matrix([0.])
            vc1_1_1 = np.matrix([qp_vb[16] * (qp_t2 - qp_t1)])
            vc1_1_2 = np.matrix([qp_vb[16] * ((t3 - qp_t1) ** 2 + (t4 - t3) ** 2)])
            vc1_1_3 = np.matrix([qp_vb[16] * ((t3 - qp_t1) ** 3 + (t4 - t3) ** 3)])
            vc1_2_0 = np.matrix([0.])
            vc1_2_1 = np.matrix([qp_vb[16] * (qp_t3 - qp_t2)])
            vc1_2_2 = np.matrix([qp_vb[16] * ((t5 - qp_t2) ** 2 + (t6 - t5) ** 2)])
            vc1_2_3 = np.matrix([qp_vb[16] * ((t5 - qp_t2) ** 3 + (t6 - t5) ** 3)])
            vc1_3_0 = np.matrix([0.])
            vc1_3_1 = np.matrix([qp_vb[16] * (qp_t4 - qp_t3)])
            vc1_3_2 = np.matrix([qp_vb[16] * ((t7 - qp_t3) ** 2 + (t8 - t7) ** 2)])
            vc1_3_3 = np.matrix([qp_vb[16] * ((t7 - qp_t3) ** 3 + (t8 - t7) ** 3)])
            vc1_4_0 = np.matrix([0.])
            vc1_4_1 = np.matrix([qp_vb[16] * (qp_t5 - qp_t4)])
            vc1_4_2 = np.matrix([qp_vb[16] * ((t9 - qp_t4) ** 2 + (t10 - t9) ** 2)])
            vc1_4_3 = np.matrix([qp_vb[16] * ((t9 - qp_t4) ** 3 + (t10 - t9) ** 3)])
            qp_vc1 = np.vstack((vc1_0_0, vc1_0_1, vc1_0_2, vc1_0_3, vc1_1_0, vc1_1_1, vc1_1_2, vc1_1_3, vc1_2_0, vc1_2_1, vc1_2_2, vc1_2_3, vc1_3_0, vc1_3_1, vc1_3_2, vc1_3_3,
                                vc1_4_0, vc1_4_1, vc1_4_2, vc1_4_3))
            # QP Speed ➤ Objective Functional ➤ Objective Functional 2
            vG_2 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 4 * qp_t1_t0, 6 * qp_t1_t0 ** 2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 6 * qp_t1_t0 ** 2, 12 * qp_t1_t0 ** 3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 4 * qp_t2_t1, 6 * qp_t2_t1 ** 2, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 6 * qp_t2_t1 ** 2, 12 * qp_t2_t1 ** 3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4 * qp_t3_t2, 6 * qp_t3_t2 ** 2, 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6 * qp_t3_t2 ** 2, 12 * qp_t3_t2 ** 3, 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4 * qp_t4_t3, 6 * qp_t4_t3 ** 2, 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6 * qp_t4_t3 ** 2, 12 * qp_t4_t3 ** 3, 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4 * qp_t5_t4, 6 * qp_t5_t4 ** 2],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6 * qp_t5_t4 ** 2, 12 * qp_t5_t4 ** 3]])
            # QP Speed ➤ Objective Functional ➤ Objective Functional 3
            vG_3 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 36 * qp_t1_t0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 36 * qp_t2_t1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 36 * qp_t3_t2, 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 36 * qp_t4_t3, 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 36 * qp_t5_t4]])
            # QP Speed ➤ Objective Functional ➤ Objective Functional 0 + 1 + 2 + 3
            qp_vG = qp_speed_w0 * vG_0 + qp_speed_w1 * vG_1 + qp_speed_w2 * vG_2 + qp_speed_w3 * vG_3
            qp_vc = -(qp_speed_w0 * qp_vc0 + qp_speed_w1 * qp_vc1)

            # CVXOPT solver
            """
            min 1/2 x^T * P * x + q^T * x
             x 
            subject to Gx <= h, Ax = b
            """
            vP_ = matrix(qp_vG)
            vq_ = matrix(qp_vc)
            vG_ = matrix(-qp_vIA)
            vh_ = matrix(-qp_vIb)
            idx = list(range(0, 15))
            # idx.append(15)  # s
            # idx.append(16)  # s'
            idx.append(17)  # s"
            vA_ = matrix(qp_vA[idx, :])
            vb_ = matrix(qp_vb[idx, :])
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    try:
                        vsol = solvers.qp(vP_, vq_, vG_, vh_, vA_, vb_)
                    except:
                        vsol = {'x': None, 'status': 'fail'}
            qp_vres_P = np.array(vsol['x'])
            if vsol['status'] != 'optimal':
                qp_speed_succeeded = False
            else:
                qp_speed_succeeded = True
            # print(f'QP speed succeeded for plan {i_plan}: {qp_speed_succeeded}')

            if qp_speed_succeeded:
                qp_vres_P = np.squeeze(qp_vres_P)
                p_0 = qp_vres_P[0:4]
                p_1 = qp_vres_P[4:8]
                p_2 = qp_vres_P[8:12]
                p_3 = qp_vres_P[12:16]
                p_4 = qp_vres_P[16:]
                s_t0 = polynomial(self.t[0:num_points_each_piece], p_0)
                s_t1 = polynomial(self.t[0:num_points_each_piece], p_1)
                s_t2 = polynomial(self.t[0:num_points_each_piece], p_2)
                s_t3 = polynomial(self.t[0:num_points_each_piece], p_3)
                s_t4 = polynomial(self.t[0:num_points_each_piece + 1], p_4)
                qp_vres_s = np.hstack((s_t0, s_t1, s_t2, s_t3, s_t4))
                ds_dt0 = polynomial_derivative(self.t[0:num_points_each_piece], p_0, order=1)
                ds_dt1 = polynomial_derivative(self.t[0:num_points_each_piece], p_1, order=1)
                ds_dt2 = polynomial_derivative(self.t[0:num_points_each_piece], p_2, order=1)
                ds_dt3 = polynomial_derivative(self.t[0:num_points_each_piece], p_3, order=1)
                ds_dt4 = polynomial_derivative(self.t[0:num_points_each_piece + 1], p_4, order=1)
                qp_vres_ds_dt = np.hstack((ds_dt0, ds_dt1, ds_dt2, ds_dt3, ds_dt4))
                qp_vres_s_t_vs = np.vstack((self.t, qp_vres_s, qp_vres_ds_dt))
                cost0 = 1/2 * qp_speed_w0 * qp_vres_P.T.dot(vG_0.dot(qp_vres_P)) + qp_vc.T.dot(qp_vres_P)
                cost1 = 1/2 * qp_speed_w1 * qp_vres_P.T.dot(vG_1.dot(qp_vres_P))
                cost2 = 1/2 * qp_speed_w2 * qp_vres_P.T.dot(vG_2.dot(qp_vres_P))
                cost3 = 1/2 * qp_speed_w3 * qp_vres_P.T.dot(vG_3.dot(qp_vres_P))
                qp_speed_cost = float(cost0 + cost1 + cost2 + cost3)

            else:
                qp_vres_s_t_vs = np.vstack((self.t, ego_s_t[i_plan], ego_vs_t[i_plan]))
                qp_speed_cost = None

            qp_speeds.append(qp_vres_s_t_vs.transpose(1, 0))
            qp_speeds_succeeded.append(qp_speed_succeeded)
            qp_speeds_costs.append(qp_speed_cost)

        return qp_speeds, qp_speeds_succeeded, qp_speeds_costs

    def idm(self,
            current_ego_state,
            current_observations,
            current_agents_poses_frenet,
            ego_dim,
            obs_dim,
            frenet_frame,
            reference_line_lanes,
            traffic_light_data,
            qp_path_frenet):
        current_ego_pose_frenet = current_agents_poses_frenet[0]
        current_ego_pose_cartesian = np.array(current_ego_state.center.serialize())
        v_max = min([lane.speed_limit_mps if lane.speed_limit_mps is not None else 15. for lane in reference_line_lanes[:3]])
        if self._previous_v_max:
            v_max = self._v_alpha * self._previous_v_max + (1 - self._v_alpha) * v_max
        self._previous_v_max = v_max
        # consider traffic rule
        red_lane_connector_ids = [str(tl_data.lane_connector_id) for tl_data in traffic_light_data if tl_data.status.name == 'RED']
        red_reference_lanes = [pl for pl in reference_line_lanes if pl.id in red_lane_connector_ids]
        distance_to_red = None
        if len(red_reference_lanes) > 0:
            red_pl_points = [pl.baseline_path.discrete_path[0].point for pl in red_reference_lanes]
            red_pl_stations = frenet_frame.get_nearest_station_from_position(red_pl_points)
            front_mask = red_pl_stations > current_ego_pose_frenet[0]
            if np.any(front_mask):
                min_red_pl_station = min(red_pl_stations[front_mask])
                min_red_pl_station -= ego_dim['length'] / 2
                distance_to_red = min_red_pl_station - current_ego_pose_frenet[0]

        # idm
        v = current_ego_state.dynamic_car_state.rear_axle_velocity_2d.x
        agents_v = np.array([np.sqrt(agent.velocity.x ** 2 + agent.velocity.y ** 2) for agent in current_observations])
        same_lane_mask = np.abs(current_agents_poses_frenet[:1, 1] - current_agents_poses_frenet[1:, 1]) <= ego_dim['width'] / 2 + obs_dim['width'] / 2
        front_mask = current_agents_poses_frenet[:1, 0] < current_agents_poses_frenet[1:, 0]
        same_lane_front_mask = same_lane_mask & front_mask
        if any(same_lane_front_mask):
            distances = (current_agents_poses_frenet[1:, 0][same_lane_front_mask] - current_agents_poses_frenet[:1, 0])
            argmin_dis = np.argmin(distances)
            distance_to_leading_vehicle = min(999., distances[argmin_dis]
                                              - ego_dim['length'] / 2
                                              - obs_dim['length'][same_lane_front_mask][argmin_dis] / 2)
            v_leading = agents_v[same_lane_front_mask][argmin_dis]
        else:
            distance_to_leading_vehicle = 999.
            v_leading = 0.
        delta_v = v - v_leading
        margin_desired = self.safety_margin + max(
            0.,
            max(0., v) * self.time_headway + (v * delta_v) / (2 * np.sqrt(self.acc_limit * self.dec_limit))
        )
        if distance_to_red is not None:
            if distance_to_red < distance_to_leading_vehicle:
                delta_v = v
                margin_desired = (v * delta_v) / (2 * np.sqrt(self.acc_limit * self.dec_limit))
                distance_to_leading_vehicle = max(0., distance_to_red)
        distance_to_leading_vehicle = max(distance_to_leading_vehicle, 1e-8)
        acceleration = self.acc_limit * (1 - (v / v_max) ** self.acc_exponent - (margin_desired / distance_to_leading_vehicle) ** 2)
        acceleration = max(-self.dec_limit, acceleration)
        acceleration = min(self.acc_limit, acceleration)

        # Generate plan assuming following centerline and keep the acceleration computed by idm for 1 s.
        tracking_horizon = 10
        velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=v,
            derivatives=np.ones(tracking_horizon, dtype=np.float64) * acceleration,
            discretization_time=self.dt,
        )[1:]
        velocity_profile = np.maximum(0., velocity_profile)
        slow = velocity_profile[0] < 2.
        station_profile = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=current_ego_pose_frenet[0],
            derivatives=velocity_profile,
            discretization_time=self.dt,
        )[1:]
        lateral_profile = np.zeros_like(station_profile)
        heading_profile = np.zeros_like(station_profile)
        poses_frenet = np.stack([station_profile, lateral_profile, heading_profile])
        velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=velocity_profile[-1],
            derivatives=np.zeros(self.num_future_steps - tracking_horizon, dtype=np.float64),
            discretization_time=self.dt,
        )[1:]
        station_profile = _generate_profile_from_initial_condition_and_derivatives(
            initial_condition=station_profile[-1],
            derivatives=velocity_profile,
            discretization_time=self.dt,
        )[1:]
        lateral_profile = np.zeros_like(station_profile)
        heading_profile = np.zeros_like(station_profile)
        poses_frenet = np.concatenate([
            poses_frenet,
            np.stack([station_profile, lateral_profile, heading_profile])
        ], axis=1)

        if qp_path_frenet is not None:
            f_s_l = interp1d(qp_path_frenet[:, 0], qp_path_frenet[:, 1], fill_value='extrapolate')
            f_s_heading = interp1d(qp_path_frenet[:, 0], qp_path_frenet[:, 2], fill_value='extrapolate')
            lateral_profile = f_s_l(poses_frenet[0])
            heading_profile = f_s_heading(poses_frenet[0])
            poses_frenet[1] = lateral_profile
            poses_frenet[2] = heading_profile
        else:
            # connect current ego point to curve
            if not slow:
                connection_horizon = 10
                initial_lateral = current_ego_pose_frenet[1]
                initial_lateral_v = v * np.sin(current_ego_pose_frenet[2])
                a = current_ego_state.dynamic_car_state.rear_axle_acceleration_2d.x
                initial_lateral_a = a * np.sin(current_ego_pose_frenet[2])
                target_lateral = 0.
                target_lateral_v = 0.
                target_lateral_a = 0.
                # 5th-order Polynomial l(t)
                A_l = np.matrix([[1., 0., 0., 0., 0., 0.],
                                 [0., 1., 0., 0., 0., 0.],
                                 [0., 0., 2., 0., 0., 0.],
                                 [1., self.t[connection_horizon + 1], self.t[connection_horizon + 1] ** 2, self.t[connection_horizon + 1] ** 3, self.t[connection_horizon + 1] ** 4, self.t[connection_horizon + 1] ** 5],
                                 [0., 1., 2 * self.t[connection_horizon + 1], 3 * self.t[connection_horizon + 1] ** 2, 4 * self.t[connection_horizon + 1] ** 3, 5 * self.t[connection_horizon + 1] ** 4],
                                 [0., 0., 2., 6 * self.t[connection_horizon + 1], 12 * self.t[connection_horizon + 1] ** 2, 20 * self.t[connection_horizon + 1] ** 3]])
                Q_l = np.array([initial_lateral,
                                initial_lateral_v,
                                initial_lateral_a,
                                target_lateral,
                                target_lateral_v,
                                target_lateral_a
                                ]).reshape((-1, 1))
                P_l = A_l.I * Q_l
                l_t = polynomial(self.t[:connection_horizon + 1], P_l.A)
                poses_frenet[1, :connection_horizon] = l_t[:, 1:]
                poses_frenet[2, :-1] = np.arctan(
                    (poses_frenet[1, 1:] - poses_frenet[1, :-1]) / (poses_frenet[0, 1:] - poses_frenet[0, :-1] + 1e-8))

        # frenet to cartesian
        poses_cartesian = frenet_frame.frenet_to_cartesian(np.expand_dims(poses_frenet, axis=0))['pose_cartesian'][0].T

        # global to local frame
        relative_poses_cartesian = efficient_absolute_to_relative_poses(current_ego_pose_cartesian.reshape((1, -1)),
                                                                        poses_cartesian[np.newaxis, np.newaxis])
        relative_poses_cartesian = np.squeeze(relative_poses_cartesian, axis=1)

        return poses_cartesian, relative_poses_cartesian

    def visualize(self,
                  current_input,
                  scenario,
                  features,
                  frenet_frame,
                  overlap_region,
                  trajs,
                  ego_dim,
                  obs_dim,
                  show=True,
                  save=False):
        # define
        TunnelColor = [1., 0.64, 0.16]
        TunnelLineWidth = 1.
        OverlapColor = 'tab:red'
        OverlapLineWidth = 1
        TrajMarkerColor = 'tab:cyan'
        CandidatePathColor = [0.78, 0.82, 0.59]
        CandidatePathWidth = 0.25
        knotColor = [0.78, 0.78, 0.17]
        knotWidth = 0.025
        PredTrajColor = [0.43, 0.69, 0.85]  # [0.86, 0.35, 0.38]
        PredTrajWidth = 1.
        RefinedTrajColor = 'tab:purple'
        QPPathColor = 'tab:green'
        QPTrajColor = [0.19, 0.50, 0.85]
        QPTrajWidth = 1.5

        features = {
            key: value.to_device(torch.device('cpu'))
            for key, value in features.items()
        }
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
                points = map_data['map_point']['position'][sample_idx][mask]
                points_sides=  map_data['map_point']['side'][sample_idx][mask]
                coords_diff = points[1:, :] - points[0:-1, :]
                coords_diff = torch.cat([coords_diff[0:1, :], coords_diff], dim=0)
                coords_diff = torch.abs(coords_diff)
                jump_mask = torch.logical_or(coords_diff[:, 0] > coords_diff[:, 0].median() * 5,
                                             coords_diff[:, 1] > coords_diff[:, 1].median() * 5)
                jump_idx = torch.where(jump_mask)[0]
                knots_idx = torch.cat([torch.zeros(1,), jump_idx, torch.ones(1,) * jump_mask.shape[0]], dim=0)
                knots_idx = knots_idx.to(dtype=torch.long).numpy().tolist()
                split_sizes = [knots_idx[i] - knots_idx[i-1] for i in range(1, len(knots_idx))]
                split_points = torch.split(points, split_sizes)
                split_points_sides = torch.split(points_sides, split_sizes)
                for pts, sides in zip(split_points, split_points_sides):
                    if pts.shape[0] > 0:
                        if sides[0] == 2:
                            plt.plot(pts[:, 0], pts[:, 1], alpha=0.8)
                        else:
                            plt.plot(pts[:, 0], pts[:, 1], color='tab:gray', alpha=0.5)
                plt.scatter(points[:, 0], points[:, 1], s=1)

            # plot agents
            num_agents = agent_data['position'][sample_idx].shape[0]
            anchorx = None
            anchory = None
            for i_a in range(num_agents):
                if i_a == 0:
                    length = ego_dim['length']
                    width = ego_dim['width']
                else:
                    length = obs_dim['length'][i_a - 1]
                    width = obs_dim['width'][i_a - 1]
                _, _, _, _, xC, yC, _, _ = Env.get_vertices_from_center(agent_data['position'][sample_idx][i_a, -1, 0],
                                                                         agent_data['position'][sample_idx][i_a, -1, 1],
                                                                         agent_data['heading'][sample_idx][i_a, -1],
                                                                         length,
                                                                         width,
                                                                         rad=True)
                heading = agent_data['heading'][sample_idx][i_a, -1]
                xy = (xC, yC)
                if i_a == 0:
                    color = [0.15, 0.53, 0.79]
                else:
                    color = [0.52, 0.52, 0.52]
                rect = patches.Rectangle(xy, length, width, angle=np.rad2deg(heading.numpy()), color=color)
                ax.add_patch(rect)
                plt.text(xC, yC, s=f'{i_a-1}', color='tab:blue', alpha=0.8)
                if anchorx is None:
                    anchorx = xC
                    anchory = yC

            # plot other agents' trajectories
            obs_trajs = trajs['obs_trajs']
            for i, traj in enumerate(obs_trajs):
                for tj in traj:
                    plt.plot(tj[:, 0],
                             tj[:, 1],
                             linewidth=0.5,
                             marker='>',
                             alpha=0.5,
                             markersize=1)
                    # plt.scatter(tj[0::8, 0],
                    #             tj[0::8, 1],
                    #             color=TrajMarkerColor,
                    #             s=1)
                    plt.text(tj[0, 0], tj[0, 1], s=f'{i}', color='tab:red', alpha=0.8)

            # reference line
            xy_ref = np.array(frenet_frame.reference_line.xy)

            for i_plan in range(1):
                # plot feasible tunnel
                lateral_lb_positions_frenet = np.vstack((self.s[i_plan], self.lateral_lb[i_plan]))
                diff = np.diff(lateral_lb_positions_frenet, axis=1)
                lateral_lb_heading = np.arctan2(diff[1], diff[0])
                lateral_lb_heading = np.hstack((lateral_lb_heading, lateral_lb_heading[-1]))
                lateral_lb_poses_frenet = np.vstack((lateral_lb_positions_frenet, lateral_lb_heading))
                lateral_lb_poses_frenet = np.expand_dims(lateral_lb_poses_frenet, axis=0)
                lb = frenet_frame.frenet_to_cartesian(lateral_lb_poses_frenet)['pose_cartesian'][0]
                lateral_rb_positions_frenet = np.vstack((self.s[i_plan], self.lateral_rb[i_plan]))
                diff = np.diff(lateral_rb_positions_frenet, axis=1)
                lateral_rb_heading = np.arctan2(diff[1], diff[0])
                lateral_rb_heading = np.hstack((lateral_rb_heading, lateral_rb_heading[-1]))
                lateral_rb_poses_frenet = np.vstack((lateral_rb_positions_frenet, lateral_rb_heading))
                lateral_rb_poses_frenet = np.expand_dims(lateral_rb_poses_frenet, axis=0)
                rb = frenet_frame.frenet_to_cartesian(lateral_rb_poses_frenet)['pose_cartesian'][0]
                bd = np.hstack((lb, rb[:, ::-1]))
                ax.plot(lb[0], lb[1], color=TunnelColor, linewidth=TunnelLineWidth)
                ax.plot(rb[0], rb[1], color=TunnelColor, linewidth=TunnelLineWidth)
                ax.fill(bd[0], bd[1], facecolor=TunnelColor, edgecolor='none', alpha=0.5)

                # plot overlap regions
                # rectangle_vertices_s = np.vstack((
                #     overlap_region['smin'][i_plan],
                #     overlap_region['smin'][i_plan],
                #     overlap_region['smax'][i_plan],
                #     overlap_region['smax'][i_plan],
                # ))
                # rectangle_vertices_l = np.vstack((
                #     overlap_region['lmin'][i_plan],
                #     overlap_region['lmax'][i_plan],
                #     overlap_region['lmax'][i_plan],
                #     overlap_region['lmin'][i_plan]
                # ))
                # if rectangle_vertices_s.shape[1] > 0 and rectangle_vertices_l.shape[1] > 0 and np.any(overlap_region['overlaps'][i_plan]):
                #     rectangle_vertices_heading = np.zeros_like(rectangle_vertices_s)
                #     rectangle_vertices = np.stack((rectangle_vertices_s, rectangle_vertices_l, rectangle_vertices_heading), axis=0)
                #     rectangle_vertices = rectangle_vertices[:, :, overlap_region['overlaps'][i_plan]]
                #     rectangle_vertices = rectangle_vertices.transpose((2, 0, 1))
                #     rectangle_vertices = frenet_frame.frenet_to_cartesian(rectangle_vertices)['pose_cartesian']
                #     for i_agent in range(rectangle_vertices.shape[0]):
                #         ax.fill(rectangle_vertices[i_agent, 0, :],
                #                 rectangle_vertices[i_agent, 1, :],
                #                 facecolor=OverlapColor,
                #                 edgecolor=OverlapColor,
                #                 linewidth=OverlapLineWidth,
                #                 alpha=0.3)

                rectangle_vertices_s = np.stack((
                    overlap_region['smin'][i_plan],
                    overlap_region['smin'][i_plan],
                    overlap_region['smax'][i_plan],
                    overlap_region['smax'][i_plan],
                ))
                rectangle_vertices_l = np.stack((
                    overlap_region['lmin'][i_plan],
                    overlap_region['lmax'][i_plan],
                    overlap_region['lmax'][i_plan],
                    overlap_region['lmin'][i_plan]
                ))
                if rectangle_vertices_s.shape[1] > 0 and rectangle_vertices_l.shape[1] > 0 and np.any(overlap_region['overlaps'][i_plan]):
                    rectangle_vertices_heading = np.zeros_like(rectangle_vertices_s)
                    rectangle_vertices = np.stack((rectangle_vertices_s, rectangle_vertices_l, rectangle_vertices_heading), axis=0)
                    rectangle_vertices = rectangle_vertices[:, :, overlap_region['overlaps'][i_plan]]
                    rectangle_vertices = rectangle_vertices.transpose((1, 0, 2))
                    rectangle_vertices = frenet_frame.frenet_to_cartesian(rectangle_vertices)['pose_cartesian']
                    for i in range(rectangle_vertices.shape[2]):
                        ax.fill(rectangle_vertices[:, 0, i],
                                rectangle_vertices[:, 1, i],
                                facecolor=OverlapColor,
                                edgecolor=OverlapColor,
                                linewidth=OverlapLineWidth,
                                alpha=0.3)

                # plot planned trajectory
                ego_traj_pred = trajs['ego_traj_pred']
                plt.plot(ego_traj_pred[i_plan, :, 0],
                         ego_traj_pred[i_plan, :, 1],
                         color=PredTrajColor,
                         marker='>',
                         markersize=1.5,
                         alpha=0.8,
                         label='predicted')
                # plt.scatter(ego_traj_pred[i_plan, 0::8, 0],
                #             ego_traj_pred[i_plan, 0::8, 1],
                #             color=TrajMarkerColor,
                #             s=1.5)
                if trajs['refined_pred'] is not None:
                    plt.plot(trajs['refined_pred'][:, 0],
                             trajs['refined_pred'][:, 1],
                             color=RefinedTrajColor,
                             marker='>',
                             markersize=1.5,
                             alpha=0.8,
                             label='refined')
                    # plt.scatter(ego_traj_pred[i_plan, 0::8, 0],
                    #             ego_traj_pred[i_plan, 0::8, 1],
                    #             color=TrajMarkerColor,
                    #             alpha=0.8,
                    #             s=1.5)
                if trajs['qp_path'] is not None:
                    plt.plot(trajs['qp_path'][:, 0],
                             trajs['qp_path'][:, 1],
                             color=QPPathColor,
                             marker='>',
                             markersize=1.5,
                             alpha=0.8,
                             label='qp_path')
                    # plt.scatter(trajs['qp_path'][0::8, 0],
                    #             trajs['qp_path'][0::8, 1],
                    #             color=QPPathColor,
                    #             s=1.5)
                try:
                    ego_traj_planned = trajs['ego_traj_optimized']
                    plt.plot(ego_traj_planned[:, 0],
                             ego_traj_planned[:, 1],
                             color=QPTrajColor,
                             linewidth=QPTrajWidth,
                             marker='>',
                             alpha=0.8,
                             markersize=1.5,
                             label='optimized')
                    # plt.scatter(ego_traj_planned[0::8, 0],
                    #             ego_traj_planned[0::8, 1],
                    #             color=TrajMarkerColor,
                    #             s=1.5)
                except:
                    pass
                plt.legend()

            window = 50
            plt.xlim(anchorx - window, anchorx + window)
            plt.ylim(anchory - window, anchory + window)
            # plt.axis('equal')
            if save:
                path = './debug_simulation_figs'
                if not os.path.exists(path):
                    os.makedirs(path)
                # plt.savefig(f'{path}/iteration: {current_input.iteration.index}.pdf')
                plt.savefig(f'{path}/log_name: {scenario.log_name} - scenario_token: {scenario.token} - iteration: {current_input.iteration.index}.pdf')
                # plt.savefig(f'{path}/log_name: {scenario.log_name} - scenario_token: {scenario.token} - iteration: {current_input.iteration.index}.png', dpi=600)
            if show:
                plt.show()
            plt.close()

    @staticmethod
    def get_overlap_sl(sA1, sB1, sC1, sD1,
                       sA2, sB2, sC2, sD2,
                       lA2, lB2, lC2, lD2, ):
        num_obs, num_poses = sA2.shape[0], sA2.shape[1]
        overlap_s = np.zeros((num_obs, 4, num_poses))
        overlap_l = np.ones((num_obs, 4, num_poses)) * 999.

        s1 = np.array([sA1, sB1, sC1, sD1])
        s2 = np.array([sA2, sB2, sC2, sD2])
        l2 = np.array([lA2, lB2, lC2, lD2])
        s1min = np.amin(s1, axis=0)
        s1max = np.amax(s1, axis=0)
        s2min = np.amin(s2, axis=0)
        s2max = np.amax(s2, axis=0)
        s1min = np.broadcast_to(s1min, shape=(s2min.shape[0], s2min.shape[1]))
        s1max = np.broadcast_to(s1max, shape=(s2max.shape[0], s2max.shape[1]))
        overlap_mask = np.logical_or(
            np.logical_and(s2min <= s1min, s1min <= s2max),
            np.logical_and(s2min <= s1max, s1max <= s2max),
            np.logical_and(s2max <= s1max, s1min <= s2min),
        )
        overlap_s[:, 0, :][overlap_mask] = s2[0][overlap_mask]
        overlap_s[:, 1, :][overlap_mask] = s2[1][overlap_mask]
        overlap_s[:, 2, :][overlap_mask] = s2[2][overlap_mask]
        overlap_s[:, 3, :][overlap_mask] = s2[3][overlap_mask]
        overlap_l[:, 0, :][overlap_mask] = l2[0][overlap_mask]
        overlap_l[:, 1, :][overlap_mask] = l2[1][overlap_mask]
        overlap_l[:, 2, :][overlap_mask] = l2[2][overlap_mask]
        overlap_l[:, 3, :][overlap_mask] = l2[3][overlap_mask]

        return overlap_s, overlap_l

    @staticmethod
    def trapz(delta_x, y):
        return 1/2 * delta_x * (y[0] + y[-1]) + delta_x * sum(y[1:-1])

    @staticmethod
    def center_to_rear_axle(poses, rear_axle_to_center_dist):
        """
        Transform reference point from center to rear axle.
        :param poses: shape (num_plan, num_poses, 3)
        :param rear_axle_to_center_dist: distance between rear axles and center
        :return: poses at rear axle
        """
        new_poses = copy.deepcopy(poses)
        new_poses[:, :, 0] = new_poses[:, :, 0] - rear_axle_to_center_dist * np.cos(new_poses[:, :, 2])
        new_poses[:, :, 1] = new_poses[:, :, 1] - rear_axle_to_center_dist * np.sin(new_poses[:, :, 2])
        return new_poses

    @staticmethod
    def rear_axle_to_center(poses, rear_axle_to_center_dist):
        """
        Transform reference point from center to rear axle.
        :param poses: shape (num_plan, num_poses, 3)
        :param rear_axle_to_center_dist: distance between rear axles and center
        :return: poses at rear axle
        """
        new_poses = copy.deepcopy(poses)
        if len(new_poses.shape) == 3:
            new_poses[:, :, 0] = new_poses[:, :, 0] + rear_axle_to_center_dist * np.cos(new_poses[:, :, 2])
            new_poses[:, :, 1] = new_poses[:, :, 1] + rear_axle_to_center_dist * np.sin(new_poses[:, :, 2])
        else:
            new_poses[:, 0] = new_poses[:, 0] + rear_axle_to_center_dist * np.cos(new_poses[:, 2])
            new_poses[:, 1] = new_poses[:, 1] + rear_axle_to_center_dist * np.sin(new_poses[:, 2])
        return new_poses

    def _get_candidate_trajs(self, current_input, ego_poses_frenet, reference_line_lanes, num_l_tgt_per_lane=11):
        """
        Generate trajectories for left changing, lane keeping, and right changing based on piece-wise polynomials
        :param current_input: current input
        :param ego_poses_frenet: ego poses in Frenet frame predicted by actor
        :param reference_line_lanes: reference line lanes
        :param num_l_tgt_per_lane: number of lateral targets per lane
        :return: candidate trajectories
        """
        out = []
        for i_plan in range(ego_poses_frenet.shape[0]):
            dp_trajs = {}
            current_ego_lateral = ego_poses_frenet[i_plan, 0, 1]
            current_ego_heading = ego_poses_frenet[i_plan, 0, 2]
            ego_state = current_input.history.ego_states[-1]
            ego_velocity_local = ego_state.dynamic_car_state.rear_axle_velocity_2d
            ego_acceleration_local = ego_state.dynamic_car_state.rear_axle_acceleration_2d
            min_dp_planning_length = 80.
            s = np.linspace(
                ego_poses_frenet[i_plan, 0, 0],
                max(ego_poses_frenet[i_plan, -1, 0], ego_poses_frenet[i_plan, 0, 0] + min_dp_planning_length),
                num=self.num_future_steps + 1
            )
            lane_width = np.linalg.norm(reference_line_lanes[0].left_boundary.discrete_path[0].array
                                        - reference_line_lanes[0].right_boundary.discrete_path[0].array)
            delta_l = lane_width / (num_l_tgt_per_lane - 1)
            l_target = np.arange(1, num_l_tgt_per_lane * 3 // 2 + 1) * delta_l
            l_target = np.hstack((-l_target[::-1], np.zeros((1,)), l_target))
            l_right_target, l_keep_target, l_left_target = np.array_split(l_target, 3)

            # generate paths
            s_keep_seg0, s_keep_seg1, s_keep_seg2, P_keep_0, P_keep_1, P_keep_2 = self._get_lane_keeping_paths(
                s,
                current_ego_lateral,
                current_ego_heading,
                l_keep_target,
                ego_velocity_local,
                ego_acceleration_local
            )
            s_left_seg0, s_left_seg1, P_left_0, P_left_1 = self._get_lane_change_paths(
                s,
                current_ego_lateral,
                current_ego_heading,
                l_left_target,
                ego_velocity_local,
                ego_acceleration_local
            )
            s_right_seg0, s_right_seg1, P_right_0, P_right_1 = self._get_lane_change_paths(
                s,
                current_ego_lateral,
                current_ego_heading,
                l_right_target,
                ego_velocity_local,
                ego_acceleration_local
            )

            # paths -> trajs
            # paths -> trajs: keep
            ego_stations = ego_poses_frenet[i_plan, :, 0]
            seg0_mask = ego_stations < s_keep_seg0[-1]
            seg1_mask = (s_keep_seg1[0] <= ego_stations) & (ego_stations < s_keep_seg1[-1])
            seg2_mask = (s_keep_seg2[0] <= ego_stations)
            s_keep = {
                'seg0': ego_stations[seg0_mask],
                'seg1': None,
                'seg2': None,
            }
            l_s_keep = {
                'seg0': [polynomial(ego_stations[seg0_mask], P.A) for P in P_keep_0],
                'seg1': None,
                'seg2': None,
            }
            if np.any(seg1_mask):
                s_keep['seg1'] = ego_stations[seg1_mask]
                l_s_keep['seg1'] = [polynomial(ego_stations[seg1_mask], P.A) for P in P_keep_1]
            if np.any(seg2_mask):
                s_keep['seg2'] = ego_stations[seg2_mask]
                l_s_keep['seg2'] = [polynomial(ego_stations[seg2_mask], P.A) for P in P_keep_2]
            # paths -> trajs: left change
            seg0_mask = ego_stations < s_left_seg0[-1]
            seg1_mask = s_left_seg1[0] <= ego_stations
            s_left = {
                'seg0': ego_stations[seg0_mask],
                'seg1': None,
            }
            l_s_left = {
                'seg0': [polynomial(ego_stations[seg0_mask], P.A) for P in P_left_0],
                'seg1': None,
            }
            if np.any(seg1_mask):
                s_left['seg1'] = ego_stations[seg1_mask]
                l_s_left['seg1'] = [polynomial(ego_stations[seg1_mask], P.A) for P in P_left_1]
            # paths -> trajs: right change
            seg0_mask = ego_stations < s_right_seg0[-1]
            seg1_mask = s_right_seg1[0] <= ego_stations
            s_right = {
                'seg0': ego_stations[seg0_mask],
                'seg1': None,
            }
            l_s_right = {
                'seg0': [polynomial(ego_stations[seg0_mask], P.A) for P in P_right_0],
                'seg1': None,
            }
            if np.any(seg1_mask):
                s_right['seg1'] = ego_stations[seg1_mask]
                l_s_right['seg1'] = [polynomial(ego_stations[seg1_mask], P.A) for P in P_right_1]

            dp_trajs['s_keep'] = s_keep
            dp_trajs['s_left'] = s_left
            dp_trajs['s_right'] = s_right
            dp_trajs['l_s_keep'] = l_s_keep
            dp_trajs['l_s_left'] = l_s_left
            dp_trajs['l_s_right'] = l_s_right
            out.append(dp_trajs)

            return out

    def _get_lane_keeping_paths(
            self,
            s,
            current_ego_lateral,
            current_ego_heading,
            l_target,
            ego_velocity_local,
            ego_acceleration_local,
    ):
        cos_heading = np.cos(current_ego_heading)
        sin_heading = np.sin(current_ego_heading)
        num_pt_per_seg = s.shape[0] // 3
        s_seg0 = s[:num_pt_per_seg + 1]
        s_seg1 = s[num_pt_per_seg:num_pt_per_seg * 2 + 1]
        s_seg2 = s[num_pt_per_seg * 2:]
        # 5th-order Polynomial l(s): segment 0
        vs_frenet = ego_velocity_local.x * cos_heading - ego_velocity_local.y * sin_heading
        vl_frenet = ego_velocity_local.x * sin_heading + ego_velocity_local.y * cos_heading
        as_frenet = ego_acceleration_local.x * cos_heading - ego_acceleration_local.y * sin_heading
        al_frenet = ego_acceleration_local.x * sin_heading + ego_acceleration_local.y * cos_heading
        # dl_ds = vl_frenet / (vs_frenet + 1e-8)
        if vs_frenet < 0.1:
            ddl_dds = 0.
        else:
            ddl_dds = (al_frenet * vs_frenet - vl_frenet * as_frenet) / (vs_frenet ** 3 + 1e-8)
        P_l_0 = interpolate_polynomial(
            deg=5,
            x_0=s_seg0[:1],
            x_1=s_seg0[-1:],
            y_0=np.broadcast_to(current_ego_lateral, l_target.shape),
            # y_prime_0=np.broadcast_to(dl_ds, l_target.shape),
            y_prime_0=np.ones_like(l_target) * current_ego_heading,
            y_pprime_0=np.broadcast_to(ddl_dds, l_target.shape),
            y_1=l_target,
            y_prime_1=np.zeros_like(l_target),
            y_pprime_1=np.zeros_like(l_target)
        )
        # 5th-order Polynomial l(s): segment 1
        P_l_1 = interpolate_polynomial(
            deg=5,
            x_0=np.broadcast_to(s_seg1[:1], l_target.shape),
            x_1=np.broadcast_to(s_seg1[-1:], l_target.shape),
            y_0=np.array_split(l_target, l_target.shape[0]),
            y_prime_0=[np.array([0])] * l_target.shape[0],
            y_pprime_0=[np.array([0])] * l_target.shape[0],
            y_1=l_target,
            y_prime_1=np.zeros_like(l_target),
            y_pprime_1=np.zeros_like(l_target)
        )
        # 5th-order Polynomial l(s): segment 2
        P_l_2 = interpolate_polynomial(
            deg=5,
            x_0=np.broadcast_to(s_seg2[:1], l_target.shape),
            x_1=np.broadcast_to(s_seg2[-1:], l_target.shape),
            y_0=np.array_split(l_target, l_target.shape[0]),
            y_prime_0=[np.array([0])] * l_target.shape[0],
            y_pprime_0=[np.array([0])] * l_target.shape[0],
            y_1=l_target,
            y_prime_1=np.zeros_like(l_target),
            y_pprime_1=np.zeros_like(l_target)
        )

        return s_seg0, s_seg1, s_seg2, P_l_0, P_l_1, P_l_2

    def _get_lane_change_paths(
            self,
            s,
            current_ego_lateral,
            current_ego_heading,
            l_target,
            ego_velocity_local,
            ego_acceleration_local,
    ):
        cos_heading = np.cos(current_ego_heading)
        sin_heading = np.sin(current_ego_heading)
        num_pt_per_seg = s.shape[0] // 2
        s_seg0 = s[:num_pt_per_seg + 1]
        s_seg1 = s[num_pt_per_seg:]
        # 5th-order Polynomial l(s): segment 0
        vs_frenet = ego_velocity_local.x * cos_heading - ego_velocity_local.y * sin_heading
        vl_frenet = ego_velocity_local.x * sin_heading + ego_velocity_local.y * cos_heading
        as_frenet = ego_acceleration_local.x * cos_heading - ego_acceleration_local.y * sin_heading
        al_frenet = ego_acceleration_local.x * sin_heading + ego_acceleration_local.y * cos_heading
        # dl_ds = vl_frenet / (vs_frenet + 1e-8)
        ddl_dds = (al_frenet * vs_frenet - vl_frenet * as_frenet) / (vs_frenet ** 3 + 1e-8)
        P_l_0 = interpolate_polynomial(
            deg=5,
            x_0=s_seg0[:1],
            x_1=s_seg0[-1:],
            y_0=np.broadcast_to(current_ego_lateral, l_target.shape),
            # y_prime_0=np.broadcast_to(dl_ds, l_target.shape),
            y_prime_0=np.ones_like(l_target) * current_ego_heading,
            y_pprime_0=np.broadcast_to(ddl_dds, l_target.shape),
            y_1=l_target,
            y_prime_1=np.zeros_like(l_target),
            y_pprime_1=np.zeros_like(l_target)
        )
        # 5th-order Polynomial l(s): segment 1
        P_l_1 = interpolate_polynomial(
            deg=5,
            x_0=np.broadcast_to(s_seg1[:1], l_target.shape),
            x_1=np.broadcast_to(s_seg1[-1:], l_target.shape),
            y_0=np.array_split(l_target, l_target.shape[0]),
            y_prime_0=[np.array([0])] * l_target.shape[0],
            y_pprime_0=[np.array([0])] * l_target.shape[0],
            y_1=l_target,
            y_prime_1=np.zeros_like(l_target),
            y_pprime_1=np.zeros_like(l_target)
        )

        return s_seg0, s_seg1, P_l_0, P_l_1

    def _get_ego_baseline_path(self, reference_lines, ego_state: EgoState):
        init_ref_points = np.array([r[0] for r in reference_lines], dtype=np.float64)

        init_distance = np.linalg.norm(
            init_ref_points[:, :2] - ego_state.rear_axle.array, axis=-1
        )
        nearest_idx = np.argmin(init_distance)
        reference_line = reference_lines[nearest_idx]
        baseline_path = shapely.LineString(reference_line[:, :2])

        return baseline_path

    def _get_obs_info(self, obs_track_tokens, tracked_objects, obs_dim, agents_absolute_poses):
        # exclude uninterested objects
        interest_mask = np.array([
            True if tracked_objects[pred_object_id].tracked_object_type in self.interested_objects_types else False
            for pred_object_id in obs_track_tokens
        ])
        has_interested_objects = np.any(interest_mask)
        if has_interested_objects:
            # obs_tokens
            obs_tokens = [id for i, id in enumerate(obs_track_tokens) if interest_mask[i]]
            # obs_shape (width, length)
            obs_shape = np.stack((obs_dim['width'], obs_dim['length'])).transpose()[interest_mask]
            # obs_category
            obs_category = np.array([
                self.interested_objects_types.index(
                    tracked_objects[pred_object_id].tracked_object_type
                )
                for i, pred_object_id in enumerate(obs_track_tokens)
                if interest_mask[i]
            ])
            # obs_predictions (global)
            obs_predictions = agents_absolute_poses[1:, 0][interest_mask]
            # obs_velocity (global)
            obs_velocity = (
                    np.linalg.norm(np.diff(obs_predictions[..., :2], axis=-2), axis=-1)
                    / 0.1
            )
            obs_current_velocity = np.array([
                np.linalg.norm(tracked_objects[token].velocity.array)
                for token in obs_tokens
            ])
            obs_velocity = np.concatenate([obs_current_velocity[..., None], obs_velocity], axis=-1)
        else:
            obs_tokens = []
            obs_shape = np.array([])
            obs_category = np.array([])
            obs_predictions = np.empty((0, agents_absolute_poses.shape[1], agents_absolute_poses.shape[2], agents_absolute_poses.shape[3]))
            obs_velocity = np.array([])

        return {
            "tokens": obs_tokens,
            "shape": obs_shape,
            "category": obs_category,
            "velocity": obs_velocity,
            "predictions": obs_predictions,
        }