from __future__ import annotations

from typing import Dict, List, Tuple, Type, Union, Optional, Any

import torch
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from _datetime import datetime
import shapely.geometry as geom
from shapely.geometry.point import Point

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.common.maps.nuplan_map.roadblock import NuPlanRoadBlock
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatuses,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    MapObjectPolylines,
    VectorFeatureLayer,
    LaneSegmentCoords,
    LaneSegmentTrafficLightData,
    get_neighbor_vector_map,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType
from nuplan.common.maps.nuplan_map.utils import (
    extract_roadblock_objects,
    compute_curvature,
)

from nuplan_zigned.training.preprocessing.feature_builders.avrl_feature_builder_utils import (
    get_neighbor_vector_set_map,
)
from nuplan_zigned.training.preprocessing.feature_builders.qcmae_feature_builder_utils import get_neighbor_vector_set_map as qcmae_get_neighbor_vector_set_map
from nuplan_zigned.training.preprocessing.feature_builders.avrl_vector_set_map_builder_utils import (
    get_centerline_coords,
    visualize,
)
from nuplan_zigned.training.preprocessing.features.avrl_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.features.qcmae_vector_set_map import VectorSetMap as qcmaeVectorSetMap
from nuplan_zigned.training.preprocessing.utils.avrl_vector_preprocessing import convert_feature_layer_to_fixed_size
from nuplan_zigned.training.preprocessing.utils.qcmae_vector_preprocessing import convert_lane_layers_to_consistent_size
from nuplan_zigned.utils.frenet_frame_object import FrenetFrame
from nuplan_zigned.utils.trajectory_sampler import TrajectorySampler
from nuplan_zigned.utils.utils import (
    safe_list_index,
)


class VectorSetMapFeatureBuilder(ScriptableFeatureBuilder):
    """
    Feature builder for constructing map features in a vector set representation, similar to that of
        VectorNet ("VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation").
    """

    def __init__(
        self,
        map_features: List[str],
        max_elements: Dict[str, int],
        max_points: Dict[str, int],
        radius: float,
        interpolation_method: str,
        num_poses: int,
        time_horizon: float,
        frenet_radius: float,
    ) -> None:
        """
        Initialize vector set map builder with configuration parameters.
        :param map_features: name of map features to be extracted.
        :param max_elements: maximum number of elements to extract per feature layer.
        :param max_points: maximum number of points per feature to extract per feature layer.
        :param radius:  [m] The query radius scope relative to the current ego-pose.
        :param interpolation_method: Interpolation method to apply when interpolating to maintain fixed size
            map elements.
        :param num_poses: number of poses in future trajectory in addition to initial state.
        :param time_horizon: [s] time horizon of all poses.
        :param frenet_radius: [m] The minimum query radius scope relative to the current ego-pose. Used when building
            Frenet frame. Will be adjusted according to speed.
        :return: Vector set map data including map element coordinates and traffic light status info.
        """
        super().__init__()
        self.map_features = map_features
        self.max_elements = max_elements
        self.max_points = max_points
        self.radius = radius
        self.interpolation_method = interpolation_method
        self._traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()
        self.num_poses = num_poses
        self.time_horizon = time_horizon
        self.frenet_radius = frenet_radius

        self.v_max = 90 / 3.6
        self.a_min = -4.
        self.a_max = 4.
        self.lane_width = 3.66  # [m], 12 feet, defaut lane width
        self.delta_l = self.lane_width / 4
        self.deduplicate_radius = 2.

        self._polygon_types = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        self._polygon_is_intersections = [True, False, None]
        self._point_types = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER', 'UNKNOWN']
        self._point_traffic_light_statuses = ['GREEN', 'YELLOW', 'RED', 'UNKNOWN']
        self._traffic_light_one_hot_decoding = {
            (1, 0, 0, 0): 0,  # GREEN
            (0, 1, 0, 0): 1,  # YELLOW
            (0, 0, 1, 0): 2,  # RED
            (0, 0, 0, 1): 3,  # UNKNOWN
        }
        self._traffic_light_statuses = ['GREEN', 'YELLOW', 'RED', 'UNKNOWN']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

        # Sanitize feature building parameters
        for feature_name in self.map_features:
            try:
                VectorFeatureLayer[feature_name]
            except KeyError:
                raise ValueError(f"Object representation for layer: {feature_name} is unavailable!")
            if feature_name not in self.max_elements:
                raise RuntimeError(f"Max elements unavailable for {feature_name} feature layer!")
            if feature_name not in self.max_points:
                raise RuntimeError(f"Max points unavailable for {feature_name} feature layer!")

    @torch.jit.unused
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return VectorSetMap  # type: ignore

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "vector_set_map"

    @torch.jit.unused
    def get_scriptable_input_from_scenario(
        self, scenario: AbstractScenario
    ) -> Tuple[
        List[
            List[
                List[Dict[str, torch.Tensor] | Dict[str, List[torch.Tensor]] | Dict[str, List[List[torch.Tensor]]]]
            ]
        ],
        Dict[str, FeatureDataType],
    ]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        future_ego_states = list(
            scenario.get_ego_future_trajectory(iteration=0, time_horizon=self.time_horizon, num_samples=self.num_poses))
        future_ego_coords = [Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y) for ego_state in future_ego_states]
        future_tl = list(
            scenario.get_future_traffic_light_status_history(iteration=0, time_horizon=self.time_horizon, num_samples=self.num_poses))
        future_timestamps = list(scenario.get_future_timestamps(iteration=0, time_horizon=self.time_horizon, num_samples=self.num_poses))
        route_roadblock_ids = scenario.get_route_roadblock_ids()

        dict_query_feature: Dict[int, Dict[Point2D, List]] = {}  # used to record features of query ego positions
        future_features = []
        trajectory_features = []
        for ego_state, ego_coords, traffic_light_status, timestamp in zip(
                future_ego_states, future_ego_coords, future_tl, future_timestamps):
            if timestamp.time_us not in dict_query_feature.keys():
                dict_query_feature[timestamp.time_us] = {}
            if len(dict_query_feature[timestamp.time_us].keys()) > 0:
                collected_points = np.array([coords.array for coords in dict_query_feature[timestamp.time_us].keys()])
                distance_to_previous_points = np.linalg.norm(collected_points - ego_coords.array, ord=2, axis=1)
                need_to_compute = np.amin(distance_to_previous_points) > self.deduplicate_radius
                if not need_to_compute:
                    idx = np.argmin(distance_to_previous_points)
                    query_point = list(dict_query_feature[timestamp.time_us].keys())[idx]
            else:
                need_to_compute = True
            if need_to_compute:
                coords, traffic_light_data = get_neighbor_vector_set_map(
                    scenario.map_api,
                    self.map_features,
                    ego_coords,
                    self.radius,
                    route_roadblock_ids,
                    [traffic_light_status],
                    [timestamp]
                )
                tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
                    coords, list(traffic_light_data.values())[0], ego_state.rear_axle
                )
                dict_query_feature[timestamp.time_us][ego_coords] = [tensor, list_tensor, list_list_tensor]
            else:
                tensor, list_tensor, list_list_tensor = dict_query_feature[timestamp.time_us][query_point]
            trajectory_features.append([tensor, list_tensor, list_list_tensor])
        future_features.append(trajectory_features)

        # ----sparse sampling----
        # find neighbor map objects within neighbor_radius
        current_ego_state = scenario.initial_ego_state

        # find centerline
        reference_line_coords, sth_to_plot = get_centerline_coords(scenario,
                                                                   self.radius,
                                                                   self.v_max,
                                                                   self.a_max,
                                                                   self.time_horizon,
                                                                   current_ego_state)

        # build Frenet frame
        frenet_frame = FrenetFrame(reference_line_coords)

        # sparse sampling
        trajectory_sampler = TrajectorySampler(
            frenet_frame,
            current_ego_state,
            self.lane_width,
            self.delta_l,
            self.time_horizon,
            self.num_poses
        )
        trajectory_samples_frenet = trajectory_sampler.get_trajectory_samples(caching_avrl_features=True)
        trajectory_samples_cartesian = frenet_frame.frenet_to_cartesian(
            trajectory_samples_frenet['poses_frenet'],
            trajectory_samples_frenet['t'],
            trajectory_samples_frenet['vs_frenet'],
            trajectory_samples_frenet['vl_frenet'],
            trajectory_samples_frenet['as_frenet'],
            trajectory_samples_frenet['al_frenet'],
        )

        # translate the trajectory_samples_cartesian to match the current ego position
        trajectory_samples_cartesian['pose_cartesian'][:, 0, :] = (trajectory_samples_cartesian['pose_cartesian'][:, 0, :] +
                                                                   (current_ego_state.rear_axle.x - trajectory_samples_cartesian['pose_cartesian'][:, 0, 0])[:, np.newaxis])
        trajectory_samples_cartesian['pose_cartesian'][:, 1, :] = (trajectory_samples_cartesian['pose_cartesian'][:, 1, :] +
                                                                   (current_ego_state.rear_axle.y - trajectory_samples_cartesian['pose_cartesian'][:, 1, 0])[:, np.newaxis])
        trajectory_samples_cartesian['geo_center_pose_cartesian'] = np.array([
            trajectory_samples_cartesian['pose_cartesian'][:, 0, :] + current_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center * np.cos(
                trajectory_samples_cartesian['pose_cartesian'][:, 2, :]),
            trajectory_samples_cartesian['pose_cartesian'][:, 1, :] + current_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center * np.sin(
                trajectory_samples_cartesian['pose_cartesian'][:, 2, :]),
            trajectory_samples_cartesian['pose_cartesian'][:, 2, :]
        ]).transpose(1, 0, 2)

        # ----sparse sampling end----

        # generate sampled future ego states
        for i_traj in range(trajectory_samples_cartesian['pose_cartesian'].shape[0]):
            rear_axle_velocity_x = trajectory_samples_cartesian['rear_axle_velocity_x'][i_traj, 1:]
            rear_axle_velocity_y = trajectory_samples_cartesian['rear_axle_velocity_y'][i_traj, 1:]
            rear_axle_acceleration_x = trajectory_samples_cartesian['rear_axle_acceleration_x'][i_traj, 1:]
            rear_axle_acceleration_y = trajectory_samples_cartesian['rear_axle_acceleration_y'][i_traj, 1:]
            angular_velocity = trajectory_samples_cartesian['angular_velocity'][i_traj, 1:]
            geo_center_traj = trajectory_samples_cartesian['geo_center_pose_cartesian'][i_traj, :, 1:]
            trajectory_features = []
            for t in range(trajectory_samples_cartesian['pose_cartesian'][:, :, 1:].shape[2]):
                # generate car_footprint
                car_footprint = CarFootprint(StateSE2(geo_center_traj[0, t],
                                                      geo_center_traj[1, t],
                                                      geo_center_traj[2, t]),
                                             current_ego_state.car_footprint.vehicle_parameters)
                dynamic_car_state = DynamicCarState(
                    current_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center,
                    StateVector2D(rear_axle_velocity_x[t],
                                  rear_axle_velocity_y[t]),
                    StateVector2D(rear_axle_acceleration_x[t],
                                  rear_axle_acceleration_y[t]),
                    angular_velocity=angular_velocity[t]
                )
                ego_state = EgoState(car_footprint,
                                     dynamic_car_state,
                                     tire_steering_angle=0.,
                                     is_in_auto_mode=True,
                                     time_point=future_ego_states[t].time_point)
                ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
                traffic_light_status = future_tl[t]

                # coords, traffic_light_data = get_neighbor_vector_set_map(
                #     scenario.map_api,
                #     self.map_features,
                #     ego_coords,
                #     self.radius,
                #     route_roadblock_ids,
                #     [traffic_light_status],
                #     [future_timestamps[t]]
                # )
                # tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
                #     coords, list(traffic_light_data.values())[0], ego_state.rear_axle
                # )

                timestamp = future_ego_states[t].time_point
                if timestamp.time_us not in dict_query_feature.keys():
                    dict_query_feature[timestamp.time_us] = {}
                if len(dict_query_feature[timestamp.time_us].keys()) > 0:
                    collected_points = np.array([coords.array for coords in dict_query_feature[timestamp.time_us].keys()])
                    distance_to_previous_points = np.linalg.norm(collected_points - ego_coords.array, ord=2, axis=1)
                    need_to_compute = np.amin(distance_to_previous_points) > self.deduplicate_radius
                    if not need_to_compute:
                        idx = np.argmin(distance_to_previous_points)
                        query_point = list(dict_query_feature[timestamp.time_us].keys())[idx]
                else:
                    need_to_compute = True
                if need_to_compute:
                    coords, traffic_light_data = get_neighbor_vector_set_map(
                        scenario.map_api,
                        self.map_features,
                        ego_coords,
                        self.radius,
                        route_roadblock_ids,
                        [traffic_light_status],
                        [timestamp]
                    )
                    tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
                        coords, list(traffic_light_data.values())[0], ego_state.rear_axle
                    )
                    dict_query_feature[timestamp.time_us][ego_coords] = [tensor, list_tensor, list_list_tensor]
                else:
                    tensor, list_tensor, list_list_tensor = dict_query_feature[timestamp.time_us][query_point]

                trajectory_features.append([tensor, list_tensor, list_list_tensor])
            future_features.append(trajectory_features)

        # # TODO debug only: visualization
        # sth_to_plot['scenario'] = scenario
        # sth_to_plot['anchor_ego_state'] = current_ego_state
        # sth_to_plot['future_ego_states'] = future_ego_states
        # sth_to_plot['trajectory_samples_cartesian'] = trajectory_samples_cartesian
        # visualize(sth_to_plot)

        return future_features, trajectory_samples_cartesian

    @torch.jit.unused
    def get_scriptable_input_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization, scenario: AbstractScenario
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]], Dict[str, Any]]:
        """
        Extract the input for the scriptable forward method from the simulation objects
        :param current_input: planner input from sim
        :param initialization: planner initialization from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        radius = 150.
        time_horizon = self.time_horizon
        num_poses = self.num_poses

        ego_state = current_input.history.ego_states[-1]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(current_input.iteration.index))

        coords, traffic_light_data = qcmae_get_neighbor_vector_set_map(
            scenario.map_api,
            self._polygon_types,
            ego_coords,
            radius,
            route_roadblock_ids,
            [TrafficLightStatuses(traffic_light_data)],
        )

        tensor, list_tensor, list_list_tensor = self.qcmae_pack_to_feature_tensor_dict(
            coords, traffic_light_data[0], ego_state.rear_axle
        )

        # ----sparse sampling----
        # find neighbor map objects within neighbor_radius
        current_ego_state = current_input.history.current_state[0]

        # find centerline
        reference_line_coords, sth_to_plot = get_centerline_coords(scenario,
                                                                   radius,
                                                                   self.v_max,
                                                                   self.a_max,
                                                                   time_horizon,
                                                                   current_ego_state)

        # build Frenet frame
        frenet_frame = FrenetFrame(reference_line_coords)

        # sparse sampling
        trajectory_sampler = TrajectorySampler(
            frenet_frame,
            current_ego_state,
            self.lane_width,
            self.delta_l,
            time_horizon,
            num_poses
        )
        trajectory_samples_frenet = trajectory_sampler.get_trajectory_samples(caching_avrl_features=True)
        trajectory_samples_cartesian = frenet_frame.frenet_to_cartesian(
            trajectory_samples_frenet['poses_frenet'],
            trajectory_samples_frenet['t'],
            trajectory_samples_frenet['vs_frenet'],
            trajectory_samples_frenet['vl_frenet'],
            trajectory_samples_frenet['as_frenet'],
            trajectory_samples_frenet['al_frenet'],
        )

        # translate the trajectory_samples_cartesian to match the current ego position
        trajectory_samples_cartesian['pose_cartesian'][:, 0, :] = (trajectory_samples_cartesian['pose_cartesian'][:, 0, :] +
                                                                   (current_ego_state.rear_axle.x - trajectory_samples_cartesian['pose_cartesian'][:, 0, 0])[:, np.newaxis])
        trajectory_samples_cartesian['pose_cartesian'][:, 1, :] = (trajectory_samples_cartesian['pose_cartesian'][:, 1, :] +
                                                                   (current_ego_state.rear_axle.y - trajectory_samples_cartesian['pose_cartesian'][:, 1, 0])[:, np.newaxis])
        trajectory_samples_cartesian['geo_center_pose_cartesian'] = np.array([
            trajectory_samples_cartesian['pose_cartesian'][:, 0, :] + current_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center * np.cos(
                trajectory_samples_cartesian['pose_cartesian'][:, 2, :]),
            trajectory_samples_cartesian['pose_cartesian'][:, 1, :] + current_ego_state.car_footprint.vehicle_parameters.rear_axle_to_center * np.sin(
                trajectory_samples_cartesian['pose_cartesian'][:, 2, :]),
            trajectory_samples_cartesian['pose_cartesian'][:, 2, :]
        ]).transpose(1, 0, 2)

        # ----sparse sampling end----

        # # TODO debug only: visualization
        # sth_to_plot['scenario'] = scenario
        # sth_to_plot['anchor_ego_state'] = current_ego_state
        # sth_to_plot['trajectory_samples_cartesian'] = trajectory_samples_cartesian
        # visualize(sth_to_plot)

        return tensor, list_tensor, list_list_tensor, trajectory_samples_cartesian

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorSetMap:
        """Inherited, see superclass."""
        future_features, trajectory_samples = self.get_scriptable_input_from_scenario(scenario)
        processed_future_features = []
        for trajectory_features in future_features:
            processed_trajectory_features = []
            for tensor, list_tensor, list_list_tensor in trajectory_features:
                tensor_data, list_tensor_data, list_list_tensor_data, map_obj_ids, neighbor_ids = self.scriptable_forward(
                    tensor, list_tensor, list_list_tensor
                )
                assert neighbor_ids is not None, 'neighbor_ids is None'
                processed_trajectory_features.append([tensor_data, list_tensor_data, list_list_tensor_data, map_obj_ids, neighbor_ids])
            processed_future_features.append(processed_trajectory_features)

        return self._unpack_feature_to_vector_set_map(processed_future_features, trajectory_samples)

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization, scenario: AbstractScenario
    ) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data, trajectory_samples_cartesian = self.get_scriptable_input_from_simulation(
            current_input, initialization, scenario
        )
        tensor_data, list_tensor_data, list_list_tensor_data = self.qcmae_scriptable_forward(
            tensor_data, list_tensor_data, list_list_tensor_data
        )

        return self.qcmae_unpack_feature_to_vector_set_map(tensor_data, list_tensor_data, list_list_tensor_data, trajectory_samples_cartesian)

    @torch.jit.unused
    def _unpack_feature_to_vector_set_map(
        self,
        future_features: List[List[List[Dict[str, torch.Tensor] |
                                        Dict[str, List[torch.Tensor]] |
                                        Dict[str, List[List[torch.Tensor]]] |
                                        Dict[str, List[str]]]]],
        trajectory_samples: Dict[str, FeatureDataType],
    ) -> VectorSetMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorSetMap object.
        :param future_features: The data to unpack.
        :param trajectory_samples: The trajectory samples in Cartesian frame.
        :param geo_center_trajectory_samples: The geometric center trajectory samples in Cartesian frame.
        :return: The unpacked VectorSetMap.
        """
        coords: Dict[str, List[FeatureDataType]] = {}
        traffic_light_data: Dict[str, List[FeatureDataType]] = {}
        availabilities: Dict[str, List[FeatureDataType]] = {}
        map_obj_ids: Dict[str, Dict[int, Dict[int, List[str]]]] = {}
        neighbor_ids: Dict[int, Dict[int, Dict[str, List]]] = {}
        lane_types: Dict[int, Dict[int, List]] = {}
        num_trajs = len(future_features)
        num_poses = self.num_poses

        for i_traj in range(num_trajs):
            for i_pose in range(num_poses):
                tensor_data, list_tensor_data, list_list_tensor_data, map_obj_id, neighbor_data = future_features[i_traj][i_pose]
                lane_type = neighbor_data['LANE_TYPE']
                neighbor_id = {key: value for key, value in list(neighbor_data.items())[1:]}
                for key in list_tensor_data:
                    if key.startswith("vector_set_map.coords."):
                        feature_name = key[len("vector_set_map.coords.") :]
                        if feature_name not in coords.keys():
                            coords[feature_name] = [
                                np.zeros((num_trajs,
                                          num_poses,
                                          self.max_elements[feature_name],
                                          self.max_points[feature_name], 2))
                            ]
                        coords[feature_name][0][i_traj, i_pose] = list_tensor_data[key][0].detach().numpy()
                        if feature_name not in map_obj_ids.keys():
                            map_obj_ids[feature_name] = {}
                        if i_traj not in map_obj_ids[feature_name].keys():
                            map_obj_ids[feature_name][i_traj] = {}
                        map_obj_ids[feature_name][i_traj][i_pose] = map_obj_id[f'vector_set_map.map_obj_ids.{feature_name}']
                        if feature_name == 'LANE':
                            if i_traj not in neighbor_ids.keys():
                                neighbor_ids[i_traj] = {}
                            neighbor_ids[i_traj][i_pose] = neighbor_id
                            if i_traj not in lane_types.keys():
                                lane_types[i_traj] = {}
                            lane_types[i_traj][i_pose] = lane_type
                    if key.startswith("vector_set_map.traffic_light_data."):
                        feature_name = key[len("vector_set_map.traffic_light_data.") :]
                        if feature_name not in traffic_light_data.keys():
                            traffic_light_data[feature_name] = [
                                np.zeros((num_trajs, num_poses,
                                          self.max_elements[feature_name],
                                          self.max_points[feature_name],
                                          self._traffic_light_encoding_dim),)
                            ]
                        traffic_light_data[feature_name][0][i_traj, i_pose] = list_tensor_data[key][0].detach().numpy()
                    if key.startswith("vector_set_map.availabilities."):
                        feature_name = key[len("vector_set_map.availabilities.") :]
                        if feature_name not in availabilities.keys():
                            availabilities[feature_name] = [
                                np.zeros((num_trajs,
                                          num_poses,
                                          self.max_elements[feature_name],
                                          self.max_points[feature_name]),)
                            ]
                        availabilities[feature_name][0][i_traj, i_pose] = list_tensor_data[key][0].detach().numpy()

        map_data = self.get_map_data(
            VectorSetMap(
                coords=coords,
                traffic_light_data=traffic_light_data,
                availabilities=availabilities,
                map_obj_ids=map_obj_ids,
                lane_types=lane_types,
                neighbor_ids=neighbor_ids,
                trajectory_samples=trajectory_samples,
            ).to_feature_tensor()
        )

        return VectorSetMap(
            coords=coords,
            traffic_light_data=traffic_light_data,
            availabilities=availabilities,
            map_data=map_data,
            map_obj_ids=map_obj_ids,
            trajectory_samples=trajectory_samples,
        )

    @torch.jit.unused
    def qcmae_unpack_feature_to_vector_set_map(
            self,
            tensor_data: Dict[str, torch.Tensor],
            list_tensor_data: Dict[str, List[Dict[str, torch.Tensor]]],
            list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
            trajectory_samples_cartesian: Dict[str, Any],
    ) -> qcmaeVectorSetMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorSetMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorSetMap.
        """

        map_data = self.ritp_get_map_data(list_tensor_data)

        return qcmaeVectorSetMap(map_data=map_data, trajectory_samples_cartesian=trajectory_samples_cartesian)

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        coords: Dict[str, MapObjectPolylines],
        traffic_light_data: Dict[str, LaneSegmentTrafficLightData],
        anchor_state: StateSE2,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Union[Dict[str, torch.Tensor], List]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature transform.
        :param coords: Dictionary mapping feature name to polyline vector sets.
        :param traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
        :param anchor_state: The ego state to transform to vector.
        :return
           tensor_data: Packed tensor data.
           list_tensor_data: Packed Dict[tensor] data.
           list_list_tensor_data: Packed List[List[tensor]] data.
        """
        tensor_data: Dict[str, torch.Tensor] = {}
        # anchor state
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        tensor_data["anchor_state"] = anchor_state_tensor

        list_tensor_data: Dict[str, Union[Dict[str, torch.Tensor], List]] = {}

        for feature_name, feature_coords in coords.items():
            if feature_name in ['LANE_TYPE', 'LEFT_NEIGHBOR_LANE_ID', 'RIGHT_NEIGHBOR_LANE_ID', 'PREDECESSOR_ID', 'SUCCESSOR_ID']:
                continue
            list_feature_coords: Dict[str, torch.Tensor] = {}

            # Pack coords into tensor list
            for map_obj_id, element_coords in zip(feature_coords.polylines.keys(), feature_coords.to_vector()):
                list_feature_coords[map_obj_id] = torch.tensor(element_coords, dtype=torch.float64)
            list_tensor_data[f"coords.{feature_name}"] = list_feature_coords

            # Pack traffic light data into tensor list if it exists
            if feature_name in traffic_light_data:
                list_feature_tl_data: Dict[str, torch.Tensor] = {}

                for map_obj_id, element_tl_data in zip(
                        traffic_light_data[feature_name].traffic_lights.keys(),
                        traffic_light_data[feature_name].to_vector()
                ):
                    list_feature_tl_data[map_obj_id] = torch.tensor(element_tl_data, dtype=torch.float32)
                list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data

        # Pack neighbor ids into tensor list
        for feature_name in ['LANE_TYPE', 'LEFT_NEIGHBOR_LANE_ID', 'RIGHT_NEIGHBOR_LANE_ID', 'PREDECESSOR_ID', 'SUCCESSOR_ID']:
            list_tensor_data[f"coords.{feature_name}"] = coords[feature_name]

        return (
            tensor_data,
            list_tensor_data,
            {},
        )

    @torch.jit.unused
    def qcmae_pack_to_feature_tensor_dict(
            self,
            coords: Dict[str, MapObjectPolylines],
            traffic_light_data: Dict[str, LaneSegmentTrafficLightData],
            anchor_state: StateSE2,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Union[Dict[str, torch.Tensor], List]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Transforms the provided map and actor state primitives into scriptable types.
        This is to prepare for the scriptable portion of the feature transform.
        :param coords: Dictionary mapping feature name to polyline vector sets.
        :param traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
        :param anchor_state: The ego state to transform to vector.
        :return
           tensor_data: Packed tensor data.
           list_tensor_data: Packed Dict[tensor] data.
           list_list_tensor_data: Packed List[List[tensor]] data.
        """
        tensor_data: Dict[str, torch.Tensor] = {}
        # anchor state
        anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float64)
        tensor_data["anchor_state"] = anchor_state_tensor

        list_tensor_data: Dict[str, Union[Dict[str, torch.Tensor], List]] = {}

        for feature_name, feature_coords in coords.items():
            if feature_name in ['LANE_TYPE', 'LEFT_NEIGHBOR_LANE_ID', 'RIGHT_NEIGHBOR_LANE_ID', 'PREDECESSOR_ID', 'SUCCESSOR_ID']:
                continue
            list_feature_coords: Dict[str, torch.Tensor] = {}

            # Pack coords into tensor list
            for map_obj_id, element_coords in zip(feature_coords.polylines.keys(), feature_coords.to_vector()):
                list_feature_coords[map_obj_id] = torch.tensor(element_coords, dtype=torch.float64)
            list_tensor_data[f"coords.{feature_name}"] = list_feature_coords

            # Pack traffic light data into tensor list if it exists
            if feature_name in traffic_light_data:
                list_feature_tl_data: Dict[str, torch.Tensor] = {}

                for map_obj_id, element_tl_data in zip(
                        traffic_light_data[feature_name].traffic_lights.keys(),
                        traffic_light_data[feature_name].to_vector()
                ):
                    list_feature_tl_data[map_obj_id] = torch.tensor(element_tl_data, dtype=torch.float32)
                list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data

        # Pack neighbor ids into tensor list
        for feature_name in ['LANE_TYPE', 'LEFT_NEIGHBOR_LANE_ID', 'RIGHT_NEIGHBOR_LANE_ID', 'PREDECESSOR_ID', 'SUCCESSOR_ID']:
            list_tensor_data[f"{feature_name}"] = coords[feature_name]

        return (
            tensor_data,
            list_tensor_data,
            {},
        )

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Implemented. See interface.
        """
        tensor_output: Dict[str, torch.Tensor] = {}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {}
        map_obj_ids_output: Dict[str, List[str]] = {}
        neighbor_ids_output: Optional[Dict[str, List]] = None

        anchor_state = tensor_data["anchor_state"]

        for feature_name in self.map_features:
            if f"coords.{feature_name}" in list_tensor_data:
                feature_coords = list_tensor_data[f"coords.{feature_name}"]
                feature_tl_data = (
                    [list_tensor_data[f"traffic_light_data.{feature_name}"]]
                    if f"traffic_light_data.{feature_name}" in list_tensor_data
                    else None
                )
                if feature_name == 'LANE':
                    neighbor_data = {
                        'LANE_TYPE': list_tensor_data['coords.LANE_TYPE'],
                        'LEFT_NEIGHBOR_LANE_ID': list_tensor_data['coords.LEFT_NEIGHBOR_LANE_ID'],
                        'RIGHT_NEIGHBOR_LANE_ID': list_tensor_data['coords.RIGHT_NEIGHBOR_LANE_ID'],
                        'PREDECESSOR_ID': list_tensor_data['coords.PREDECESSOR_ID'],
                        'SUCCESSOR_ID': list_tensor_data['coords.SUCCESSOR_ID'],
                    }
                else:
                    neighbor_data = None

                coords, tl_data, avails, map_obj_ids, neighbor_ids = convert_feature_layer_to_fixed_size(
                    feature_coords,
                    feature_tl_data,
                    self.max_elements[feature_name],
                    self.max_points[feature_name],
                    self._traffic_light_encoding_dim,
                    interpolation=self.interpolation_method  # apply interpolation only for lane features
                    if feature_name
                    in [
                        VectorFeatureLayer.LANE.name,
                        VectorFeatureLayer.LEFT_BOUNDARY.name,
                        VectorFeatureLayer.RIGHT_BOUNDARY.name,
                        VectorFeatureLayer.ROUTE_LANES.name,
                    ]
                    else None,
                    neighbor_data=neighbor_data
                )
                if neighbor_ids is not None:
                    neighbor_ids_output = neighbor_ids

                # # disable because conflicts with feature compressing when caching features.
                # coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state)

                list_tensor_output[f"vector_set_map.coords.{feature_name}"] = [coords]
                list_tensor_output[f"vector_set_map.availabilities.{feature_name}"] = [avails]

                if tl_data is not None:
                    list_tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = [tl_data[0]]

                map_obj_ids_output[f"vector_set_map.map_obj_ids.{feature_name}"] = map_obj_ids

        return tensor_output, list_tensor_output, list_list_tensor_output, map_obj_ids_output, neighbor_ids_output

    @torch.jit.export
    def qcmae_scriptable_forward(
            self,
            tensor_data: Dict[str, torch.Tensor],
            list_tensor_data: Dict[str, List[torch.Tensor]],
            list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Implemented. See interface.
        """
        tensor_output: Dict[str, torch.Tensor] = {}
        list_tensor_output: Dict[str, List[torch.Tensor]] = {}
        list_list_tensor_output: Dict[str, List[List[torch.Tensor]]] = {}

        anchor_state = tensor_data["anchor_state"]

        for feature_name in self.map_features:
            if f"coords.{feature_name}" in list_tensor_data:
                feature_coords = list_tensor_data[f"coords.{feature_name}"]
                feature_tl_data = (
                    [list_tensor_data[f"traffic_light_data.{feature_name}"]]
                    if f"traffic_light_data.{feature_name}" in list_tensor_data
                    else None
                )

                list_tensor_output[f"coords.{feature_name}"] = [feature_coords]

                if feature_tl_data is not None:
                    list_tensor_output[f"traffic_light_data.{feature_name}"] = feature_tl_data

        centerline_coords, left_boundary_coords, right_boundary_coords = convert_lane_layers_to_consistent_size(
            list_tensor_output["coords.LANE"],
            list_tensor_output["coords.LEFT_BOUNDARY"],
            list_tensor_output["coords.RIGHT_BOUNDARY"],
            max_points=self.max_points,
            interpolation=self.interpolation_method
        )
        list_tensor_output["coords.LANE"] = centerline_coords
        list_tensor_output["coords.LEFT_BOUNDARY"] = left_boundary_coords
        list_tensor_output["coords.RIGHT_BOUNDARY"] = right_boundary_coords

        list_tensor_output["LANE_TYPE"] = [list_tensor_data["LANE_TYPE"]]
        list_tensor_output["LEFT_NEIGHBOR_LANE_ID"] = [list_tensor_data["LEFT_NEIGHBOR_LANE_ID"]]
        list_tensor_output["RIGHT_NEIGHBOR_LANE_ID"] = [list_tensor_data["RIGHT_NEIGHBOR_LANE_ID"]]
        list_tensor_output["PREDECESSOR_ID"] = [list_tensor_data["PREDECESSOR_ID"]]
        list_tensor_output["SUCCESSOR_ID"] = [list_tensor_data["SUCCESSOR_ID"]]

        return tensor_output, list_tensor_output, list_list_tensor_output

    def get_map_data(self, vector_set_map: VectorSetMap) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        list_num_polygons = []
        list_polygon_position = []
        list_polygon_orientation = []
        list_polygon_type = []
        list_polygon_is_intersection = []
        list_num_points = []
        list_point_position = []
        list_point_orientation = []
        list_point_magnitude = []
        list_point_type = []
        list_point_side = []
        list_point_tl_statuses = []
        list_point_to_polygon_edge_index = []
        list_polygon_to_polygon_edge_index = []
        list_polygon_to_polygon_type = []
        num_pl_detail = {}
        num_pl_to_pl_edge_index_detail = {}

        batch_size = vector_set_map.batch_size
        num_trajs = vector_set_map.coords['LANE'][0].shape[0]
        num_poses = self.num_poses

        for sample_idx in range(batch_size):
            if sample_idx not in num_pl_detail.keys():
                num_pl_detail[sample_idx] = {}
            if sample_idx not in num_pl_to_pl_edge_index_detail.keys():
                num_pl_to_pl_edge_index_detail[sample_idx] = {}
            avails_lane = vector_set_map.availabilities['LANE'][sample_idx]
            avails_stop_line = vector_set_map.availabilities['STOP_LINE'][sample_idx]
            avails_crosswalk = vector_set_map.availabilities['CROSSWALK'][sample_idx]
            avails_route_lanes = vector_set_map.availabilities['ROUTE_LANES'][sample_idx]
            valid_mask_lane = torch.sum(avails_lane, dim=-1) != 0
            valid_mask_stop_line = torch.sum(avails_stop_line, dim=-1) != 0
            valid_mask_crosswalk = torch.sum(avails_crosswalk, dim=-1) != 0
            valid_mask_route_lanes = torch.sum(avails_route_lanes, dim=-1) != 0
            num_polygons = int(valid_mask_lane.sum() +
                               valid_mask_stop_line.sum() +
                               valid_mask_crosswalk.sum() +
                               valid_mask_route_lanes.sum())
            # num_points = int(avails_lane.sum() +
            #                  avails_stop_line.sum() +
            #                  avails_crosswalk.sum() +
            #                  avails_route_lanes.sum())
            _, _, max_elements, max_points, dim = vector_set_map.coords['LANE'][0].shape
            # initialization
            polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
            polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
            polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
            polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
            point_position: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_orientation: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_magnitude: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_type: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_side: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_tl_statuses: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            # point_to_polygon_edge_index = torch.zeros((2, num_points), dtype=torch.long)
            point_to_polygon_edge_index_1 = []  # point_to_polygon_edge_index[1]
            polygon_to_polygon_edge_index = []
            polygon_to_polygon_type = []

            idx_offset_pl = 0
            idx_offset_pt = 0
            # point_to_polygon_edge_index[0] = torch.arange(num_points, dtype=torch.long)
            for i_traj in range(num_trajs):
                if i_traj not in num_pl_detail[sample_idx].keys():
                    num_pl_detail[sample_idx][i_traj] = {}
                if i_traj not in num_pl_to_pl_edge_index_detail[sample_idx].keys():
                    num_pl_to_pl_edge_index_detail[sample_idx][i_traj] = {}
                for i_pose in range(num_poses):
                    if i_pose not in num_pl_detail[sample_idx][i_traj].keys():
                        num_pl_detail[sample_idx][i_traj][i_pose] = {}
                    num_lanes = int((torch.sum(avails_lane[i_traj][i_pose], dim=-1) != 0).sum())
                    num_stop_lines = int((torch.sum(avails_stop_line[i_traj][i_pose], dim=-1) != 0).sum())
                    num_crosswalks = int((torch.sum(avails_crosswalk[i_traj][i_pose], dim=-1) != 0).sum())
                    num_route_lanes = int((torch.sum(avails_route_lanes[i_traj][i_pose], dim=-1) != 0).sum())
                    num_elements = num_lanes + num_stop_lines + num_crosswalks + num_route_lanes

                    num_pl_detail[sample_idx][i_traj][i_pose]['num_lanes'] = num_lanes
                    num_pl_detail[sample_idx][i_traj][i_pose]['num_stop_lines'] = num_stop_lines
                    num_pl_detail[sample_idx][i_traj][i_pose]['num_crosswalks'] = num_crosswalks
                    num_pl_detail[sample_idx][i_traj][i_pose]['num_route_lanes'] = num_route_lanes
                    num_pl_detail[sample_idx][i_traj][i_pose]['num_polygons'] = num_elements
                    num_pl_to_pl_edge_index_detail[sample_idx][i_traj][i_pose] = num_lanes

                    # 'LANE'
                    for i_element in range(num_lanes):
                        num_points_i = int((avails_lane[i_traj][i_pose][i_element] != 0).sum())
                        centerline = vector_set_map.coords['LANE'][sample_idx][i_traj, i_pose, i_element, 0:num_points_i, :]
                        polygon_position[idx_offset_pl + i_element] = centerline[0]
                        polygon_orientation[idx_offset_pl + i_element] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                                     centerline[1, 0] - centerline[0, 0])
                        polygon_type[idx_offset_pl + i_element] = self._polygon_types.index('LANE')
                        if vector_set_map.lane_types[i_traj][i_pose][i_element] == 'LANE':
                            polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(False)
                        elif vector_set_map.lane_types[i_traj][i_pose][i_element] == 'LANE_CONNECTOR':
                            polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(True)

                        left_boundary = vector_set_map.coords['LEFT_BOUNDARY'][sample_idx][i_traj, i_pose, i_element, 0:num_points_i, :]
                        right_boundary = vector_set_map.coords['RIGHT_BOUNDARY'][sample_idx][i_traj, i_pose, i_element, 0:num_points_i, :]
                        point_position[idx_offset_pl + i_element] = torch.cat([left_boundary[:-1, :],
                                                                               right_boundary[:-1, :],
                                                                               centerline[:-1, :]], dim=0)
                        left_vectors = left_boundary[1:] - left_boundary[:-1]
                        right_vectors = right_boundary[1:] - right_boundary[:-1]
                        center_vectors = centerline[1:] - centerline[:-1]
                        point_orientation[idx_offset_pl + i_element] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                                                  torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                                                  torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                                                 dim=0)
                        point_magnitude[idx_offset_pl + i_element] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                                           right_vectors[:, :2],
                                                                                           center_vectors[:, :2]], dim=0), p=2, dim=-1)
                        left_type = self._point_types.index('LEFT_BOUNDARY')
                        right_type = self._point_types.index('RIGHT_BOUNDARY')
                        center_type = self._point_types.index('LANE')
                        point_type[idx_offset_pl + i_element] = torch.cat(
                            [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                             torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                             torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
                        point_side[idx_offset_pl + i_element] = torch.cat(
                            [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                             torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                             torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
                        tl_status = vector_set_map.traffic_light_data['LANE'][sample_idx][i_traj, i_pose, i_element, 0:num_points_i, :]
                        tl_status = tl_status.type(dtype=torch.uint8).numpy()
                        tl_status = torch.tensor([self._traffic_light_one_hot_decoding[tuple(status)] for status in tl_status[:-1, :]], dtype=torch.uint8)
                        point_tl_statuses[idx_offset_pl + i_element] = torch.cat([tl_status, tl_status, tl_status], dim=0)

                        # point_to_polygon_edge_index[1, idx_offset_pt:idx_offset_pt + num_points_i] \
                        #     = torch.full((num_points_i,), idx_offset_pl + i_element, dtype=torch.long)
                        point_to_polygon_edge_index_1.append(torch.full(((num_points_i - 1) * 3,), idx_offset_pl + i_element, dtype=torch.long))

                        lane_ids = vector_set_map.map_obj_ids['LANE'][i_traj][i_pose]
                        pred_ids = vector_set_map.neighbor_ids[i_traj][i_pose]['PREDECESSOR_ID'][i_element]
                        pred_indexes = []
                        if pred_ids is not None:
                            for pred_id in pred_ids:
                                index = safe_list_index(lane_ids, pred_id)
                                if index is not None:
                                    pred_indexes.append(idx_offset_pl + safe_list_index(lane_ids, pred_id))
                        if len(pred_indexes) > 0:
                            polygon_to_polygon_edge_index.append(
                                torch.stack([torch.tensor(pred_indexes, dtype=torch.long),
                                             torch.full((len(pred_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                            polygon_to_polygon_type.append(
                                torch.full((len(pred_indexes),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
                        succ_ids = vector_set_map.neighbor_ids[i_traj][i_pose]['SUCCESSOR_ID'][i_element]
                        succ_indexes = []
                        if succ_ids is not None:
                            for succ_id in succ_ids:
                                index = safe_list_index(lane_ids, succ_id)
                                if index is not None:
                                    succ_indexes.append(idx_offset_pl + safe_list_index(lane_ids, succ_id))
                        if len(succ_indexes) > 0:
                            polygon_to_polygon_edge_index.append(
                                torch.stack([torch.tensor(succ_indexes, dtype=torch.long),
                                             torch.full((len(succ_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                            polygon_to_polygon_type.append(
                                torch.full((len(succ_indexes),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
                        left_lane_ids = vector_set_map.neighbor_ids[i_traj][i_pose]['LEFT_NEIGHBOR_LANE_ID'][i_element]
                        left_lane_indexes = []
                        if left_lane_ids is not None:
                            index = safe_list_index(lane_ids, left_lane_ids)
                            if index is not None:
                                left_lane_indexes.append(idx_offset_pl + safe_list_index(lane_ids, left_lane_ids))
                        if len(left_lane_indexes) > 0:
                            polygon_to_polygon_edge_index.append(
                                torch.stack([torch.tensor(left_lane_indexes, dtype=torch.long),
                                             torch.full((len(left_lane_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                            polygon_to_polygon_type.append(
                                torch.full((len(left_lane_indexes),), self._polygon_to_polygon_types.index('LEFT'), dtype=torch.uint8))
                        right_lane_ids = vector_set_map.neighbor_ids[i_traj][i_pose]['RIGHT_NEIGHBOR_LANE_ID'][i_element]
                        right_lane_indexes = []
                        if right_lane_ids is not None:
                            index = safe_list_index(lane_ids, right_lane_ids)
                            if index is not None:
                                right_lane_indexes.append(idx_offset_pl + safe_list_index(lane_ids, right_lane_ids))
                        if len(right_lane_indexes) > 0:
                            polygon_to_polygon_edge_index.append(
                                torch.stack([torch.tensor(right_lane_indexes, dtype=torch.long),
                                             torch.full((len(right_lane_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                            polygon_to_polygon_type.append(
                                torch.full((len(right_lane_indexes),), self._polygon_to_polygon_types.index('RIGHT'), dtype=torch.uint8))

                        if len(pred_indexes) == 0 and len(succ_indexes) == 0 and len(left_lane_indexes) == 0 and len(right_lane_indexes) == 0:
                            num_pl_to_pl_edge_index_detail[sample_idx][i_traj][i_pose] = num_pl_to_pl_edge_index_detail[sample_idx][i_traj][i_pose] - 1

                        idx_offset_pt = idx_offset_pt + (num_points_i - 1) * 3

                    idx_offset_pl = idx_offset_pl + num_lanes

                    # 'STOP_LINE'
                    for i_element in range(num_stop_lines):
                        num_points_i = int((avails_stop_line[i_traj][i_pose][i_element] != 0).sum())
                        edge = vector_set_map.coords['STOP_LINE'][sample_idx][i_traj, i_pose, i_element, 0:num_points_i, :]
                        center_position = edge[:-1].mean(dim=0)
                        polygon_position[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = center_position
                        polygon_type[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = self._polygon_types.index('STOP_LINE')
                        polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(None)

                        point_position[idx_offset_pl + i_element] = edge[:-1]
                        edge_vectors = edge[1:] - edge[:-1]
                        point_orientation[idx_offset_pl + i_element] = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
                        point_magnitude[idx_offset_pl + i_element] = torch.norm(edge_vectors[:, :2], p=2, dim=-1)
                        edge_type = self._point_types.index('STOP_LINE')
                        point_type[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), edge_type, dtype=torch.uint8)
                        point_side[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), self._point_sides.index('UNKNOWN'), dtype=torch.uint8)
                        point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors),),
                                                                                  self._point_traffic_light_statuses.index('UNKNOWN'),
                                                                                  dtype=torch.uint8)

                        # point_to_polygon_edge_index[1, idx_offset_pt:idx_offset_pt + num_points_i] \
                        #     = torch.full((num_points_i,), idx_offset_pl + i_element, dtype=torch.long)
                        point_to_polygon_edge_index_1.append(torch.full((num_points_i - 1,), idx_offset_pl + i_element, dtype=torch.long))
                        idx_offset_pt = idx_offset_pt + num_points_i - 1
                    idx_offset_pl = idx_offset_pl + num_stop_lines

                    # 'CROSSWALK'
                    for i_element in range(num_crosswalks):
                        num_points_i = int((avails_crosswalk[i_traj][i_pose][i_element] != 0).sum())
                        edge = vector_set_map.coords['CROSSWALK'][sample_idx][i_traj, i_pose, i_element, 0:num_points_i, :]
                        center_position = edge[:-1].mean(dim=0)
                        polygon_position[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = center_position
                        polygon_type[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = self._polygon_types.index('CROSSWALK')
                        polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(None)

                        point_position[idx_offset_pl + i_element] = edge[:-1]
                        edge_vectors = edge[1:] - edge[:-1]
                        point_orientation[idx_offset_pl + i_element] = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
                        point_magnitude[idx_offset_pl + i_element] = torch.norm(edge_vectors[:, :2], p=2, dim=-1)
                        edge_type = self._point_types.index('CROSSWALK')
                        point_type[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), edge_type, dtype=torch.uint8)
                        point_side[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), self._point_sides.index('UNKNOWN'), dtype=torch.uint8)
                        point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors),),
                                                                                  self._point_traffic_light_statuses.index('UNKNOWN'),
                                                                                  dtype=torch.uint8)

                        # point_to_polygon_edge_index[1, idx_offset_pt:idx_offset_pt + num_points_i] \
                        #     = torch.full((num_points_i,), idx_offset_pl + i_element, dtype=torch.long)
                        point_to_polygon_edge_index_1.append(torch.full((num_points_i - 1,), idx_offset_pl + i_element, dtype=torch.long))
                        idx_offset_pt = idx_offset_pt + num_points_i - 1
                    idx_offset_pl = idx_offset_pl + num_crosswalks

                    # 'ROUTE_LANES'
                    for i_element in range(num_route_lanes):
                        num_points_i = int((avails_route_lanes[i_traj][i_pose][i_element] != 0).sum())
                        edge = vector_set_map.coords['ROUTE_LANES'][sample_idx][i_traj, i_pose, i_element, 0:num_points_i, :]
                        start_position = edge[0]
                        polygon_position[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = start_position
                        polygon_type[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = self._polygon_types.index('ROUTE_LANES')
                        polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(None)

                        point_position[idx_offset_pl + i_element] = edge[:-1]
                        edge_vectors = edge[1:] - edge[:-1]
                        point_orientation[idx_offset_pl + i_element] = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
                        point_magnitude[idx_offset_pl + i_element] = torch.norm(edge_vectors[:, :2], p=2, dim=-1)
                        edge_type = self._point_types.index('ROUTE_LANES')
                        point_type[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), edge_type, dtype=torch.uint8)
                        point_side[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)
                        point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors),),
                                                                                  self._point_traffic_light_statuses.index('UNKNOWN'),
                                                                                  dtype=torch.uint8)

                        # point_to_polygon_edge_index[1, idx_offset_pt:idx_offset_pt + num_points_i] \
                        #     = torch.full((num_points_i,), idx_offset_pl + i_element, dtype=torch.long)
                        point_to_polygon_edge_index_1.append(torch.full((num_points_i - 1,), idx_offset_pl + i_element, dtype=torch.long))
                        idx_offset_pt = idx_offset_pt + num_points_i - 1

                    idx_offset_pl = idx_offset_pl + num_route_lanes

            num_points = idx_offset_pt
            point_to_polygon_edge_index_0 = torch.arange(num_points, dtype=torch.long)
            point_to_polygon_edge_index_1 = torch.cat(point_to_polygon_edge_index_1, dim=0)
            point_to_polygon_edge_index = torch.stack([point_to_polygon_edge_index_0, point_to_polygon_edge_index_1], dim=0)


            if len(polygon_to_polygon_edge_index) == 0:
                idx_offset_pl = 0
                for i_traj in range(num_trajs):
                    for i_pose in range(num_poses):
                        num_lanes = int((torch.sum(avails_lane[i_traj][i_pose], dim=-1) != 0).sum())
                        num_stop_lines = int((torch.sum(avails_stop_line[i_traj][i_pose], dim=-1) != 0).sum())
                        num_crosswalks = int((torch.sum(avails_crosswalk[i_traj][i_pose], dim=-1) != 0).sum())
                        num_route_lanes = int((torch.sum(avails_route_lanes[i_traj][i_pose], dim=-1) != 0).sum())
                        all_indexes = list(range(num_lanes))
                        for i_element in range(num_lanes):
                            rest_indexes = all_indexes.copy()
                            rest_indexes.remove(i_element)
                            rest_indexes = [index + idx_offset_pl for index in rest_indexes]
                            polygon_to_polygon_edge_index.append(
                                torch.stack([torch.tensor(rest_indexes, dtype=torch.long),
                                             torch.full((len(rest_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                            polygon_to_polygon_type.append(
                                torch.full((len(rest_indexes),), self._polygon_to_polygon_types.index('NONE'), dtype=torch.uint8))

                        num_pl_to_pl_edge_index_detail[sample_idx][i_traj][i_pose] = num_lanes

                        idx_offset_pl = idx_offset_pl + num_lanes
                        idx_offset_pl = idx_offset_pl + num_stop_lines
                        idx_offset_pl = idx_offset_pl + num_crosswalks
                        idx_offset_pl = idx_offset_pl + num_route_lanes

            list_num_polygons.append(num_polygons)
            list_polygon_position.append(polygon_position)
            list_polygon_orientation.append(polygon_orientation)
            list_polygon_type.append(polygon_type)
            list_polygon_is_intersection.append(polygon_is_intersection)
            list_num_points.append(num_points)
            list_point_position.append(point_position)
            list_point_orientation.append(point_orientation)
            list_point_magnitude.append(point_magnitude)
            list_point_type.append(point_type)
            list_point_side.append(point_side)
            list_point_tl_statuses.append(point_tl_statuses)
            list_point_to_polygon_edge_index.append(point_to_polygon_edge_index)
            list_polygon_to_polygon_edge_index.append(polygon_to_polygon_edge_index)
            list_polygon_to_polygon_type.append(polygon_to_polygon_type)

        num_polygons = sum(list_num_polygons)
        polygon_position = torch.cat(list_polygon_position, dim=0)
        polygon_orientation = torch.cat(list_polygon_orientation)
        polygon_type = torch.cat(list_polygon_type)
        polygon_is_intersection = torch.cat(list_polygon_is_intersection)
        num_points = sum(list_num_points)
        point_position = torch.cat([torch.cat(point_pos, dim=0) for point_pos in list_point_position], dim=0).float()
        point_orientation = torch.cat([torch.cat(point_ori) for point_ori in list_point_orientation]).float()
        point_magnitude = torch.cat([torch.cat(point_mag) for point_mag in list_point_magnitude]).float()
        point_type = torch.cat([torch.cat(point_type) for point_type in list_point_type])
        point_side = torch.cat([torch.cat(point_side) for point_side in list_point_side])
        point_tl_statuses = torch.cat([torch.cat(tl_status) for tl_status in list_point_tl_statuses])
        list_polygon_to_polygon_edge_index = [torch.cat(index, dim=1) for index in list_polygon_to_polygon_edge_index]
        for sample_idx in range(batch_size):
            if sample_idx == 0:
                continue
            else:
                list_point_to_polygon_edge_index[sample_idx][0, :] \
                    = list_point_to_polygon_edge_index[sample_idx][0, :] + list_point_to_polygon_edge_index[sample_idx - 1][0, -1] + 1
                list_point_to_polygon_edge_index[sample_idx][1, :] \
                    = list_point_to_polygon_edge_index[sample_idx][1, :] + list_point_to_polygon_edge_index[sample_idx - 1][1, -1] + 1
                list_polygon_to_polygon_edge_index[sample_idx] \
                    = list_polygon_to_polygon_edge_index[sample_idx] + sum(list_num_polygons[:sample_idx])
        point_to_polygon_edge_index = torch.cat(list_point_to_polygon_edge_index, dim=1)
        polygon_to_polygon_edge_index = torch.cat(list_polygon_to_polygon_edge_index, dim=1)
        polygon_to_polygon_type = torch.cat([torch.cat(index) for index in list_polygon_to_polygon_type])

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
            'num_pl_detail': num_pl_detail,
            'num_pl_to_pl_edge_index_detail': num_pl_to_pl_edge_index_detail
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position.detach().numpy()
        map_data['map_polygon']['orientation'] = polygon_orientation.detach().numpy()
        map_data['map_polygon']['type'] = polygon_type.detach().numpy()
        map_data['map_polygon']['is_intersection'] = polygon_is_intersection.detach().numpy()
        map_data['map_point']['num_nodes'] = num_points
        map_data['map_point']['position'] = point_position.detach().numpy()
        map_data['map_point']['orientation'] = point_orientation.detach().numpy()
        map_data['map_point']['magnitude'] = point_magnitude.detach().numpy()
        map_data['map_point']['type'] = point_type.detach().numpy()
        map_data['map_point']['side'] = point_side.detach().numpy()
        map_data['map_point']['tl_statuses'] = point_tl_statuses.detach().numpy()
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index.detach().numpy()
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index.detach().numpy()
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type.detach().numpy()

        return map_data

    def ritp_get_map_data(self, list_tensor_data: Dict[str, List[Union[Dict[str, torch.Tensor], List[Any]]]]) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        """Modified version of get_map_data function in ritp_vector_set_map_feature_builder.py"""
        list_num_polygons = []
        list_polygon_position = []
        list_polygon_orientation = []
        list_polygon_type = []
        list_polygon_is_intersection = []
        list_polygon_tl_statuses = []  # used for QCMAE
        list_num_points = []
        list_point_position = []
        list_point_orientation = []
        list_point_magnitude = []
        list_point_type = []
        list_point_side = []
        list_point_tl_statuses = []  # used for RewardFormer
        list_point_to_polygon_edge_index = []
        list_polygon_to_polygon_edge_index = []
        list_polygon_to_polygon_type = []
        num_pl_detail = {}
        num_pl_to_pl_edge_index_detail = {}

        batch_size = len(list_tensor_data['coords.LANE'])

        for sample_idx in range(batch_size):
            if sample_idx not in num_pl_detail.keys():
                num_pl_detail[sample_idx] = {}
            if sample_idx not in num_pl_to_pl_edge_index_detail.keys():
                num_pl_to_pl_edge_index_detail[sample_idx] = {}
            num_lanes = len(list_tensor_data['coords.LANE'][sample_idx])
            num_stop_lines = len(list_tensor_data['coords.STOP_LINE'][sample_idx])
            num_crosswalks = len(list_tensor_data['coords.CROSSWALK'][sample_idx])
            num_route_lanes = len(list_tensor_data['coords.ROUTE_LANES'][sample_idx])
            num_polygons = num_lanes + num_stop_lines + num_crosswalks + num_route_lanes
            lane_ids = list(list_tensor_data['coords.LANE'][sample_idx].keys())
            stop_line_ids = list(list_tensor_data['coords.STOP_LINE'][sample_idx].keys())
            crosswalk_ids = list(list_tensor_data['coords.CROSSWALK'][sample_idx].keys())
            route_lane_ids = list(list_tensor_data['coords.ROUTE_LANES'][sample_idx].keys())
            num_lane_points_detail = [coords.shape[0] for coords in list_tensor_data['coords.LANE'][sample_idx].values()]
            num_lane_points = sum(num_lane_points_detail)
            num_stop_line_points_detail = [coords.shape[0] for coords in list_tensor_data['coords.STOP_LINE'][sample_idx].values()]
            num_stop_line_points = sum(num_stop_line_points_detail)
            num_crosswalk_points_detail = [coords.shape[0] for coords in list_tensor_data['coords.CROSSWALK'][sample_idx].values()]
            num_crosswalk_points = sum(num_crosswalk_points_detail)
            num_route_lane_points_detail = [coords.shape[0] for coords in list_tensor_data['coords.ROUTE_LANES'][sample_idx].values()]
            num_route_lane_points = sum(num_route_lane_points_detail)
            num_points = num_lane_points + num_stop_line_points + num_crosswalk_points + num_route_lane_points

            # initialization
            polygon_position = torch.zeros(num_polygons, 2, dtype=torch.float)
            polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
            polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
            polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
            polygon_tl_statuses = torch.zeros(num_polygons, dtype=torch.uint8)
            point_position: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_orientation: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_magnitude: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_type: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_side: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_tl_statuses: Union[List[Optional[torch.Tensor]], torch.Tensor] = [None] * num_polygons
            point_to_polygon_edge_index_1 = []  # point_to_polygon_edge_index[1]
            polygon_to_polygon_edge_index = []
            polygon_to_polygon_type = []

            idx_offset_pl = 0
            idx_offset_pt = 0

            num_pl_detail[sample_idx]['num_lanes'] = num_lanes
            num_pl_detail[sample_idx]['num_stop_lines'] = num_stop_lines
            num_pl_detail[sample_idx]['num_crosswalks'] = num_crosswalks
            num_pl_detail[sample_idx]['num_route_lanes'] = num_route_lanes
            num_pl_detail[sample_idx]['num_polygons'] = num_polygons
            num_pl_to_pl_edge_index_detail[sample_idx] = num_lanes

            # 'LANE'
            for i_element in range(num_lanes):
                num_points_i = num_lane_points_detail[i_element]
                centerline = list_tensor_data['coords.LANE'][sample_idx][lane_ids[i_element]]
                polygon_position[idx_offset_pl + i_element] = centerline[0]
                polygon_orientation[idx_offset_pl + i_element] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                             centerline[1, 0] - centerline[0, 0])
                polygon_type[idx_offset_pl + i_element] = self._polygon_types.index('LANE')
                if list_tensor_data['LANE_TYPE'][sample_idx][i_element] == 'LANE':
                    polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(False)
                elif list_tensor_data['LANE_TYPE'][sample_idx][i_element] == 'LANE_CONNECTOR':
                    polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(True)
                tl_status = list_tensor_data['traffic_light_data.LANE'][sample_idx][lane_ids[i_element]]
                tl_status = tl_status.type(dtype=torch.uint8).numpy()
                tl_status = self._traffic_light_one_hot_decoding[tuple(tl_status)]
                polygon_tl_statuses[idx_offset_pl + i_element] = tl_status

                left_boundary = list_tensor_data['coords.LEFT_BOUNDARY'][sample_idx][lane_ids[i_element]]
                right_boundary = list_tensor_data['coords.RIGHT_BOUNDARY'][sample_idx][lane_ids[i_element]]
                point_position[idx_offset_pl + i_element] = torch.cat([left_boundary[:-1, :],
                                                                       right_boundary[:-1, :],
                                                                       centerline[:-1, :]], dim=0)
                left_vectors = left_boundary[1:] - left_boundary[:-1]
                right_vectors = right_boundary[1:] - right_boundary[:-1]
                center_vectors = centerline[1:] - centerline[:-1]
                point_orientation[idx_offset_pl + i_element] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                                          torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                                          torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                                         dim=0)
                point_magnitude[idx_offset_pl + i_element] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                                   right_vectors[:, :2],
                                                                                   center_vectors[:, :2]], dim=0), p=2, dim=-1)
                left_type = self._point_types.index('LEFT_BOUNDARY')
                right_type = self._point_types.index('RIGHT_BOUNDARY')
                center_type = self._point_types.index('LANE')
                point_type[idx_offset_pl + i_element] = torch.cat(
                    [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                     torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                     torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
                point_side[idx_offset_pl + i_element] = torch.cat(
                    [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                     torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                     torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
                tl_status = torch.full((len(left_vectors),), tl_status, dtype=torch.uint8)
                point_tl_statuses[idx_offset_pl + i_element] = torch.cat([tl_status, tl_status, tl_status], dim=0)

                point_to_polygon_edge_index_1.append(torch.full(((num_points_i - 1) * 3,), idx_offset_pl + i_element, dtype=torch.long))

                pred_ids = list_tensor_data['PREDECESSOR_ID'][sample_idx][i_element]
                pred_indexes = []
                if pred_ids is not None:
                    for pred_id in pred_ids:
                        index = safe_list_index(lane_ids, pred_id)
                        if index is not None:
                            pred_indexes.append(idx_offset_pl + safe_list_index(lane_ids, pred_id))
                if len(pred_indexes) > 0:
                    polygon_to_polygon_edge_index.append(
                        torch.stack([torch.tensor(pred_indexes, dtype=torch.long),
                                     torch.full((len(pred_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                    polygon_to_polygon_type.append(
                        torch.full((len(pred_indexes),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
                succ_ids = list_tensor_data['SUCCESSOR_ID'][sample_idx][i_element]
                succ_indexes = []
                if succ_ids is not None:
                    for succ_id in succ_ids:
                        index = safe_list_index(lane_ids, succ_id)
                        if index is not None:
                            succ_indexes.append(idx_offset_pl + safe_list_index(lane_ids, succ_id))
                if len(succ_indexes) > 0:
                    polygon_to_polygon_edge_index.append(
                        torch.stack([torch.tensor(succ_indexes, dtype=torch.long),
                                     torch.full((len(succ_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                    polygon_to_polygon_type.append(
                        torch.full((len(succ_indexes),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
                left_lane_ids = list_tensor_data['LEFT_NEIGHBOR_LANE_ID'][sample_idx][i_element]
                left_lane_indexes = []
                if left_lane_ids is not None:
                    index = safe_list_index(lane_ids, left_lane_ids)
                    if index is not None:
                        left_lane_indexes.append(idx_offset_pl + safe_list_index(lane_ids, left_lane_ids))
                if len(left_lane_indexes) > 0:
                    polygon_to_polygon_edge_index.append(
                        torch.stack([torch.tensor(left_lane_indexes, dtype=torch.long),
                                     torch.full((len(left_lane_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                    polygon_to_polygon_type.append(
                        torch.full((len(left_lane_indexes),), self._polygon_to_polygon_types.index('LEFT'), dtype=torch.uint8))
                right_lane_ids = list_tensor_data['RIGHT_NEIGHBOR_LANE_ID'][sample_idx][i_element]
                right_lane_indexes = []
                if right_lane_ids is not None:
                    index = safe_list_index(lane_ids, right_lane_ids)
                    if index is not None:
                        right_lane_indexes.append(idx_offset_pl + safe_list_index(lane_ids, right_lane_ids))
                if len(right_lane_indexes) > 0:
                    polygon_to_polygon_edge_index.append(
                        torch.stack([torch.tensor(right_lane_indexes, dtype=torch.long),
                                     torch.full((len(right_lane_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                    polygon_to_polygon_type.append(
                        torch.full((len(right_lane_indexes),), self._polygon_to_polygon_types.index('RIGHT'), dtype=torch.uint8))

                if len(pred_indexes) == 0 and len(succ_indexes) == 0 and len(left_lane_indexes) == 0 and len(right_lane_indexes) == 0:
                    num_pl_to_pl_edge_index_detail[sample_idx] = num_pl_to_pl_edge_index_detail[sample_idx] - 1

                idx_offset_pt = idx_offset_pt + (num_points_i - 1) * 3

            idx_offset_pl = idx_offset_pl + num_lanes

            # 'STOP_LINE'
            for i_element in range(num_stop_lines):
                num_points_i = num_stop_line_points_detail[i_element]
                edge = list_tensor_data['coords.STOP_LINE'][sample_idx][stop_line_ids[i_element]]
                center_position = edge[:-1].mean(dim=0)
                polygon_position[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = center_position
                polygon_type[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = self._polygon_types.index('STOP_LINE')
                polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(None)
                polygon_tl_statuses[idx_offset_pl + i_element] = self._traffic_light_statuses.index('UNKNOWN')

                point_position[idx_offset_pl + i_element] = edge[:-1]
                edge_vectors = edge[1:] - edge[:-1]
                point_orientation[idx_offset_pl + i_element] = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
                point_magnitude[idx_offset_pl + i_element] = torch.norm(edge_vectors[:, :2], p=2, dim=-1)
                edge_type = self._point_types.index('STOP_LINE')
                point_type[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), edge_type, dtype=torch.uint8)
                point_side[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), self._point_sides.index('UNKNOWN'), dtype=torch.uint8)
                point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors),),
                                                                          self._traffic_light_statuses.index('UNKNOWN'),
                                                                          dtype=torch.uint8)

                point_to_polygon_edge_index_1.append(torch.full((num_points_i - 1,), idx_offset_pl + i_element, dtype=torch.long))
                idx_offset_pt = idx_offset_pt + num_points_i - 1
            idx_offset_pl = idx_offset_pl + num_stop_lines

            # 'CROSSWALK'
            for i_element in range(num_crosswalks):
                num_points_i = num_crosswalk_points_detail[i_element]
                edge = list_tensor_data['coords.CROSSWALK'][sample_idx][crosswalk_ids[i_element]]
                center_position = edge[:-1].mean(dim=0)
                polygon_position[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = center_position
                polygon_type[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = self._polygon_types.index('CROSSWALK')
                polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(None)
                polygon_tl_statuses[idx_offset_pl + i_element] = self._traffic_light_statuses.index('UNKNOWN')

                point_position[idx_offset_pl + i_element] = edge[:-1]
                edge_vectors = edge[1:] - edge[:-1]
                point_orientation[idx_offset_pl + i_element] = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
                point_magnitude[idx_offset_pl + i_element] = torch.norm(edge_vectors[:, :2], p=2, dim=-1)
                edge_type = self._point_types.index('CROSSWALK')
                point_type[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), edge_type, dtype=torch.uint8)
                point_side[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), self._point_sides.index('UNKNOWN'), dtype=torch.uint8)
                point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors),),
                                                                          self._traffic_light_statuses.index('UNKNOWN'),
                                                                          dtype=torch.uint8)

                point_to_polygon_edge_index_1.append(torch.full((num_points_i - 1,), idx_offset_pl + i_element, dtype=torch.long))
                idx_offset_pt = idx_offset_pt + num_points_i - 1
            idx_offset_pl = idx_offset_pl + num_crosswalks

            # 'ROUTE_LANES'
            for i_element in range(num_route_lanes):
                num_points_i = num_route_lane_points_detail[i_element]
                edge = list_tensor_data['coords.ROUTE_LANES'][sample_idx][route_lane_ids[i_element]]
                start_position = edge[0]
                polygon_position[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = start_position
                polygon_type[idx_offset_pl + i_element:idx_offset_pl + i_element + 1] = self._polygon_types.index('ROUTE_LANES')
                polygon_is_intersection[idx_offset_pl + i_element] = self._polygon_is_intersections.index(None)
                polygon_tl_statuses[idx_offset_pl + i_element] = self._traffic_light_statuses.index('UNKNOWN')

                point_position[idx_offset_pl + i_element] = edge[:-1]
                edge_vectors = edge[1:] - edge[:-1]
                point_orientation[idx_offset_pl + i_element] = torch.atan2(edge_vectors[:, 1], edge_vectors[:, 0])
                point_magnitude[idx_offset_pl + i_element] = torch.norm(edge_vectors[:, :2], p=2, dim=-1)
                edge_type = self._point_types.index('ROUTE_LANES')
                point_type[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), edge_type, dtype=torch.uint8)
                point_side[idx_offset_pl + i_element] = torch.full((len(edge_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)
                point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors),),
                                                                          self._traffic_light_statuses.index('UNKNOWN'),
                                                                          dtype=torch.uint8)

                point_to_polygon_edge_index_1.append(torch.full((num_points_i - 1,), idx_offset_pl + i_element, dtype=torch.long))
                idx_offset_pt = idx_offset_pt + num_points_i - 1

            idx_offset_pl = idx_offset_pl + num_route_lanes

            num_points = idx_offset_pt
            point_to_polygon_edge_index_0 = torch.arange(num_points, dtype=torch.long)
            point_to_polygon_edge_index_1 = torch.cat(point_to_polygon_edge_index_1, dim=0)
            point_to_polygon_edge_index = torch.stack([point_to_polygon_edge_index_0, point_to_polygon_edge_index_1], dim=0)

            if len(polygon_to_polygon_edge_index) == 0:
                idx_offset_pl = 0
                all_indexes = list(range(num_lanes))
                for i_element in range(num_lanes):
                    rest_indexes = all_indexes.copy()
                    rest_indexes.remove(i_element)
                    rest_indexes = [index + idx_offset_pl for index in rest_indexes]
                    polygon_to_polygon_edge_index.append(
                        torch.stack([torch.tensor(rest_indexes, dtype=torch.long),
                                     torch.full((len(rest_indexes),), idx_offset_pl + i_element, dtype=torch.long)], dim=0))
                    polygon_to_polygon_type.append(
                        torch.full((len(rest_indexes),), self._polygon_to_polygon_types.index('NONE'), dtype=torch.uint8))

                num_pl_to_pl_edge_index_detail[sample_idx] = num_lanes

                idx_offset_pl = idx_offset_pl + num_lanes
                idx_offset_pl = idx_offset_pl + num_stop_lines
                idx_offset_pl = idx_offset_pl + num_crosswalks
                idx_offset_pl = idx_offset_pl + num_route_lanes

            list_num_polygons.append(num_polygons)
            list_polygon_position.append(polygon_position.detach().numpy())
            list_polygon_orientation.append(polygon_orientation.detach().numpy())
            list_polygon_type.append(polygon_type.detach().numpy())
            list_polygon_is_intersection.append(polygon_is_intersection.detach().numpy())
            list_polygon_tl_statuses.append(polygon_tl_statuses.detach().numpy())
            list_num_points.append(num_points)
            list_point_position.append(torch.cat(point_position, dim=0).detach().numpy())
            list_point_orientation.append(torch.cat(point_orientation, dim=0).detach().numpy())
            list_point_magnitude.append(torch.cat(point_magnitude, dim=0).detach().numpy())
            list_point_type.append(torch.cat(point_type, dim=0).detach().numpy())
            list_point_side.append(torch.cat(point_side, dim=0).detach().numpy())
            list_point_tl_statuses.append(torch.cat(point_tl_statuses, dim=0).detach().numpy())
            list_point_to_polygon_edge_index.append(point_to_polygon_edge_index.detach().numpy())
            list_polygon_to_polygon_edge_index.append(torch.cat(polygon_to_polygon_edge_index, dim=1).detach().numpy())
            list_polygon_to_polygon_type.append(torch.cat(polygon_to_polygon_type, dim=0).detach().numpy())

        # num_polygons = sum(list_num_polygons)
        # polygon_position = torch.cat(list_polygon_position, dim=0)
        # polygon_orientation = torch.cat(list_polygon_orientation)
        # polygon_type = torch.cat(list_polygon_type)
        # polygon_is_intersection = torch.cat(list_polygon_is_intersection)
        # polygon_tl_statuses = torch.cat(list_polygon_tl_statuses)
        # num_points = sum(list_num_points)
        # point_position = torch.cat([torch.cat(point_pos, dim=0) for point_pos in list_point_position], dim=0).float()
        # point_orientation = torch.cat([torch.cat(point_ori) for point_ori in list_point_orientation]).float()
        # point_magnitude = torch.cat([torch.cat(point_mag) for point_mag in list_point_magnitude]).float()
        # point_type = torch.cat([torch.cat(point_type) for point_type in list_point_type])
        # point_side = torch.cat([torch.cat(point_side) for point_side in list_point_side])
        # list_polygon_to_polygon_edge_index = [torch.cat(index, dim=1) for index in list_polygon_to_polygon_edge_index]
        # for sample_idx in range(batch_size):
        #     if sample_idx == 0:
        #         continue
        #     else:
        #         list_point_to_polygon_edge_index[sample_idx][0, :] \
        #             = list_point_to_polygon_edge_index[sample_idx][0, :] + list_point_to_polygon_edge_index[sample_idx - 1][0, -1] + 1
        #         list_point_to_polygon_edge_index[sample_idx][1, :] \
        #             = list_point_to_polygon_edge_index[sample_idx][1, :] + list_point_to_polygon_edge_index[sample_idx - 1][1, -1] + 1
        #         list_polygon_to_polygon_edge_index[sample_idx] \
        #             = list_polygon_to_polygon_edge_index[sample_idx] + sum(list_num_polygons[:sample_idx])
        # point_to_polygon_edge_index = torch.cat(list_point_to_polygon_edge_index, dim=1)
        # polygon_to_polygon_edge_index = torch.cat(list_polygon_to_polygon_edge_index, dim=1)
        # polygon_to_polygon_type = torch.cat([torch.cat(index) for index in list_polygon_to_polygon_type])

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
            'num_pl_detail': [detail for detail in num_pl_detail.values()],
            'num_pl_to_pl_edge_index_detail': [detail for detail in num_pl_to_pl_edge_index_detail.values()]
        }
        map_data['map_polygon']['num_nodes'] = list_num_polygons
        map_data['map_polygon']['position'] = list_polygon_position
        map_data['map_polygon']['orientation'] = list_polygon_orientation
        map_data['map_polygon']['type'] = list_polygon_type
        map_data['map_polygon']['is_intersection'] = list_polygon_is_intersection
        map_data['map_polygon']['tl_statuses'] = list_polygon_tl_statuses
        map_data['map_point']['num_nodes'] = list_num_points
        map_data['map_point']['position'] = list_point_position
        map_data['map_point']['orientation'] = list_point_orientation
        map_data['map_point']['magnitude'] = list_point_magnitude
        map_data['map_point']['type'] = list_point_type
        map_data['map_point']['side'] = list_point_side
        map_data['map_point']['tl_statuses'] = list_point_tl_statuses
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = list_point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = list_polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = list_polygon_to_polygon_type

        return map_data

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Implemented. See Interface.
        """
        empty: Dict[str, str] = {}
        max_elements: List[str] = [
            f"{feature_name}.{feature_max_elements}" for feature_name, feature_max_elements in self.max_elements.items()
        ]
        max_points: List[str] = [
            f"{feature_name}.{feature_max_points}" for feature_name, feature_max_points in self.max_points.items()
        ]

        return {
            "neighbor_vector_set_map": {
                "radius": str(self.radius),
                "interpolation_method": self.interpolation_method,
                "map_features": ",".join(self.map_features),
                "max_elements": ",".join(max_elements),
                "max_points": ",".join(max_points),
            },
            "initial_ego_state": empty,
        }
