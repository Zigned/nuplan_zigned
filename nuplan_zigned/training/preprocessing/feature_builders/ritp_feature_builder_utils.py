from __future__ import annotations

from typing import List, Tuple, Dict, Set, cast, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
from _datetime import datetime
from shapely import Point
from dataclasses import dataclass

from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
LaneSegmentRoadBlockIDs,
prune_route_by_connectivity,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
EgoInternalIndex,
)
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatuses,
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.nuplan_map.lane_connector import NuPlanLaneConnector
from nuplan.common.maps.nuplan_map.utils import (
    get_distance_between_map_object_and_point,
    extract_polygon_from_map_object,
)
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint, StateSE2, StateVector2D, Point2D
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    VectorFeatureLayer,
    VectorFeatureLayerMapping,
    LaneSegmentLaneIDs,
)


@dataclass
class MapObjectPolylines:
    """
    Collection of map object polylines, each represented as a list of x, y coords
    [num_elements, num_points_in_element (variable size), 2].
    """

    polylines: Dict[str, List[Point2D]]  # lane_id: List[Point2D]

    def to_vector(self) -> List[List[List[float]]]:
        """
        Returns data in vectorized form
        :return: vectorized coords of map object polylines as [num_elements, num_points_in_element (variable size), 2].
        """
        return [[[coord.x, coord.y] for coord in polygon] for polygon in self.polylines.values()]


@dataclass
class LaneSegmentTrafficLightData:
    """
    Traffic light data represented as one-hot encoding per segment [num_lane_segment, 4].
    The one-hot encoding: green [1, 0, 0, 0], yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1].
    """

    traffic_lights: Dict[str, Tuple[int, int, int, int]]

    _one_hot_encoding = {
        TrafficLightStatusType.GREEN: (1, 0, 0, 0),
        TrafficLightStatusType.YELLOW: (0, 1, 0, 0),
        TrafficLightStatusType.RED: (0, 0, 1, 0),
        TrafficLightStatusType.UNKNOWN: (0, 0, 0, 1),
    }
    _encoding_dim: int = 4

    def to_vector(self) -> List[List[float]]:
        """
        Returns data in vectorized form.
        :return: vectorized traffic light data per segment as [num_lane_segment, 4].
        """
        return [list(data) for data in self.traffic_lights.values()]

    @classmethod
    def encode(cls, traffic_light_type: TrafficLightStatusType) -> Tuple[int, int, int, int]:
        """
        One-hot encoding of TrafficLightStatusType: green [1, 0, 0, 0], yellow [0, 1, 0, 0], red [0, 0, 1, 0],
            unknown [0, 0, 0, 1].
        """
        return cls._one_hot_encoding[traffic_light_type]

    @classmethod
    def encoding_dim(cls) -> int:
        """
        Dimensionality of associated data encoding.
        :return: encoding dimensionality.
        """
        return cls._encoding_dim


def extract_proximal_roadblock_objects(map_api: AbstractMap, point: Point2D, radius: float) -> List[RoadBlockGraphEdgeMapObject]:
    """
    Extract roadblocks/roadblock connectors objects within the given radius around the point x, y.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius [m] floating number about vector map query range.
    :return List of roadblocks/roadblock connectors.
    """
    layer_names = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)
    roadblock = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.ROADBLOCK])
    if len(roadblock[SemanticMapLayer.ROADBLOCK]) > 0:

        return roadblock[SemanticMapLayer.ROADBLOCK]
    else:
        roadblock_conns = map_api.get_proximal_map_objects(point, radius, [SemanticMapLayer.ROADBLOCK_CONNECTOR])

        return cast(List[RoadBlockGraphEdgeMapObject], roadblock_conns[SemanticMapLayer.ROADBLOCK_CONNECTOR])


def get_on_route_indices(
    route_roadblock_ids: List[str], roadblock_ids: LaneSegmentRoadBlockIDs
) -> np.ndarray:
    """
    Identify whether given lane segments lie within goal route.
    :param route_roadblock_ids: List of ids of roadblocks (lane groups) within goal route.
    :param roadblock_ids: Roadblock ids (lane group associations) pertaining to associated lane segments.
    :return on_route_indices: indices of input roadblock id that lie within goal route.
    """
    if route_roadblock_ids:
        # prune route to extracted roadblocks maintaining connectivity
        route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, set(roadblock_ids.roadblock_ids))

        # Get indices of the segments that lie on the route
        on_route_indices = np.arange(len(roadblock_ids.roadblock_ids))[
            np.in1d(roadblock_ids.roadblock_ids, route_roadblock_ids)
        ]

    else:
        # set on route indices to None if no route available
        on_route_indices = None

    return on_route_indices


def future_ego_states_to_tensor(future_ego_states: List[EgoState]) -> torch.Tensor:
    """
    Converts a list of N ego states into a N x 8 tensor. The 8 fields are as defined in `AgentFeatureIndex`
    :param future_ego_states: The ego states to convert.
    :return: The converted tensor.
    """
    # output = torch.zeros((len(future_ego_states), EgoInternalIndex.dim()), dtype=torch.float32)
    # for i in range(0, len(future_ego_states), 1):
    #     # output[i, EgoInternalIndex.x()] = future_ego_states[i].rear_axle.x
    #     # output[i, EgoInternalIndex.y()] = future_ego_states[i].rear_axle.y
    #     # output[i, EgoInternalIndex.heading()] = future_ego_states[i].rear_axle.heading
    #     # output[i, EgoInternalIndex.vx()] = future_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x
    #     # output[i, EgoInternalIndex.vy()] = future_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y
    #     # output[i, EgoInternalIndex.ax()] = future_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x
    #     # output[i, EgoInternalIndex.ay()] = future_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y
    #
    # return output

    # output = torch.zeros((len(future_ego_states), AgentFeatureIndex.dim()), dtype=torch.float32)
    # for i in range(0, len(future_ego_states), 1):
    #     output[i, AgentFeatureIndex.x()] = future_ego_states[i].rear_axle.x
    #     output[i, AgentFeatureIndex.y()] = future_ego_states[i].rear_axle.y
    #     output[i, AgentFeatureIndex.heading()] = future_ego_states[i].rear_axle.heading
    #     output[i, AgentFeatureIndex.vx()] = future_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x
    #     output[i, AgentFeatureIndex.vy()] = future_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y
    #     output[i, AgentFeatureIndex.yaw_rate()] = future_ego_states[i].dynamic_car_state.angular_velocity
    #     output[i, AgentFeatureIndex.length()] = future_ego_states[i].car_footprint.vehicle_parameters.length
    #     output[i, AgentFeatureIndex.width()] = future_ego_states[i].car_footprint.vehicle_parameters.width

    output = torch.zeros((len(future_ego_states), AgentFeatureIndex.dim()), dtype=torch.float32)
    for i in range(0, len(future_ego_states), 1):
        output[i, AgentFeatureIndex.x()] = future_ego_states[i].agent.center.x
        output[i, AgentFeatureIndex.y()] = future_ego_states[i].agent.center.y
        output[i, AgentFeatureIndex.heading()] = future_ego_states[i].agent.center.heading
        # note that future_ego_states[i].agent.velocity is in the local frame (vehicle coordinate system)
        output[i, AgentFeatureIndex.vx()] = (future_ego_states[i].agent.velocity.x * np.cos(future_ego_states[i].agent.center.heading)
                                             - future_ego_states[i].agent.velocity.y * np.sin(future_ego_states[i].agent.center.heading))
        output[i, AgentFeatureIndex.vy()] = (future_ego_states[i].agent.velocity.x * np.sin(future_ego_states[i].agent.center.heading)
                                             + future_ego_states[i].agent.velocity.y * np.cos(future_ego_states[i].agent.center.heading))
        output[i, AgentFeatureIndex.yaw_rate()] = future_ego_states[i].dynamic_car_state.angular_velocity
        output[i, AgentFeatureIndex.length()] = future_ego_states[i].agent.box.length
        output[i, AgentFeatureIndex.width()] = future_ego_states[i].agent.box.width

    return output


def future_timestamps_to_tensor(future_time_stamps: List[TimePoint]) -> torch.Tensor:
    """
    Converts a list of N past timestamps into a 1-d tensor of shape [N]. The field is the timestamp in uS.
    :param future_time_stamps: The time stamps to convert.
    :return: The converted tensor.
    """
    flat = [t.time_us for t in future_time_stamps]
    return torch.tensor(flat, dtype=torch.int64)


def get_sampled_future_ego_states(
        trajectory_samples_cartesian: Dict[str, np.ndarray],
        future_ego_states: List[EgoState],
) -> List[List[EgoState]]:
    """
    Generate sampled future ego states
    :param trajectory_samples_cartesian
    :param geo_center_trajectory_samples_cartesian
    :param future_ego_states
    :return: sampled_future_ego_states
    """
    vehicle_parameters = future_ego_states[0].car_footprint.vehicle_parameters
    sampled_future_ego_states = []
    for i_traj in range(trajectory_samples_cartesian['pose_cartesian'].shape[0]):
        rear_axle_velocity_x = trajectory_samples_cartesian['rear_axle_velocity_x'][i_traj, 1:]
        rear_axle_velocity_y = trajectory_samples_cartesian['rear_axle_velocity_y'][i_traj, 1:]
        rear_axle_acceleration_x = trajectory_samples_cartesian['rear_axle_acceleration_x'][i_traj, 1:]
        rear_axle_acceleration_y = trajectory_samples_cartesian['rear_axle_acceleration_y'][i_traj, 1:]
        angular_velocity = trajectory_samples_cartesian['angular_velocity'][i_traj, 1:]
        geo_center_traj = trajectory_samples_cartesian['geo_center_pose_cartesian'][i_traj, :, 1:]
        trajectory_states = []
        for t in range(trajectory_samples_cartesian['pose_cartesian'][:, :, 1:].shape[2]):
            car_footprint = CarFootprint(StateSE2(geo_center_traj[0, t],
                                                  geo_center_traj[1, t],
                                                  geo_center_traj[2, t]),
                                         vehicle_parameters)
            dynamic_car_state = DynamicCarState(
                vehicle_parameters.rear_axle_to_center,
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
            trajectory_states.append(ego_state)
        sampled_future_ego_states.append(trajectory_states)

    return sampled_future_ego_states


def get_lane_polylines(
    map_api: AbstractMap, point: Point2D, radius: float
) -> Tuple[MapObjectPolylines, MapObjectPolylines, MapObjectPolylines, LaneSegmentLaneIDs,
    List[str],
    # List[Optional[List[Point2D]]], List[Optional[List[Point2D]]], List[Optional[List[Point2D]]],
    List[Optional[str]],
    # List[Optional[List[Point2D]]], List[Optional[List[Point2D]]], List[Optional[List[Point2D]]],
    List[Optional[str]],
    List[Optional[List[str]]], List[Optional[List[str]]]]:
    """
    Extract ids, baseline path polylines, and boundary polylines of neighbor lanes and lane connectors around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :return:
        lanes_mid: extracted lane/lane connector baseline polylines.
        lanes_left: extracted lane/lane connector left boundary polylines.
        lanes_right: extracted lane/lane connector right boundary polylines.
        lane_ids: ids of lanes/lane connector associated polylines were extracted from.
    """
    lanes_mid: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_left: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_right: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lane_ids: List[str] = []  # shape: [num_lanes]
    lane_types: List[str] = []  # shape: [num_lanes]
    # left_neighbor_lanes_mid: List[Optional[List[Point2D]]] = []
    # left_neighbor_lanes_left: List[Optional[List[Point2D]]] = []
    # left_neighbor_lanes_right: List[Optional[List[Point2D]]] = []
    left_neighbor_lane_ids: List[Optional[str]] = []
    # right_neighbor_lanes_mid: List[Optional[List[Point2D]]] = []
    # right_neighbor_lanes_left: List[Optional[List[Point2D]]] = []
    # right_neighbor_lanes_right: List[Optional[List[Point2D]]] = []
    right_neighbor_lane_ids: List[Optional[str]] = []
    predecessor_ids: List[Optional[List[str]]] = []
    successor_ids: List[Optional[List[str]]] = []
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)

    map_objects: List[MapObject] = []

    for layer_name in layer_names:
        map_objects += layers[layer_name]
    # sort by distance to query point
    map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))

    for map_obj in map_objects:
        # center lane
        baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path]
        lanes_mid.append(baseline_path_polyline)
        # boundaries
        lanes_left.append([Point2D(node.x, node.y) for node in map_obj.left_boundary.discrete_path])
        lanes_right.append([Point2D(node.x, node.y) for node in map_obj.right_boundary.discrete_path])
        # lane ids
        lane_ids.append(map_obj.id)
        # lane types
        if isinstance(map_obj, NuPlanLane):
            lane_types.append(SemanticMapLayer.LANE.name)
        elif isinstance(map_obj, NuPlanLaneConnector):
            lane_types.append(SemanticMapLayer.LANE_CONNECTOR.name)

        # left beighbor
        if map_obj.adjacent_edges[0] is not None:
            # # center of left neighbor lane
            # baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.adjacent_edges[0].baseline_path.discrete_path]
            # left_neighbor_lanes_mid.append(baseline_path_polyline)
            # # boundaries of left neighbor lane
            # left_neighbor_lanes_left.append([Point2D(node.x, node.y) for node in map_obj.adjacent_edges[0].left_boundary.discrete_path])
            # left_neighbor_lanes_right.append([Point2D(node.x, node.y) for node in map_obj.adjacent_edges[0].right_boundary.discrete_path])
            # lane ids of left neighbor lane
            left_neighbor_lane_ids.append(map_obj.adjacent_edges[0].id)
        else:
            # left_neighbor_lanes_mid.append(None)
            # left_neighbor_lanes_left.append(None)
            # left_neighbor_lanes_right.append(None)
            left_neighbor_lane_ids.append(None)

        # right beighbor
        if map_obj.adjacent_edges[1] is not None:
            # # center of right neighbor lane
            # baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.adjacent_edges[1].baseline_path.discrete_path]
            # right_neighbor_lanes_mid.append(baseline_path_polyline)
            # # boundaries of right neighbor lane
            # right_neighbor_lanes_left.append([Point2D(node.x, node.y) for node in map_obj.adjacent_edges[1].left_boundary.discrete_path])
            # right_neighbor_lanes_right.append([Point2D(node.x, node.y) for node in map_obj.adjacent_edges[1].right_boundary.discrete_path])
            # lane ids of right neighbor lane
            right_neighbor_lane_ids.append(map_obj.adjacent_edges[1].id)
        else:
            # right_neighbor_lanes_mid.append(None)
            # right_neighbor_lanes_left.append(None)
            # right_neighbor_lanes_right.append(None)
            right_neighbor_lane_ids.append(None)

        # predecessor
        if len(map_obj.incoming_edges) > 0:
            predecessor_ids.append([obj.id for obj in map_obj.incoming_edges])
        else:
            predecessor_ids.append(None)

        # successor
        if len(map_obj.outgoing_edges) > 0:
            successor_ids.append([obj.id for obj in map_obj.outgoing_edges])
        else:
            successor_ids.append(None)

    return (
        MapObjectPolylines({lane_id: lane_mid for lane_mid, lane_id in zip(lanes_mid, lane_ids)}),
        MapObjectPolylines({lane_id: lane_left for lane_left, lane_id in zip(lanes_left, lane_ids)}),
        MapObjectPolylines({lane_id: lane_right for lane_right, lane_id in zip(lanes_right, lane_ids)}),
        LaneSegmentLaneIDs(lane_ids),
        lane_types,
        # left_neighbor_lanes_mid,
        # left_neighbor_lanes_left,
        # left_neighbor_lanes_right,
        left_neighbor_lane_ids,
        # right_neighbor_lanes_mid,
        # right_neighbor_lanes_left,
        # right_neighbor_lanes_right,
        right_neighbor_lane_ids,
        predecessor_ids,
        successor_ids,
    )


def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    route_roadblock_ids: List[str],
    traffic_light_statuses_over_time: List[TrafficLightStatuses],
    timestamps: List[TimePoint]
) -> Tuple[Dict[str, MapObjectPolylines], Dict[int, Dict[str, LaneSegmentTrafficLightData]]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param route_roadblock_ids: List of ids of roadblocks/roadblock connectors (lane groups) within goal route.
    :param traffic_light_statuses_over_time: A list of available traffic light statuses data, indexed by time step.
    :param timestamps: A list of timestamps.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, Union[MapObjectPolylines, List]] = {}
    feature_layers: List[VectorFeatureLayer] = []
    traffic_light_data_over_time: Dict[int, Dict[str, LaneSegmentTrafficLightData]] = {}

    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(f"Object representation for layer: {feature_name} is unavailable")

    # extract lanes and traffic light
    if VectorFeatureLayer.LANE in feature_layers:
        (lanes_mid, lanes_left, lanes_right, lane_ids,
         lane_types,
         # left_neighbor_lanes_mid, left_neighbor_lanes_left, left_neighbor_lanes_right,
         left_neighbor_lane_ids,
         # right_neighbor_lanes_mid, right_neighbor_lanes_left, right_neighbor_lanes_right,
         right_neighbor_lane_ids,
         predecessor_ids, successor_ids)\
            = get_lane_polylines(map_api, point, radius)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)

        # lane types
        coords['LANE_TYPE'] = lane_types

        # left neighbor lane baseline paths
        # coords['LEFT_NEIGHBOR_LANE'] = left_neighbor_lanes_mid
        # coords['LEFT_NEIGHBOR_LANE_LEFT_BOUNDARY'] = left_neighbor_lanes_left
        # coords['LEFT_NEIGHBOR_LANE_RIGHT_BOUNDARY'] = left_neighbor_lanes_right
        coords['LEFT_NEIGHBOR_LANE_ID'] = left_neighbor_lane_ids

        # right neighbor lane baseline paths
        # coords['RIGHT_NEIGHBOR_LANE'] = right_neighbor_lanes_mid
        # coords['RIGHT_NEIGHBOR_LANE_LEFT_BOUNDARY'] = right_neighbor_lanes_left
        # coords['RIGHT_NEIGHBOR_LANE_RIGHT_BOUNDARY'] = right_neighbor_lanes_right
        coords['RIGHT_NEIGHBOR_LANE_ID'] = right_neighbor_lane_ids

        # predecessor and successor ids
        coords['PREDECESSOR_ID'] = predecessor_ids
        coords['SUCCESSOR_ID'] = successor_ids

        # extract traffic light
        for traffic_lights, timestamp in zip(traffic_light_statuses_over_time, timestamps):
            # lane traffic light data
            traffic_light_data_at_t: Dict[str, LaneSegmentTrafficLightData] = {}
            traffic_light_data_at_t[VectorFeatureLayer.LANE.name] = (
                get_traffic_light_encoding(lane_ids, traffic_lights.traffic_lights)
            )
            traffic_light_data_over_time[timestamp.time_us] = traffic_light_data_at_t

    # extract route
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(map_api, point, radius, route_roadblock_ids)
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines

    # extract generic map objects
    for feature_layer in feature_layers:

        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(
                map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer)
            )
            coords[feature_layer.name] = polygons

    return coords, traffic_light_data_over_time


def get_traffic_light_encoding(
    lane_seg_ids: LaneSegmentLaneIDs, traffic_light_data: List[TrafficLightStatusData]
) -> LaneSegmentTrafficLightData:
    """
    Encode the lane segments with traffic light data.
    :param lane_seg_ids: The lane_segment ids [num_lane_segment].
    :param traffic_light_data: A list of all available data at the current time step.
    :returns: Encoded traffic light data per segment.
    """
    # Initialize with all segment labels with UNKNOWN status
    traffic_light_encoding = np.full(
        (len(lane_seg_ids.lane_ids), len(TrafficLightStatusType)),
        LaneSegmentTrafficLightData.encode(TrafficLightStatusType.UNKNOWN),
    )

    # Extract ids of red and green lane connectors
    green_lane_connectors = [
        str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.GREEN
    ]
    red_lane_connectors = [
        str(data.lane_connector_id) for data in traffic_light_data if data.status == TrafficLightStatusType.RED
    ]

    # Assign segments with corresponding traffic light status
    for tl_id in green_lane_connectors:
        indices = np.argwhere(np.array(lane_seg_ids.lane_ids) == tl_id)
        traffic_light_encoding[indices] = LaneSegmentTrafficLightData.encode(TrafficLightStatusType.GREEN)

    for tl_id in red_lane_connectors:
        indices = np.argwhere(np.array(lane_seg_ids.lane_ids) == tl_id)
        traffic_light_encoding[indices] = LaneSegmentTrafficLightData.encode(TrafficLightStatusType.RED)

    traffic_light_encodings = list(map(tuple, traffic_light_encoding))
    return LaneSegmentTrafficLightData(
        {lane_id: tl_encoding for lane_id, tl_encoding in zip(lane_seg_ids.lane_ids, traffic_light_encodings)}
    )  # type: ignore


def get_route_lane_polylines_from_roadblock_ids(
    map_api: AbstractMap, point: Point2D, radius: float, route_roadblock_ids: List[str]
) -> MapObjectPolylines:
    """
    Extract route polylines from map for route specified by list of roadblock ids. Route is represented as collection of
        baseline polylines of all children lane/lane connectors or roadblock/roadblock connectors encompassing route.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param route_roadblock_ids: ids of roadblocks/roadblock connectors specifying route.
    :return: A route as sequence of lane/lane connector polylines.
    """
    route_lane_polylines: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    map_objects = []

    # extract roadblocks/connectors within query radius to limit route consideration
    layer_names = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)
    roadblock_ids: Set[str] = set()

    for layer_name in layer_names:
        roadblock_ids = roadblock_ids.union({map_object.id for map_object in layers[layer_name]})
    # prune route by connected roadblocks within query radius
    route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, roadblock_ids)

    for route_roadblock_id in route_roadblock_ids:
        # roadblock
        roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK)

        # roadblock connector
        if not roadblock_obj:
            roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)

        # represent roadblock/connector by interior lanes/connectors
        if roadblock_obj:
            map_objects += roadblock_obj.interior_edges

    # sort by distance to query point
    map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))

    for map_obj in map_objects:
        baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path]
        route_lane_polylines.append(baseline_path_polyline)

    return MapObjectPolylines({map_obj.id: polyline for polyline, map_obj in zip(route_lane_polylines, map_objects)})


def get_map_object_polygons(
    map_api: AbstractMap, point: Point2D, radius: float, layer_name: SemanticMapLayer
) -> MapObjectPolylines:
    """
    Extract polygons of neighbor map object around ego vehicle for specified semantic layers.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param layer_name: semantic layer to query.
    :return extracted map object polygons.
    """
    map_objects = map_api.get_proximal_map_objects(point, radius, [layer_name])[layer_name]
    # sort by distance to query point
    map_objects.sort(key=lambda map_obj: get_distance_between_map_object_and_point(point, map_obj))
    polygons = [extract_polygon_from_map_object(map_obj) for map_obj in map_objects]

    return MapObjectPolylines({map_obj.id: polygon for polygon, map_obj in zip(polygons, map_objects)})


def to_tensor(data) -> torch.Tensor:
    """
    Convert data to tensor
    :param data which is either int, numpy or Tensor
    :return torch.Tensor
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float()
    elif isinstance(data, int):
        # return torch.tensor(data)
        return data
    else:
        raise ValueError(f"Unknown type: {type(data)}")
