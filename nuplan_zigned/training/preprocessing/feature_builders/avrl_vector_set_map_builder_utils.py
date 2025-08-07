from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Set, Tuple, Any, Union

import os
import numpy as np
import shapely.geometry as geom
from shapely.geometry.point import Point
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatuses,
    TrafficLightStatusType,
)
from nuplan.common.maps.nuplan_map.utils import (
    build_lane_segments_from_blps_with_trim,
    connect_trimmed_lane_conn_predecessor,
    connect_trimmed_lane_conn_successor,
    extract_polygon_from_map_object,
    get_distance_between_map_object_and_point,
)
from nuplan.common.maps.nuplan_map.roadblock import NuPlanRoadBlock
from nuplan.common.maps.nuplan_map.utils import (
    extract_roadblock_objects,
    compute_curvature,
)

from nuplan_zigned.training.preprocessing.feature_builders.avrl_feature_builder_utils import (
    get_on_route_indices,
    get_roadblock_successors_given_route,
    extract_proximal_roadblock_objects,
)
from nuplan_zigned.utils.utils import (
    point_to_point_distance,
)



class OnRouteStatusType(IntEnum):
    """
    Enum for OnRouteStatusType.
    """

    OFF_ROUTE = 0
    ON_ROUTE = 1
    UNKNOWN = 2


class VectorFeatureLayer(IntEnum):
    """
    Enum for VectorFeatureLayer.
    """

    LANE = 0
    LEFT_BOUNDARY = 1
    RIGHT_BOUNDARY = 2
    STOP_LINE = 3
    CROSSWALK = 4
    ROUTE_LANES = 5

    @classmethod
    def deserialize(cls, layer: str) -> VectorFeatureLayer:
        """Deserialize the type when loading from a string."""
        return VectorFeatureLayer.__members__[layer]


@dataclass
class VectorFeatureLayerMapping:
    """
    Dataclass for associating VectorFeatureLayers with SemanticMapLayers for extracting map object polygons.
    """

    _semantic_map_layer_mapping = {
        VectorFeatureLayer.STOP_LINE: SemanticMapLayer.STOP_LINE,
        VectorFeatureLayer.CROSSWALK: SemanticMapLayer.CROSSWALK,
    }

    @classmethod
    def available_polygon_layers(cls) -> List[VectorFeatureLayer]:
        """
        List of VectorFeatureLayer for which mapping is supported.
        :return List of available layers.
        """
        return list(cls._semantic_map_layer_mapping.keys())

    @classmethod
    def semantic_map_layer(cls, feature_layer: VectorFeatureLayer) -> SemanticMapLayer:
        """
        Returns associated SemanticMapLayer for feature extraction, if exists.
        :param feature_layer: specified VectorFeatureLayer to look up.
        :return associated SemanticMapLayer.
        """
        return cls._semantic_map_layer_mapping[feature_layer]


@dataclass
class LaneOnRouteStatusData:
    """
    Route following status data represented as binary encoding per lane segment [num_lane_segment, 2].
    The binary encoding: off route [0, 1], on route [1, 0], unknown [0, 0].
    """

    on_route_status: List[Tuple[int, int]]

    _binary_encoding = {
        OnRouteStatusType.OFF_ROUTE: (0, 1),
        OnRouteStatusType.ON_ROUTE: (1, 0),
        OnRouteStatusType.UNKNOWN: (0, 0),
    }
    _encoding_dim: int = 2

    def to_vector(self) -> List[List[float]]:
        """
        Returns data in vectorized form.
        :return: vectorized on route status data per lane segment as [num_lane_segment, 2].
        """
        return [list(data) for data in self.on_route_status]

    @classmethod
    def encode(cls, on_route_status_type: OnRouteStatusType) -> Tuple[int, int]:
        """
        Binary encoding of OnRouteStatusType: off route [0, 0], on route [0, 1], unknown [1, 0].
        """
        return cls._binary_encoding[on_route_status_type]

    @classmethod
    def encoding_dim(cls) -> int:
        """
        Dimensionality of associated data encoding.
        :return: encoding dimensionality.
        """
        return cls._encoding_dim


@dataclass
class LaneSegmentCoords:
    """
    Lane-segment coordinates in format of [N, 2, 2] representing [num_lane_segment, [start coords, end coords]].
    """

    coords: List[Tuple[Point2D, Point2D]]

    def to_vector(self) -> List[List[List[float]]]:
        """
        Returns data in vectorized form.
        :return: vectorized lane segment coordinates in [num_lane_segment, 2, 2].
        """
        return [[[start.x, start.y], [end.x, end.y]] for start, end in self.coords]


@dataclass
class LaneSegmentConnections:
    """
    Lane-segment connection relations in format of [num_connection, 2] and each column in the array is
    (from_lane_segment_idx, to_lane_segment_idx).
    """

    connections: List[Tuple[int, int]]

    def to_vector(self) -> List[List[int]]:
        """
        Returns data in vectorized form.
        :return: vectorized lane segment connections as [num_lane_segment, 2, 2].
        """
        return [[start, end] for start, end in self.connections]


@dataclass
class LaneSegmentGroupings:
    """
    Lane-segment groupings in format of [num_lane, num_segment_in_lane (variable size)]
    containing a list of indices of lane segments in corresponding coords list for each lane.
    """

    groupings: List[List[int]]

    def to_vector(self) -> List[List[int]]:
        """
        Returns data in vectorized form.
        :return: vectorized groupings of lane segments as [num_lane, num_lane_segment_in_lane].
        """
        return [[segment_id for segment_id in grouping] for grouping in self.groupings]


@dataclass
class LaneSegmentLaneIDs:
    """
    IDs of lane/lane connectors that lane segment at specified index belong to.
    """

    lane_ids: List[str]


@dataclass
class LaneSegmentRoadBlockIDs:
    """
    IDs of roadblock/roadblock connectors that lane segment at specified index belong to.
    """

    roadblock_ids: List[str]


@dataclass
class LaneSegmentTrafficLightData:
    """
    Traffic light data represented as one-hot encoding per segment [num_lane_segment, 4].
    The one-hot encoding: green [1, 0, 0, 0], yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1].
    """

    traffic_lights: List[Tuple[int, int, int, int]]

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
        return [list(data) for data in self.traffic_lights]

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


@dataclass
class MapObjectPolylines:
    """
    Collection of map object polylines, each represented as a list of x, y coords
    [num_elements, num_points_in_element (variable size), 2].
    """

    polylines: List[List[Point2D]]

    def to_vector(self) -> List[List[List[float]]]:
        """
        Returns data in vectorized form
        :return: vectorized coords of map object polylines as [num_elements, num_points_in_element (variable size), 2].
        """
        return [[[coord.x, coord.y] for coord in polygon] for polygon in self.polylines]


def lane_segment_coords_from_lane_segment_vector(coords: List[List[List[float]]]) -> LaneSegmentCoords:
    """
    Convert lane segment coords [N, 2, 2] to nuPlan LaneSegmentCoords.
    :param coords: lane segment coordinates in vector form.
    :return: lane segment coordinates as LaneSegmentCoords.
    """
    return LaneSegmentCoords([(Point2D(*start), Point2D(*end)) for start, end in coords])


def prune_route_by_connectivity(route_roadblock_ids: List[str], roadblock_ids: Set[str]) -> List[str]:
    """
    Prune route by overlap with extracted roadblock elements within query radius to maintain connectivity in route
    feature. Assumes route_roadblock_ids is ordered and connected to begin with.
    :param route_roadblock_ids: List of roadblock ids representing route.
    :param roadblock_ids: Set of ids of extracted roadblocks within query radius.
    :return: List of pruned roadblock ids (connected and within query radius).
    """
    pruned_route_roadblock_ids: List[str] = []
    route_start = False  # wait for route to come into query radius before declaring broken connection

    for roadblock_id in route_roadblock_ids:

        if roadblock_id in roadblock_ids:
            pruned_route_roadblock_ids.append(roadblock_id)
            route_start = True

        elif route_start:  # connection broken
            break

    return pruned_route_roadblock_ids


def get_lane_polylines(
    map_api: AbstractMap, point: Point2D, radius: float
) -> Tuple[MapObjectPolylines, MapObjectPolylines, MapObjectPolylines, LaneSegmentLaneIDs]:
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

    return (
        MapObjectPolylines(lanes_mid),
        MapObjectPolylines(lanes_left),
        MapObjectPolylines(lanes_right),
        LaneSegmentLaneIDs(lane_ids),
    )


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

    return MapObjectPolylines(polygons)


def get_route_polygon_from_roadblock_ids(
    map_api: AbstractMap, point: Point2D, radius: float, route_roadblock_ids: List[str]
) -> MapObjectPolylines:
    """
    Extract route polygon from map for route specified by list of roadblock ids. Polygon is represented as collection of
        polygons of roadblocks/roadblock connectors encompassing route.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param route_roadblock_ids: ids of roadblocks/roadblock connectors specifying route.
    :return: A route as sequence of roadblock/roadblock connector polygons.
    """
    route_polygons: List[List[Point2D]] = []

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

        if roadblock_obj:
            polygon = extract_polygon_from_map_object(roadblock_obj)
            route_polygons.append(polygon)

    return MapObjectPolylines(route_polygons)


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

    return MapObjectPolylines(route_lane_polylines)


def get_on_route_status(
    route_roadblock_ids: List[str], roadblock_ids: LaneSegmentRoadBlockIDs
) -> LaneOnRouteStatusData:
    """
    Identify whether given lane segments lie within goal route.
    :param route_roadblock_ids: List of ids of roadblocks (lane groups) within goal route.
    :param roadblock_ids: Roadblock ids (lane group associations) pertaining to associated lane segments.
    :return on_route_status: binary encoding of on route status for each input roadblock id.
    """
    if route_roadblock_ids:
        # prune route to extracted roadblocks maintaining connectivity
        route_roadblock_ids = prune_route_by_connectivity(route_roadblock_ids, set(roadblock_ids.roadblock_ids))

        # initialize on route status as OFF_ROUTE
        on_route_status = np.full(
            (len(roadblock_ids.roadblock_ids), len(OnRouteStatusType) - 1),
            LaneOnRouteStatusData.encode(OnRouteStatusType.OFF_ROUTE),
        )

        # Get indices of the segments that lie on the route
        on_route_indices = np.arange(on_route_status.shape[0])[
            np.in1d(roadblock_ids.roadblock_ids, route_roadblock_ids)
        ]

        #  Set segments on route to ON_ROUTE
        on_route_status[on_route_indices] = LaneOnRouteStatusData.encode(OnRouteStatusType.ON_ROUTE)

    else:
        # set on route status to UNKNOWN if no route available
        on_route_status = np.full(
            (len(roadblock_ids.roadblock_ids), len(OnRouteStatusType) - 1),
            LaneOnRouteStatusData.encode(OnRouteStatusType.UNKNOWN),
        )

    return LaneOnRouteStatusData(list(map(tuple, on_route_status)))  # type: ignore


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

    return LaneSegmentTrafficLightData(list(map(tuple, traffic_light_encoding)))  # type: ignore


def get_neighbor_vector_map(
    map_api: AbstractMap, point: Point2D, radius: float
) -> Tuple[
    LaneSegmentCoords, LaneSegmentConnections, LaneSegmentGroupings, LaneSegmentLaneIDs, LaneSegmentRoadBlockIDs
]:
    """
    Extract neighbor vector map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :return
        lane_seg_coords: lane_segment coords in shape of [num_lane_segment, 2, 2].
        lane_seg_conns: lane_segment connections [start_idx, end_idx] in shape of [num_connection, 2].
        lane_seg_groupings: collection of lane_segment indices in each lane in shape of
            [num_lane, num_lane_segment_in_lane].
        lane_seg_lane_ids: lane ids of segments at given index in coords in shape of [num_lane_segment 1].
        lane_seg_roadblock_ids: roadblock ids of segments at given index in coords in shape of [num_lane_segment 1].
    """
    lane_seg_coords: List[List[List[float]]] = []  # shape: [num_lane_segment, 2, 2]
    lane_seg_conns: List[Tuple[int, int]] = []  # shape: [num_connection, 2]
    lane_seg_groupings: List[List[int]] = []  # shape: [num_lanes]
    lane_seg_lane_ids: List[str] = []  # shape: [num_lane_segment]
    lane_seg_roadblock_ids: List[str] = []  # shape: [num_lane_segment]
    cross_blp_conns: Dict[str, Tuple[int, int]] = dict()

    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    nearest_vector_map = map_api.get_proximal_map_objects(point, radius, layer_names)

    # create lane segment vectors from baseline paths
    for layer_name in layer_names:

        for map_obj in nearest_vector_map[layer_name]:
            # current number of coords needed for indexing lane segments
            start_lane_seg_idx = len(lane_seg_coords)
            # update lane segment info with info for given lane/lane connector
            trim_nodes = build_lane_segments_from_blps_with_trim(point, radius, map_obj, start_lane_seg_idx)
            if trim_nodes is not None:
                (
                    obj_coords,
                    obj_conns,
                    obj_groupings,
                    obj_lane_ids,
                    obj_roadblock_ids,
                    obj_cross_blp_conn,
                ) = trim_nodes

                lane_seg_coords += obj_coords
                lane_seg_conns += obj_conns
                lane_seg_groupings += obj_groupings
                lane_seg_lane_ids += obj_lane_ids
                lane_seg_roadblock_ids += obj_roadblock_ids
                cross_blp_conns[map_obj.id] = obj_cross_blp_conn

    # create connections between adjoining lanes and lane connectors
    for lane_conn in nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]:
        if lane_conn.id in cross_blp_conns:
            lane_seg_conns += connect_trimmed_lane_conn_predecessor(lane_seg_coords, lane_conn, cross_blp_conns)
            lane_seg_conns += connect_trimmed_lane_conn_successor(lane_seg_coords, lane_conn, cross_blp_conns)

    return (
        lane_segment_coords_from_lane_segment_vector(lane_seg_coords),
        LaneSegmentConnections(lane_seg_conns),
        LaneSegmentGroupings(lane_seg_groupings),
        LaneSegmentLaneIDs(lane_seg_lane_ids),
        LaneSegmentRoadBlockIDs(lane_seg_roadblock_ids),
    )


def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    route_roadblock_ids: List[str],
    traffic_light_statuses_over_time: List[TrafficLightStatuses],
) -> Tuple[Dict[str, MapObjectPolylines], List[Dict[str, LaneSegmentTrafficLightData]]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param route_roadblock_ids: List of ids of roadblocks/roadblock connectors (lane groups) within goal route.
    :param traffic_light_statuses_over_time: A list of available traffic light statuses data, indexed by time step.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, MapObjectPolylines] = {}
    feature_layers: List[VectorFeatureLayer] = []
    traffic_light_data_over_time: List[Dict[str, LaneSegmentTrafficLightData]] = []

    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(f"Object representation for layer: {feature_name} is unavailable")

    # extract lanes and traffic light
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(map_api, point, radius)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)

        # extract traffic light
        for traffic_lights in traffic_light_statuses_over_time:
            # lane traffic light data
            traffic_light_data_at_t: Dict[str, LaneSegmentTrafficLightData] = {}
            traffic_light_data_at_t[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
                lane_ids, traffic_lights.traffic_lights
            )
            traffic_light_data_over_time.append(traffic_light_data_at_t)

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


def get_centerline_coords(scenario: AbstractScenario,
                          radius: float,
                          v_max: float,
                          a_max: float,
                          time_horizon: float,
                          anchor_ego_state: EgoState = None,
                          return_reference_lanes: bool=False) -> Union[np.ndarray, Any]:
    """get coordinates of centerline that is the closest to ego vehicle"""
    current_ego_coords = Point2D(anchor_ego_state.rear_axle.x, anchor_ego_state.rear_axle.y)
    route_roadblock_ids = scenario.get_route_roadblock_ids()
    v_max = min(v_max, anchor_ego_state.dynamic_car_state.speed + a_max * time_horizon)
    s_max = 0.5 * (anchor_ego_state.dynamic_car_state.speed + v_max) * time_horizon
    neighbor_radius = max(s_max, radius)
    (
        lane_seg_coords,  # centerlines of lanes
        lane_seg_conns,
        lane_seg_groupings,
        lane_seg_lane_ids,  # a lane consists of many lane segments
        lane_seg_roadblock_ids,  # a roadblock consists of many lanes
    ) = get_neighbor_vector_map(scenario.map_api, current_ego_coords, neighbor_radius)

    # find proximal route roadblock
    route_roadblocks: List[NuPlanRoadBlock] = [scenario.map_api.get_map_object(id, SemanticMapLayer.ROADBLOCK) for id in route_roadblock_ids]
    route_roadblock_centroids: List[Point] = [rb.polygon.centroid for rb in route_roadblocks]
    current_ego_point = Point(anchor_ego_state.rear_axle.x, anchor_ego_state.rear_axle.y)
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
    on_point_roadblocks = extract_roadblock_objects(scenario.map_api, anchor_ego_state.rear_axle.point)
    if len(on_point_roadblocks) == 0:
        on_point_roadblocks = extract_proximal_roadblock_objects(scenario.map_api, anchor_ego_state.rear_axle.point, 3.)
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
        roadblock_successors_given_route = get_roadblock_successors_given_route(route_roadblock_ids,
                                                                                on_point_roadblocks,
                                                                                current_ego_state=anchor_ego_state)
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
            roadblock_successors_given_route = get_roadblock_successors_given_route(route_roadblock_ids,
                                                                                    on_point_roadblocks_back_up,
                                                                                    current_ego_state=anchor_ego_state)
            on_route_roadblocks = roadblock_successors_given_route['on_route_roadblocks']
            routes = roadblock_successors_given_route['routes']
        else:
            on_route_roadblocks = on_point_roadblocks_back_up
            routes = on_point_roadblocks
        on_route_roadblock_ids = [roadblock.id for roadblock in on_route_roadblocks]
        routes_roadblock_ids = [[roadblock.id for roadblock in route] for route in routes]
        on_route_indices = get_on_route_indices(routes_roadblock_ids, lane_seg_roadblock_ids, inputting_routes=True)

    # extract a baseline path that starts at the current ego position with the minimum curvature
    reference_lane_seg_groupings = []  # used to build Frenet frame
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
    current_ego_coord_array = np.array([anchor_ego_state.rear_axle.x, anchor_ego_state.rear_axle.y])

    # find the closest lane seg group (current lane seg group)
    on_route_lane_seg_grouping_coord_array = []
    on_route_lane_seg_groupings_indices = []
    for on_route_lane_seg_grouping in on_route_lane_seg_groupings:
        tmp = []
        for index in on_route_lane_seg_grouping:
            tmp.append([lane_seg_coords.coords[index][0].array, lane_seg_coords.coords[index][1].array])
        on_route_lane_seg_grouping_coord_array = on_route_lane_seg_grouping_coord_array + tmp
        if len(on_route_lane_seg_groupings_indices) == 0:
            on_route_lane_seg_groupings_indices.append(np.arange(len(tmp)))
        else:
            on_route_lane_seg_groupings_indices.append(np.arange(on_route_lane_seg_groupings_indices[-1][-1] + 1,
                                                                 on_route_lane_seg_groupings_indices[-1][
                                                                     -1] + 1 + len(tmp)))
    on_route_lane_seg_grouping_coord_array = np.array(on_route_lane_seg_grouping_coord_array)
    distance_to_ego = point_to_point_distance(current_ego_coord_array,
                                              on_route_lane_seg_grouping_coord_array[:, 0, :])
    grouping_wise_min_distance = []
    for indices in on_route_lane_seg_groupings_indices:
        grouping_wise_min_distance.append(min(distance_to_ego[indices]))
    minimum_distance = min(grouping_wise_min_distance)
    closest_lane_seg_grouping_indices = np.where(grouping_wise_min_distance == minimum_distance)[0]
    closest_lane_seg_groupings = [on_route_lane_seg_groupings[index] for index in closest_lane_seg_grouping_indices]
    if len(closest_lane_seg_groupings) > 1:
        closest_lane_seg_coords = []
        closest_lane_seg_curvatures = []
        for closest_lane_seg_grouping in closest_lane_seg_groupings:
            closest_lane_seg_coords.append(
                np.array([lane_seg_coords.coords[index][0].array for index in closest_lane_seg_grouping]))
            length = closest_lane_seg_coords[-1].shape[0]
            point1 = geom.Point(closest_lane_seg_coords[-1][0])
            point2 = geom.Point(closest_lane_seg_coords[-1][length // 2])
            point3 = geom.Point(closest_lane_seg_coords[-1][-1])
            closest_lane_seg_curvatures.append(compute_curvature(point1, point2, point3))
        reference_lane_seg_groupings.append(closest_lane_seg_groupings[np.argmin(closest_lane_seg_curvatures)])
    else:
        reference_lane_seg_groupings.append(closest_lane_seg_groupings[0])
    lane_seg_predecessor_end_coord_array = lane_seg_coords.coords[reference_lane_seg_groupings[-1][-1]][0].array

    # find lane seg grouping successors of current lane seg group
    while True:
        distance = on_route_lane_seg_grouping_start_coord_array[:, 0, :] - lane_seg_predecessor_end_coord_array
        squarred_distance = distance[:, 0] ** 2 + distance[:, 1] ** 2
        minimum_squarred_distance = squarred_distance.min()
        if len(reference_lane_seg_groupings) > 0 and minimum_squarred_distance > 1.0:
            break
        closest_lane_seg_grouping_indices = np.where(squarred_distance == minimum_squarred_distance)[0]
        closest_lane_seg_groupings = [on_route_lane_seg_groupings[index] for index in
                                      closest_lane_seg_grouping_indices]
        if len(closest_lane_seg_groupings) > 1:
            closest_lane_seg_coords = []
            closest_lane_seg_curvatures = []
            for closest_lane_seg_grouping in closest_lane_seg_groupings:
                closest_lane_seg_coords.append(
                    np.array([lane_seg_coords.coords[index][0].array for index in closest_lane_seg_grouping]))
                length = closest_lane_seg_coords[-1].shape[0]
                point1 = geom.Point(closest_lane_seg_coords[-1][0])
                point2 = geom.Point(closest_lane_seg_coords[-1][length // 2])
                point3 = geom.Point(closest_lane_seg_coords[-1][-1])
                closest_lane_seg_curvatures.append(compute_curvature(point1, point2, point3))

            lane_seg_grouping = closest_lane_seg_groupings[np.argmin(closest_lane_seg_curvatures)]
        else:
            lane_seg_grouping = closest_lane_seg_groupings[0]
        if lane_seg_grouping in reference_lane_seg_groupings:
            break
        else:
            reference_lane_seg_groupings.append(lane_seg_grouping)

        # reference_lane_seg_groupings.append(closest_lane_seg_groupings[np.argmax(closest_lane_seg_curvatures)])  # TODO debug only
        lane_seg_predecessor_end_coord_array = lane_seg_coords.coords[reference_lane_seg_groupings[-1][-1]][0].array

    if return_reference_lanes:
        reference_line_coords = []
        reference_line_lane_ids = []
        for lane_seg_grouping in reference_lane_seg_groupings:
            reference_line_coords = reference_line_coords + [lane_seg_coords.coords[index][0].array for index in
                                                             lane_seg_grouping]
            reference_line_lane_ids = reference_line_lane_ids + [lane_seg_lane_ids.lane_ids[index] for index in
                                                                 lane_seg_grouping]
        reference_line_coords = np.array(reference_line_coords)
        reference_line_lane_ids_deduplicated = list(set(reference_line_lane_ids))
        reference_line_lane_ids_deduplicated.sort(key=reference_line_lane_ids.index)
        reference_line_lanes = []  # lanes and lane connectors
        for lane_id in reference_line_lane_ids_deduplicated:
            lane_or_lane_connector = scenario.map_api._get_lane(lane_id)
            if lane_or_lane_connector is None:
                lane_or_lane_connector = scenario.map_api._get_lane_connector(lane_id)
            reference_line_lanes.append(lane_or_lane_connector)
    else:
        reference_line_coords = []
        for lane_seg_grouping in reference_lane_seg_groupings:
            reference_line_coords = reference_line_coords + [lane_seg_coords.coords[index][0].array for index in
                                                             lane_seg_grouping]
        reference_line_coords = np.array(reference_line_coords)

    # # TODO debug only: sth to plot
    # sth_to_plot = {
    #     'lane_seg_coords': lane_seg_coords,
    #     'on_route_roadblocks': on_route_roadblocks,
    #     'roadblock_successors_given_route': roadblock_successors_given_route,
    #     'lane_seg_roadblock_ids': lane_seg_roadblock_ids,
    #     'route_roadblocks': route_roadblocks,
    #     'route_roadblock_ids': route_roadblock_ids,
    #     'on_route_indices': on_route_indices,
    #     'on_route_lane_seg_grouping_start_coord_array': on_route_lane_seg_grouping_start_coord_array,
    #     'on_route_lane_seg_grouping_end_coord_array': on_route_lane_seg_grouping_end_coord_array,
    #     'reference_line_coords': reference_line_coords,
    # }
    sth_to_plot = None

    if return_reference_lanes:
        return reference_line_coords, sth_to_plot, reference_line_lanes
    else:
        return reference_line_coords, sth_to_plot


def visualize(sth_to_plot: Dict[str, Any]=None):
    scenario = sth_to_plot['scenario']
    lane_seg_coords = sth_to_plot['lane_seg_coords']
    anchor_ego_state = sth_to_plot['anchor_ego_state']
    future_ego_states = sth_to_plot['future_ego_states'] if 'future_ego_states' in sth_to_plot else None
    on_route_roadblocks = sth_to_plot['on_route_roadblocks']
    route_roadblocks = sth_to_plot['route_roadblocks']
    roadblock_successors_given_route = sth_to_plot['roadblock_successors_given_route']
    lane_seg_roadblock_ids = sth_to_plot['lane_seg_roadblock_ids']
    route_roadblock_ids = sth_to_plot['route_roadblock_ids']
    reference_line_lanes = sth_to_plot['reference_line_lanes']
    on_route_lane_seg_grouping_start_coord_array = sth_to_plot['on_route_lane_seg_grouping_start_coord_array']
    on_route_lane_seg_grouping_end_coord_array = sth_to_plot['on_route_lane_seg_grouping_end_coord_array']
    reference_line_coords = sth_to_plot['reference_line_coords']
    trajectory_samples_cartesian = sth_to_plot['trajectory_samples_cartesian'] if 'trajectory_samples_cartesian' in sth_to_plot else None
    on_route_indices = sth_to_plot['on_route_indices']

    # import time
    # suffix = str(time.time())
    suffix = scenario.log_name + '-' + scenario.token
    plt.figure(figsize=(12, 12))
    lane_seg_coord_vectors = lane_seg_coords.to_vector()
    for vector in lane_seg_coord_vectors:
        polyline = np.array(vector)
        plt.plot(polyline[:, 0], polyline[:, 1])

    plt.scatter(anchor_ego_state.rear_axle.x, anchor_ego_state.rear_axle.y, marker='*', color='tab:red', alpha=0.5)

    if future_ego_states:
        ego_traj = []
        for state in [anchor_ego_state] + future_ego_states:
            ego_traj.append(state.rear_axle.point.array)
        ego_traj = np.array(ego_traj)
        plt.plot(ego_traj[:, 0], ego_traj[:, 1], color='tab:blue', linewidth=1, alpha=0.8, zorder=1000)

    on_route_lane_seg_coords = LaneSegmentCoords([lane_seg_coords.coords[index] for index in on_route_indices])
    on_route_lane_seg_coords_vector = on_route_lane_seg_coords.to_vector()
    for vector in on_route_lane_seg_coords_vector:
        polyline = np.array(vector)
        plt.plot(polyline[:, 0], polyline[:, 1], color='k', alpha=0.3)

    for on_route_roadblock in on_route_roadblocks:
        centroid = on_route_roadblock.polygon.centroid
        id = on_route_roadblock.id
        plt.text(centroid.x, centroid.y, id)

    for lane in reference_line_lanes:
        centroid = lane.polygon.centroid
        id = lane.id
        plt.text(centroid.x, centroid.y, id, color='g', fontsize='small')

    for rb in route_roadblocks:
        centroid = rb.polygon.centroid
        id = rb.id
        plt.text(centroid.x, centroid.y, id, color='r')

    really_on_route_roadblocks = roadblock_successors_given_route['really_on_route_roadblocks']
    really_on_route_roadblock_ids = [roadblock.id for roadblock in really_on_route_roadblocks]
    really_on_route_indices = get_on_route_indices(really_on_route_roadblock_ids, lane_seg_roadblock_ids)
    try:
        really_on_route_lane_seg_coords = LaneSegmentCoords([lane_seg_coords.coords[index] for index in really_on_route_indices])
        really_on_route_lane_seg_coords_vector = really_on_route_lane_seg_coords.to_vector()
        for vector in really_on_route_lane_seg_coords_vector:
            polyline = np.array(vector)
            plt.plot(polyline[:, 0], polyline[:, 1], color='tab:gray', alpha=0.8)

        route_indices = get_on_route_indices(route_roadblock_ids, lane_seg_roadblock_ids)
        route_lane_seg_coord_array = np.array([lane_seg_coords.coords[index][0].array for index in route_indices])
        plt.scatter(route_lane_seg_coord_array[:, 0], route_lane_seg_coord_array[:, 1], color='tab:green', alpha=0.8, s=2)
    except:
        pass

    # for lane_seg_grouping in lane_seg_groupings.groupings:
    #     if lane_seg_grouping[0] in on_route_indices or lane_seg_grouping[-1] in on_route_indices:
    #         lane_seg_start = [lane_seg_coords.coords[index][0] for index in lane_seg_grouping]
    #         for point in lane_seg_start:
    #             plt.scatter(point.x, point.y, color='tab:orange', alpha=0.8, s=2)

    plt.scatter(on_route_lane_seg_grouping_start_coord_array[:, 0, 0], on_route_lane_seg_grouping_start_coord_array[:, 0, 1], color='gold', s=4, marker='o', alpha=0.3,
                zorder=1000)
    plt.scatter(on_route_lane_seg_grouping_end_coord_array[:, 0, 0], on_route_lane_seg_grouping_end_coord_array[:, 0, 1], color='dodgerblue', s=1, marker='s',
                alpha=0.3, zorder=1000)

    # on_trajectory_lane_seg_coords_vector = on_trajectory_lane_seg_coords.to_vector()
    # for vector in on_trajectory_lane_seg_coords_vector:
    #     polyline = np.array(vector)
    #     plt.plot(polyline[:, 0], polyline[:, 1], color='r')

    # on_point_lane_seg_coords_vector = on_point_lane_seg_coords.to_vector()
    # for vector in on_point_lane_seg_coords_vector:
    #     polyline = np.array(vector)
    #     plt.plot(polyline[:, 0], polyline[:, 1], color='r')

    # lane_seg_coord_array = np.array(lane_seg_coord_vectors)
    # where_jump = np.where(on_route_lane_seg_conns_array_sorted[1:, 0] - on_route_lane_seg_conns_array_sorted[0:-1, 0] > 1)
    # # for i in range(where_jump[0].shape[0]):
    # for i in range(2, 3):
    #     if i == 0:
    #         indices = on_route_lane_seg_conns_array_sorted[0:where_jump[0][0] + 1, 0]
    #     elif i == where_jump[0].shape[0] - 1:
    #         indices = on_route_lane_seg_conns_array_sorted[where_jump[0][-1] + 1:, 0]
    #     else:
    #         indices = on_route_lane_seg_conns_array_sorted[where_jump[0][i] + 1:where_jump[0][i + 1] + 1, 0]
    #     plt.plot(lane_seg_coord_array[indices, 0, 0], lane_seg_coord_array[indices, 0, 1])

    plt.plot(reference_line_coords[:, 0], reference_line_coords[:, 1], color='tab:red', linestyle='--', linewidth=1.5, alpha=0.8)

    if trajectory_samples_cartesian:
        for rear_axle_traj in trajectory_samples_cartesian['pose_cartesian'][::-1, :, :]:
            plt.plot(rear_axle_traj[0], rear_axle_traj[1], linewidth=0.3, alpha=0.8)

    # point_on_ref_line_cartesian = frenet_frame.reference_line.interpolate(trajectory_samples_frenet['poses_frenet'][0, 0, :])
    # [plt.scatter(pt.x, pt.y) for pt in point_on_ref_line_cartesian]

    plt.axis('equal')
    # plt.xlim((664400, 664500))
    # plt.ylim((3997000, 3997200))
    # plt.savefig(f'route_{suffix}.png', dpi=600)
    iteration = sth_to_plot['iteration'] if 'iteration' in sth_to_plot else ''
    path = './debug_figs'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'./debug_figs/route_{suffix}_{iteration}.pdf')
    # plt.show()
    plt.close()
    # sys.exit(0)