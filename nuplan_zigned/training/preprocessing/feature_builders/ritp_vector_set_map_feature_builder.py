from __future__ import annotations

from typing import Dict, List, Tuple, Type, Union, Optional, Any

import torch
import numpy as np
import numpy.typing as npt
import shapely.geometry as geom
import matplotlib.pyplot as plt
from _datetime import datetime

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatuses,
)
from nuplan.common.maps.nuplan_map.utils import (
    extract_roadblock_objects,
    compute_curvature,
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

from nuplan_zigned.training.preprocessing.feature_builders.ritp_feature_builder_utils import (
    get_neighbor_vector_set_map,
)
from nuplan_zigned.training.preprocessing.features.ritp_vector_set_map import VectorSetMap
from nuplan_zigned.training.preprocessing.utils.qcmae_vector_preprocessing import convert_lane_layers_to_consistent_size
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
        radius: float,
        map_features: List[str],
        max_points: Dict[str, int],
        max_elements: Dict[str, int],
        interpolation_method: str,
    ) -> None:
        """
        Initialize vector set map builder with configuration parameters.
        :param radius: [m] floating number about vector map query range.
        :param map_features: name of map features to be extracted.
        :param max_elements: maximum number of elements to extract per feature layer.
        :param max_points: maximum number of points per feature to extract per feature layer.
        :param interpolation_method: Interpolation method to apply when interpolating to maintain fixed size
            map elements.
        """
        super().__init__()
        self.radius = radius
        self.map_features = map_features
        self.max_points = max_points
        self.max_elements = max_elements
        self.interpolation_method = interpolation_method

        self._traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()
        self._traffic_light_statuses = ['GREEN', 'YELLOW', 'RED', 'UNKNOWN']
        self._traffic_light_one_hot_decoding = {
            (1, 0, 0, 0): 0,  # GREEN
            (0, 1, 0, 0): 1,  # YELLOW
            (0, 0, 1, 0): 2,  # RED
            (0, 0, 0, 1): 3,  # UNKNOWN
        }
        self._polygon_types = map_features
        self._polygon_is_intersections = [True, False, None]
        self._point_types = map_features
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER', 'UNKNOWN']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']

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
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        self.number_of_interations = scenario.get_number_of_iterations()
        self.scenario_time_horizon = self.number_of_interations * scenario.database_interval
        future_tl = list(
            scenario.get_future_traffic_light_status_history(iteration=0, time_horizon=self.scenario_time_horizon, num_samples=self.number_of_interations))
        future_timestamps = list(scenario.get_future_timestamps(iteration=0, time_horizon=self.scenario_time_horizon, num_samples=self.number_of_interations))

        coords, traffic_light_data = get_neighbor_vector_set_map(
            scenario.map_api,
            self._polygon_types,
            ego_coords,
            self.radius,
            route_roadblock_ids,
            future_tl,
            future_timestamps
        )

        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
            coords, traffic_light_data, ego_state.rear_axle
        )
        return tensor, list_tensor, list_list_tensor

    @torch.jit.unused
    def get_scriptable_input_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Extract the input for the scriptable forward method from the simulation objects
        :param current_input: planner input from sim
        :param initialization: planner initialization from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        ego_state = current_input.history.ego_states[-1]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = initialization.route_roadblock_ids

        # get traffic light status
        if current_input.traffic_light_data is None:
            raise ValueError("Cannot build VectorSetMap feature. PlannerInput.traffic_light_data is None")
        traffic_light_data = current_input.traffic_light_data

        coords, traffic_light_data = get_neighbor_vector_set_map(
            initialization.map_api,
            self.map_features,
            ego_coords,
            self.radius,
            route_roadblock_ids,
            [TrafficLightStatuses(traffic_light_data)],
        )

        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
            coords, traffic_light_data[0], ego_state.rear_axle
        )
        return tensor, list_tensor, list_list_tensor

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_scenario(scenario)
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(
            tensor_data, list_tensor_data, list_list_tensor_data
        )

        return self._unpack_feature_to_vector_set_map(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> VectorSetMap:
        """Inherited, see superclass."""
        tensor_data, list_tensor_data, list_list_tensor_data = self.get_scriptable_input_from_simulation(
            current_input, initialization
        )
        tensor_data, list_tensor_data, list_list_tensor_data = self.scriptable_forward(
            tensor_data, list_tensor_data, list_list_tensor_data
        )

        return self._unpack_feature_from_tensor_dict(tensor_data, list_tensor_data, list_list_tensor_data)

    @torch.jit.unused
    def _unpack_feature_to_vector_set_map(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[Dict[str, torch.Tensor]]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> VectorSetMap:
        """
        Unpacks the data returned from the scriptable portion of the method into a VectorSetMap object.
        :param tensor_data: The tensor data to unpack.
        :param list_tensor_data: The List[tensor] data to unpack.
        :param list_list_tensor_data: The List[List[tensor]] data to unpack.
        :return: The unpacked VectorSetMap.
        """

        map_data = self.get_map_data(list_tensor_data)

        return VectorSetMap(map_data=map_data)

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        coords: Dict[str, MapObjectPolylines],
        traffic_light_data: Dict[int, Dict[str, LaneSegmentTrafficLightData]],
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
            list_feature_tl_data: Dict[str, torch.Tensor] = {}
            for timestamp, tl_data in traffic_light_data.items():
                if feature_name in tl_data.keys():
                    for map_obj_id, element_tl_data in zip(
                            tl_data[feature_name].traffic_lights.keys(),
                            tl_data[feature_name].to_vector()
                    ):
                        if map_obj_id not in list_feature_tl_data.keys():
                            list_feature_tl_data[map_obj_id] = [element_tl_data]
                        else:
                            list_feature_tl_data[map_obj_id].append(element_tl_data)
            for map_obj_id, data in list_feature_tl_data.items():
                list_feature_tl_data[map_obj_id] = torch.tensor(data, dtype=torch.float32)
            if len(list_feature_tl_data) > 0:
                list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data

        # Pach timestamps into tensor list
        list_tensor_data[f"timestamps"] = list(traffic_light_data.keys())

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

        list_tensor_output["timestamps"] = [list_tensor_data["timestamps"]]

        return tensor_output, list_tensor_output, list_list_tensor_output

    def get_map_data(self, list_tensor_data: Dict[str, List[Union[Dict[str, torch.Tensor], List[Any]]]]) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        list_num_polygons = []
        list_polygon_position = []
        list_polygon_orientation = []
        list_polygon_type = []
        list_polygon_is_intersection = []
        list_polygon_tl_statuses = []  # used for QCMAE
        list_polygon_tl_timestamps = []
        list_num_points = []
        list_point_position = []
        list_point_orientation = []
        list_point_magnitude = []
        list_point_type = []
        list_point_side = []
        list_point_tl_statuses = []  # used for RewardFormer
        list_point_tl_timestamps = []
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
            num_timesteps = len(list_tensor_data['timestamps'][sample_idx])

            # initialization
            polygon_position = torch.zeros(num_polygons, 2, dtype=torch.float)
            polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
            polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
            polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
            polygon_tl_statuses = torch.zeros(num_polygons, num_timesteps, dtype=torch.uint8)
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
                tl_status = torch.tensor([self._traffic_light_one_hot_decoding[tuple(tl_s)] for tl_s in tl_status], dtype=torch.uint8)
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
                tl_status = tl_status.unsqueeze(dim=0).repeat(len(left_vectors), 1)
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
                point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors), num_timesteps),
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
                point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors), num_timesteps),
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
                point_tl_statuses[idx_offset_pl + i_element] = torch.full((len(edge_vectors), num_timesteps),
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

                num_pl_to_pl_edge_index_detail[sample_idx]= num_lanes

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
            list_polygon_tl_timestamps.append(list_tensor_data['timestamps'][sample_idx])
            list_num_points.append(num_points)
            list_point_position.append(torch.cat(point_position, dim=0).detach().numpy())
            list_point_orientation.append(torch.cat(point_orientation, dim=0).detach().numpy())
            list_point_magnitude.append(torch.cat(point_magnitude, dim=0).detach().numpy())
            list_point_type.append(torch.cat(point_type, dim=0).detach().numpy())
            list_point_side.append(torch.cat(point_side, dim=0).detach().numpy())
            list_point_tl_statuses.append(torch.cat(point_tl_statuses, dim=0).detach().numpy())
            list_point_tl_timestamps.append(list_tensor_data['timestamps'][sample_idx])
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
        map_data['map_polygon']['tl_timestamps'] = list_polygon_tl_timestamps
        map_data['map_point']['num_nodes'] = list_num_points
        map_data['map_point']['position'] = list_point_position
        map_data['map_point']['orientation'] = list_point_orientation
        map_data['map_point']['magnitude'] = list_point_magnitude
        map_data['map_point']['type'] = list_point_type
        map_data['map_point']['side'] = list_point_side
        map_data['map_point']['tl_statuses'] = list_point_tl_statuses
        map_data['map_point']['tl_timestamps'] = list_polygon_tl_timestamps
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
