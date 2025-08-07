from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from pyquaternion import Quaternion

from nuplan.planning.script.builders.utils.utils_type import are_the_same_type, validate_type
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneSegmentTrafficLightData,
    VectorFeatureLayer,
)
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)
from nuplan.planning.training.preprocessing.features.vector_utils import (
    rotate_coords,
    scale_coords,
    translate_coords,
    xflip_coords,
    yflip_coords,
)


@dataclass
class VectorSetMap(AbstractModelFeature):
    """
    Vector set map data structure, including:
        coords: Dict[str, List[<np.ndarray: num_trajs, num_poses, num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample in batch,
                indexed by map feature.
        traffic_light_data: Dict[str, List[<np.ndarray: num_trajs, num_poses, num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample
                in batch, indexed by map feature. Same indexing as coords.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
        availabilities: Dict[str, List[<np.ndarray: num_trajs, num_poses, num_elements, num_points>]].
            Boolean indicator of whether feature data (coords as well as traffic light status if it exists for feature)
                is available for point at given index or if it is zero-padded.
        trajectory_samples: FeatureDataType

    Feature formulation as sets of vectors for each map element similar to that of VectorNet ("VectorNet: Encoding HD
    Maps and Agent Dynamics from Vectorized Representation"), except map elements are encoded as sets of singular x, y
    points instead of start, end point pairs.

    Coords, traffic light status, and availabilities data are each keyed by map feature name, with dimensionality
    (availabilities don't include feature dimension):
    B: number of samples per batch (variable)
    num_trajs: number of trajectories per batch
    num_poses: number of poses per trajectory
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features (2 for coords, 4 for traffic light status)

    Data at the same index represent the same map element/point among coords, traffic_light_data, and availabilities,
    with traffic_light_data only optionally included. For each map feature, the top level List represents number of
    samples per batch. This is a special feature where each batch entry can have a different size. For that reason, the
    features can not be placed to a single tensor, and we batch the feature with a custom `collate` function.
    """

    coords: Optional[Dict[str, List[FeatureDataType]]] = None
    traffic_light_data: Optional[Dict[str, List[FeatureDataType]]] = None
    availabilities: Optional[Dict[str, List[FeatureDataType]]] = None
    map_data: Optional[Dict[Union[str, Tuple[str, str, str]], Any]] = None
    trajectory_samples: Union[List[Dict[str, FeatureDataType]], Dict[str, FeatureDataType]] = None
    map_obj_ids: Optional[Dict[str, Dict[int, Dict[int, Union[List[str], torch.Tensor]]]]] = None
    neighbor_ids: Optional[Dict[int, Dict[int, Dict[str, List]]]] = None
    lane_types: Optional[Dict[int, Dict[int, List]]] = None
    max_elements: Optional[Dict] = None
    max_points: Optional[Dict] = None
    sanitize: bool = True
    compressed: bool = False
    _polyline_coord_dim: int = 2
    _traffic_light_status_dim: int = LaneSegmentTrafficLightData.encoding_dim()

    def __post_init__(self) -> None:
        """
        Sanitize attributes of the dataclass.
        :raise RuntimeError if dimensions invalid.
        """
        # Check empty data
        if not len(self.coords) > 0:
            raise RuntimeError("Coords cannot be empty!")

        if not all([len(coords) > 0 for coords in self.coords.values()]):
            raise RuntimeError("Batch size has to be > 0!")

        if self.sanitize:
            self._sanitize_feature_consistency()
            self._sanitize_data_dimensionality()

    def _sanitize_feature_consistency(self) -> None:
        """
        Check data dimensionality consistent across and within map features.
        :raise RuntimeError if dimensions invalid.
        """
        # Check consistency across map features
        if not all([len(coords) == len(list(self.coords.values())[0]) for coords in self.coords.values()]):
            raise RuntimeError("Batch size inconsistent across features!")

        # Check consistency across data within map feature
        for feature_name, feature_coords in self.coords.items():
            if feature_name not in self.availabilities:
                raise RuntimeError("No matching feature in coords for availabilities data!")
            feature_avails = self.availabilities[feature_name]

            if len(feature_avails) != len(feature_coords):
                raise RuntimeError(
                    f"Batch size between coords and availabilities data inconsistent! {len(feature_coords)} != {len(feature_avails)}"
                )
            feature_size = self.feature_size(feature_name)
            if feature_size[1] == 0:
                raise RuntimeError("Features cannot be empty!")

            for coords in feature_coords:
                if coords.shape[0:2] != feature_size:
                    raise RuntimeError(
                        f"Coords for {feature_name} feature don't have consistent feature size! {coords.shape[0:2] != feature_size}"
                    )
            for avails in feature_avails:
                if avails.shape[0:2] != feature_size:
                    raise RuntimeError(
                        f"Availabilities for {feature_name} feature don't have consistent feature size! {avails.shape[0:2] != feature_size}"
                    )

        for feature_name, feature_tl_data in self.traffic_light_data.items():
            if feature_name not in self.coords:
                raise RuntimeError("No matching feature in coords for traffic light data!")
            feature_coords = self.coords[feature_name]

            if len(feature_tl_data) != len(self.coords[feature_name]):
                raise RuntimeError(
                    f"Batch size between coords and traffic light data inconsistent! {len(feature_coords)} != {len(feature_tl_data)}"
                )
            feature_size = self.feature_size(feature_name)

            for tl_data in feature_tl_data:
                if tl_data.shape[0:2] != feature_size:
                    raise RuntimeError(
                        f"Traffic light data for {feature_name} feature don't have consistent feature size! {tl_data.shape[0:2] != feature_size}"
                    )

    def _sanitize_data_dimensionality(self) -> None:
        """
        Check data dimensionality as expected.
        :raise RuntimeError if dimensions invalid.
        """
        for feature_coords in self.coords.values():
            for sample in feature_coords:
                if sample.shape[-1] != self._polyline_coord_dim:
                    raise RuntimeError("The dimension of coords is not correct!")

        for feature_tl_data in self.traffic_light_data.values():
            for sample in feature_tl_data:
                if sample.shape[-1] != self._traffic_light_status_dim:
                    raise RuntimeError("The dimension of traffic light data is not correct!")

        for feature_avails in self.availabilities.values():
            for sample in feature_avails:
                if len(sample.shape) != 4:
                    raise RuntimeError("The dimension of availabilities is not correct!")

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        if self.compressed:
            # do not validate feature when compressed
            return True
        else:
            return (
                    all([len(feature_coords) > 0 for feature_coords in self.coords.values()])
                    and all([feature_coords[0].shape[0] > 0 for feature_coords in self.coords.values()])
                    and all([feature_coords[0].shape[1] > 0 for feature_coords in self.coords.values()])
                    and all([feature_coords[0].shape[2] > 0 for feature_coords in self.coords.values()])
                    and all([feature_coords[0].shape[3] > 0 for feature_coords in self.coords.values()])
                    and all([len(feature_tl_data) > 0 for feature_tl_data in self.traffic_light_data.values()])
                    and all([feature_tl_data[0].shape[0] > 0 for feature_tl_data in self.traffic_light_data.values()])
                    and all([feature_tl_data[0].shape[1] > 0 for feature_tl_data in self.traffic_light_data.values()])
                    and all([feature_tl_data[0].shape[2] > 0 for feature_tl_data in self.traffic_light_data.values()])
                    and all([feature_tl_data[0].shape[3] > 0 for feature_tl_data in self.traffic_light_data.values()])
                    and all([len(features_avails) > 0 for features_avails in self.availabilities.values()])
                    and all([features_avails[0].shape[0] > 0 for features_avails in self.availabilities.values()])
                    and all([features_avails[0].shape[1] > 0 for features_avails in self.availabilities.values()])
                    and all([features_avails[0].shape[2] > 0 for features_avails in self.availabilities.values()])
                    and all([features_avails[0].shape[3] > 0 for features_avails in self.availabilities.values()])
            )

    @property
    def batch_size(self) -> int:
        """
        Batch size across features.
        :return: number of batches.
        """
        return len(list(self.coords.values())[0])  # All features guaranteed to have same batch size

    def feature_size(self, feature_name: str) -> Tuple[int, int]:
        """
        Number of map elements for given feature, points per element.
        :param feature_name: name of map feature to access.
        :return: [num_elements, num_points]
        :raise: RuntimeError if empty feature.
        """
        map_feature = self.coords[feature_name][0]
        if map_feature.size == 0:
            raise RuntimeError("Feature is empty!")
        return map_feature.shape[0], map_feature.shape[1]

    @classmethod
    def coord_dim(cls) -> int:
        """
        Coords dimensionality, should be 2 (x, y).
        :return: dimension of coords.
        """
        return cls._polyline_coord_dim

    @classmethod
    def traffic_light_status_dim(cls) -> int:
        """
        Traffic light status dimensionality, should be 4.
        :return: dimension of traffic light status.
        """
        return cls._traffic_light_status_dim

    def get_lane_coords(self, sample_idx: int) -> FeatureDataType:
        """
        Retrieve lane coordinates at given sample index.
        :param sample_idx: the batch index of interest.
        :return: lane coordinate features.
        """
        lane_coords = self.coords[VectorFeatureLayer.LANE.name][sample_idx][0, 0, :, :, :]
        if lane_coords.size == 0:
            raise RuntimeError("Lane feature is empty!")
        return lane_coords

    @classmethod
    def collate(cls, batch: List[VectorSetMap]) -> VectorSetMap:
        """Implemented. See interface."""
        coords: Dict[str, List[FeatureDataType]] = defaultdict(list)
        traffic_light_data: Dict[str, List[FeatureDataType]] = defaultdict(list)
        availabilities: Dict[str, List[FeatureDataType]] = defaultdict(list)
        map_obj_ids: Dict[str, List[FeatureDataType]] = defaultdict(list)
        map_data = {
            'map_polygon': [],
            'map_point': [],
            ('map_point', 'to', 'map_polygon'): [],
            ('map_polygon', 'to', 'map_polygon'): [],
            'num_pl_detail': [],
            'num_pl_to_pl_edge_index_detail': [],
        }
        trajectory_samples = []

        for sample in batch:
            # coords
            for feature_name, feature_coords in sample.coords.items():
                coords[feature_name] += feature_coords

            # traffic light data
            for feature_name, feature_tl_data in sample.traffic_light_data.items():
                traffic_light_data[feature_name] += feature_tl_data

            # availabilities
            for feature_name, feature_avails in sample.availabilities.items():
                availabilities[feature_name] += feature_avails

            # map_data
            for key, value in sample.map_data.items():
                if key not in ['num_pl_detail', 'num_pl_to_pl_edge_index_detail']:
                    map_data[key].append(value)
                else:
                    map_data[key] += list(value.values())

            # map_obj_ids
            for feature_name, feature_map_obj_ids in sample.map_obj_ids.items():
                map_obj_ids[feature_name] += [feature_map_obj_ids]
            # trajectory_samples
            trajectory_samples.append(sample.trajectory_samples)

        return VectorSetMap(coords=coords,
                            traffic_light_data=traffic_light_data,
                            availabilities=availabilities,
                            map_data=map_data,
                            map_obj_ids=map_obj_ids,
                            trajectory_samples=trajectory_samples)

    def to_feature_tensor(self) -> VectorSetMap:
        """Implemented. See interface."""
        if self.map_data is not None:
            map_data = {
                'map_polygon': {'num_nodes': self.map_data['map_polygon']['num_nodes'],
                                'position': to_tensor(self.map_data['map_polygon']['position']).contiguous(),
                                'orientation': to_tensor(self.map_data['map_polygon']['orientation']).contiguous(),
                                'type': to_tensor(self.map_data['map_polygon']['type']).contiguous(),
                                'is_intersection': to_tensor(self.map_data['map_polygon']['is_intersection']).contiguous()},
                'map_point': {'num_nodes': self.map_data['map_point']['num_nodes'],
                              'position': to_tensor(self.map_data['map_point']['position']).contiguous(),
                              'orientation': to_tensor(self.map_data['map_point']['orientation']).contiguous(),
                              'magnitude': to_tensor(self.map_data['map_point']['magnitude']).contiguous(),
                              'type': to_tensor(self.map_data['map_point']['type']).contiguous(),
                              'side': to_tensor(self.map_data['map_point']['side']).contiguous(),
                              'tl_statuses': to_tensor(self.map_data['map_point']['tl_statuses']).contiguous()},
                ('map_point', 'to', 'map_polygon'): {'edge_index': to_tensor(self.map_data['map_point', 'to', 'map_polygon']['edge_index']).contiguous()},
                ('map_polygon', 'to', 'map_polygon'): {'edge_index': to_tensor(self.map_data['map_polygon', 'to', 'map_polygon']['edge_index']).contiguous(),
                                                       'type': to_tensor(self.map_data['map_polygon', 'to', 'map_polygon']['type']).contiguous()},
                'num_pl_detail': self.map_data['num_pl_detail'],
                'num_pl_to_pl_edge_index_detail': self.map_data['num_pl_to_pl_edge_index_detail'],
            }
        else:
            map_data = None

        return VectorSetMap(
            coords={
                feature_name: [to_tensor(sample).contiguous() for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data={
                feature_name: [to_tensor(sample).contiguous() for sample in feature_tl_data]
                for feature_name, feature_tl_data in self.traffic_light_data.items()
            },
            availabilities={
                feature_name: [to_tensor(sample).contiguous() for sample in feature_avails]
                for feature_name, feature_avails in self.availabilities.items()
            },
            map_data=map_data,
            trajectory_samples={
                key: to_tensor(value).contiguous() for key, value in self.trajectory_samples.items()
            },
            map_obj_ids=self.map_obj_ids,
            neighbor_ids=self.neighbor_ids,
            lane_types=self.lane_types,
        )

    def to_device(self, device: torch.device) -> VectorSetMap:
        """Implemented. See interface."""
        if isinstance(self.trajectory_samples, list):
            trajectory_samples = [{key: value.to(device) for key, value in traj_samples.items()} if traj_samples is not None else None
                                  for traj_samples in self.trajectory_samples]
        elif isinstance(self.trajectory_samples, dict):
            trajectory_samples = {key: value.to(device) for key, value in self.trajectory_samples.items()}
        else:
            trajectory_samples = None

        if self.map_data is not None:
            map_data = {
                'map_polygon': [],
                'map_point': [],
                ('map_point', 'to', 'map_polygon'): [],
                ('map_polygon', 'to', 'map_polygon'): [],
                'num_pl_detail': [],
                'num_pl_to_pl_edge_index_detail': [],
            }
            for sample_idx in range(len(self.map_data['map_polygon'])):
                map_data['map_polygon'].append(
                    {'num_nodes': self.map_data['map_polygon'][sample_idx]['num_nodes'],
                     'position': self.map_data['map_polygon'][sample_idx]['position'].to(device),
                     'orientation': self.map_data['map_polygon'][sample_idx]['orientation'].to(device),
                     'type': self.map_data['map_polygon'][sample_idx]['type'].to(device),
                     'is_intersection': self.map_data['map_polygon'][sample_idx]['is_intersection'].to(device)}
                )
                map_data['map_point'].append(
                    {'num_nodes': self.map_data['map_point'][sample_idx]['num_nodes'],
                     'position': self.map_data['map_point'][sample_idx]['position'].to(device),
                     'orientation': self.map_data['map_point'][sample_idx]['orientation'].to(device),
                     'magnitude': self.map_data['map_point'][sample_idx]['magnitude'].to(device),
                     'type': self.map_data['map_point'][sample_idx]['type'].to(device),
                     'side': self.map_data['map_point'][sample_idx]['side'].to(device),
                     'tl_statuses': self.map_data['map_point'][sample_idx]['tl_statuses'].to(device)}
                )
                map_data[('map_point', 'to', 'map_polygon')].append(
                    {'edge_index': self.map_data['map_point', 'to', 'map_polygon'][sample_idx]['edge_index'].to(device)}
                )
                map_data[('map_polygon', 'to', 'map_polygon')].append(
                    {
                        'edge_index': self.map_data['map_polygon', 'to', 'map_polygon'][sample_idx]['edge_index'].to(device),
                        'type': self.map_data['map_polygon', 'to', 'map_polygon'][sample_idx]['type'].to(device)
                    }
                )
                map_data['num_pl_detail'].append(self.map_data['num_pl_detail'][sample_idx]),
                map_data['num_pl_to_pl_edge_index_detail'].append(self.map_data['num_pl_to_pl_edge_index_detail'][sample_idx]),
        else:
            map_data = None

        return VectorSetMap(
            coords={
                feature_name: [sample.to(device=device) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data={
                feature_name: [sample.to(device=device) for sample in feature_tl_data]
                for feature_name, feature_tl_data in self.traffic_light_data.items()
            },
            availabilities={
                feature_name: [sample.to(device=device) for sample in feature_avails]
                for feature_name, feature_avails in self.availabilities.items()
            },
            map_data=map_data,
            trajectory_samples=trajectory_samples,
            map_obj_ids=self.map_obj_ids,
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(
            coords=data["coords"],
            traffic_light_data=data["traffic_light_data"],
            availabilities=data["availabilities"],
            map_data=data["map_data"],
            trajectory_samples=data["trajectory_samples"],
            map_obj_ids=data["map_obj_ids"],
            max_elements=data["max_elements"],
            max_points=data["max_points"],
            sanitize=False,
        )

    def unpack(self) -> List[VectorSetMap]:
        """Implemented. See interface."""
        return [
            VectorSetMap(
                {feature_name: [feature_coords[sample_idx]] for feature_name, feature_coords in self.coords.items()},
                {
                    feature_name: [feature_tl_data[sample_idx]]
                    for feature_name, feature_tl_data in self.traffic_light_data.items()
                },
                {
                    feature_name: [feature_avails[sample_idx]]
                    for feature_name, feature_avails in self.availabilities.items()
                },
            )
            for sample_idx in range(self.batch_size)
        ]

    def rotate(self, quaternion: Quaternion) -> VectorSetMap:
        """
        Rotate the vector set map.
        :param quaternion: Rotation to apply.
        :return rotated VectorSetMap.
        """
        # Function only works with numpy arrays
        for feature_coords in self.coords.values():
            for sample in feature_coords:
                validate_type(sample, np.ndarray)

        return VectorSetMap(
            coords={
                feature_name: [rotate_coords(sample, quaternion) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def translate(self, translation_value: FeatureDataType) -> VectorSetMap:
        """
        Translate the vector set map.
        :param translation_value: Translation in x, y, z.
        :return translated VectorSetMap.
        :raise ValueError if translation_value dimensions invalid.
        """
        if translation_value.size != 3:
            raise ValueError(
                f"Translation value has incorrect dimensions: {translation_value.size}! Expected: 3 (x, y, z)"
            )
        are_the_same_type(translation_value, list(self.coords.values())[0])

        return VectorSetMap(
            coords={
                feature_name: [
                    translate_coords(sample_coords, translation_value, sample_avails)
                    for sample_coords, sample_avails in zip(
                        self.coords[feature_name], self.availabilities[feature_name]
                    )
                ]
                for feature_name in self.coords
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def scale(self, scale_value: FeatureDataType) -> VectorSetMap:
        """
        Scale the vector set map.
        :param scale_value: <np.float: 3,>. Scale in x, y, z.
        :return scaled VectorSetMap.
        :raise ValueError if scale_value dimensions invalid.
        """
        if scale_value.size != 3:
            raise ValueError(f"Scale value has incorrect dimensions: {scale_value.size}! Expected: 3 (x, y, z)")
        are_the_same_type(scale_value, list(self.coords.values())[0])

        return VectorSetMap(
            coords={
                feature_name: [scale_coords(sample, scale_value) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def xflip(self) -> VectorSetMap:
        """
        Flip the vector set map along the X-axis.
        :return flipped VectorSetMap.
        """
        return VectorSetMap(
            coords={
                feature_name: [xflip_coords(sample) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def yflip(self) -> VectorSetMap:
        """
        Flip the vector set map along the Y-axis.
        :return flipped VectorSetMap.
        """
        return VectorSetMap(
            coords={
                feature_name: [yflip_coords(sample) for sample in feature_coords]
                for feature_name, feature_coords in self.coords.items()
            },
            traffic_light_data=self.traffic_light_data,
            availabilities=self.availabilities,
        )

    def compress(self) -> VectorSetMap:
        """
        Compress feature to save storage space when caching feature.
        :return: VectorSetMap
        """
        coords_unique: Dict[str, Dict[str, FeatureDataType]] = {}
        traffic_light_data_unique: Dict[str, Dict[str, Dict[str, FeatureDataType]]] = {}
        # availabilities_unique: Dict[str, Dict[str, FeatureDataType]] = {}
        feature_names = ['LANE', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        for feature_name in feature_names:
            if feature_name in self.map_obj_ids.keys():
                for i_traj in self.map_obj_ids[feature_name].keys():
                    for i_pose in self.map_obj_ids[feature_name][i_traj].keys():
                        map_obj_ids = self.map_obj_ids[feature_name][i_traj][i_pose]
                        for i_obj in range(len(map_obj_ids)):
                            if feature_name not in coords_unique.keys():
                                coords_unique[feature_name] = {}
                            if feature_name == 'LANE':
                                if feature_name not in traffic_light_data_unique.keys():
                                    traffic_light_data_unique[feature_name] = {}
                                if i_pose not in traffic_light_data_unique[feature_name].keys():
                                    traffic_light_data_unique[feature_name][i_pose] = {}
                            # if feature_name not in availabilities_unique.keys():
                            #     availabilities_unique[feature_name] = {}

                            # coords_unique
                            if map_obj_ids[i_obj] not in coords_unique[feature_name].keys():
                                coords_unique[feature_name][map_obj_ids[i_obj]] = (
                                    self.coords[feature_name][0][i_traj][i_pose][i_obj])

                                if feature_name == 'LANE':
                                    if 'LEFT_BOUNDARY' not in coords_unique.keys():
                                        coords_unique['LEFT_BOUNDARY'] = {}
                                    coords_unique['LEFT_BOUNDARY'][map_obj_ids[i_obj]] = (
                                        self.coords['LEFT_BOUNDARY'][0][i_traj][i_pose][i_obj])

                                    if 'RIGHT_BOUNDARY' not in coords_unique.keys():
                                        coords_unique['RIGHT_BOUNDARY'] = {}
                                    coords_unique['RIGHT_BOUNDARY'][map_obj_ids[i_obj]] = (
                                        self.coords['RIGHT_BOUNDARY'][0][i_traj][i_pose][i_obj])

                            # traffic_light_data_unique
                            if feature_name == 'LANE':
                                if map_obj_ids[i_obj] not in traffic_light_data_unique[feature_name][i_pose].keys():
                                    traffic_light_data_unique[feature_name][i_pose][map_obj_ids[i_obj]] = (
                                        self.traffic_light_data[feature_name][0][i_traj][i_pose][i_obj])

                            # # availabilities_unique: seems wrong
                            # if map_obj_ids[i_obj] not in availabilities_unique[feature_name].keys():
                            #     availabilities_unique[feature_name][map_obj_ids[i_obj]] = (
                            #         self.availabilities[feature_name][0][i_traj][i_pose][i_obj])
                            #
                            #     if feature_name == 'LANE':
                            #         if 'LEFT_BOUNDARY' not in availabilities_unique.keys():
                            #             availabilities_unique['LEFT_BOUNDARY'] = {}
                            #         availabilities_unique['LEFT_BOUNDARY'][map_obj_ids[i_obj]] = (
                            #             self.availabilities['LEFT_BOUNDARY'][0][i_traj][i_pose][i_obj])
                            #
                            #         if 'RIGHT_BOUNDARY' not in availabilities_unique.keys():
                            #             availabilities_unique['RIGHT_BOUNDARY'] = {}
                            #         availabilities_unique['RIGHT_BOUNDARY'][map_obj_ids[i_obj]] = (
                            #             self.availabilities['RIGHT_BOUNDARY'][0][i_traj][i_pose][i_obj])

        # # str to int
        # map_obj_ids_int: Dict[str, Dict[int, Dict[int, torch.Tensor]]] = {}
        # for feature_name in self.map_obj_ids.keys():
        #     map_obj_ids_int[feature_name] = {}
        #     for i_traj in self.map_obj_ids[feature_name].keys():
        #         map_obj_ids_int[feature_name][i_traj] = {}
        #         for i_pose in self.map_obj_ids[feature_name][i_traj].keys():
        #             ids_list = self.map_obj_ids[feature_name][i_traj][i_pose]
        #             ids_list = [int(id) for id in ids_list]
        #             map_obj_ids_int[feature_name][i_traj][i_pose] = torch.tensor(
        #                 ids_list,
        #                 dtype=torch.int32
        #             )

        max_elements: Dict[str, int] = {}
        max_points: Dict[str, int] = {}
        for feature_name in self.map_obj_ids.keys():
            max_elements[feature_name] = self.coords[feature_name][0][0][0].shape[0]
            max_points[feature_name] = self.coords[feature_name][0][0][0].shape[1]

        return VectorSetMap(
            coords=coords_unique,
            traffic_light_data=traffic_light_data_unique,
            availabilities=self.availabilities,
            map_data=self.map_data,
            map_obj_ids=self.map_obj_ids,
            neighbor_ids=self.neighbor_ids,
            trajectory_samples=self.trajectory_samples,
            max_elements=max_elements,
            max_points=max_points,
            sanitize=False,
            compressed=True,
        )

    def decompress(self) -> VectorSetMap:
        """
        Decompress loaded feature to original feature.
        :return: VectorSetMap
        """
        coords: Dict[str, List[FeatureDataType]] = {}
        traffic_light_data: Dict[str, List[FeatureDataType]] = {}
        # availabilities: Dict[str, List[FeatureDataType]] = {}
        num_trajs = len(self.map_obj_ids['LANE'])
        num_poses = len(self.map_obj_ids['LANE'][0])
        max_elements = self.max_elements
        max_points = self.max_points
        feature_names = ['LANE', 'STOP_LINE', 'CROSSWALK', 'ROUTE_LANES']
        for feature_name in feature_names:
            # coords
            coords[feature_name] = [
                np.zeros((num_trajs,
                          num_poses,
                          max_elements[feature_name],
                          max_points[feature_name], 2),)
            ]
            if feature_name == 'LANE':
                coords['LEFT_BOUNDARY'] = [
                    np.zeros((num_trajs,
                              num_poses,
                              max_elements[feature_name],
                              max_points[feature_name], 2),)
                ]
                coords['RIGHT_BOUNDARY'] = [
                    np.zeros((num_trajs,
                              num_poses,
                              max_elements[feature_name],
                              max_points[feature_name], 2),)
                ]

            # traffic_light_data
            if feature_name == 'LANE':
                traffic_light_data[feature_name] = [
                    np.zeros((num_trajs,
                              num_poses,
                              max_elements[feature_name],
                              max_points[feature_name], self._traffic_light_status_dim),)
                ]

            # # availabilities
            # traffic_light_data[feature_name] = [
            #     np.zeros((num_trajs,
            #               num_poses,
            #               max_elements[feature_name],
            #               max_points[feature_name]),)
            # ]

            for i_traj in range(num_trajs):
                for i_pose in range(num_poses):
                    map_obj_ids = self.map_obj_ids[feature_name][i_traj][i_pose]
                    coords_tmp = []
                    for map_obj_id in map_obj_ids:
                        coords_tmp.append(self.coords[feature_name][map_obj_id])
                    if len(coords_tmp) > 0:
                        coords[feature_name][0][i_traj, i_pose][:len(coords_tmp)] = torch.tensor(coords_tmp)

                    if feature_name == 'LANE':
                        coords_left_tmp = []
                        coords_right_tmp = []
                        traffic_light_data_tmp = []
                        for map_obj_id in map_obj_ids:
                            coords_left_tmp.append(self.coords['LEFT_BOUNDARY'][map_obj_id])
                            coords_right_tmp.append(self.coords['RIGHT_BOUNDARY'][map_obj_id])
                            traffic_light_data_tmp.append(self.traffic_light_data[feature_name][i_pose][map_obj_id])
                        if len(coords_left_tmp) > 0:
                            coords['LEFT_BOUNDARY'][0][i_traj, i_pose][:len(coords_left_tmp)] = torch.tensor(coords_left_tmp)
                        if len(coords_right_tmp) > 0:
                            coords['RIGHT_BOUNDARY'][0][i_traj, i_pose][:len(coords_right_tmp)] = torch.tensor(coords_right_tmp)
                        if len(traffic_light_data_tmp) > 0:
                            traffic_light_data[feature_name][0][i_traj][i_pose][:len(traffic_light_data_tmp)] = torch.tensor(traffic_light_data_tmp)

        return VectorSetMap(
            coords=coords,
            traffic_light_data=traffic_light_data,
            availabilities=self.availabilities,
            map_data=self.map_data,
            map_obj_ids=self.map_obj_ids,
            neighbor_ids=self.neighbor_ids,
            trajectory_samples=self.trajectory_samples,
            max_elements=max_elements,
            max_points=max_points,
            sanitize=True,
        )

