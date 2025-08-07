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
)

from nuplan_zigned.training.preprocessing.feature_builders.qcmae_feature_builder_utils import to_tensor


@dataclass
class VectorSetMap(AbstractModelFeature):
    """
    Vector set map data structure, including:
        map_data: Dict[Union[str, Tuple[str, str, str]], Dict[str, List[Any]]].

    For each map feature, the top level List represents number of samples per batch.
    This is a special feature where each batch entry can have a different size. For that reason, the
    features can not be placed to a single tensor, and we batch the feature with a custom `collate` function.
    """

    map_data: Dict[Union[str, Tuple[str, str, str]], Dict[str, List[Any]]]
    _polyline_coord_dim: int = 2
    _traffic_light_status_dim: int = LaneSegmentTrafficLightData.encoding_dim()
    enable_to_device = True

    def __post_init__(self) -> None:
        """
        Sanitize attributes of the dataclass.
        :raise RuntimeError if dimensions invalid.
        """
        # Check empty data
        if not len(self.map_data) > 0:
            raise RuntimeError("map_data cannot be empty!")

        if not all([[len(d) > 0 for d in data.values()] if isinstance(data, dict) else True for data in self.map_data.values()]):
            raise RuntimeError("Batch size has to be > 0!")

        self._sanitize_feature_consistency()

    def _sanitize_feature_consistency(self) -> None:
        """
        Check data dimensionality consistent across and within map features.
        :raise RuntimeError if dimensions invalid.
        """
        # Check consistency across map features
        for data in self.map_data.values():
            if isinstance(data, dict):
                for data_data in data.values():
                    if len(data_data) != self.batch_size:
                        raise RuntimeError("Batch size inconsistent across features!")

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return self.batch_size > 0

    @property
    def batch_size(self) -> int:
        """
        Batch size across features.
        :return: number of batches.
        """
        return len(self.map_data['map_polygon']['num_nodes'])  # All features guaranteed to have same batch size

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

    @classmethod
    def collate(cls, batch: List[VectorSetMap]) -> VectorSetMap:
        """Implemented. See interface."""
        map_data: Dict[str, Dict[str, List[FeatureDataType]]] = {}

        for sample in batch:
            for feature_name, feature_data in sample.map_data.items():
                if feature_name not in ['num_pl_detail', 'num_pl_to_pl_edge_index_detail']:
                    if feature_name not in map_data.keys():
                        map_data[feature_name] = defaultdict(list)
                    for key, value in feature_data.items():
                        map_data[feature_name][key] += value
                else:
                    if feature_name not in map_data.keys():
                        map_data[feature_name] = []
                    map_data[feature_name] += [feature_data[0]]

        return VectorSetMap(map_data=map_data)

    def to_feature_tensor(self) -> VectorSetMap:
        """Implemented. See interface."""
        map_data: Dict[str, Dict[str, List[FeatureDataType]]] = {}

        for feature_name, feature_data in self.map_data.items():
            if feature_name not in map_data.keys():
                map_data[feature_name] = defaultdict(list)
            if feature_name not in ['num_pl_detail', 'num_pl_to_pl_edge_index_detail']:
                for data_name, data_value in feature_data.items():
                    map_data[feature_name][data_name].append(to_tensor(data_value[0]))

        map_data['num_pl_detail'] = self.map_data['num_pl_detail']
        map_data['num_pl_to_pl_edge_index_detail'] = self.map_data['num_pl_to_pl_edge_index_detail']

        return VectorSetMap(map_data=map_data)

    def to_device(self, device: torch.device) -> VectorSetMap:
        """Implemented. See interface."""
        if self.enable_to_device:
            map_data: Dict[str, Dict[str, List[FeatureDataType]]] = {}

            for feature_name, feature_data in self.map_data.items():
                if feature_name not in map_data.keys():
                    map_data[feature_name] = defaultdict(list)
                if feature_name not in ['num_pl_detail', 'num_pl_to_pl_edge_index_detail']:
                    for data_name, data_value in feature_data.items():
                        for sample in data_value:
                            if isinstance(sample, torch.Tensor):
                                tensor = sample.to(device)
                            else:
                                tensor = sample
                            map_data[feature_name][data_name].append(tensor)

            map_data['num_pl_detail'] = self.map_data['num_pl_detail']
            map_data['num_pl_to_pl_edge_index_detail'] = self.map_data['num_pl_to_pl_edge_index_detail']

            return VectorSetMap(map_data=map_data)

        else:
            return self

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> VectorSetMap:
        """Implemented. See interface."""
        return VectorSetMap(map_data=data["map_data"])

    def unpack(self) -> List[VectorSetMap]:
        """Implemented. See interface."""
        features = []
        for sample_idx in range(self.batch_size):
            map_data: Dict[str, Dict[str, List[FeatureDataType]]] = {}
            for feature_name, feature_data in self.map_data.items():
                if feature_name not in ['num_pl_detail', 'num_pl_to_pl_edge_index_detail']:
                    if feature_name not in map_data.keys():
                        map_data[feature_name] = defaultdict(list)
                    for key, value in feature_data.items():
                        map_data[feature_name][key] = [value[sample_idx]]
                else:
                    map_data[feature_name] = [feature_data[sample_idx]]
            features.append(VectorSetMap(map_data=map_data))

        return features

    def pseudo_feature(self) -> VectorSetMap:
        map_data: Dict[str, Dict[str, List[FeatureDataType]]] = {}

        for feature_name, feature_data in self.map_data.items():
            if feature_name not in map_data.keys():
                map_data[feature_name] = defaultdict(list)
            if feature_name not in ['num_pl_detail', 'num_pl_to_pl_edge_index_detail']:
                for data_name, data_value in feature_data.items():
                    for sample in data_value:
                        if isinstance(sample, torch.Tensor):
                            tensor = torch.Tensor([0.])
                        else:
                            tensor = sample
                        map_data[feature_name][data_name].append(tensor)

        map_data['num_pl_detail'] = self.map_data['num_pl_detail']
        map_data['num_pl_to_pl_edge_index_detail'] = self.map_data['num_pl_to_pl_edge_index_detail']

        return VectorSetMap(map_data=map_data)