from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
)

from nuplan_zigned.training.preprocessing.feature_builders.qcmae_feature_builder_utils import to_tensor


@dataclass
class GenericAgents(AbstractModelFeature):
    """
    Model input feature representing the present and past states of the ego and agents.

    The structure includes:
        ego: List[<np.ndarray: num_frames, 9>].
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the ego pose (x, y, heading) velocities (vx, vy) width length timestep track_token at time t.
            Example dimensions: 8 (batch_size) x 5 (1 present + 4 past frames) x 9
        agents: Dict[str, List[List[<np.ndarray: num_frames, 9>]]].
            Agent features indexed by agent feature type.
            The outer list is the batch dimension.
            The num_frames includes both present and past frames.
            The last dimension is the ego pose (x, y, heading) velocities (vx, vy) width length timestep track_token at time t.

    The present/past frames dimension is populated in increasing chronological order, i.e. (t_-N, ..., t_-1, t_0)
    where N is the number of frames in the feature

    In both cases, the outer List represent number of batches. This is a special feature where each batch entry
    can have different size. For that reason, the feature can not be placed to a single tensor,
    and we batch the feature with a custom `collate` function
    """

    ego: List[FeatureDataType] = None
    agents: Dict[str, List[List[FeatureDataType]]] = None
    agent_data: Dict = None
    tracked_token_ids: List[Dict[str, Union[Dict[str, int], Any]]] = None
    enable_to_device = True

    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        if self.ego is not None and self.agents is not None:
            if not all([len(self.ego) == len(agent) for agent in self.agents.values()]):
                raise AssertionError("Batch size inconsistent across features!")

            if len(self.ego) == 0:
                raise AssertionError("Batch size has to be > 0!")

            if self.ego[0].ndim != 2:
                raise AssertionError(
                    "Ego feature samples does not conform to feature dimensions! "
                    f"Got ndim: {self.ego[0].ndim} , expected 2 [num_frames, 7]"
                )

            if 'EGO' in self.agents.keys():
                raise AssertionError("EGO not a valid agents feature type!")
            for feature_name in self.agents.keys():
                if feature_name not in TrackedObjectType._member_names_:
                    raise ValueError(f"Object representation for layer: {feature_name} is unavailable!")

            for agent in self.agents.values():
                for agent_i in agent:
                    if agent_i[0].ndim != 2:
                        raise AssertionError(
                            "Agent feature samples does not conform to feature dimensions! "
                            f"Got ndim: {agent_i[0].ndim} , "
                            f"expected 2 [num_frames, 9]"
                        )

    def _validate_ego_query(self, sample_idx: int) -> None:
        """
        Validate ego sample query is valid.
        :param sample_idx: the batch index of interest.
        :raise
            ValueError if sample_idx invalid.
            RuntimeError if feature at given sample index is empty.
        """
        if self.batch_size < sample_idx:
            raise ValueError(f"Requsted sample index {sample_idx} larger than batch size {self.batch_size}!")
        if self.ego[sample_idx].size == 0:
            raise RuntimeError("Feature is empty!")

    def _validate_agent_query(self, agent_type: str, sample_idx: int) -> None:
        """
        Validate agent type, sample query is valid.
        :param agent_type: agent feature type.
        :param sample_idx: the batch index of interest.
        :raise ValueError if agent_type or sample_idx invalid.
        """
        if agent_type not in TrackedObjectType._member_names_:
            raise ValueError(f"Invalid agent type: {agent_type}")
        if agent_type not in self.agents.keys():
            raise ValueError(f"Agent type: {agent_type} is unavailable!")
        if self.batch_size < sample_idx:
            raise ValueError(f"Requsted sample index {sample_idx} larger than batch size {self.batch_size}!")

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return self.batch_size > 0

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches.
        """
        return len(self.agent_data['num_nodes']) if self.agent_data is not None else None

    @classmethod
    def collate(cls, batch: List[GenericAgents]) -> GenericAgents:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        agent_data: Dict[str, List[Union[int, List, FeatureDataType]]] = defaultdict(list)
        for sample in batch:
            for agent_name, agent in sample.agent_data.items():
                agent_data[agent_name] += agent
        return GenericAgents(agent_data=agent_data)

    def to_feature_tensor(self) -> GenericAgents:
        """Implemented. See interface."""
        agent_data: Dict[str, List[Union[int, List, FeatureDataType]]] = defaultdict(list)
        agent_data['num_nodes'] = self.agent_data['num_nodes']
        agent_data['av_index'] = self.agent_data['av_index']
        agent_data['valid_mask'] = [to_tensor(sample) for sample in self.agent_data['valid_mask']]
        agent_data['predict_mask'] = [to_tensor(sample) for sample in self.agent_data['predict_mask']]
        agent_data['id'] = self.agent_data['id']
        agent_data['type'] = [to_tensor(sample) for sample in self.agent_data['type']]
        agent_data['position'] = [to_tensor(sample) for sample in self.agent_data['position']]
        agent_data['heading'] = [to_tensor(sample) for sample in self.agent_data['heading']]
        agent_data['velocity'] = [to_tensor(sample) for sample in self.agent_data['velocity']]
        agent_data['length'] = [to_tensor(sample) for sample in self.agent_data['length']]
        agent_data['width'] = [to_tensor(sample) for sample in self.agent_data['width']]
        return GenericAgents(agent_data=agent_data)

    def to_device(self, device: torch.device) -> GenericAgents:
        """Implemented. See interface."""
        if self.enable_to_device:
            agent_data: Dict[str, List[Union[int, List, FeatureDataType]]] = defaultdict(list)
            agent_data['num_nodes'] = self.agent_data['num_nodes']
            agent_data['av_index'] = self.agent_data['av_index']
            agent_data['valid_mask'] = [sample.to(device) for sample in self.agent_data['valid_mask']]
            agent_data['predict_mask'] = [sample.to(device) for sample in self.agent_data['predict_mask']]
            agent_data['id'] = self.agent_data['id']
            agent_data['type'] = [sample.to(device) for sample in self.agent_data['type']]
            agent_data['position'] = [sample.to(device) for sample in self.agent_data['position']]
            agent_data['heading'] = [sample.to(device) for sample in self.agent_data['heading']]
            agent_data['velocity'] = [sample.to(device) for sample in self.agent_data['velocity']]
            agent_data['length'] = [sample.to(device) for sample in self.agent_data['length']]
            agent_data['width'] = [sample.to(device) for sample in self.agent_data['width']]
            return GenericAgents(agent_data=agent_data)
        else:
            return self

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> GenericAgents:
        """Implemented. See interface."""
        return GenericAgents(agent_data=data["agent_data"])

    def unpack(self) -> List[GenericAgents]:
        """Implemented. See interface."""
        features: List[GenericAgents] = []
        for sample_idx in range(self.batch_size):
            agent_data: Dict[str, List[Union[int, List, FeatureDataType]]] = defaultdict(list)
            for agent_name, agent in self.agent_data.items():
                agent_data[agent_name] = [agent[sample_idx]]
            features.append(GenericAgents(agent_data=agent_data))

        return features

    @staticmethod
    def agents_states_dim() -> int:
        """
        :return: agent state dimension.
        """
        return GenericAgentFeatureIndex.dim()

    def pseudo_feature(self) -> GenericAgents:
        agent_data: Dict[str, List[Union[int, List, FeatureDataType]]] = defaultdict(list)
        agent_data['num_nodes'] = self.agent_data['num_nodes']
        agent_data['av_index'] = self.agent_data['av_index']
        agent_data['valid_mask'] = [torch.Tensor([0.]) for sample in self.agent_data['valid_mask']]
        agent_data['predict_mask'] = [torch.Tensor([0.]) for sample in self.agent_data['predict_mask']]
        agent_data['id'] = self.agent_data['id']
        agent_data['type'] = [torch.Tensor([0.]) for sample in self.agent_data['type']]
        agent_data['position'] = [torch.Tensor([0.]) for sample in self.agent_data['position']]
        agent_data['heading'] = [torch.Tensor([0.]) for sample in self.agent_data['heading']]
        agent_data['velocity'] = [torch.Tensor([0.]) for sample in self.agent_data['velocity']]
        agent_data['length'] = [torch.Tensor([0.]) for sample in self.agent_data['length']]
        agent_data['width'] = [torch.Tensor([0.]) for sample in self.agent_data['width']]
        return GenericAgents(agent_data=agent_data)


class GenericEgoFeatureIndex:
    """
    A convenience class for assigning semantic meaning to the tensor index
        in the final output ego feature.

    It is intended to be used like an IntEnum, but supported by TorchScript.
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x coordinate of the ego.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y coordinate of the ego.
        :return: index
        """
        return 1

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the ego.
        :return: index
        """
        return 2

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the ego.
        :return: index
        """
        return 3

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the ego.
        :return: index
        """
        return 4

    @staticmethod
    def ax() -> int:
        """
        The dimension corresponding to the x acceleration of the ego.
        :return: index
        """
        return 5

    @staticmethod
    def ay() -> int:
        """
        The dimension corresponding to the y acceleration of the ego.
        :return: index
        """
        return 6

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the EgoFeature.
        :return: number of features.
        """
        return 7


class GenericAgentFeatureIndex:
    """
    A convenience class for assigning semantic meaning to the tensor indexes
        in the final output agents feature.

    It is intended to be used like an IntEnum, but supported by TorchScript.
    """

    def __init__(self) -> None:
        """
        Init method.
        """
        raise ValueError("This class is not to be instantiated.")

    @staticmethod
    def x() -> int:
        """
        The dimension corresponding to the x coordinate of the agent.
        :return: index
        """
        return 0

    @staticmethod
    def y() -> int:
        """
        The dimension corresponding to the y coordinate of the agent.
        :return: index
        """
        return 1

    @staticmethod
    def heading() -> int:
        """
        The dimension corresponding to the heading of the agent.
        :return: index
        """
        return 2

    @staticmethod
    def vx() -> int:
        """
        The dimension corresponding to the x velocity of the agent.
        :return: index
        """
        return 3

    @staticmethod
    def vy() -> int:
        """
        The dimension corresponding to the y velocity of the agent.
        :return: index
        """
        return 4

    @staticmethod
    def length() -> int:
        """
        The dimension corresponding to the length of the agent.
        :return: index
        """
        return 5

    @staticmethod
    def width() -> int:
        """
        The dimension corresponding to the width of the agent.
        :return: index
        """
        return 6

    @staticmethod
    def timestep() -> int:
        """
        The dimension corresponding to the timestep of the agent.
        :return: index
        """
        return 7

    @staticmethod
    def track_token() -> int:
        """
        The dimension corresponding to the track token of the agent.
        :return: index
        """
        return 8

    @staticmethod
    def dim() -> int:
        """
        The number of features present in the AgentsFeature.
        :return: number of features.
        """
        return 9
