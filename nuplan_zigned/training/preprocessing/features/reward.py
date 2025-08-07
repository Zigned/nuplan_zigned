from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor


@dataclass
class Reward(AbstractModelFeature):
    """
    Dataclass that holds reward signals produced from the model for supervision.

    :param data: either a [num_batches, num_trajs, num_poses] or [num_trajs, num_poses].
    """

    data: FeatureDataType

    def __post_init__(self) -> None:
        """Sanitize attributes of the dataclass."""
        array_dims = self.num_dimensions

        if (array_dims != 2) and (array_dims != 3):
            raise RuntimeError(f'Invalid trajectory array. Expected 2 or 3 dims, got {array_dims}.')

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return self.data.shape[0] > 0 and self.data.shape[1] > 0 and self.data.shape[2] > 0

    def to_device(self, device: torch.device) -> Reward:
        """Implemented. See interface."""
        validate_type(self.data, torch.Tensor)
        return Reward(data=self.data.to(device=device))

    def to_feature_tensor(self) -> Reward:
        """Inherited, see superclass."""
        return Reward(data=to_tensor(self.data))

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> Reward:
        """Implemented. See interface."""
        return Reward(data=data["data"])

    def unpack(self) -> List[Reward]:
        """Implemented. See interface."""
        return [Reward(data[None]) for data in self.data]

    @property
    def num_dimensions(self) -> int:
        """
        :return: dimensions of underlying data
        """
        return len(self.data.shape)

    @property
    def num_of_trajs(self) -> int:
        """
        :return: number of sampled trajectories
        """
        return int(self.data.shape[1])

    @property
    def num_batches(self) -> Optional[int]:
        """
        :return: number of batches in the reward, None if reward does not have batch dimension
        """
        return None if self.num_dimensions <= 2 else self.data.shape[0]
