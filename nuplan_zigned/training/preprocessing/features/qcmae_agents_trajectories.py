from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import cached_property
import torch
from collections import defaultdict

from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType, to_tensor

from nuplan.planning.training.preprocessing.features.trajectory import Trajectory

@dataclass
class AgentsTrajectories(AbstractModelFeature):
    """
    Collection of ego's and other agents' Trajectory.
    """

    trajectories: List[Dict[str, Optional[Trajectory, FeatureDataType]]]
    track_token_ids: List[List[str]]
    objects_types: List[List[str]]
    predict_mask: List[FeatureDataType]
    trajectories_global: Optional[List[Dict[str, Optional[Trajectory, FeatureDataType]]]] = None
    velocity_global:Optional[List[Dict[str, Optional[FeatureDataType]]]] = None
    enable_to_device = True

    @cached_property
    def is_valid(self) -> bool:
        """Inherited, see superclass."""
        return self.batch_size > 0

    @property
    def batch_size(self) -> int:
        return len(self.track_token_ids)

    @classmethod
    def collate(cls, batch: List[AgentsTrajectories]) -> AgentsTrajectories:
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        if isinstance(batch[0].trajectories, dict):
            trajectories = defaultdict(list)
        else:
            trajectories = []
        if isinstance(batch[0].trajectories_global, dict):
            trajectories_global = defaultdict(list)
        else:
            trajectories_global = []
        if isinstance(batch[0].velocity_global, dict):
            velocity_global = defaultdict(list)
        else:
            velocity_global = []
        track_token_ids = []
        objects_types = []
        predict_mask = []
        for item in batch:
            if isinstance(item.trajectories, dict):
                for key, value in item.trajectories.items():
                    trajectories[key] += value
            else:
                trajectories += item.trajectories
            if isinstance(item.trajectories_global, dict):
                for key, value in item.trajectories_global.items():
                    trajectories_global[key] += value
            elif item.trajectories_global is None:
                trajectories_global = None
            else:
                trajectories_global += item.trajectories_global
            if isinstance(item.velocity_global, dict):
                for key, value in item.velocity_global.items():
                    velocity_global[key] += value
            elif item.velocity_global is None:
                velocity_global = None
            else:
                velocity_global += item.velocity_global
            track_token_ids += item.track_token_ids
            objects_types += item.objects_types
            predict_mask += item.predict_mask
        return AgentsTrajectories(
            trajectories=trajectories,
            trajectories_global=trajectories_global,
            velocity_global=velocity_global,
            track_token_ids=track_token_ids,
            objects_types=objects_types,
            predict_mask=predict_mask,
        )

    def to_feature_tensor(self) -> AgentsTrajectories:
        """Implemented. See interface."""
        try:
            return AgentsTrajectories(
                trajectories=[
                    {id: to_tensor(data['data']) for id, data in sample.items()}
                    for sample in self.trajectories
                ],
                trajectories_global=[
                    {id: to_tensor(data['data']) for id, data in sample.items()}
                    for sample in self.trajectories_global
                ] if self.trajectories_global is not None else None,
                velocity_global=[
                    {id: to_tensor(data) for id, data in sample.items()}
                    for sample in self.velocity_global
                ] if self.velocity_global is not None else None,
                track_token_ids=self.track_token_ids,
                objects_types=self.objects_types,
                predict_mask=[to_tensor(data) for data in self.predict_mask]
            )
        except:
            return AgentsTrajectories(
                trajectories=[
                    {id: to_tensor(data.data) for id, data in sample.items()}
                    for sample in self.trajectories
                ],
                trajectories_global=[
                    {id: to_tensor(data.data) for id, data in sample.items()}
                    for sample in self.trajectories_global
                ] if self.trajectories_global is not None else None,
                velocity_global=[
                    {id: to_tensor(data) for id, data in sample.items()}
                    for sample in self.velocity_global
                ] if self.velocity_global is not None else None,
                track_token_ids=self.track_token_ids,
                objects_types=self.objects_types,
                predict_mask=[to_tensor(data) for data in self.predict_mask]
            )

    def to_device(self, device: torch.device) -> AgentsTrajectories:
        """Implemented. See interface."""
        if self.enable_to_device:
            trajectories_global = None
            velocity_global = None
            if isinstance(self.trajectories, list):
                trajectories = [
                    {id: traj.to(device) for id, traj in sample.items()}
                    for sample in self.trajectories
                ]
            elif isinstance(self.trajectories, dict):
                trajectories = {
                    key: [sample.to(device) for sample in value]
                    for key, value in self.trajectories.items()
                }
            if isinstance(self.trajectories_global, list):
                trajectories_global = [
                    {id: traj.to(device) for id, traj in sample.items()}
                    for sample in self.trajectories_global
                ]
            elif isinstance(self.trajectories_global, dict):
                trajectories_global = {
                    key: [sample.to(device) for sample in value]
                    for key, value in self.trajectories_global.items()
                }
            if isinstance(self.velocity_global, list):
                velocity_global = [
                    {id: traj.to(device) for id, traj in sample.items()}
                    for sample in self.velocity_global
                ]
            elif isinstance(self.velocity_global, dict):
                velocity_global = {
                    key: [sample.to(device) for sample in value]
                    for key, value in self.velocity_global.items()
                }

            return AgentsTrajectories(
                trajectories=trajectories,
                trajectories_global=trajectories_global,
                velocity_global=velocity_global,
                track_token_ids=self.track_token_ids,
                objects_types=self.objects_types,
                predict_mask=[data.to(device) for data in self.predict_mask]
            )

        else:
            return self

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> AgentsTrajectories:
        """Implemented. See interface."""
        return AgentsTrajectories(
            trajectories=data['trajectories'],
            trajectories_global=data['trajectories_global'] if 'trajectories_global' in data.keys() else None,
            velocity_global=data['velocity_global'] if 'velocity_global' in data.keys() else None,
            track_token_ids=data['track_token_ids'],
            objects_types=data['objects_types'],
            predict_mask=data['predict_mask']
        )

    def unpack(self) -> List[AgentsTrajectories]:
        """Implemented. See interface."""
        return [
            AgentsTrajectories(
                trajectories=[self.trajectories[i]],
                trajectories_global=[self.trajectories_global[i]],
                velocity_global=[self.velocity_global[i]],
                track_token_ids=[self.track_token_ids[i]],
                objects_types=[self.objects_types[i]],
                predict_mask=[self.predict_mask[i]],
            )
            for i in range(self.batch_size)
        ]

    def pseudo_feature(self) -> AgentsTrajectories:
        trajectories_global = None
        velocity_global = None
        trajectories = [
            {'AV': torch.Tensor([0.])}
            for sample in self.trajectories
        ]
        trajectories_global = [
            {'AV': torch.Tensor([0.])}
            for sample in self.trajectories_global
        ]
        velocity_global = [
            {'AV': torch.Tensor([0.])}
            for sample in self.velocity_global
        ]

        return AgentsTrajectories(
            trajectories=trajectories,
            trajectories_global=trajectories_global,
            velocity_global=velocity_global,
            track_token_ids=self.track_token_ids,
            objects_types=self.objects_types,
            predict_mask=[torch.Tensor([0.]) for data in self.predict_mask]
        )