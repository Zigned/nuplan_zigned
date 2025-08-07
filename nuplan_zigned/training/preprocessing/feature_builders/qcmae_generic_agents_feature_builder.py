#   Adapted from:
#   https://github.com/ZikangZhou/QCNet (Apache License 2.0)

from typing import Dict, List, Tuple, Type, cast, Union, Any, Optional

import numpy as np
import torch

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    sampled_past_timestamps_to_tensor,
)

from nuplan_zigned.training.preprocessing.utils.qcmae_agents_preprocessing import (
    get_agent_states,
    pack_agents_tensor,
    sampled_past_ego_states_to_tensor,
    sampled_tracked_objects_to_tensor_list,
    filter_agents_tensor,
    filter_agents_by_distance,
)
from nuplan_zigned.training.preprocessing.features.qcmae_generic_agents import (
    GenericAgents,
    GenericAgentFeatureIndex,
)


class GenericAgentsFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self,
                 agent_features: List[str],
                 trajectory_sampling: TrajectorySampling,
                 num_future_steps: int,
                 a2a_radius: float,
                 max_agents: Dict[str, int]) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param trajectory_sampling: Parameters of the sampled trajectory of every agent
        """
        super().__init__()
        self.agent_features = agent_features
        self.num_past_steps = trajectory_sampling.num_poses
        self.past_time_horizon = trajectory_sampling.time_horizon
        self.num_historical_steps = self.num_past_steps + 1  # hist = past + present
        self.num_future_steps = num_future_steps
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self.a2a_radius = a2a_radius
        self.max_agents = max_agents

        self._agents_states_dim = GenericAgents.agents_states_dim()
        self._agent_types = agent_features

        # Sanitize feature building parameters
        if 'EGO' in self.agent_features:
            raise AssertionError("EGO not valid agents feature type!")
        for feature_name in self.agent_features:
            if feature_name not in TrackedObjectType._member_names_:
                raise ValueError(f"Object representation for layer: {feature_name} is unavailable!")

    @torch.jit.unused
    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "generic_agents"

    @torch.jit.unused
    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return GenericAgents  # type: ignore

    @torch.jit.unused
    def get_scriptable_input_from_scenario(
        self, scenario: AbstractScenario
    ) -> Tuple[Dict[str, torch.Tensor],
    Dict[str, List[torch.Tensor]],
    Dict[str, List[List[torch.Tensor]]],
    Dict[str, Union[Dict[str, int], Any]]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        anchor_ego_state = scenario.initial_ego_state

        past_ego_states = scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_steps, time_horizon=self.past_time_horizon
        )
        sampled_past_ego_states = list(past_ego_states) + [anchor_ego_state]
        time_stamps = list(
            scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_steps, time_horizon=self.past_time_horizon
            )
        ) + [scenario.start_time]
        # Retrieve past/present agent boxes
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_steps
            )
        ]

        # Extract and pad features
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]

        assert len(sampled_past_ego_states) == len(sampled_past_observations), (
            "Expected the trajectory length of ego and agent to be equal. "
            f"Got ego: {len(sampled_past_ego_states)} and agent: {len(sampled_past_observations)}"
        )

        assert len(sampled_past_observations) > 2, (
            "Trajectory of length of " f"{len(sampled_past_observations)} needs to be at least 3"
        )

        tensor, list_tensor, list_list_tensor, tracked_token_ids = self._pack_to_feature_tensor_dict(
            sampled_past_ego_states, time_stamps, sampled_past_observations
        )
        return tensor, list_tensor, list_list_tensor, tracked_token_ids

    @torch.jit.unused
    def get_scriptable_input_from_simulation(
        self, current_input: PlannerInput
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, List[torch.Tensor]],
        Dict[str, List[List[torch.Tensor]]],
        Dict[str, Union[Dict[str, int], Any]]
    ]:
        """
        Extract the input for the scriptable forward method from the simulation input
        :param current_input: planner input from sim
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        history = current_input.history
        assert isinstance(
            history.observations[0], DetectionsTracks
        ), f"Expected observation of type DetectionTracks, got {type(history.observations[0])}"

        present_ego_state, present_observation = history.current_state

        past_observations = history.observations[:-1]
        past_ego_states = history.ego_states[:-1]

        assert history.sample_interval, "SimulationHistoryBuffer sample interval is None"

        indices = sample_indices_with_time_horizon(self.num_past_steps, self.past_time_horizon, history.sample_interval)

        try:
            sampled_past_observations = [
                cast(DetectionsTracks, past_observations[-idx]).tracked_objects for idx in reversed(indices)
            ]
            sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
        except IndexError:
            raise RuntimeError(
                f"SimulationHistoryBuffer duration: {history.duration} is "
                f"too short for requested past_time_horizon: {self.past_time_horizon}. "
                f"Please increase the simulation_buffer_duration in default_simulation.yaml"
            )

        sampled_past_observations = sampled_past_observations + [
            cast(DetectionsTracks, present_observation).tracked_objects
        ]
        sampled_past_ego_states = sampled_past_ego_states + [present_ego_state]
        time_stamps = [state.time_point for state in sampled_past_ego_states]

        tensor, list_tensor, list_list_tensor, tracked_token_ids = self._pack_to_feature_tensor_dict(
            sampled_past_ego_states, time_stamps, sampled_past_observations
        )
        return tensor, list_tensor, list_list_tensor, tracked_token_ids

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> GenericAgents:
        """Inherited, see superclass."""
        # Retrieve present/past ego states and agent boxes
        with torch.no_grad():
            tensors, list_tensors, list_list_tensors, tracked_token_ids = self.get_scriptable_input_from_scenario(scenario)
            tensors, list_tensors, list_list_tensors, tracked_token_ids = self.scriptable_forward(tensors, list_tensors, list_list_tensors, tracked_token_ids)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors, tracked_token_ids)
            return output

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> GenericAgents:
        """Inherited, see superclass."""
        with torch.no_grad():
            tensors, list_tensors, list_list_tensors, tracked_token_ids = self.get_scriptable_input_from_simulation(current_input)
            tensors, list_tensors, list_list_tensors, tracked_token_ids = self.scriptable_forward(tensors, list_tensors, list_list_tensors, tracked_token_ids)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_tensors, list_list_tensors, tracked_token_ids)
            return output

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        past_ego_states: List[EgoState],
        past_time_stamps: List[TimePoint],
        past_tracked_objects: List[TrackedObjects],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]], Dict[str, Union[Dict[str, int], Any]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param past_ego_states: The past states of the ego vehicle.
        :param past_time_stamps: The past time stamps of the input data.
        :param past_tracked_objects: The past tracked objects.
        :return: The packed tensors.
        """
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(past_ego_states)
        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)
        track_token_ids = {}

        for feature_name in self.agent_features:
            past_tracked_objects_tensor_list, track_token_ids[feature_name] = sampled_tracked_objects_to_tensor_list(
                past_tracked_objects, TrackedObjectType[feature_name]
            )
            list_tensor_data[f"past_tracked_objects.{feature_name}"] = past_tracked_objects_tensor_list

        return (
            {
                "past_ego_states": past_ego_states_tensor,
                "past_time_stamps": past_time_stamps_tensor,
            },
            list_tensor_data,
            {},
            track_token_ids,
        )

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
        tracked_token_ids: Dict[str, Union[Dict[str, int], Any]]
    ) -> GenericAgents:
        """
        Unpacks the data returned from the scriptable core into an GenericAgents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_tensor_data: The List[tensor] data output from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed GenericAgents object.
        """
        ego_features = [list_tensor_data["generic_agents.ego"][0].detach().numpy()]
        agent_features = {}
        for key in list_tensor_data:
            if key.startswith("generic_agents.agents."):
                feature_name = key[len("generic_agents.agents.") :]
                agent_features[feature_name] = [[data.detach().numpy() for data in list_tensor_data[key][0]]]

        agent_data = self.get_agent_data(ego_features, agent_features, tracked_token_ids)

        return GenericAgents(agent_data=agent_data, tracked_token_ids=[tracked_token_ids])

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
        tracked_token_ids: Dict[str, Union[Dict[str, int], Any]],
    ) -> Tuple[Dict[str, torch.Tensor],
    Dict[str, List[torch.Tensor]],
    Dict[str, List[List[torch.Tensor]]],
    Dict[str, Union[Dict[str, int], Any]]]:
        """
        Inherited. See interface.
        """
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, Union[List[torch.Tensor], List[List[torch.Tensor]]]] = {}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}

        ego_history: torch.Tensor = tensor_data["past_ego_states"]
        time_stamps: torch.Tensor = tensor_data["past_time_stamps"]
        anchor_ego_state = ego_history[-1, :].squeeze()

        # ego features
        output_list_dict["generic_agents.ego"] = [ego_history]

        # agent features
        for feature_name in self.agent_features:

            if f"past_tracked_objects.{feature_name}" in list_tensor_data:
                agents: List[torch.Tensor] = list_tensor_data[f"past_tracked_objects.{feature_name}"]
                agent_history, tracked_token_ids[feature_name] = filter_agents_tensor(agents,
                                                                                      tracked_token_ids[feature_name],
                                                                                      reverse=True)

                if agent_history[-1].shape[0] == 0:
                    # Return zero array when there are no agents in the scene
                    agents_tensor: List[torch.Tensor] = [torch.zeros((0, self._agents_states_dim)).float()]
                else:
                    list_agent_states = get_agent_states(agent_history)
                    list_agent_states, tracked_token_ids[feature_name] = filter_agents_by_distance(
                        ego_history,
                        list_agent_states,
                        tracked_token_ids[feature_name],
                        self.a2a_radius,
                        self.max_agents[feature_name] if feature_name in self.max_agents else None
                    )
                    agents_tensor = pack_agents_tensor(list_agent_states)

                output_list_dict[f"generic_agents.agents.{feature_name}"] = [agents_tensor]

        return output_dict, output_list_dict, output_list_list_dict, tracked_token_ids

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {
            "past_ego_states": {
                "iteration": "0",
                "num_samples": str(self.num_past_steps),
                "time_horizon": str(self.past_time_horizon),
            },
            "past_time_stamps": {
                "iteration": "0",
                "num_samples": str(self.num_past_steps),
                "time_horizon": str(self.past_time_horizon),
            },
            "past_tracked_objects": {
                "iteration": "0",
                "time_horizon": str(self.past_time_horizon),
                "num_samples": str(self.num_past_steps),
                "agent_features": ",".join(self.agent_features),
            },
        }

    def get_agent_data(self,
                       ego_features: List[np.ndarray],
                       agent_features: Dict[str, List[List[np.ndarray]]],
                       tracked_token_ids: Dict[str, Union[Dict[str, int], Any]]) -> Dict[str, List[np.ndarray]]:
        agent_features['VEHICLE'][0] = ego_features + agent_features['VEHICLE'][0]  # combine ego and agent features
        agent_ids = ['AV',]
        num_agents_detail = {}
        agents_data = []
        agent_types = []
        for feature_name, feature in agent_features.items():
            num_agents_detail[feature_name] = len(feature)  # VEHICLE: EGO + other VEHICLE
            agent_ids += list(tracked_token_ids[feature_name].keys())
            list_features = []
            for f in feature[0]:
                if f.shape[0] > 0:
                    list_features.append(f)
            agents_data += list_features
            agent_types += [self._agent_types.index(feature_name)] * len(list_features)
        agent_types = np.array(agent_types, dtype=int)

        num_agents = len(agent_ids)
        av_idx = agent_ids.index('AV')

        # initialization
        valid_mask = np.zeros((num_agents, self.num_steps), dtype=bool)
        current_valid_mask = np.zeros((num_agents,), dtype=bool)
        predict_mask = np.zeros((num_agents, self.num_steps), dtype=bool)
        position = np.zeros((num_agents, self.num_historical_steps, 2), dtype=float)
        heading = np.zeros((num_agents, self.num_historical_steps), dtype=float)
        velocity = np.zeros((num_agents, self.num_historical_steps, 2), dtype=float)

        for agent_id, agent_data in zip(agent_ids, agents_data):
            agent_idx = agent_ids.index(agent_id)
            agent_steps = agent_data[:, GenericAgentFeatureIndex.timestep()].astype(int)

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx] = True  # default to predict all future steps

            # a time step t is valid only when both t and t-1 are valid
            valid_mask[agent_idx, 1: self.num_historical_steps] = (
                    valid_mask[agent_idx, :self.num_historical_steps - 1] &
                    valid_mask[agent_idx, 1: self.num_historical_steps])
            valid_mask[agent_idx, 0] = False

            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            position[agent_idx, agent_steps, :2] = np.stack([agent_data[:, GenericAgentFeatureIndex.x()],
                                                             agent_data[:, GenericAgentFeatureIndex.y()]],
                                                            axis=-1)
            heading[agent_idx, agent_steps] = agent_data[:, GenericAgentFeatureIndex.heading()]
            velocity[agent_idx, agent_steps, :2] = np.stack([agent_data[:, GenericAgentFeatureIndex.vx()],
                                                             agent_data[:, GenericAgentFeatureIndex.vy()]],
                                                            axis=-1)

        # if self.split == 'test':
        #     predict_mask[current_valid_mask
        #                  | (agent_category == 2)
        #                  | (agent_category == 3), self.num_historical_steps:] = True

        return {
            'num_nodes': [num_agents],
            'av_index': [av_idx],
            'valid_mask': [valid_mask],
            'predict_mask': [predict_mask],
            'id': [agent_ids],
            'type': [agent_types],
            'position': [position],
            'heading': [heading],
            'velocity': [velocity],
        }