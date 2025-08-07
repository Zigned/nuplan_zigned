from __future__ import annotations

from typing import Dict, List, Tuple, Type, cast, Union, Any, Optional

import torch
import numpy as np

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
from nuplan.planning.training.preprocessing.features.abstract_model_feature import FeatureDataType
from nuplan.planning.training.preprocessing.features.agents import AgentFeatureIndex
from nuplan.planning.training.preprocessing.feature_builders.scriptable_feature_builder import ScriptableFeatureBuilder
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    convert_absolute_quantities_to_relative,
    filter_agents_tensor,
    pad_agent_states,
    sampled_tracked_objects_to_tensor_list,
)

from nuplan_zigned.training.preprocessing.feature_builders.avrl_feature_builder_utils import (
    future_ego_states_to_tensor,
    future_timestamps_to_tensor,
    get_sampled_future_ego_states,
)
from nuplan_zigned.training.preprocessing.features.avrl_generic_agents import GenericAgents
from nuplan_zigned.training.preprocessing.utils.avrl_agents_preprocessing import (
    build_generic_ego_features_from_tensor,
    compute_yaw_rate_from_state_tensors,
    pack_agents_tensor,
    AgentInternalIndex,
)
from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories
from nuplan_zigned.training.preprocessing.features.qcmae_trajectory import Trajectory


class GenericAgentsFeatureBuilder(ScriptableFeatureBuilder):
    """Builder for constructing agent features during training and simulation."""

    def __init__(self, agent_features: List[str], num_poses: int, time_horizon: float, max_agents: int) -> None:
        """
        Initializes AgentsFeatureBuilder.
        :param agent_features: agent type name of interest
        :param num_poses: number of poses in future trajectory in addition to initial state.
        :param time_horizon: [s] time horizon of all poses.
        """
        super().__init__()
        self.agent_features = agent_features
        self.num_poses = num_poses
        self.time_horizon = time_horizon
        self.max_agents = max_agents

        self._agents_states_dim = GenericAgents.agents_states_dim()

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
    ) -> Tuple[EgoState, List[
            List[Dict[str, torch.Tensor] | Dict[str, List[torch.Tensor]] | Dict[str, List[List[torch.Tensor]]]]
        ]]:
        """
        Extract the input for the scriptable forward method from the scenario object
        :param scenario: planner input from training
        :returns: Tensor data + tensor list data to be used in scriptable forward
        """
        anchor_ego_state = scenario.initial_ego_state
        future_ego_states = list(
            scenario.get_ego_future_trajectory(
                iteration=0, time_horizon=self.time_horizon, num_samples=self.num_poses
            )
        )
        sampled_future_ego_states = get_sampled_future_ego_states(
            scenario.data_across_builders['trajectory_samples'],
            future_ego_states
        )
        time_stamps = list(
            scenario.get_future_timestamps(
                iteration=0, num_samples=self.num_poses, time_horizon=self.time_horizon
            )
        )
        # Retrieve future agent boxes
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.time_horizon, num_samples=self.num_poses
            )
        ]

        # Extract features
        assert len(future_ego_states) == len(future_tracked_objects), (
            "Expected the trajectory length of ego and agent to be equal. "
            f"Got ego: {len(future_ego_states)} and agent: {len(future_tracked_objects)}"
        )

        assert len(future_tracked_objects) > 2, (
            "Trajectory of length of " f"{len(future_tracked_objects)} needs to be at least 3"
        )

        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
            future_ego_states, time_stamps, future_tracked_objects
        )
        future_features = [[tensor, list_tensor, list_list_tensor]]
        for sampled_future_ego_state in sampled_future_ego_states:
            tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
                sampled_future_ego_state, time_stamps, future_tracked_objects
            )
            future_features.append([tensor, list_tensor, list_list_tensor])

        return anchor_ego_state, future_features

    @torch.jit.unused
    def get_scriptable_input_from_simulation(
        self, current_input: PlannerInput
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
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

        indices = sample_indices_with_time_horizon(self.num_past_poses, self.past_time_horizon, history.sample_interval)

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

        tensor, list_tensor, list_list_tensor = self._pack_to_feature_tensor_dict(
            sampled_past_ego_states, time_stamps, sampled_past_observations
        )
        return tensor, list_tensor, list_list_tensor

    @torch.jit.unused
    def get_features_from_scenario(self, scenario: AbstractScenario) -> GenericAgents:
        """Inherited, see superclass."""
        # Retrieve present/future ego states and agent boxes
        with torch.no_grad():
            anchor_ego_state, future_features = self.get_scriptable_input_from_scenario(scenario)
            list_dict_tensor = []
            for future_feature in future_features:
                tensors, list_tensors, list_list_tensors = future_feature
                tensors, list_tensors, list_list_tensors = self.scriptable_forward(tensors, list_tensors, list_list_tensors)
                list_dict_tensor.append(list_tensors)
            output: GenericAgents = self._unpack_feature_from_tensor_dict(tensors, list_dict_tensor, list_list_tensors)
            return output

    @torch.jit.unused
    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization, scenario: AbstractScenario
    ) -> AgentsTrajectories:
        """Inherited, see superclass."""
        with torch.no_grad():
            output: AgentsTrajectories = self.qcmae_get_agents_targets(current_input, scenario)
            return output

    def qcmae_get_agents_targets(self, current_input: PlannerInput, scenario: AbstractScenario) -> AgentsTrajectories:
        """Modified version of get_targets in qcmae_agents_trajectories_target_builder."""
        """
        --------------------------------|-|--------------------------------------
        |<-------num_past_steps-------->|-|<---------num_future_steps---------->|
                     ^past               ^               ^future
                                         ^current (initial)
        """
        # EGO and AGENTs current
        anchor_ego_state = current_input.history.ego_states[-1]
        anchor_agents_states = scenario.get_tracked_objects_at_iteration(current_input.iteration.index)
        anchor_states = {'AV': anchor_ego_state.agent}
        anchor_states.update({agent.track_token: agent for agent in anchor_agents_states.tracked_objects.tracked_objects})

        # EGO future
        trajectory_absolute_states = list(scenario.get_ego_future_trajectory(
            iteration=current_input.iteration.index, num_samples=self.num_poses, time_horizon=self.time_horizon
        ))
        future_ego_trajectory = [[state.agent.center.x,
                                  state.agent.center.y,
                                  state.agent.center.heading,
                                  ] for state in trajectory_absolute_states]
        future_ego_velocity = [[state.dynamic_car_state.center_velocity_2d.x,
                                state.dynamic_car_state.center_velocity_2d.y,
                                ] for state in trajectory_absolute_states]
        future_ego_velocity = np.array(future_ego_velocity)
        future_ego_trajectory = Trajectory(data=np.array(future_ego_trajectory))

        # other AGENTs future
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=current_input.iteration.index, time_horizon=self.time_horizon, num_samples=self.num_poses
            )
        ]
        future_agents_trajectories = [
            [
                [[agent.center.x,
                 agent.center.y,
                 agent.center.heading],
                 [agent.track_token, agent.tracked_object_type.name],
                 [agent.velocity.x, agent.velocity.y]] for agent in agents.tracked_objects
            ] for agents in future_tracked_objects
        ]  # [num_frames, num_agents, [pose, [track_token, tracked_object_type.name]]]

        # reshape to [num_agents, num_frames, pose]
        reshaped_future_agents_trajectories: Dict[str, Dict[int, List]] = {}
        reshaped_future_agents_velocities: Dict[str, Dict[int, List]] = {}
        objects_types = ['AV', ]
        for timestep in range(len(future_agents_trajectories)):
            for agent in future_agents_trajectories[timestep]:
                if agent[1][0] not in reshaped_future_agents_trajectories.keys():
                    reshaped_future_agents_trajectories[agent[1][0]] = {}
                    reshaped_future_agents_velocities[agent[1][0]] = {}
                    objects_types.append(agent[1][1])
                reshaped_future_agents_trajectories[agent[1][0]][timestep] = agent[0]
                reshaped_future_agents_velocities[agent[1][0]][timestep] = agent[2]

        trajectories = {'AV': future_ego_trajectory}
        velocities = {'AV': future_ego_velocity}
        for agent_id, agent_poses in reshaped_future_agents_trajectories.items():
            traj = np.zeros_like(future_ego_trajectory.data)
            timestamp_range = list(agent_poses.keys())
            traj[timestamp_range] = np.array(list(agent_poses.values()))
            trajectories[agent_id] = Trajectory(data=traj)

            agent_velocities = reshaped_future_agents_velocities[agent_id]
            vel = np.zeros_like(future_ego_velocity)
            vel[timestamp_range] = np.array(list(agent_velocities.values()))
            velocities[agent_id] = vel

        # transform the future trajectory of each agent to its local coordinate system
        transformed_trajectories = {}
        for id, traj in trajectories.items():
            if id in anchor_states.keys():
                transformed_trajectories[id] = Trajectory.transform_to_local_frame_given_anchor_state(data=traj.data, anchor_state=anchor_states[id])
            else:  # those don't exist in the current step will not be transformed
                transformed_trajectories[id] = traj

        # predict mask
        predict_mask = np.zeros((len(reshaped_future_agents_trajectories) + 1, self.num_poses), dtype=bool)
        predict_mask[0, :] = True  # for EGO
        for agent_id, value in reshaped_future_agents_trajectories.items():
            agent_idx = list(reshaped_future_agents_trajectories.keys()).index(agent_id)
            predict_mask[agent_idx + 1, np.array(list(value.keys()))] = True

        return AgentsTrajectories(
            trajectories=[transformed_trajectories],
            trajectories_global=[trajectories],
            velocity_global=[velocities],
            track_token_ids=[['AV']+list(reshaped_future_agents_trajectories.keys())],
            objects_types=[objects_types],
            predict_mask=[predict_mask]
        )

    @torch.jit.unused
    def _pack_to_feature_tensor_dict(
        self,
        future_ego_states: List[EgoState],
        future_time_stamps: List[TimePoint],
        future_tracked_objects: List[TrackedObjects],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Packs the provided objects into tensors to be used with the scriptable core of the builder.
        :param future_ego_states: The future states of the ego vehicle.
        :param future_time_stamps: The future time stamps of the input data.
        :param future_tracked_objects: The future tracked objects.
        :return: The packed tensors.
        """
        list_tensor_data: Dict[str, List[torch.Tensor]] = {}
        future_ego_states_tensor = future_ego_states_to_tensor(future_ego_states)
        future_time_stamps_tensor = future_timestamps_to_tensor(future_time_stamps)

        for feature_name in self.agent_features:
            future_tracked_objects_tensor_list = sampled_tracked_objects_to_tensor_list(
                future_tracked_objects, TrackedObjectType[feature_name]
            )
            list_tensor_data[f"future_tracked_objects.{feature_name}"] = future_tracked_objects_tensor_list

        # consider max number of each type of agents
        for i_pose in range(self.num_poses):
            for feature_name in self.agent_features:
                objects = list_tensor_data[f"future_tracked_objects.{feature_name}"][i_pose]
                distance_to_ego = torch.norm(
                    future_ego_states_tensor[i_pose, [AgentFeatureIndex.x(), AgentFeatureIndex.y()]] -
                    objects[:, [AgentInternalIndex.x(), AgentInternalIndex.y()]],
                    p=2,
                    dim=1
                )
                _, indices = torch.sort(distance_to_ego, descending=False, dim=0)
                objects = objects[indices][:self.max_agents[feature_name]]
                list_tensor_data[f"future_tracked_objects.{feature_name}"][i_pose] = objects

        return (
            {
                "future_ego_states": future_ego_states_tensor,
                "future_time_stamps": future_time_stamps_tensor,
            },
            list_tensor_data,
            {},
        )

    @torch.jit.unused
    def _unpack_feature_from_tensor_dict(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_dict_list_tensor_data: List[Dict[str, List[List[torch.Tensor]]]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> GenericAgents:
        """
        Unpacks the data returned from the scriptable core into an GenericAgents feature class.
        :param tensor_data: The tensor data output from the scriptable core.
        :param list_dict_list_tensor_data: The List[tensor] data outputs from the scriptable core.
        :param list_tensor_data: The List[List[tensor]] data output from the scriptable core.
        :return: The packed GenericAgents object.
        """
        ego_tmp: List[List[FeatureDataType]] = []
        agents_tmp: Dict[str, List[List[FeatureDataType]]] = {}
        for dict_list_tensor_data in list_dict_list_tensor_data:
            ego_features = [data.detach().numpy() for data in dict_list_tensor_data["generic_agents.ego"][0]]
            ego_tmp.append(ego_features)
            for key in dict_list_tensor_data:
                if key.startswith("generic_agents.agents."):
                    feature_name = key[len("generic_agents.agents."):]
                    list_values = []
                    for value in dict_list_tensor_data[key][0]:
                        list_values.append(value.detach().numpy())
                    if feature_name not in agents_tmp.keys():
                        agents_tmp[feature_name] = [list_values]
                    else:
                        agents_tmp[feature_name].append(list_values)

        # add a batch dimension
        ego: List[List[List[FeatureDataType]]] = [ego_tmp]
        agents: Dict[str, List[List[List[FeatureDataType]]]] = {}
        for key, value in agents_tmp.items():
            agents[key] = [value]

        return GenericAgents(ego=ego, agents=agents)

    @torch.jit.export
    def scriptable_forward(
        self,
        tensor_data: Dict[str, torch.Tensor],
        list_tensor_data: Dict[str, List[torch.Tensor]],
        list_list_tensor_data: Dict[str, List[List[torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, List[List[torch.Tensor]]], Dict[str, List[List[torch.Tensor]]]]:
        """
        Inherited. See interface.
        """
        output_dict: Dict[str, torch.Tensor] = {}
        output_list_dict: Dict[str, List[List[torch.Tensor]]] = {}
        output_list_list_dict: Dict[str, List[List[torch.Tensor]]] = {}

        ego_future: torch.Tensor = tensor_data["future_ego_states"]
        time_stamps: torch.Tensor = tensor_data["future_time_stamps"]

        # ego features
        output_list_dict["generic_agents.ego"] = [[ego_future[i, :].unsqueeze(dim=0) for i in range(ego_future.shape[0])]]

        # agent features
        for feature_name in self.agent_features:

            if f"future_tracked_objects.{feature_name}" in list_tensor_data:
                agent_future: List[torch.Tensor] = list_tensor_data[f"future_tracked_objects.{feature_name}"]
                # agent_history = filter_agents_tensor(agents, reverse=True)

                # Calculate yaw rate
                frame_agent_yaw_rates = compute_yaw_rate_from_state_tensors(agent_future, time_stamps)

                agents_tensors, agents_ids = pack_agents_tensor(agent_future, frame_agent_yaw_rates)

                output_list_dict[f"generic_agents.agents.{feature_name}"] = [agents_tensors]
                output_list_dict[f"generic_agents.agents.{feature_name}" + "_ids"] = [agents_ids]

        return output_dict, output_list_dict, output_list_list_dict

    @torch.jit.export
    def precomputed_feature_config(self) -> Dict[str, Dict[str, str]]:
        """
        Inherited. See interface.
        """
        return {
            "past_ego_states": {
                "iteration": "0",
                "num_samples": str(self.num_past_poses),
                "time_horizon": str(self.past_time_horizon),
            },
            "past_time_stamps": {
                "iteration": "0",
                "num_samples": str(self.num_past_poses),
                "time_horizon": str(self.past_time_horizon),
            },
            "past_tracked_objects": {
                "iteration": "0",
                "time_horizon": str(self.past_time_horizon),
                "num_samples": str(self.num_past_poses),
                "agent_features": ",".join(self.agent_features),
            },
        }
