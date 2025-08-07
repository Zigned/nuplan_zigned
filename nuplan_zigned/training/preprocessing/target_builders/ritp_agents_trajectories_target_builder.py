from __future__ import annotations

from typing import Type, Dict, List
import numpy as np
import copy

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan_zigned.training.preprocessing.features.qcmae_trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories


class AgentTrajectoryTargetBuilder(AbstractTargetBuilder):
    """Trajectory builders constructed the desired agents' trajectory from a scenario."""

    def __init__(self, agent_featrues: List[str], num_past_steps: int, future_trajectory_sampling: TrajectorySampling) -> None:
        """
        Initializes the class.
        :param agent_featrues: agents of interested types to be predicted.
        :param num_past_steps: number of past steps (not including present).
        :param future_trajectory_sampling: parameters for sampled future trajectory.
        """
        self._agent_featrues = agent_featrues
        self._num_historical_steps = num_past_steps + 1
        self._num_future_steps = future_trajectory_sampling.num_poses
        self._time_horizon = future_trajectory_sampling.time_horizon

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "agents_trajectories"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return AgentsTrajectories  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> AgentsTrajectories:
        """Inherited, see superclass."""
        """
        --------------------------------|-|--------------------------------------
        |<-------num_past_steps-------->|-|<---------num_future_steps---------->|
                     ^past               ^               ^future
                                         ^current (initial)
        """
        self.number_of_interations = scenario.get_number_of_iterations()
        self.scenario_time_horizon = self.number_of_interations * scenario.database_interval

        # EGO and AGENTs current
        anchor_ego_state = scenario.initial_ego_state
        anchor_agents_states = scenario.initial_tracked_objects
        anchor_states = {'AV': anchor_ego_state.agent}
        anchor_states.update({agent.track_token: agent for agent in anchor_agents_states.tracked_objects.tracked_objects})

        # EGO future
        trajectory_absolute_states = list(scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.number_of_interations, time_horizon=self.scenario_time_horizon
        ))
        future_ego_trajectory = [[state.agent.center.x,
                                  state.agent.center.y,
                                  state.agent.center.heading] for state in trajectory_absolute_states]
        future_ego_trajectory = Trajectory(data=np.array(future_ego_trajectory))
        future_ego_velocity = np.array(
            [[state.agent.velocity.x * np.cos(state.agent.center.heading)
              - state.agent.velocity.y * np.sin(state.agent.center.heading),
              state.agent.velocity.x * np.sin(state.agent.center.heading)
              + state.agent.velocity.y * np.cos(state.agent.center.heading)]
             for state in trajectory_absolute_states]
        )

        # other AGENTs future
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.scenario_time_horizon, num_samples=self.number_of_interations
            )
        ]
        future_agents_trajectories = [
            [
                [[agent.center.x,
                 agent.center.y,
                 agent.center.heading], [agent.track_token, agent.tracked_object_type.name]] for agent in agents.tracked_objects
            ] for agents in future_tracked_objects
        ]  # [num_frames, num_agents, [pose, [track_token, tracked_object_type.name]]]
        future_agents_velocity = [
            [
                [agent.velocity.x * np.cos(agent.center.heading)
                 - agent.velocity.y * np.sin(agent.center.heading),
                 agent.velocity.x * np.sin(agent.center.heading)
                 + agent.velocity.y * np.cos(agent.center.heading)] for agent in agents.tracked_objects
            ] for agents in future_tracked_objects
        ]

        # reshape to [num_agents, num_frames, pose]
        reshaped_future_agents_trajectories: Dict[str, Dict[int, List]] = {}
        reshaped_future_agents_velocity: Dict[str, Dict[int, List]] = {}
        objects_types = ['AV', ]
        for timestep in range(len(future_agents_trajectories)):
            for agent, velocity in zip(future_agents_trajectories[timestep], future_agents_velocity[timestep]):
                if agent[1][0] not in reshaped_future_agents_trajectories.keys():
                    reshaped_future_agents_trajectories[agent[1][0]] = {}
                    reshaped_future_agents_velocity[agent[1][0]] = {}
                    objects_types.append(agent[1][1])
                reshaped_future_agents_trajectories[agent[1][0]][timestep] = agent[0]
                reshaped_future_agents_velocity[agent[1][0]][timestep] = velocity

        trajectories = {'AV': future_ego_trajectory}
        for agent_id, agent_poses in reshaped_future_agents_trajectories.items():
            trajectories[agent_id] = Trajectory(data=np.array(list(agent_poses.values())))
        velocity = {'AV': future_ego_velocity}
        for agent_id, agent_velocity in reshaped_future_agents_velocity.items():
            velocity[agent_id] = np.array(list(agent_velocity.values()))
        trajectories_copy = copy.deepcopy(trajectories)

        # transform the future trajectory of each agent to its local coordinate system
        transformed_trajectories = {}
        for id, traj in trajectories.items():
            if id in anchor_states.keys():
                transformed_trajectories[id] = Trajectory.transform_to_local_frame_given_anchor_state(data=traj.data, anchor_state=anchor_states[id])
            else:  # those don't exist in the current step will not be transformed
                transformed_trajectories[id] = traj

        # predict mask
        predict_mask = np.zeros((len(reshaped_future_agents_trajectories) + 1,
                                 self.number_of_interations,
                                 self._num_future_steps), dtype=bool)
        predict_mask[0, :, :] = True  # for EGO
        for agent_id, value in reshaped_future_agents_trajectories.items():
            agent_idx = list(reshaped_future_agents_trajectories.keys()).index(agent_id)
            agent_steps = np.array(list(value.keys()))
            for iteration in range(self.number_of_interations):
                agent_steps_at_iter = agent_steps[(agent_steps >= iteration) &
                                                  (agent_steps < iteration + self._num_future_steps)]
                predict_mask[agent_idx + 1, iteration, agent_steps_at_iter - iteration] = True
        predict_mask = np.concatenate((np.zeros((len(reshaped_future_agents_trajectories) + 1,
                                                 self.number_of_interations,
                                                 self._num_historical_steps), dtype=bool),
                                       predict_mask),
                                      axis=2)

        # pad agents' trajectories and velocity
        padded_transformed_trajectories = {'AV': transformed_trajectories['AV']}
        for id, traj in transformed_trajectories.items():
            if id == 'AV':
                continue
            traj_temp = np.zeros_like(transformed_trajectories['AV'].data)
            traj_temp[list(reshaped_future_agents_trajectories[id].keys()), :] = traj.data
            padded_transformed_trajectories[id] = Trajectory(data=traj_temp)
        padded_trajectories = {'AV': trajectories_copy['AV']}
        for id, traj in trajectories_copy.items():
            if id == 'AV':
                continue
            traj_temp = np.zeros_like(transformed_trajectories['AV'].data)
            traj_temp[list(reshaped_future_agents_trajectories[id].keys()), :] = traj.data
            padded_trajectories[id] = Trajectory(data=traj_temp)
        padded_velocity = {'AV': velocity['AV']}
        for id, vel in velocity.items():
            if id == 'AV':
                continue
            vel_temp = np.zeros((transformed_trajectories['AV'].data.shape[0], 2))
            vel_temp[list(reshaped_future_agents_velocity[id].keys()), :] = vel.data
            padded_velocity[id] = vel_temp

        return AgentsTrajectories(
            trajectories=[padded_transformed_trajectories],
            trajectories_global=[padded_trajectories],
            velocity_global=[padded_velocity],
            track_token_ids=[['AV']+list(reshaped_future_agents_trajectories.keys())],
            objects_types=[objects_types],
            predict_mask=[predict_mask]
        )
