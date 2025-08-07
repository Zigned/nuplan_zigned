from __future__ import annotations

from typing import Type, Dict, List
import numpy as np

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
        # EGO and AGENTs current
        anchor_ego_state = scenario.initial_ego_state
        anchor_agents_states = scenario.initial_tracked_objects
        anchor_states = {'AV': anchor_ego_state.agent}
        anchor_states.update({agent.track_token: agent for agent in anchor_agents_states.tracked_objects.tracked_objects})

        # EGO future
        trajectory_absolute_states = scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self._num_future_steps, time_horizon=self._time_horizon
        )
        future_ego_trajectory = [[state.agent.center.x,
                                  state.agent.center.y,
                                  state.agent.center.heading] for state in trajectory_absolute_states]
        future_ego_trajectory = Trajectory(data=np.array(future_ego_trajectory))

        # other AGENTs future
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self._time_horizon, num_samples=self._num_future_steps
            )
        ]
        future_agents_trajectories = [
            [
                [[agent.center.x,
                 agent.center.y,
                 agent.center.heading], [agent.track_token, agent.tracked_object_type.name]] for agent in agents.tracked_objects
            ] for agents in future_tracked_objects
        ]  # [num_frames, num_agents, [pose, [track_token, tracked_object_type.name]]]

        # reshape to [num_agents, num_frames, pose]
        reshaped_future_agents_trajectories: Dict[str, Dict[int, List]] = {}
        objects_types = ['AV', ]
        for timestep in range(len(future_agents_trajectories)):
            for agent in future_agents_trajectories[timestep]:
                if agent[1][0] not in reshaped_future_agents_trajectories.keys():
                    reshaped_future_agents_trajectories[agent[1][0]] = {}
                    objects_types.append(agent[1][1])
                reshaped_future_agents_trajectories[agent[1][0]][timestep] = agent[0]

        trajectories = {'AV': future_ego_trajectory}
        for agent_id, agent_poses in reshaped_future_agents_trajectories.items():
            trajectories[agent_id] = Trajectory(data=np.array(list(agent_poses.values())))

        # transform the future trajectory of each agent to its local coordinate system
        transformed_trajectories = {}
        for id, traj in trajectories.items():
            if id in anchor_states.keys():
                transformed_trajectories[id] = Trajectory.transform_to_local_frame_given_anchor_state(data=traj.data, anchor_state=anchor_states[id])
            else:  # those don't exist in the current step will not be transformed
                transformed_trajectories[id] = traj

        # predict mask
        predict_mask = np.zeros((len(reshaped_future_agents_trajectories) + 1, self._num_future_steps), dtype=bool)
        predict_mask[0, :] = True  # for EGO
        for agent_id, value in reshaped_future_agents_trajectories.items():
            agent_idx = list(reshaped_future_agents_trajectories.keys()).index(agent_id)
            predict_mask[agent_idx + 1, np.array(list(value.keys()))] = True
        predict_mask = np.hstack((np.zeros((len(reshaped_future_agents_trajectories) + 1, self._num_historical_steps), dtype=bool),
                                  predict_mask))

        return AgentsTrajectories(
            trajectories=[transformed_trajectories],
            track_token_ids=[['AV']+list(reshaped_future_agents_trajectories.keys())],
            objects_types=[objects_types],
            predict_mask=[predict_mask]
        )
