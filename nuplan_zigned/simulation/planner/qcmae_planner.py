import time
from typing import List, Optional, Type, cast, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.ml_planner.ml_planner import MLPlanner
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from nuplan_zigned.simulation.planner.model_loader import ModelLoader
from nuplan_zigned.simulation.planner.avrl_transform_utils import transform_predictions_to_states
from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories


class QCMAEPlanner(AbstractPlanner):

    def __init__(self, model: TorchModuleWrapper) -> None:
        """
        Initializes the QCMAE planner class.
        :param model: Model to use for inference.
        """
        self._future_horizon = model.future_trajectory_sampling.time_horizon
        self._step_interval = model.future_trajectory_sampling.step_time
        self._num_output_dim = model.future_trajectory_sampling.num_poses

        self._model_loader = ModelLoader(model)

        self._initialization: Optional[PlannerInitialization] = None

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

    def _infer_model(self, features: FeaturesType, current_input: PlannerInput, initialization: PlannerInitialization) -> npt.NDArray[np.float64]:
        """
        Makes a single inference on a Pytorch/Torchscript model.

        :param features: dictionary of feature types
        :param current_input: Iteration specific inputs for building the feature.
        :return: predicted trajectory poses as a numpy array
        """
        # Propagate model
        predictions = self._model_loader.infer(features, current_input, initialization)

        # Extract trajectory prediction
        trajectory_predicted = cast(AgentsTrajectories, predictions['agents_trajectories'])
        assert trajectory_predicted.batch_size == 1
        pred = trajectory_predicted.trajectories
        sample_idx = 0
        ego_index = 0
        pi = pred['pi'][sample_idx]
        prob = F.softmax(pi[ego_index], dim=-1)
        ego_most_likely_mode = prob.argmax(dim=-1)
        all_modes = pred['loc_refine_pos'][sample_idx][ego_index]
        num_modes = all_modes.shape[0]
        heading = torch.atan2(all_modes[:, 1:, 1] - all_modes[:, 0:-1, 1],
                              all_modes[:, 1:, 0] - all_modes[:, 0:-1, 0])
        heading = torch.cat([heading, heading[:, -1:]], dim=1)
        all_modes = torch.cat([all_modes, heading.unsqueeze(-1)], dim=-1)
        ego_most_likely_traj = all_modes[ego_most_likely_mode]
        trajectory = ego_most_likely_traj.cpu().double().numpy()

        rest_modes = list(range(num_modes))
        rest_modes.remove(ego_most_likely_mode.item())
        rest_modes = all_modes[rest_modes]
        rest_modes = rest_modes.cpu().double().numpy()

        return cast(npt.NDArray[np.float64], trajectory), cast(npt.NDArray[np.float64], rest_modes)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._model_loader.initialize()
        self._initialization = initialization

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> Tuple[AbstractTrajectory, List[AbstractTrajectory]]:
        """
        Infer relative trajectory poses from model and convert to absolute agent states wrapped in a trajectory.
        Inherited, see superclass.
        """
        # Extract history
        history = current_input.history

        # Construct input features
        start_time = time.perf_counter()
        features = self._model_loader.build_features(current_input, self._initialization)
        self._feature_building_runtimes.append(time.perf_counter() - start_time)

        # Infer model
        start_time = time.perf_counter()
        predictions, rest_modes = self._infer_model(features, current_input, self._initialization)
        self._inference_runtimes.append(time.perf_counter() - start_time)

        # Convert relative poses to absolute states and wrap in a trajectory object.
        states = transform_predictions_to_states(
            predictions, history.ego_states, self._future_horizon, self._step_interval
        )
        trajectory = InterpolatedTrajectory(states)

        list_rest_modes = []
        for traj in rest_modes:
            states = transform_predictions_to_states(
                traj, history.ego_states, self._future_horizon, self._step_interval
            )
            mode = InterpolatedTrajectory(states)
            list_rest_modes.append(mode)

        return trajectory, list_rest_modes

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []
        return report
