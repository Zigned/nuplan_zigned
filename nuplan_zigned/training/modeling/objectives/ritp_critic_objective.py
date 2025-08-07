from typing import Dict, List, Any

import torch
import torch.nn.functional as F

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper


class CriticObjective(AbstractObjective):

    def __init__(self, scenario_type_loss_weighting: Dict[str, float]):
        """
        Initializes the class
        """
        self._name = 'critic_objective'

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectories"]

    def compute(self,
                predictions: Dict[str, Any],
                targets: TargetsType,
                scenarios: ScenarioListType,
                optimizer_idx: int,
                ) -> torch.Tensor:
        """
        Computes the TD3 objective's loss.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        loss = None

        if optimizer_idx == 0:  # critic loss
            target_Q = predictions['td3']['target_Q']
            current_Q1 = predictions['td3']['current_Q1']
            current_Q2 = predictions['td3']['current_Q2']
            if target_Q is None:
                loss = None
            else:
                loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        return loss
