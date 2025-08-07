from typing import List, Optional

import torch
from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType

class PseudoMetric(AbstractTrainingMetric):

    def __init__(self,
                 name: str = 'PseudoMetric') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the training_metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectories"]

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes the training_metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: training_metric scalar tensor
        """

        return torch.Tensor([-1.])

