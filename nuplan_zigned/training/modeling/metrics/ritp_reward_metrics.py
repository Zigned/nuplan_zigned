from typing import List, Optional

import torch
from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType


class NumSafeSteps(AbstractTrainingMetric):

    def __init__(self,
                 name: str = 'num_safe_steps') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self._num_safe_steps = None
        self._previous = None

    def name(self) -> str:
        """
        Name of the training_metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes average number of safe steps over batch_size episodes.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: training_metric scalar tensor
        """
        if 'reward_log' not in predictions.keys():
            return None
        else:
            if predictions['reward_log'] is None:
                return None
        if self._num_safe_steps is None:
            self._num_safe_steps = torch.zeros(len(predictions['reward_log']['done']))
        else:
            for sample_idx in range(len(predictions['reward_log']['done'])):
                self._num_safe_steps[sample_idx] += float(not predictions['reward_log']['done'][sample_idx])

        if all(predictions['reward_log']['done']) or all(predictions['reward_log']['episode_end']):
            average_num_safe_steps = torch.mean(self._num_safe_steps)
            self._num_safe_steps = None
            self._previous = average_num_safe_steps
            return average_num_safe_steps.item()
        else:
            return self._previous.item() if self._previous is not None else None


class RewardMean(AbstractTrainingMetric):

    def __init__(self,
                 name: str = 'reward_mean') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self._reward_mean = None
        self._previous = None

    def name(self) -> str:
        """
        Name of the training_metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes average accumulated reward mean over batch_size episodes.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: training_metric scalar tensor
        """
        if 'reward_log' not in predictions.keys():
            return None
        else:
            if predictions['reward_log'] is None:
                return None

        if self._reward_mean is None:
            self._reward_mean = predictions['reward_log']['reward_mean']
        else:
            self._reward_mean += predictions['reward_log']['reward_mean']

        if all(predictions['reward_log']['done']) or all(predictions['reward_log']['episode_end']):
            average_reward_mean = torch.mean(self._reward_mean)
            self._reward_mean = None
            self._previous = average_reward_mean
            return average_reward_mean.item()
        else:
            return self._previous.item() if self._previous is not None else None


class RewardVariance(AbstractTrainingMetric):

    def __init__(self,
                 name: str = 'reward_variance') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self._reward_variance = None
        self._previous = None

    def name(self) -> str:
        """
        Name of the training_metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes average accumulated reward variance over batch_size episodes.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: training_metric scalar tensor
        """
        if 'reward_log' not in predictions.keys():
            return None
        else:
            if predictions['reward_log'] is None:
                return None

        if self._reward_variance is None:
            self._reward_variance = predictions['reward_log']['reward_variance']
        else:
            self._reward_variance += predictions['reward_log']['reward_variance']

        if all(predictions['reward_log']['done']) or all(predictions['reward_log']['episode_end']):
            average_reward_variance = torch.mean(self._reward_variance)
            self._reward_variance = None
            self._previous = average_reward_variance
            return average_reward_variance.item()
        else:
            return self._previous.item() if self._previous is not None else None


class ExponentialPenalizedReward(AbstractTrainingMetric):

    def __init__(self,
                 name: str = 'exponential_penalized_reward') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self._exponential_penalized_reward = None
        self._previous = None

    def name(self) -> str:
        """
        Name of the training_metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return []

    def compute(self, predictions: TargetsType, targets: TargetsType) -> torch.Tensor:
        """
        Computes average accumulated exponential penalized reward over batch_size episodes.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: training_metric scalar tensor
        """
        if 'reward_log' not in predictions.keys():
            return None
        else:
            if predictions['reward_log'] is None:
                return None

        if self._exponential_penalized_reward is None:
            self._exponential_penalized_reward = predictions['reward_log']['exponential_penalized_reward']
        else:
            self._exponential_penalized_reward += predictions['reward_log']['exponential_penalized_reward']

        if all(predictions['reward_log']['done']) or all(predictions['reward_log']['episode_end']):
            average_exponential_penalized_reward = torch.mean(self._exponential_penalized_reward)
            self._exponential_penalized_reward = None
            self._previous = average_exponential_penalized_reward
            return average_exponential_penalized_reward.item()
        else:
            return self._previous.item() if self._previous is not None else None