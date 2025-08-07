import warnings
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import pytorch_lightning as pl
import torch
import numpy as np
from hydra.utils import instantiate
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from copy import deepcopy

from nuplan.planning.script.builders.lr_scheduler_builder import build_lr_scheduler
from nuplan.planning.training.modeling.metrics.planning_metrics import AbstractTrainingMetric
from nuplan.planning.training.modeling.objectives.abstract_objective import aggregate_objectives
from nuplan.planning.training.modeling.objectives.imitation_objective import AbstractObjective
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories

logger = logging.getLogger(__name__)


class LightningModuleWrapper(pl.LightningModule):
    """
    Lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(
        self,
        model: TorchModuleWrapper,
        objectives: List[AbstractObjective],
        metrics: List[AbstractTrainingMetric],
        batch_size: int,
        optimizer: Optional[DictConfig] = None,
        lr_scheduler: Optional[DictConfig] = None,
        warm_up_lr_scheduler: Optional[DictConfig] = None,
        objective_aggregate_mode: str = 'mean',
        augmentors: Optional[List[AbstractAugmentor]] = None,
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.objectives = objectives
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warm_up_lr_scheduler = warm_up_lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode
        self.augmentors = augmentors

        self.beginning_of_scenario = None
        self.batch_of_this_scenario = None
        self.batch_of_this_step = None

    def _index_batch_data(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType]):
        # generate features and targets for current iteration
        features, targets, scenarios = batch

        num_of_iterations = [scenario.get_number_of_iterations() for scenario in scenarios]
        num_of_iterations = min(num_of_iterations)
        polygon_tl_statuses = features['vector_set_map'].map_data['map_polygon']['tl_statuses']
        point_tl_statuses = features['vector_set_map'].map_data['map_point']['tl_statuses']
        valid_mask = features['generic_agents'].agent_data['valid_mask']
        predict_mask = features['generic_agents'].agent_data['predict_mask']
        position = features['generic_agents'].agent_data['position']
        heading = features['generic_agents'].agent_data['heading']
        velocity = features['generic_agents'].agent_data['velocity']

        predict_mask_tg = targets['agents_trajectories'].predict_mask
        trajectories_tg = targets['agents_trajectories'].trajectories
        trajectories_global_tg = targets['agents_trajectories'].trajectories_global
        velocity_global_tg = targets['agents_trajectories'].velocity_global

        for sample_idx in range(features['vector_set_map'].batch_size):
            iteration = self.env.iteration[sample_idx]
            polygon_tl_statuses[sample_idx] = polygon_tl_statuses[sample_idx][:, iteration]
            point_tl_statuses[sample_idx] = point_tl_statuses[sample_idx][:, iteration]
            valid_mask[sample_idx] = valid_mask[sample_idx][:, iteration, :]
            predict_mask[sample_idx] = predict_mask[sample_idx][:, iteration, :]
            position[sample_idx] = position[sample_idx][:, iteration, :, :]
            heading[sample_idx] = heading[sample_idx][:, iteration, :]
            velocity[sample_idx] = velocity[sample_idx][:, iteration, :, :]

            predict_mask_tg[sample_idx] = predict_mask_tg[sample_idx][:, iteration, :]
            traj_tg = {}
            traj_global_tg = {}
            vel_global_tg = {}
            num_steps = predict_mask[sample_idx].shape[-1]  # num_historical_steps + num_future_steps
            num_historical_steps = position[sample_idx].shape[1]
            num_future_steps = num_steps - num_historical_steps
            for (id, traj), traj_global, vel_global in zip(trajectories_tg[sample_idx].items(),
                                trajectories_global_tg[sample_idx].values(),
                                velocity_global_tg[sample_idx].values()):
                if iteration == 0:
                    trajectory = traj[iteration: iteration + num_future_steps]
                else:
                    trajectory = traj[iteration - 1: iteration + num_future_steps]  # traj[iteration - 1] will be used as origin pose
                trajectory_global = traj_global[iteration: iteration + num_future_steps]
                velocity_global = vel_global[iteration: iteration + num_future_steps]
                if iteration + num_future_steps >= num_of_iterations:
                    # skip rest iterations
                    self.env.episode_end[sample_idx] = True
                traj_tg[id] = trajectory.to(device=traj.device, dtype=traj.dtype)
                traj_global_tg[id] = trajectory_global.to(device=traj.device, dtype=traj.dtype)
                vel_global_tg[id] = velocity_global.to(device=traj.device, dtype=traj.dtype)
            trajectories_tg[sample_idx] = traj_tg
            trajectories_global_tg[sample_idx] = traj_global_tg
            velocity_global_tg[sample_idx] = vel_global_tg

            if iteration == 0:
                self.env.previous_state = None
                self.env.previous_action = None
                self.env.previous_target = None
                self.env.previous_reward = None
                self.env.previous_done = None
                self.env.state = None
                self.env.action = None
                self.env.reward = None
                self.env.reward_log = None
                self.env.next_state = None
                self.env.done = [False for _ in self.env.done]
                self.env.target = None
                self.env.episode_end = [False for _ in self.env.episode_end]

                self.env.ego_historical_position[sample_idx] = position[sample_idx][0, :, :].clone()
                self.env.ego_historical_heading[sample_idx] = heading[sample_idx][0, :].clone()
                self.env.ego_historical_velocity[sample_idx] = velocity[sample_idx][0, :, :].clone()
                self.env.time_point[sample_idx] = self.env.time_points[sample_idx][iteration]

            # replace ego historical trajectory with rolled-out trajectory
            elif iteration > 0:
                position[sample_idx][0, :, :] = self.env.ego_historical_position[sample_idx]
                heading[sample_idx][0, :] = self.env.ego_historical_heading[sample_idx]
                velocity[sample_idx][0, :, :] = self.env.ego_historical_velocity[sample_idx]
                # update time_point (no use)
                self.env.time_point[sample_idx] = self.env.time_points[sample_idx][iteration]

    def _step(self,
              batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
              optimizer_idx: Optional[int],
              evaluation: bool,
              prefix: str) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param optimizer_idx: optimizer's index
        :param replay_buffer: replay buffer for RL
        :param env: environment for RL
        :param evaluation: whether in evaluation loop or not
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch

        predictions = self.forward(features, targets, optimizer_idx, evaluation)
        if self.model.finetune_mode != 'pseudo_ground_truth':
            predictions = self._remove_ego_prediction(predictions)

        if 'evaluation' in predictions.keys():
            if predictions['evaluation']:
                self.model.convert_absolute_to_relative_targets(targets)

        # compute objectives
        if self.model.finetune_mode != 'pseudo_ground_truth':
            if 'targets' in predictions.keys():
                if predictions['targets'] is not None:
                    # targets for computing objectives
                    targets = {
                        'agents_trajectories': AgentsTrajectories.collate([pred['agents_trajectories']
                                                                           for pred in predictions['targets']])
                    }
                    targets = self._remove_ego_target(targets)
        else:
            if 'pseudo_targets' in predictions.keys():
                if predictions['pseudo_targets'] is not None:
                    targets = {
                        'agents_trajectories': AgentsTrajectories.collate([tg['agents_trajectories']
                                                                           for tg in predictions['pseudo_targets']])
                    }
        objectives = self._compute_objectives(predictions, targets, scenarios, optimizer_idx)
        if objectives['actor_objective'] is None and objectives['critic_objective'] is None:
            loss = None
        else:
            objectives = {obj_name: obj for obj_name, obj in objectives.items() if obj is not None}
            loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

        # compute metrics
        if 'targets' in predictions['td3'].keys():
            if predictions['td3']['targets'] is not None:
                # targets for computing metrics
                targets = {
                    'agents_trajectories': AgentsTrajectories.collate([pred['agents_trajectories']
                                                                       for pred in predictions['td3']['targets']])
                }
                targets = self._remove_ego_target(targets)
        if optimizer_idx == 0:
            predictions['reward_log'] = self.model.env.reward_log
        metrics = self._compute_metrics(predictions, targets)

        self._log_step(loss, objectives, metrics, prefix)

        if optimizer_idx == 1:
            self.model.t += features['vector_set_map'].batch_size

        return loss

    def _compute_objectives(
        self,
        predictions: TargetsType,
        targets: TargetsType,
        scenarios: ScenarioListType,
        optimizer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: model's output signal
        :param targets: supervisory signal
        :return: dictionary of objective names and values
        """
        return {objective.name(): objective.compute(predictions,
                                                    targets,
                                                    scenarios,
                                                    optimizer_idx) for objective in self.objectives}

    def _compute_metrics(self, predictions: TargetsType, targets: TargetsType) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        return {metric.name(): metric.compute(predictions, targets) for metric in self.metrics}

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = 'loss',
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        if loss is not None:
            self.log(f'loss/{prefix}_{loss_name}', loss)

        for key, value in objectives.items():
            if value is not None:
                self.log(f'objectives/{prefix}_{key}', value)

        for key, value in metrics.items():
            if value is not None:
                self.log(f'metrics/{prefix}_{key}', value)

    def training_step(
            self,
            batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
            batch_idx: int,
            optimizer_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :param optimizer_idx: optimizer's index
        :param replay_buffer: replay buffer for RL
        :param env: environment for RL
        :return: model's loss tensor
        """
        # save effort for moving tensors to GPU
        if np.any(self.beginning_of_scenario) and optimizer_idx == 0:
            self.batch_of_this_scenario = deepcopy(batch)
        batch = deepcopy(self.batch_of_this_scenario)

        if optimizer_idx == 0:
            self._index_batch_data(batch)
            self.batch_of_this_step = batch

        return self._step(self.batch_of_this_step, optimizer_idx, evaluation=False, prefix='train')

    def validation_step(
            self,
            batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
            batch_idx: int,
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        # save effort for moving tensors to GPU
        if np.any(self.beginning_of_scenario):
            self.batch_of_this_scenario = deepcopy(batch)
        else:
            batch = deepcopy(self.batch_of_this_scenario)

        self._index_batch_data(batch)

        return self._step(batch, optimizer_idx=None, evaluation=True, prefix='val')

    def test_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        # save effort for moving tensors to GPU
        if np.any(self.beginning_of_scenario):
            self.batch_of_this_scenario = deepcopy(batch)
        else:
            batch = deepcopy(self.batch_of_this_scenario)

        self._index_batch_data(batch)

        return self._step(batch, optimizer_idx=None, evaluation=True, prefix='test')

    def forward(self,
                features: FeaturesType,
                targets: TargetsType = None,
                optimizer_idx: int = None,
                evaluation: bool = None) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :param targets: supervisory signal in global frame
        :param optimizer_idx: optimizer's index
        :param replay_buffer: replay buffer for RL
        :param env: environment for RL
        :param evaluation: whether in evaluation loop or not
        :return: model's predictions
        """
        return self.model(features, targets, optimizer_idx, evaluation)

    def configure_optimizers(
        self,
    ) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        # Get optimizer
        critic_optimizer: Optimizer = instantiate(
            config=self.optimizer,
            params=self.model.critic.parameters(),
            lr=self.optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )
        actor_optimizer: Optimizer = instantiate(
            config=self.optimizer,
            params=self.model.actor.parameters(),
            lr=self.optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )
        # Log the optimizer used
        logger.info(f'Using optimizer: {self.optimizer._target_}')

        # Get lr_scheduler
        critic_lr_scheduler_params: Dict[str, Union[_LRScheduler, str, int]] = build_lr_scheduler(
            optimizer=critic_optimizer,
            lr=self.optimizer.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler,
            lr_scheduler_cfg=self.lr_scheduler,
        )
        actor_lr_scheduler_params: Dict[str, Union[_LRScheduler, str, int]] = build_lr_scheduler(
            optimizer=critic_optimizer,
            lr=self.optimizer.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler,
            lr_scheduler_cfg=self.lr_scheduler,
        )

        optimizer_list = [critic_optimizer, actor_optimizer]
        lr_scheduler_list = []
        if critic_lr_scheduler_params or actor_lr_scheduler_params:
            logger.info(f'Using lr_schedulers {critic_lr_scheduler_params} for critic and {actor_lr_scheduler_params} for actor')
            lr_scheduler_list = [critic_lr_scheduler_params, actor_lr_scheduler_params]

        return optimizer_list if len(lr_scheduler_list) > 0 else optimizer_list

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # update critic optimizer
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

        # update actor optimizer
        if optimizer_idx == 1:
            optimizer.step(closure=optimizer_closure)

    def _remove_ego_prediction(self, prediction: TargetsType) -> TargetsType:
        if prediction['agents_trajectories'] is not None:
            if prediction['agents_trajectories'].trajectories is not None:
                objects_types = prediction['agents_trajectories'].objects_types
                predict_mask = prediction['agents_trajectories'].predict_mask
                track_token_ids = prediction['agents_trajectories'].track_token_ids
                trajectories = prediction['agents_trajectories'].trajectories

                objects_types = [type[1:] for type in objects_types]
                predict_mask = [mask[1:] for mask in predict_mask]
                track_token_ids = [id[1:] for id in track_token_ids]
                trajectories = {
                    key: [v[1:] for v in value]
                    for key, value in trajectories.items()
                }

                prediction['agents_trajectories'].objects_types = objects_types
                prediction['agents_trajectories'].predict_mask = predict_mask
                prediction['agents_trajectories'].track_token_ids = track_token_ids
                prediction['agents_trajectories'].trajectories = trajectories

        return prediction

    def _remove_ego_target(self, target: TargetsType) -> TargetsType:
        objects_types = target['agents_trajectories'].objects_types
        predict_mask = target['agents_trajectories'].predict_mask
        track_token_ids = target['agents_trajectories'].track_token_ids
        trajectories = target['agents_trajectories'].trajectories
        trajectories_global = target['agents_trajectories'].trajectories_global
        velocity_global = target['agents_trajectories'].velocity_global

        objects_types = [type[1:] for type in objects_types]
        predict_mask = [mask[1:] for mask in predict_mask]
        track_token_ids = [id[1:] for id in track_token_ids]
        [traj_dict.pop('AV') for traj_dict in trajectories]
        [traj_dict.pop('AV') for traj_dict in trajectories_global]
        [vel_dict.pop('AV') for vel_dict in velocity_global]

        target['agents_trajectories'].objects_types = objects_types
        target['agents_trajectories'].predict_mask = predict_mask
        target['agents_trajectories'].track_token_ids = track_token_ids
        target['agents_trajectories'].trajectories = trajectories
        target['agents_trajectories'].trajectories_global = trajectories_global
        target['agents_trajectories'].velocity_global = velocity_global

        return target