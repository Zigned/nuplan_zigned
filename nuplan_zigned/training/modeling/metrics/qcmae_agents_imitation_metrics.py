#   Adapted from:
#   https://github.com/jchengai/pluto (Apache License 2.0)

import traceback
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType

from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories
from nuplan_zigned.utils.utils import safe_list_index, wrap_angle
from nuplan_zigned.training.modeling.metrics.utils import valid_filter, topk


class Brier(AbstractTrainingMetric):

    def __init__(self,
                 output_dim: int,
                 output_head: bool,
                 num_past_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 self_distillation: bool,
                 only_evaluate_av: bool,
                 name: str = 'Brier') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_past_steps = num_past_steps
        self.num_historical_steps = num_past_steps + 1
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.self_distillation = self_distillation
        self.only_evaluate_av = only_evaluate_av

        self.last_data = None
        self.sum = 0.
        self.count = 0

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
        if 'td3' in predictions.keys():
            if 'target_Q' not in predictions['td3'].keys():  # optimizer_idx == 1
                return None
            if 'evaluation' in predictions.keys():
                # t < start_timesteps and evaluation == False
                if predictions['td3']['t'] < predictions['td3']['start_timesteps'] and not predictions['evaluation']:
                    return None

        targets_before_collating = targets
        if self.last_data is not None:  # self-distillation
            targets = {
                'agents_trajectories': AgentsTrajectories.collate(batch=[self.last_data['agents_trajectories'], targets['agents_trajectories']])
            }
        if self.self_distillation:
            self.last_data = targets_before_collating

        predicted_agents: AgentsTrajectories = predictions["agents_trajectories"]
        target_agents: AgentsTrajectories = targets["agents_trajectories"]
        batch_size = predicted_agents.batch_size
        self.sum = 0.
        self.count = 0

        for sample_idx in range(batch_size):
            # avoid "line 99, RuntimeError: "bitwise_or_cpu" not implemented for 'Float'" when only ego in target_agents
            if sample_idx == batch_size - 1:
                if len(target_agents.objects_types[sample_idx]) == 0:
                    if self.count == 0:
                        return None
            if len(target_agents.objects_types[sample_idx]) == 0:
                continue

            if self.only_evaluate_av:
                # only evaluate AV
                eval_mask = torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]])
            else:
                # only evaluate AV VEHICLE, PEDESTRIAN, and BICYCLE
                try:
                    eval_mask = (
                            (torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]]))
                            | (torch.tensor([object_type == 'VEHICLE' for object_type in target_agents.objects_types[sample_idx]]))
                            | (torch.tensor([object_type == 'PEDESTRIAN' for object_type in target_agents.objects_types[sample_idx]]))
                            | (torch.tensor([object_type == 'BICYCLE' for object_type in target_agents.objects_types[sample_idx]]))
                    )
                except:
                    traceback.print_exc()
            reg_mask = target_agents.predict_mask[sample_idx][:, self.num_historical_steps:].bool()

            agent_in_target_mask, coexisting_agent_ids_in_target = [], []
            for tg_agent_id in target_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(predicted_agents.track_token_ids[sample_idx], tg_agent_id)
                if idx is not None:
                    agent_in_target_mask.append(True)
                    coexisting_agent_ids_in_target.append(tg_agent_id)
                else:
                    agent_in_target_mask.append(False)
            agent_in_pred_mask = []
            for pred_agent_id in predicted_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(coexisting_agent_ids_in_target, pred_agent_id)
                if idx is not None:
                    agent_in_pred_mask.append(True)
                else:
                    agent_in_pred_mask.append(False)
            eval_mask = eval_mask[agent_in_target_mask]
            reg_mask = reg_mask[agent_in_target_mask]
            eval_mask = eval_mask.to(reg_mask.device)
            valid_mask_eval = reg_mask[eval_mask]
            pred = {
                key: value[sample_idx][agent_in_pred_mask] for key, value in predicted_agents.trajectories.items()
            }

            num_agents = reg_mask.shape[0]
            gt = torch.zeros((num_agents, self.num_future_steps, self.output_dim), device=reg_mask.device, dtype=torch.float)
            masked_pred_ids = list(np.array(predicted_agents.track_token_ids[sample_idx])[agent_in_pred_mask])
            for i_agent in range(len(coexisting_agent_ids_in_target)):
                agent_id = coexisting_agent_ids_in_target[i_agent]
                agent_idx_in_pred = safe_list_index(masked_pred_ids, agent_id)
                if agent_idx_in_pred is None:
                    continue
                agent_idx_in_target = i_agent
                gt[agent_idx_in_pred][reg_mask[agent_idx_in_target]] = target_agents.trajectories[sample_idx][agent_id][:, :self.output_dim].float()


            if self.output_head:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['loc_refine_head'],
                                         pred['scale_refine_pos'][..., :self.output_dim],
                                         pred['conc_refine_head']], dim=-1)
            else:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            pi = pred['pi']
            gt = torch.cat([gt[..., :self.output_dim], gt[..., -1:]], dim=-1)

            traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
            if not self.output_head:
                traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                         traj_eval[..., :2]], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
            pi_eval = F.softmax(pi[eval_mask], dim=-1)
            gt_eval = gt[eval_mask]

            self.compute_Brier(pred=traj_eval[..., :self.output_dim],
                               target=gt_eval[..., :self.output_dim],
                               prob=pi_eval,
                               valid_mask=valid_mask_eval)

        return self.sum / self.count

    def compute_Brier(self,
                      pred: torch.Tensor,
                      target: torch.Tensor,
                      prob: Optional[torch.Tensor] = None,
                      valid_mask: Optional[torch.Tensor] = None,
                      keep_invalid_final_step: bool = True,
                      min_criterion: str = 'FDE') -> None:
        device = pred.device
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, prob_topk = topk(self.num_modes, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=device)).argmax(dim=-1)
            inds_best = torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                                   target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                   p=2, dim=-1).argmin(dim=-1)
        elif min_criterion == 'ADE':
            inds_best = (torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                         valid_mask.unsqueeze(1)).sum(dim=-1).argmin(dim=-1)
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.sum += (1.0 - prob_topk[torch.arange(pred.size(0)), inds_best]).pow(2).sum()
        self.count += pred.size(0)

class minADE(AbstractTrainingMetric):

    def __init__(self,
                 output_dim: int,
                 output_head: bool,
                 num_past_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 self_distillation: bool,
                 only_evaluate_av: bool,
                 num_modes_eval: int,
                 name: str = 'minADE') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_past_steps = num_past_steps
        self.num_historical_steps = num_past_steps + 1
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.self_distillation = self_distillation
        self.only_evaluate_av = only_evaluate_av
        self.num_modes_eval = num_modes_eval

        self.last_data = None
        self.sum = 0.
        self.count = 0

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

        if 'td3' in predictions.keys():
            if 'target_Q' not in predictions['td3'].keys():  # optimizer_idx == 1
                return None
            if 'evaluation' in predictions.keys():
                # t < start_timesteps and evaluation == False
                if predictions['td3']['t'] < predictions['td3']['start_timesteps'] and not predictions['evaluation']:
                    return None

        targets_before_collating = targets
        if self.last_data is not None:  # self-distillation
            targets = {
                'agents_trajectories': AgentsTrajectories.collate(batch=[self.last_data['agents_trajectories'], targets['agents_trajectories']])
            }
        if self.self_distillation:
            self.last_data = targets_before_collating

        predicted_agents: AgentsTrajectories = predictions["agents_trajectories"]
        target_agents: AgentsTrajectories = targets["agents_trajectories"]
        batch_size = predicted_agents.batch_size
        self.sum = 0.
        self.count = 0

        for sample_idx in range(batch_size):
            # avoid "RuntimeError: "bitwise_or_cpu" not implemented for 'Float'" when only ego in target_agents
            if sample_idx == batch_size - 1:
                if len(target_agents.objects_types[sample_idx]) == 0:
                    if self.count == 0:
                        return None
            if len(target_agents.objects_types[sample_idx]) == 0:
                continue

            if self.only_evaluate_av:
                # only evaluate AV
                eval_mask = torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]])
            else:
                # only evaluate AV VEHICLE, PEDESTRIAN, and BICYCLE
                eval_mask = (
                        (torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'VEHICLE' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'PEDESTRIAN' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'BICYCLE' for object_type in target_agents.objects_types[sample_idx]]))
                )
            reg_mask = target_agents.predict_mask[sample_idx][:, self.num_historical_steps:].bool()

            agent_in_target_mask, coexisting_agent_ids_in_target = [], []
            for tg_agent_id in target_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(predicted_agents.track_token_ids[sample_idx], tg_agent_id)
                if idx is not None:
                    agent_in_target_mask.append(True)
                    coexisting_agent_ids_in_target.append(tg_agent_id)
                else:
                    agent_in_target_mask.append(False)
            agent_in_pred_mask = []
            for pred_agent_id in predicted_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(coexisting_agent_ids_in_target, pred_agent_id)
                if idx is not None:
                    agent_in_pred_mask.append(True)
                else:
                    agent_in_pred_mask.append(False)
            eval_mask = eval_mask[agent_in_target_mask]
            reg_mask = reg_mask[agent_in_target_mask]
            eval_mask = eval_mask.to(reg_mask.device)
            valid_mask_eval = reg_mask[eval_mask]
            pred = {
                key: value[sample_idx][agent_in_pred_mask] for key, value in predicted_agents.trajectories.items()
            }

            num_agents = reg_mask.shape[0]
            gt = torch.zeros((num_agents, self.num_future_steps, self.output_dim), device=reg_mask.device, dtype=torch.float)
            masked_pred_ids = list(np.array(predicted_agents.track_token_ids[sample_idx])[agent_in_pred_mask])
            for i_agent in range(len(coexisting_agent_ids_in_target)):
                agent_id = coexisting_agent_ids_in_target[i_agent]
                agent_idx_in_pred = safe_list_index(masked_pred_ids, agent_id)
                if agent_idx_in_pred is None:
                    continue
                agent_idx_in_target = i_agent
                gt[agent_idx_in_pred][reg_mask[agent_idx_in_target]] = target_agents.trajectories[sample_idx][agent_id][:, :self.output_dim].float()

            if self.output_head:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['loc_refine_head'],
                                         pred['scale_refine_pos'][..., :self.output_dim],
                                         pred['conc_refine_head']], dim=-1)
            else:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            pi = pred['pi']
            gt = torch.cat([gt[..., :self.output_dim], gt[..., -1:]], dim=-1)

            traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
            if not self.output_head:
                traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                         traj_eval[..., :2]], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
            pi_eval = F.softmax(pi[eval_mask], dim=-1)
            gt_eval = gt[eval_mask]
            if self.num_modes_eval == 1:
                most_likely_mode = pi_eval.argmax(dim=-1)
            else:
                most_likely_mode = torch.arange(self.num_modes)

            self.compute_minADE(pred=traj_eval[..., :self.output_dim][:, most_likely_mode, :, :],
                               target=gt_eval[..., :self.output_dim],
                               prob=pi_eval[:, most_likely_mode],
                               valid_mask=valid_mask_eval)

        return self.sum / self.count

    def compute_minADE(self,
                      pred: torch.Tensor,
                      target: torch.Tensor,
                      prob: Optional[torch.Tensor] = None,
                      valid_mask: Optional[torch.Tensor] = None,
                      keep_invalid_final_step: bool = True,
                      min_criterion: str = 'FDE') -> None:
        device = pred.device
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.num_modes, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=device)).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
            self.sum += ((torch.norm(pred_topk[torch.arange(pred.size(0)), inds_best] - target, p=2, dim=-1) *
                          valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        elif min_criterion == 'ADE':
            self.sum += ((torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1) *
                          valid_mask.unsqueeze(1)).sum(dim=-1).min(dim=-1)[0] / valid_mask.sum(dim=-1)).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.count += pred.size(0)

class minAHE(AbstractTrainingMetric):

    def __init__(self,
                 output_dim: int,
                 output_head: bool,
                 num_past_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 self_distillation: bool,
                 only_evaluate_av: bool,
                 num_modes_eval: int,
                 name: str = 'minAHE') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_past_steps = num_past_steps
        self.num_historical_steps = num_past_steps + 1
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.self_distillation = self_distillation
        self.only_evaluate_av = only_evaluate_av
        self.num_modes_eval = num_modes_eval

        self.last_data = None
        self.sum = 0.
        self.count = 0

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

        if 'td3' in predictions.keys():
            if 'target_Q' not in predictions['td3'].keys():  # optimizer_idx == 1
                return None
            if 'evaluation' in predictions.keys():
                # t < start_timesteps and evaluation == False
                if predictions['td3']['t'] < predictions['td3']['start_timesteps'] and not predictions['evaluation']:
                    return None

        targets_before_collating = targets
        if self.last_data is not None:  # self-distillation
            targets = {
                'agents_trajectories': AgentsTrajectories.collate(batch=[self.last_data['agents_trajectories'], targets['agents_trajectories']])
            }
        if self.self_distillation:
            self.last_data = targets_before_collating

        predicted_agents: AgentsTrajectories = predictions["agents_trajectories"]
        target_agents: AgentsTrajectories = targets["agents_trajectories"]
        batch_size = predicted_agents.batch_size
        self.sum = 0.
        self.count = 0

        for sample_idx in range(batch_size):
            # avoid "RuntimeError: "bitwise_or_cpu" not implemented for 'Float'" when only ego in target_agents
            if sample_idx == batch_size - 1:
                if len(target_agents.objects_types[sample_idx]) == 0:
                    if self.count == 0:
                        return None
            if len(target_agents.objects_types[sample_idx]) == 0:
                continue

            if self.only_evaluate_av:
                # only evaluate AV
                eval_mask = torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]])
            else:
                # only evaluate AV VEHICLE, PEDESTRIAN, and BICYCLE
                eval_mask = (
                        (torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'VEHICLE' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'PEDESTRIAN' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'BICYCLE' for object_type in target_agents.objects_types[sample_idx]]))
                )
            reg_mask = target_agents.predict_mask[sample_idx][:, self.num_historical_steps:].bool()

            agent_in_target_mask, coexisting_agent_ids_in_target = [], []
            for tg_agent_id in target_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(predicted_agents.track_token_ids[sample_idx], tg_agent_id)
                if idx is not None:
                    agent_in_target_mask.append(True)
                    coexisting_agent_ids_in_target.append(tg_agent_id)
                else:
                    agent_in_target_mask.append(False)
            agent_in_pred_mask = []
            for pred_agent_id in predicted_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(coexisting_agent_ids_in_target, pred_agent_id)
                if idx is not None:
                    agent_in_pred_mask.append(True)
                else:
                    agent_in_pred_mask.append(False)
            eval_mask = eval_mask[agent_in_target_mask]
            reg_mask = reg_mask[agent_in_target_mask]
            eval_mask = eval_mask.to(reg_mask.device)
            valid_mask_eval = reg_mask[eval_mask]
            pred = {
                key: value[sample_idx][agent_in_pred_mask] for key, value in predicted_agents.trajectories.items()
            }

            num_agents = reg_mask.shape[0]
            gt = torch.zeros((num_agents, self.num_future_steps, self.output_dim), device=reg_mask.device, dtype=torch.float)
            masked_pred_ids = list(np.array(predicted_agents.track_token_ids[sample_idx])[agent_in_pred_mask])
            for i_agent in range(len(coexisting_agent_ids_in_target)):
                agent_id = coexisting_agent_ids_in_target[i_agent]
                agent_idx_in_pred = safe_list_index(masked_pred_ids, agent_id)
                if agent_idx_in_pred is None:
                    continue
                agent_idx_in_target = i_agent
                gt[agent_idx_in_pred][reg_mask[agent_idx_in_target]] = target_agents.trajectories[sample_idx][agent_id][:, :self.output_dim].float()

            if self.output_head:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['loc_refine_head'],
                                         pred['scale_refine_pos'][..., :self.output_dim],
                                         pred['conc_refine_head']], dim=-1)
            else:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            pi = pred['pi']
            gt = torch.cat([gt[..., :self.output_dim], gt[..., -1:]], dim=-1)

            traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
            if not self.output_head:
                traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                         traj_eval[..., :2]], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
            pi_eval = F.softmax(pi[eval_mask], dim=-1)
            gt_eval = gt[eval_mask]
            if self.num_modes_eval == 1:
                most_likely_mode = pi_eval.argmax(dim=-1)
            else:
                most_likely_mode = torch.arange(self.num_modes)

            self.compute_minAHE(pred=traj_eval[..., :self.output_dim][:, most_likely_mode, :, :],
                               target=gt_eval[..., :self.output_dim],
                               prob=pi_eval[:, most_likely_mode],
                               valid_mask=valid_mask_eval)

        return self.sum / self.count

    def compute_minAHE(self,
                      pred: torch.Tensor,
                      target: torch.Tensor,
                      prob: Optional[torch.Tensor] = None,
                      valid_mask: Optional[torch.Tensor] = None,
                      keep_invalid_final_step: bool = True,
                      min_criterion: str = 'FDE') -> None:
        device = pred.device
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.num_modes, pred, prob)
        if min_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=device)).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last, :-1] -
                target[torch.arange(pred.size(0)), inds_last, :-1].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
        elif min_criterion == 'ADE':
            inds_best = (torch.norm(pred_topk[..., :-1] - target[..., :-1].unsqueeze(1), p=2, dim=-1) *
                         valid_mask.unsqueeze(1)).sum(dim=-1).argmin(dim=-1)
        else:
            raise ValueError('{} is not a valid criterion'.format(min_criterion))
        self.sum += ((wrap_angle(pred_topk[torch.arange(pred.size(0)), inds_best, :, -1] - target[..., -1]).abs() *
                      valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
        self.count += pred.size(0)

class minFDE(AbstractTrainingMetric):

    def __init__(self,
                 output_dim: int,
                 output_head: bool,
                 num_past_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 self_distillation: bool,
                 only_evaluate_av: bool,
                 num_modes_eval: int,
                 name: str = 'minFDE') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_past_steps = num_past_steps
        self.num_historical_steps = num_past_steps + 1
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.self_distillation = self_distillation
        self.only_evaluate_av = only_evaluate_av
        self.num_modes_eval = num_modes_eval

        self.last_data = None
        self.sum = 0.
        self.count = 0

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

        if 'td3' in predictions.keys():
            if 'target_Q' not in predictions['td3'].keys():  # optimizer_idx == 1
                return None
            if 'evaluation' in predictions.keys():
                # t < start_timesteps and evaluation == False
                if predictions['td3']['t'] < predictions['td3']['start_timesteps'] and not predictions['evaluation']:
                    return None

        targets_before_collating = targets
        if self.last_data is not None:  # self-distillation
            targets = {
                'agents_trajectories': AgentsTrajectories.collate(batch=[self.last_data['agents_trajectories'], targets['agents_trajectories']])
            }
        if self.self_distillation:
            self.last_data = targets_before_collating

        predicted_agents: AgentsTrajectories = predictions["agents_trajectories"]
        target_agents: AgentsTrajectories = targets["agents_trajectories"]
        batch_size = predicted_agents.batch_size
        self.sum = 0.
        self.count = 0

        for sample_idx in range(batch_size):
            # avoid "RuntimeError: "bitwise_or_cpu" not implemented for 'Float'" when only ego in target_agents
            if sample_idx == batch_size - 1:
                if len(target_agents.objects_types[sample_idx]) == 0:
                    if self.count == 0:
                        return None
            if len(target_agents.objects_types[sample_idx]) == 0:
                continue

            if self.only_evaluate_av:
                # only evaluate AV
                eval_mask = torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]])
            else:
                # only evaluate AV VEHICLE, PEDESTRIAN, and BICYCLE
                eval_mask = (
                        (torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'VEHICLE' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'PEDESTRIAN' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'BICYCLE' for object_type in target_agents.objects_types[sample_idx]]))
                )
            reg_mask = target_agents.predict_mask[sample_idx][:, self.num_historical_steps:].bool()

            agent_in_target_mask, coexisting_agent_ids_in_target = [], []
            for tg_agent_id in target_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(predicted_agents.track_token_ids[sample_idx], tg_agent_id)
                if idx is not None:
                    agent_in_target_mask.append(True)
                    coexisting_agent_ids_in_target.append(tg_agent_id)
                else:
                    agent_in_target_mask.append(False)
            agent_in_pred_mask = []
            for pred_agent_id in predicted_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(coexisting_agent_ids_in_target, pred_agent_id)
                if idx is not None:
                    agent_in_pred_mask.append(True)
                else:
                    agent_in_pred_mask.append(False)
            eval_mask = eval_mask[agent_in_target_mask]
            reg_mask = reg_mask[agent_in_target_mask]
            eval_mask = eval_mask.to(reg_mask.device)
            valid_mask_eval = reg_mask[eval_mask]
            pred = {
                key: value[sample_idx][agent_in_pred_mask] for key, value in predicted_agents.trajectories.items()
            }

            num_agents = reg_mask.shape[0]
            gt = torch.zeros((num_agents, self.num_future_steps, self.output_dim), device=reg_mask.device, dtype=torch.float)
            masked_pred_ids = list(np.array(predicted_agents.track_token_ids[sample_idx])[agent_in_pred_mask])
            for i_agent in range(len(coexisting_agent_ids_in_target)):
                agent_id = coexisting_agent_ids_in_target[i_agent]
                agent_idx_in_pred = safe_list_index(masked_pred_ids, agent_id)
                if agent_idx_in_pred is None:
                    continue
                agent_idx_in_target = i_agent
                gt[agent_idx_in_pred][reg_mask[agent_idx_in_target]] = target_agents.trajectories[sample_idx][agent_id][:, :self.output_dim].float()

            if self.output_head:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['loc_refine_head'],
                                         pred['scale_refine_pos'][..., :self.output_dim],
                                         pred['conc_refine_head']], dim=-1)
            else:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            pi = pred['pi']
            gt = torch.cat([gt[..., :self.output_dim], gt[..., -1:]], dim=-1)

            traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
            if not self.output_head:
                traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                         traj_eval[..., :2]], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
            pi_eval = F.softmax(pi[eval_mask], dim=-1)
            gt_eval = gt[eval_mask]
            if self.num_modes_eval == 1:
                most_likely_mode = pi_eval.argmax(dim=-1)
            else:
                most_likely_mode = torch.arange(self.num_modes)

            self.compute_minFDE(pred=traj_eval[..., :self.output_dim][:, most_likely_mode, :, :],
                               target=gt_eval[..., :self.output_dim],
                               prob=pi_eval[:, most_likely_mode],
                               valid_mask=valid_mask_eval)

        return self.sum / self.count

    def compute_minFDE(self,
                      pred: torch.Tensor,
                      target: torch.Tensor,
                      prob: Optional[torch.Tensor] = None,
                      valid_mask: Optional[torch.Tensor] = None,
                      keep_invalid_final_step: bool = True,
                      min_criterion: str = 'FDE') -> None:
        device = pred.device
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.num_modes, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=device)).argmax(dim=-1)
        self.sum += torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                               target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                               p=2, dim=-1).min(dim=-1)[0].sum()
        self.count += pred.size(0)

class minFHE(AbstractTrainingMetric):

    def __init__(self,
                 output_dim: int,
                 output_head: bool,
                 num_past_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 self_distillation: bool,
                 only_evaluate_av: bool,
                 num_modes_eval: int,
                 name: str = 'minFHE') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_past_steps = num_past_steps
        self.num_historical_steps = num_past_steps + 1
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.self_distillation = self_distillation
        self.only_evaluate_av = only_evaluate_av
        self.num_modes_eval = num_modes_eval

        self.last_data = None
        self.sum = 0.
        self.count = 0

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

        if 'td3' in predictions.keys():
            if 'target_Q' not in predictions['td3'].keys():  # optimizer_idx == 1
                return None
            if 'evaluation' in predictions.keys():
                # t < start_timesteps and evaluation == False
                if predictions['td3']['t'] < predictions['td3']['start_timesteps'] and not predictions['evaluation']:
                    return None

        targets_before_collating = targets
        if self.last_data is not None:  # self-distillation
            targets = {
                'agents_trajectories': AgentsTrajectories.collate(batch=[self.last_data['agents_trajectories'], targets['agents_trajectories']])
            }
        if self.self_distillation:
            self.last_data = targets_before_collating

        predicted_agents: AgentsTrajectories = predictions["agents_trajectories"]
        target_agents: AgentsTrajectories = targets["agents_trajectories"]
        batch_size = predicted_agents.batch_size
        self.sum = 0.
        self.count = 0

        for sample_idx in range(batch_size):
            # avoid "RuntimeError: "bitwise_or_cpu" not implemented for 'Float'" when only ego in target_agents
            if sample_idx == batch_size - 1:
                if len(target_agents.objects_types[sample_idx]) == 0:
                    if self.count == 0:
                        return None
            if len(target_agents.objects_types[sample_idx]) == 0:
                continue

            if self.only_evaluate_av:
                # only evaluate AV
                eval_mask = torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]])
            else:
                # only evaluate AV VEHICLE, PEDESTRIAN, and BICYCLE
                eval_mask = (
                        (torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'VEHICLE' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'PEDESTRIAN' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'BICYCLE' for object_type in target_agents.objects_types[sample_idx]]))
                )
            reg_mask = target_agents.predict_mask[sample_idx][:, self.num_historical_steps:].bool()

            agent_in_target_mask, coexisting_agent_ids_in_target = [], []
            for tg_agent_id in target_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(predicted_agents.track_token_ids[sample_idx], tg_agent_id)
                if idx is not None:
                    agent_in_target_mask.append(True)
                    coexisting_agent_ids_in_target.append(tg_agent_id)
                else:
                    agent_in_target_mask.append(False)
            agent_in_pred_mask = []
            for pred_agent_id in predicted_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(coexisting_agent_ids_in_target, pred_agent_id)
                if idx is not None:
                    agent_in_pred_mask.append(True)
                else:
                    agent_in_pred_mask.append(False)
            eval_mask = eval_mask[agent_in_target_mask]
            reg_mask = reg_mask[agent_in_target_mask]
            eval_mask = eval_mask.to(reg_mask.device)
            valid_mask_eval = reg_mask[eval_mask]
            pred = {
                key: value[sample_idx][agent_in_pred_mask] for key, value in predicted_agents.trajectories.items()
            }

            num_agents = reg_mask.shape[0]
            gt = torch.zeros((num_agents, self.num_future_steps, self.output_dim), device=reg_mask.device, dtype=torch.float)
            masked_pred_ids = list(np.array(predicted_agents.track_token_ids[sample_idx])[agent_in_pred_mask])
            for i_agent in range(len(coexisting_agent_ids_in_target)):
                agent_id = coexisting_agent_ids_in_target[i_agent]
                agent_idx_in_pred = safe_list_index(masked_pred_ids, agent_id)
                if agent_idx_in_pred is None:
                    continue
                agent_idx_in_target = i_agent
                gt[agent_idx_in_pred][reg_mask[agent_idx_in_target]] = target_agents.trajectories[sample_idx][agent_id][:, :self.output_dim].float()

            if self.output_head:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['loc_refine_head'],
                                         pred['scale_refine_pos'][..., :self.output_dim],
                                         pred['conc_refine_head']], dim=-1)
            else:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            pi = pred['pi']
            gt = torch.cat([gt[..., :self.output_dim], gt[..., -1:]], dim=-1)

            traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
            if not self.output_head:
                traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                         traj_eval[..., :2]], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
            pi_eval = F.softmax(pi[eval_mask], dim=-1)
            gt_eval = gt[eval_mask]
            if self.num_modes_eval == 1:
                most_likely_mode = pi_eval.argmax(dim=-1)
            else:
                most_likely_mode = torch.arange(self.num_modes)

            self.compute_minFHE(pred=traj_eval[..., :self.output_dim][:, most_likely_mode, :, :],
                               target=gt_eval[..., :self.output_dim],
                               prob=pi_eval[:, most_likely_mode],
                               valid_mask=valid_mask_eval)

        return self.sum / self.count

    def compute_minFHE(self,
                      pred: torch.Tensor,
                      target: torch.Tensor,
                      prob: Optional[torch.Tensor] = None,
                      valid_mask: Optional[torch.Tensor] = None,
                      keep_invalid_final_step: bool = True,
                      min_criterion: str = 'FDE') -> None:
        device = pred.device
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.num_modes, pred, prob)
        inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=device)).argmax(dim=-1)
        inds_best = torch.norm(
            pred_topk[torch.arange(pred.size(0)), :, inds_last, :-1] -
            target[torch.arange(pred.size(0)), inds_last, :-1].unsqueeze(-2), p=2, dim=-1).argmin(dim=-1)
        self.sum += wrap_angle(pred_topk[torch.arange(pred.size(0)), inds_best, inds_last, -1] -
                               target[torch.arange(pred.size(0)), inds_last, -1]).abs().sum()
        self.count += pred.size(0)

class MR(AbstractTrainingMetric):

    def __init__(self,
                 output_dim: int,
                 output_head: bool,
                 num_past_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 self_distillation: bool,
                 only_evaluate_av: bool,
                 num_modes_eval: int,
                 name: str = 'MR') -> None:
        """
        Initializes the class.

        :param name: the name of the training_metric (used in logger)
        """
        self._name = name
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_past_steps = num_past_steps
        self.num_historical_steps = num_past_steps + 1
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.self_distillation = self_distillation
        self.only_evaluate_av = only_evaluate_av
        self.num_modes_eval = num_modes_eval

        self.last_data = None
        self.sum = 0.
        self.count = 0

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

        if 'td3' in predictions.keys():
            if 'target_Q' not in predictions['td3'].keys():  # optimizer_idx == 1
                return None
            if 'evaluation' in predictions.keys():
                # t < start_timesteps and evaluation == False
                if predictions['td3']['t'] < predictions['td3']['start_timesteps'] and not predictions['evaluation']:
                    return None

        targets_before_collating = targets
        if self.last_data is not None:  # self-distillation
            targets = {
                'agents_trajectories': AgentsTrajectories.collate(batch=[self.last_data['agents_trajectories'], targets['agents_trajectories']])
            }
        if self.self_distillation:
            self.last_data = targets_before_collating

        predicted_agents: AgentsTrajectories = predictions["agents_trajectories"]
        target_agents: AgentsTrajectories = targets["agents_trajectories"]
        batch_size = predicted_agents.batch_size
        self.sum = 0.
        self.count = 0

        for sample_idx in range(batch_size):
            # avoid "RuntimeError: "bitwise_or_cpu" not implemented for 'Float'" when only ego in target_agents
            if sample_idx == batch_size - 1:
                if len(target_agents.objects_types[sample_idx]) == 0:
                    if self.count == 0:
                        return None
            if len(target_agents.objects_types[sample_idx]) == 0:
                continue

            if self.only_evaluate_av:
                # only evaluate AV
                eval_mask = torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]])
            else:
                # only evaluate AV VEHICLE, PEDESTRIAN, and BICYCLE
                eval_mask = (
                        (torch.tensor([object_type == 'AV' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'VEHICLE' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'PEDESTRIAN' for object_type in target_agents.objects_types[sample_idx]]))
                        | (torch.tensor([object_type == 'BICYCLE' for object_type in target_agents.objects_types[sample_idx]]))
                )
            reg_mask = target_agents.predict_mask[sample_idx][:, self.num_historical_steps:].bool()

            agent_in_target_mask, coexisting_agent_ids_in_target = [], []
            for tg_agent_id in target_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(predicted_agents.track_token_ids[sample_idx], tg_agent_id)
                if idx is not None:
                    agent_in_target_mask.append(True)
                    coexisting_agent_ids_in_target.append(tg_agent_id)
                else:
                    agent_in_target_mask.append(False)
            agent_in_pred_mask = []
            for pred_agent_id in predicted_agents.track_token_ids[sample_idx]:
                idx = safe_list_index(coexisting_agent_ids_in_target, pred_agent_id)
                if idx is not None:
                    agent_in_pred_mask.append(True)
                else:
                    agent_in_pred_mask.append(False)
            eval_mask = eval_mask[agent_in_target_mask]
            reg_mask = reg_mask[agent_in_target_mask]
            eval_mask = eval_mask.to(reg_mask.device)
            valid_mask_eval = reg_mask[eval_mask]
            pred = {
                key: value[sample_idx][agent_in_pred_mask] for key, value in predicted_agents.trajectories.items()
            }

            num_agents = reg_mask.shape[0]
            gt = torch.zeros((num_agents, self.num_future_steps, self.output_dim), device=reg_mask.device, dtype=torch.float)
            masked_pred_ids = list(np.array(predicted_agents.track_token_ids[sample_idx])[agent_in_pred_mask])
            for i_agent in range(len(coexisting_agent_ids_in_target)):
                agent_id = coexisting_agent_ids_in_target[i_agent]
                agent_idx_in_pred = safe_list_index(masked_pred_ids, agent_id)
                if agent_idx_in_pred is None:
                    continue
                agent_idx_in_target = i_agent
                gt[agent_idx_in_pred][reg_mask[agent_idx_in_target]] = target_agents.trajectories[sample_idx][agent_id][:, :self.output_dim].float()

            if self.output_head:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['loc_refine_head'],
                                         pred['scale_refine_pos'][..., :self.output_dim],
                                         pred['conc_refine_head']], dim=-1)
            else:
                traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                         pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
            pi = pred['pi']
            gt = torch.cat([gt[..., :self.output_dim], gt[..., -1:]], dim=-1)

            traj_eval = traj_refine[eval_mask, :, :, :self.output_dim + self.output_head]
            if not self.output_head:
                traj_2d_with_start_pos_eval = torch.cat([traj_eval.new_zeros((traj_eval.size(0), self.num_modes, 1, 2)),
                                                         traj_eval[..., :2]], dim=-2)
                motion_vector_eval = traj_2d_with_start_pos_eval[:, :, 1:] - traj_2d_with_start_pos_eval[:, :, :-1]
                head_eval = torch.atan2(motion_vector_eval[..., 1], motion_vector_eval[..., 0])
                traj_eval = torch.cat([traj_eval, head_eval.unsqueeze(-1)], dim=-1)
            pi_eval = F.softmax(pi[eval_mask], dim=-1)
            gt_eval = gt[eval_mask]
            if self.num_modes_eval == 1:
                most_likely_mode = pi_eval.argmax(dim=-1)
            else:
                most_likely_mode = torch.arange(self.num_modes)

            self.compute_MR(pred=traj_eval[..., :self.output_dim][:, most_likely_mode, :, :],
                            target=gt_eval[..., :self.output_dim],
                            prob=pi_eval[:, most_likely_mode],
                            valid_mask=valid_mask_eval)

        return self.sum / self.count

    def compute_MR(self,
                   pred: torch.Tensor,
                   target: torch.Tensor,
                   prob: Optional[torch.Tensor] = None,
                   valid_mask: Optional[torch.Tensor] = None,
                   keep_invalid_final_step: bool = True,
                   miss_criterion: str = 'FDE',
                   miss_threshold: float = 2.0) -> None:
        device = pred.device
        pred, target, prob, valid_mask, _ = valid_filter(pred, target, prob, valid_mask, None, keep_invalid_final_step)
        pred_topk, _ = topk(self.num_modes, pred, prob)
        if miss_criterion == 'FDE':
            inds_last = (valid_mask * torch.arange(1, valid_mask.size(-1) + 1, device=device)).argmax(dim=-1)
            self.sum += (torch.norm(pred_topk[torch.arange(pred.size(0)), :, inds_last] -
                                    target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                                    p=2, dim=-1).min(dim=-1)[0] > miss_threshold).sum()
        elif miss_criterion == 'MAXDE':
            self.sum += (((torch.norm(pred_topk - target.unsqueeze(1),
                                      p=2, dim=-1) * valid_mask.unsqueeze(1)).max(dim=-1)[0]).min(dim=-1)[0] >
                         miss_threshold).sum()
        else:
            raise ValueError('{} is not a valid criterion'.format(miss_criterion))
        self.count += pred.size(0)