from typing import Dict, List, cast, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType

from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories
from nuplan_zigned.training.modeling.objectives.losses.qcmae_nll_loss import NLLLoss
from nuplan_zigned.training.modeling.objectives.losses.qcmae_mixture_nll_loss import MixtureNLLLoss
from nuplan_zigned.utils.utils import safe_list_index


class AgentsImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(
        self,
        scenario_type_loss_weighting: Dict[str, float],
        name: str,
        weight: float,
        output_dim: int,
        output_head: bool,
        num_past_steps: int,
        self_distillation: bool,
        sd_temperature: float,
        sd_alpha: float,
        pretrain: bool,
        pretrain_map_encoder: bool,
        pretrain_agent_encoder: bool,
        prob_pretrain_mask: float,
        prob_pretrain_mask_mask: float,
        prob_pretrain_mask_random: float,
        prob_pretrain_mask_unchanged: float,
        sd_margin: float=0.1,
    ):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = name
        self._weight = weight
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_past_steps + 1
        self.self_distillation = self_distillation
        self.sd_temperature = sd_temperature
        self.sd_alpha = sd_alpha
        self.sd_margin = sd_margin
        self.pretrain = pretrain
        self.pretrain_map_encoder = pretrain_map_encoder
        self.pretrain_agent_encoder = pretrain_agent_encoder
        self.prob_pretrain_mask = prob_pretrain_mask
        self.prob_pretrain_mask_mask = prob_pretrain_mask_mask
        self.prob_pretrain_mask_random = prob_pretrain_mask_random
        self.prob_pretrain_mask_unchanged = prob_pretrain_mask_unchanged

        self.last_data = None
        self.last_traj_propose = None
        self.last_traj_refine = None
        self.last_pi = None

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["agents_trajectories"]

    def compute(self,
                predictions: FeaturesType,
                targets: TargetsType,
                scenarios: ScenarioListType,
                optimizer_idx: Optional[int]=None,) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        if optimizer_idx == 0:
            return None
        elif optimizer_idx == 1:
            if predictions['agents_trajectories'] is None:
                return None

        targets_before_collating = targets

        # reset self.last_data when starting validation
        if self.last_data is not None:
            if predictions["agents_trajectories"].batch_size < targets['agents_trajectories'].batch_size * 2:
                self.last_data = None
                self.last_traj_propose = None
                self.last_traj_refine = None
                self.last_pi = None

        if self.last_data is not None:  # self-distillation
            targets = {
                'agents_trajectories': AgentsTrajectories.collate(batch=[self.last_data['agents_trajectories'], targets['agents_trajectories']])
            }
        if self.self_distillation:
            self.last_data = targets_before_collating

        predicted_trajectory = cast(AgentsTrajectories, predictions["agents_trajectories"])
        targets_trajectory = cast(AgentsTrajectories, targets["agents_trajectories"])

        batch_size = predicted_trajectory.batch_size

        loss = []
        list_traj_propose = []
        list_traj_refine = []
        list_pi = []
        list_gt = []
        list_traj_propose_best = []
        list_traj_refine_best = []
        list_reg_mask = []
        list_cls_mask = []
        list_l2_norm = []
        list_batch_info = []

        for sample_idx in range(batch_size):
            # use predict_mask in targets
            reg_mask = targets_trajectory.predict_mask[sample_idx][:, self.num_historical_steps:]

            cls_mask = targets_trajectory.predict_mask[sample_idx][:, -1]
            agent_in_target_mask, coexisting_agent_ids_in_target = [], []
            for tg_agent_id in targets_trajectory.track_token_ids[sample_idx]:
                idx = safe_list_index(predicted_trajectory.track_token_ids[sample_idx], tg_agent_id)
                if idx is not None:
                    agent_in_target_mask.append(True)
                    coexisting_agent_ids_in_target.append(tg_agent_id)
                else:
                    agent_in_target_mask.append(False)
            agent_in_pred_mask = []
            for pred_agent_id in predicted_trajectory.track_token_ids[sample_idx]:
                idx = safe_list_index(coexisting_agent_ids_in_target, pred_agent_id)
                if idx is not None:
                    agent_in_pred_mask.append(True)
                else:
                    agent_in_pred_mask.append(False)
            reg_mask = reg_mask[agent_in_target_mask]
            cls_mask = cls_mask[agent_in_target_mask]
            num_agents = reg_mask.shape[0]
            num_future_steps = reg_mask.shape[1]
            list_reg_mask.append(reg_mask)
            list_cls_mask.append(cls_mask)

            gt = torch.zeros((num_agents, num_future_steps, self.output_dim), device=reg_mask.device, dtype=torch.float)
            masked_pred_ids = list(np.array(predicted_trajectory.track_token_ids[sample_idx])[agent_in_pred_mask])
            for i_agent in range(len(coexisting_agent_ids_in_target)):
                agent_id = coexisting_agent_ids_in_target[i_agent]
                agent_idx_in_pred = safe_list_index(masked_pred_ids, agent_id)
                if agent_idx_in_pred is None:
                    continue
                agent_idx_in_target = i_agent
                gt[agent_idx_in_pred][reg_mask[agent_idx_in_target]] = targets_trajectory.trajectories[sample_idx][agent_id][:, :self.output_dim].float()

            if self.pretrain:
                pred = {
                    key: value[sample_idx] for key, value in predicted_trajectory.trajectories.items()
                }
                if self.pretrain_map_encoder and not self.pretrain_agent_encoder:
                    l2_norm = torch.norm(pred['x_pl_predicted'] - pred['x_pl_before_masking'], p=2, dim=-1)
                    loss.append(l2_norm.mean())
                elif self.pretrain_agent_encoder and not self.pretrain_map_encoder:
                    l2_norm = torch.norm(pred['x_a_predicted'] - pred['x_a_before_masking'], p=2, dim=-1)
                    loss.append(l2_norm.mean())
                elif self.pretrain_map_encoder and self.pretrain_agent_encoder:
                    l2_norm1 = torch.norm(pred['x_pl_predicted'] - pred['x_pl_before_masking'], p=2, dim=-1)
                    l2_norm2 = torch.norm(pred['x_a_predicted'] - pred['x_a_before_masking'], p=2, dim=-1)
                    loss1 = l2_norm1.mean()
                    loss2 = l2_norm2.mean()
                    loss.append(loss1 + loss2)
            else:
                pred = {
                    key: value[sample_idx][agent_in_pred_mask] for key, value in predicted_trajectory.trajectories.items()
                }
                if self.output_head:
                    traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                              pred['loc_propose_head'],
                                              pred['scale_propose_pos'][..., :self.output_dim],
                                              pred['conc_propose_head']], dim=-1)
                    traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                             pred['loc_refine_head'],
                                             pred['scale_refine_pos'][..., :self.output_dim],
                                             pred['conc_refine_head']], dim=-1)
                else:
                    traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                              pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
                    traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                             pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
                pi = pred['pi']
                gt = torch.cat([gt[..., :self.output_dim], gt[..., -1:]], dim=-1)
                l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                                      gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
                best_mode = l2_norm.argmin(dim=-1)
                traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
                traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]

                list_traj_propose.append(traj_propose)
                list_traj_refine.append(traj_refine)
                list_pi.append(pi)
                list_gt.append(gt)
                list_traj_propose_best.append(traj_propose_best)
                list_traj_refine_best.append(traj_refine_best)
                list_l2_norm.append(l2_norm)
                list_batch_info.append(torch.full(size=(traj_propose.size(0), ), fill_value=sample_idx, dtype=torch.uint8))

        if self.pretrain:
            loss = sum(loss) / batch_size
        else:
            traj_propose = torch.cat(list_traj_propose, dim=0)
            traj_refine = torch.cat(list_traj_refine, dim=0)
            pi = torch.cat(list_pi, dim=0)
            gt = torch.cat(list_gt, dim=0)
            traj_propose_best = torch.cat(list_traj_propose_best, dim=0)
            traj_refine_best = torch.cat(list_traj_refine_best, dim=0)
            reg_mask = torch.cat(list_reg_mask, dim=0)
            cls_mask = torch.cat(list_cls_mask, dim=0)
            l2_norm = torch.cat(list_l2_norm, dim=0)
            batch_info = torch.cat(list_batch_info, dim=0)

            if self.last_traj_propose is not None:  # self-distillation
                half_batch_size = batch_size // 2
                stu_mask = batch_info < half_batch_size
                traj_propose_stu = traj_propose[stu_mask]
                traj_refine_stu = traj_refine[stu_mask]
                reg_mask_stu = reg_mask[stu_mask]
                cls_mask_stu = cls_mask[stu_mask]
                pi_last = pi[stu_mask]
                gt_stu = gt[stu_mask]
                l2_norm_propose_stu = l2_norm[stu_mask]

                # v3 202311031129 self-bounded regression loss, laplace nll loss
                best_mode_stu = l2_norm_propose_stu.argmin(dim=-1)
                l2_norm_propose_tch = (torch.norm(self.last_traj_propose[..., :self.output_dim] -
                                                  gt_stu[..., :self.output_dim].unsqueeze(1), p=2,
                                                  dim=-1) * reg_mask_stu.unsqueeze(1)).sum(dim=-1)
                l2_norm_propose_best_stu = l2_norm_propose_stu[torch.arange(l2_norm_propose_stu.size(0)), best_mode_stu]
                l2_norm_propose_best_tch = l2_norm_propose_tch[torch.arange(l2_norm_propose_stu.size(0)), best_mode_stu]
                sd_mask_propose = (l2_norm_propose_best_stu + self.sd_margin > l2_norm_propose_best_tch)
                sd_loss_propose = self.reg_loss(traj_propose_stu[torch.arange(l2_norm_propose_stu.size(0)), best_mode_stu],
                                                gt_stu[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask_stu
                sd_loss_propose = (sd_loss_propose.sum(dim=1) / reg_mask_stu.sum(dim=1).clamp_(min=1)) * sd_mask_propose
                sd_loss_propose = sd_loss_propose.mean()
                l2_norm_refine_stu = (torch.norm(traj_refine_stu[..., :self.output_dim] -
                                                 gt_stu[..., :self.output_dim].unsqueeze(1), p=2,
                                                 dim=-1) * reg_mask_stu.unsqueeze(1)).sum(dim=-1)
                l2_norm_refine_tch = (torch.norm(self.last_traj_refine[..., :self.output_dim] -
                                                 gt_stu[..., :self.output_dim].unsqueeze(1), p=2,
                                                 dim=-1) * reg_mask_stu.unsqueeze(1)).sum(dim=-1)
                l2_norm_refine_best_stu = l2_norm_refine_stu[torch.arange(l2_norm_refine_stu.size(0)), best_mode_stu]
                l2_norm_refine_best_tch = l2_norm_refine_tch[torch.arange(l2_norm_refine_stu.size(0)), best_mode_stu]
                sd_mask_refine = (l2_norm_refine_best_stu + self.sd_margin > l2_norm_refine_best_tch)
                sd_loss_refine = self.reg_loss(traj_refine_stu[torch.arange(l2_norm_refine_stu.size(0)), best_mode_stu],
                                               gt_stu[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask_stu
                sd_loss_refine = (sd_loss_refine.sum(dim=1) / reg_mask_stu.sum(dim=1).clamp_(min=1)) * sd_mask_refine
                sd_loss_refine = sd_loss_refine.mean()
                # self-distillation loss DKL(TCH||STU)
                sd_loss_cls = (
                        F.kl_div(
                            F.log_softmax(pi_last / self.sd_temperature, dim=1),  # ln(Q) in DKL(P||Q)
                            F.softmax(self.last_pi / self.sd_temperature, dim=1),  # P in DKL(P||Q)
                            reduction="batchmean",
                        )
                        * self.sd_temperature
                        * self.sd_temperature
                )

            reg_loss_propose = self.reg_loss(traj_propose_best,
                                             gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
            reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_propose = reg_loss_propose.mean()
            reg_loss_refine = self.reg_loss(traj_refine_best,
                                            gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
            reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
            reg_loss_refine = reg_loss_refine.mean()
            cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                     target=gt[:, -1:, :self.output_dim + self.output_head],
                                     prob=pi,
                                     mask=reg_mask[:, -1:]) * cls_mask
            cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1)
            if self.self_distillation and self.last_traj_propose is not None:
                loss = reg_loss_propose + reg_loss_refine + cls_loss + self.sd_alpha * (sd_loss_propose + sd_loss_refine + sd_loss_cls)

            else:
                loss = reg_loss_propose + reg_loss_refine + cls_loss
            if self.self_distillation:
                if self.last_traj_propose is None:
                    self.last_traj_propose = traj_propose.detach()
                    self.last_traj_refine = traj_refine.detach()
                    self.last_pi = pi.detach()
                else:
                    half_batch_size = batch_size // 2
                    tch_mask = batch_info >= half_batch_size
                    self.last_traj_propose = traj_propose[tch_mask].detach()
                    self.last_traj_refine = traj_refine[tch_mask].detach()
                    self.last_pi = pi[tch_mask].detach()

        return self._weight * loss
