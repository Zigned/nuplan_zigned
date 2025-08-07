from typing import Dict, List, cast, Optional, Any

import torch
import torch.nn.functional as F
import numpy as np

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from nuplan_zigned.training.preprocessing.features.qcmae_agents_trajectories import AgentsTrajectories
from nuplan_zigned.training.modeling.objectives.losses.qcmae_nll_loss import NLLLoss
from nuplan_zigned.training.modeling.objectives.losses.qcmae_mixture_nll_loss import MixtureNLLLoss
from nuplan_zigned.utils.utils import safe_list_index


class ActorObjective(AbstractObjective):

    def __init__(self,
                 scenario_type_loss_weighting: Dict[str, float],
                 output_dim: int,
                 output_head: bool,
                 num_past_steps: int,
                 only_ego: bool
                 ):
        """
        Initializes the class
        """
        self._name = 'actor_objective'

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * output_dim + ['von_mises'] * output_head,
                                       reduction='none')

        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_past_steps + 1
        self.only_ego = only_ego

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

        if optimizer_idx == 1:  # actor loss
            if predictions['td3']['actor_loss'] == 'to_be_computed_using_pseudo_ground_truth':
                predicted_trajectory = cast(AgentsTrajectories, predictions["agents_trajectories"])
                targets_trajectory = cast(AgentsTrajectories, targets["agents_trajectories"])

                batch_size = predicted_trajectory.batch_size

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
                            if self.only_ego:
                                if tg_agent_id == 'AV':
                                    agent_in_target_mask.append(True)
                                    coexisting_agent_ids_in_target.append(tg_agent_id)
                                else:
                                    agent_in_target_mask.append(False)
                            else:
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

                    list_traj_refine.append(traj_refine)
                    list_pi.append(pi)
                    list_gt.append(gt)
                    list_traj_propose_best.append(traj_propose_best)
                    list_traj_refine_best.append(traj_refine_best)
                    list_l2_norm.append(l2_norm)
                    list_batch_info.append(torch.full(size=(traj_propose.size(0),), fill_value=sample_idx, dtype=torch.uint8))

                traj_refine = torch.cat(list_traj_refine, dim=0)
                pi = torch.cat(list_pi, dim=0)
                gt = torch.cat(list_gt, dim=0)
                traj_propose_best = torch.cat(list_traj_propose_best, dim=0)
                traj_refine_best = torch.cat(list_traj_refine_best, dim=0)
                reg_mask = torch.cat(list_reg_mask, dim=0)
                cls_mask = torch.cat(list_cls_mask, dim=0)

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
                loss = reg_loss_propose + reg_loss_refine + cls_loss

            else:
                actor_loss = predictions['td3']['actor_loss']
                loss = actor_loss

        return loss
