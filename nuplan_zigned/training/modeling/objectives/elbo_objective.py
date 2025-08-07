import logging
from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

logger = logging.getLogger(__name__)


class ELBOObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, reward: TorchModuleWrapper, scenario_type_loss_weighting: Dict[str, float], base: float=2.718):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'elbo_objective'
        self._reward = reward
        self._reg = Regularization(model=reward,
                                   weight_decay=(1-reward.dropout)/(2 * reward.sigma_prior**2 * (reward.num_training_scenarios / reward.batch_size)),
                                   bias_decay=1/(2 * reward.sigma_prior**2 * reward.num_training_scenarios),
                                   p=2)
        self._base = base

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """

        if predictions is None:
            return None

        predicted_reward = cast(torch.Tensor, predictions["reward"].data)  # torch.Size([batch_size, num_trajs: 1+27, num_poses: 10])

        # estimate the negative expected log likelihood with a single sample
        r_hat_demon = predicted_reward[:, 0, :]  # torch.Size([batch_size, num_poses])
        R_hat_demon = r_hat_demon.sum(-1)  # torch.Size([batch_size])
        r_hat_actor = predicted_reward[:, 1:, :]  # torch.Size([batch_size, 27, 10])
        R_hat_actor = r_hat_actor.sum(-1)  # torch.Size([batch_size, 27])
        if self._base == 2.718:
            pow_R_hat_demon = torch.exp(R_hat_demon)
            pow_R_hat_actor = torch.exp(R_hat_actor)
        else:
            pow_R_hat_demon = torch.pow(self._base, R_hat_demon)
            pow_R_hat_actor = torch.pow(self._base, R_hat_actor)

        neg_expected_log_lik_action = - torch.log(pow_R_hat_demon / (pow_R_hat_actor.sum(-1) + pow_R_hat_demon))

        # Approximate KL divergence to the prior
        weight_list, bias_list = self._reg.get_params(self._reward)
        kl_loss = self._reg.regularization_loss(weight_list, bias_list, p=2)

        # ELBO
        loss = neg_expected_log_lik_action.mean() + kl_loss

        return loss


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, bias_decay, p=2):
        """
        :param model:
        :param weight_decay:
        :param bias_decay:
        :param p: the order of norm. Default: 2.
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            logger.info("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.bias_decay = bias_decay
        self.p = p
        self.weight_list, self.bias_list = self.get_params(model)
        self.params_info(self.weight_list, self.bias_list)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        # self.weight_list = self.get_weight(model)  # 获得最新的权重
        # reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        self.weight_list, self.bias_list = self.get_params(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.bias_list, p=self.p)
        return reg_loss

    def get_params(self, model):
        weight_list = []
        bias_list = []
        # map encoder
        if hasattr(model, 'map_encoder'):
            for name, param in model.map_encoder.pt2pl_layers.named_parameters():
                if 'weight' in name and 'norm' not in name:
                    weight = (name, param)
                    weight_list.append(weight)
                elif 'bias' in name and 'norm' not in name:
                    bias = (name, param)
                    bias_list.append(bias)
            for name, param in model.map_encoder.pl2pl_layers.named_parameters():
                if 'weight' in name and 'norm' not in name:
                    weight = (name, param)
                    weight_list.append(weight)
                elif 'bias' in name and 'norm' not in name:
                    bias = (name, param)
                    bias_list.append(bias)

        # agent encoder
        if hasattr(model, 'agent_encoder'):
            for name, param in model.agent_encoder.pl2a_attn_layers.named_parameters():
                if 'weight' in name and 'norm' not in name:
                    weight = (name, param)
                    weight_list.append(weight)
                elif 'bias' in name and 'norm' not in name:
                    bias = (name, param)
                    bias_list.append(bias)
            for name, param in model.agent_encoder.a2a_attn_layers.named_parameters():
                if 'weight' in name and 'norm' not in name:
                    weight = (name, param)
                    weight_list.append(weight)
                elif 'bias' in name and 'norm' not in name:
                    bias = (name, param)
                    bias_list.append(bias)

        # reward encoder
        if hasattr(model, 'reward_encoder'):
            for name, param in model.reward_encoder.fc.named_parameters():
                if 'weight' in name:
                    weight = (name, param)
                    weight_list.append(weight)
                elif 'bias' in name:
                    bias = (name, param)
                    bias_list.append(bias)

        return weight_list, bias_list

    def regularization_loss(self, weight_list, bias_list, p=2):
        weight_reg_loss = 0
        bias_reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.linalg.norm(w.view((-1)), ord=p) ** 2
            weight_reg_loss = weight_reg_loss + l2_reg
        for name, b in bias_list:
            l2_reg = torch.linalg.norm(b.view((-1)), ord=p) ** 2
            bias_reg_loss = bias_reg_loss + l2_reg
        reg_loss = self.weight_decay * weight_reg_loss + self.bias_decay * bias_reg_loss
        return reg_loss

    def params_info(self, weight_list, bias_list):
        try:
            logger.info("---------------regularization weight---------------")
            for name, w in weight_list:
                logger.info(name)
            logger.info("---------------------------------------------------")
            logger.info("---------------regularization bias---------------")
            for name, w in bias_list:
                logger.info(name)
            logger.info("---------------------------------------------------")
        except:
            print("---------------regularization weight---------------")
            for name, w in weight_list:
                print(name)
            print("---------------------------------------------------")
            print("---------------regularization bias---------------")
            for name, w in bias_list:
                print(name)
            print("---------------------------------------------------")
