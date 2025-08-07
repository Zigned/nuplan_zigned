from dataclasses import dataclass
from typing import Dict, List, Any

import torch
import time

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput, PlannerInitialization

from nuplan_zigned.training.preprocessing.target_builders.ritp_agents_trajectories_target_builder import AgentTrajectoryTargetBuilder
from nuplan_zigned.training.modeling.modules.ritp_actor import Actor
from nuplan_zigned.simulation.modules.ritp_post_optimizer import PostOptimizer
from nuplan_zigned.training.modeling.modules.ritp_critic import Critic
from nuplan_zigned.training.preprocessing.feature_builders.qcmae_vector_set_map_feature_builder import VectorSetMapFeatureBuilder
from nuplan_zigned.training.preprocessing.feature_builders.qcmae_generic_agents_feature_builder import GenericAgentsFeatureBuilder


@dataclass
class RITPPlannerFeatureParams:
    """
    Parameters for RITP planner features.
    """
    radius: float
    map_features: List[str]
    max_elements: Dict[str, int]
    max_points: Dict[str, int]
    max_agents: Dict[str, int]
    agent_features: List[str]
    max_agents: Dict[str, int]
    interpolation_method: str
    num_past_steps: int
    num_future_steps: int
    past_trajectory_sampling: TrajectorySampling
    future_trajectory_sampling: TrajectorySampling


@dataclass
class RITPPlannerModelParams:
    """
    Parameters for RITP planner
    :param gpus: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node. gpus = lightning.trainer.params.gpus
    """
    finetune_range: str
    hybrid_driven: bool
    frenet_radius: float
    num_plans: int
    num_modes_for_eval: int
    step_interval_for_eval: int
    acc_limit: float
    acc_exponent: float
    dec_limit: float
    time_headway: float
    safety_margin: float
    use_rule_based_refine: bool
    map_location: str


@dataclass
class QCMAEFeatureParams:
    """
    Parameters for QCMAE features.
    """
    dim: int
    num_past_steps: int
    num_future_steps: int
    radius: float
    map_features: List[str]
    max_elements: Dict[str, int]
    max_points: Dict[str, int]
    max_agents: Dict[str, int]
    predict_unseen_agents: bool
    vector_repr: bool
    interpolation_method: str
    agent_features: List[str]
    past_trajectory_sampling: TrajectorySampling


@dataclass
class QCMAETargetParams:
    """
    Parameters for QCMAE targets.
        agent_featrues: agents of interested types to be predicted.
        future_trajectory_sampling: Sampling parameters for future trajectory.
    """
    agent_featrues: List[str]
    num_past_steps: int
    future_trajectory_sampling: TrajectorySampling


@dataclass
class QCMAEModelParams:
    """
    Parameters for QCMAE
    """
    input_dim: int
    hidden_dim: int
    output_dim: int
    output_head: bool
    num_past_steps: int
    num_future_steps: int
    num_modes: int
    num_recurrent_steps: int
    num_freq_bands: int
    num_map_layers: int
    num_agent_layers: int
    num_dec_layers: int
    num_heads: int
    head_dim: int
    dropout: float
    pl2pl_radius: int
    time_span: int
    pl2a_radius: int
    a2a_radius: int
    num_t2m_steps: int
    pl2m_radius: int
    a2m_radius: int
    pretrained_model_dir: str


class RITPPlannerModel(TorchModuleWrapper):
    """
    Wrapper around transformer-based RITP planner.
    """
    def __init__(
        self,
        model_name: str,
        ritp_planner_feature_params: RITPPlannerFeatureParams,
        ritp_planner_model_params: RITPPlannerModelParams,
        qcmae_feature_params: QCMAEFeatureParams,
        qcmae_target_params: QCMAETargetParams,
        qcmae_model_params: QCMAEModelParams,
    ):
        """
        Initialize model.
        """
        super().__init__(
            feature_builders=[
                VectorSetMapFeatureBuilder(
                    dim=qcmae_feature_params.dim,
                    radius=qcmae_feature_params.radius,
                    map_features=qcmae_feature_params.map_features,
                    max_elements=qcmae_feature_params.max_elements,
                    max_points=qcmae_feature_params.max_points,
                    interpolation_method=qcmae_feature_params.interpolation_method,
                ),
                GenericAgentsFeatureBuilder(
                    agent_features=qcmae_feature_params.agent_features,
                    trajectory_sampling=qcmae_feature_params.past_trajectory_sampling,
                    num_future_steps=qcmae_feature_params.num_future_steps,
                    a2a_radius=qcmae_model_params.a2a_radius,
                    max_agents=qcmae_feature_params.max_agents,
                ),
            ],
            target_builders=[AgentTrajectoryTargetBuilder(agent_featrues=qcmae_target_params.agent_featrues,
                                                          num_past_steps=qcmae_target_params.num_past_steps,
                                                          future_trajectory_sampling=qcmae_target_params.future_trajectory_sampling)],
            future_trajectory_sampling=qcmae_target_params.future_trajectory_sampling,
        )
        self.model_name = model_name
        self.finetune_range = ritp_planner_model_params.finetune_range
        self.hybrid_driven = ritp_planner_model_params.hybrid_driven
        self.use_rule_based_refine = ritp_planner_model_params.use_rule_based_refine
        self.map_location = ritp_planner_model_params.map_location
        self.device = torch.device(self.map_location)
        self._scenario = None

        # -------------------------------------
        #             Actor Network
        # -------------------------------------
        self.actor = Actor(
            input_dim=qcmae_model_params.input_dim,
            hidden_dim=qcmae_model_params.hidden_dim,
            num_historical_steps=qcmae_model_params.num_past_steps + 1,
            pl2pl_radius=qcmae_model_params.pl2pl_radius,
            time_span=qcmae_model_params.time_span,
            pl2a_radius=qcmae_model_params.pl2a_radius,
            a2a_radius=qcmae_model_params.a2a_radius,
            num_freq_bands=qcmae_model_params.num_freq_bands,
            num_map_layers=qcmae_model_params.num_map_layers,
            num_agent_layers=qcmae_model_params.num_agent_layers,
            num_heads=qcmae_model_params.num_heads,
            head_dim=qcmae_model_params.head_dim,
            dropout=qcmae_model_params.dropout,
            map_features=qcmae_feature_params.map_features,
            agent_features=qcmae_feature_params.agent_features,
            output_dim=qcmae_model_params.output_dim,
            output_head=qcmae_model_params.output_head,
            num_future_steps=qcmae_model_params.num_future_steps,
            num_modes=qcmae_model_params.num_modes,
            num_recurrent_steps=qcmae_model_params.num_recurrent_steps,
            num_t2m_steps=qcmae_model_params.num_t2m_steps,
            pl2m_radius=qcmae_model_params.pl2m_radius,
            a2m_radius=qcmae_model_params.a2m_radius,
            num_decoder_layers=qcmae_model_params.num_dec_layers,
            radius=qcmae_feature_params.radius,
            max_agents=qcmae_feature_params.max_agents,
            pretrained_model_dir=qcmae_model_params.pretrained_model_dir,
            map_location=self.map_location,
            finetune_range=self.finetune_range,
        )

        # -------------------------------------
        #             Critic Network
        # -------------------------------------
        self.critic = Critic(
            input_dim=qcmae_model_params.input_dim,
            hidden_dim=qcmae_model_params.hidden_dim,
            num_historical_steps=qcmae_model_params.num_past_steps + 1,
            num_future_steps=qcmae_model_params.num_future_steps,
            pl2pl_radius=qcmae_model_params.pl2pl_radius,
            pl2a_radius=qcmae_model_params.pl2a_radius,
            a2a_radius=qcmae_model_params.a2a_radius,
            num_t2action_steps=qcmae_model_params.num_t2m_steps,
            pl2action_radius=qcmae_model_params.pl2m_radius,
            a2action_radius=qcmae_model_params.a2m_radius,
            num_freq_bands=qcmae_model_params.num_freq_bands,
            num_map_layers=qcmae_model_params.num_map_layers,
            num_agent_layers=qcmae_model_params.num_agent_layers,
            num_action_layers=qcmae_model_params.num_dec_layers,
            num_heads=qcmae_model_params.num_heads,
            head_dim=qcmae_model_params.head_dim,
            dropout=qcmae_model_params.dropout,
            time_span=qcmae_model_params.time_span,
            map_features=qcmae_feature_params.map_features,
            agent_features=qcmae_feature_params.agent_features,
            pretrained_model_dir=qcmae_model_params.pretrained_model_dir,
            map_location=self.map_location,
        )

        # -------------------------------------
        #             Post Optimizer
        # -------------------------------------
        self.post_optimizer = PostOptimizer(
            map_features=qcmae_feature_params.map_features,
            num_future_steps=qcmae_model_params.num_future_steps,
            frenet_radius=ritp_planner_model_params.frenet_radius,
            num_plans=ritp_planner_model_params.num_plans,
            num_modes_for_eval=ritp_planner_model_params.num_modes_for_eval,
            step_interval_for_eval=ritp_planner_model_params.step_interval_for_eval,
            acc_limit=ritp_planner_model_params.acc_limit,
            acc_exponent=ritp_planner_model_params.acc_exponent,
            dec_limit=ritp_planner_model_params.dec_limit,
            time_headway=ritp_planner_model_params.time_headway,
            safety_margin=ritp_planner_model_params.safety_margin,
            use_rule_based_refine=ritp_planner_model_params.use_rule_based_refine,
        )

    def forward(self,
                features: FeaturesType,
                current_input: PlannerInput,
                initialization: PlannerInitialization) -> Dict[str, Any]:
        """
        Predict rewards.
        :param features: input features containing
                        {
                            "vector_set_map": VectorSetMap,
                            "generic_agents": GenericAgents,
                        }
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: pred: prediction from planner
                        Dict[str, List[torch.Tensor]]
                        or
                        {
                            "trajectory": Trajectory,
                        }
        """
        start_time = time.perf_counter()

        pred = self.actor(features)

        end_time1 = time.perf_counter()

        if self.hybrid_driven:
            pred = self.post_optimizer.forward(features, pred, current_input, initialization, scenario=self._scenario)

        end_time2 = time.perf_counter()

        pred['inference_runtimes_detail'] = {
            'data-driven': end_time1 - start_time,
            'model-driven': end_time2 - end_time1,
            'total': end_time2 - start_time
        }

        return pred