import os
import hydra

from nuplan_zigned.script.run_simulation import run_simulation as main_simulation

USER_NAME = os.getlogin()
os.chdir(f'/home/{USER_NAME}/nuplan_zigned/scripts/simulation')  # set working directory
CONFIG_PATH = '../../../nuplan-devkit/nuplan/planning/script/config/simulation'
CONFIG_NAME = 'default_simulation'
COMMON_DIR = 'pkg://nuplan.planning.script.config.common'
EXPERIMENT_DIR = 'pkg://nuplan.planning.script.experiments'
ZIGNED_COMMON_DIR = 'pkg://nuplan_zigned.config.common'
ZIGNED_SIMULATION_DIR = 'pkg://nuplan_zigned.config.simulation'
# MODELTRAINING_TIME = '2024.02.23.13.06.53'
MODELTRAINING_TIME = '2024.04.26.10.35.16'
# CHECKPOINT_PATH = f"../../../nuplan/exp/qcmae/training_qcmae_experiment/finetune_qcmae/2024.02.23.13.06.53/best_model/epoch\=40-step\=30913.ckpt"  # train with pretrained map encoder prob_pretrain_mask=0.15
CHECKPOINT_PATH = f"../../../nuplan/exp/qcmae/training_qcmae_experiment/finetune_qcmae_with_sd/2024.04.26.10.35.16/best_model/epoch\=58-step\=88971.ckpt"  # train with pretrained map encoder prob_pretrain_mask=0.15 and with self-distillation sd_temperature=0.5, sd_alpha=1.5

# Create a directory to store the simulation artifacts
SAVE_DIR = f'/home/{USER_NAME}/nuplan/exp/qcmae'
EXPERIMENT = 'simulating_qcmae_experiment'
# EXPERIMENT = 'simulating_qcmae_experiment_debug'

# Select simulation parameters
EGO_CONTROLLER = 'two_stage_controller'  # [log_play_back_controller, perfect_tracking_controller, two_stage_controller]
OBSERVATION = 'box_observation'  # [box_observation, idm_agents_observation, ego_centric_ml_agents_observation, lidar_pc_observation]
CHALLENGE = 'closed_loop_nonreactive_agents'  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
METRIC = 'simulation_closed_loop_nonreactive_agents'  # [simulation_open_loop_boxes, simulation_closed_loop_nonreactive_agents, simulation_closed_loop_reactive_agents]
PARAMS = 'all-0'
JOB_NAME = f'model_{MODELTRAINING_TIME}-{CHALLENGE}-{PARAMS}'
DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',  # use nuplan mini database (2.5h of 8 autolabeled logs in Las Vegas)
    'scenario_filter=mini_split',
    # "scenario_filter.log_names=['2021.10.06.07.26.10_veh-52_00006_00398']",
    # "scenario_filter.log_names=['2021.08.30.14.54.34_veh-40_00439_00835']",
    # "scenario_filter.scenario_tokens=['2bebc9bdf0b35c93']",
    # "scenario_filter.scenario_tokens=['010ab1328f5b5847']",
    'scenario_filter.limit_total_scenarios=10000',
    # 'scenario_filter.timestamp_threshold_s=4',
    '+scenario_filter_num_batches=3',  # divide scenarios into num_batches batches to save RAM.
    '+scenario_filter_sample_idx=0',
    '+val_test_scenarios="val"',
    # '+val_test_scenarios="test"',
]

os.environ['NUPLAN_DATA_ROOT'] = f'/home/{USER_NAME}/nuplan/dataset'
os.environ['NUPLAN_MAPS_ROOT'] = f'/home/{USER_NAME}/nuplan/dataset/maps'
os.environ['NUPLAN_DB_FILES'] = f'/home/{USER_NAME}/nuplan/dataset/nuplan-v1.1/splits/mini'
os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'group={SAVE_DIR}',
    f'experiment_name={EXPERIMENT}',
    f'job_name={JOB_NAME}',
    'experiment=${experiment_name}/${job_name}',
    'model=qcmae',
    # 'model.model_params.gpus="0,"',
    'model.model_params.gpus=0',  # cpu
    'planner=qcmae_planner',
    f'planner.qcmae_planner.checkpoint_path={CHECKPOINT_PATH}',
    f'+simulation={CHALLENGE}',
    f'simulation_metric={METRIC}',
    'simulation_history_buffer_duration=5.0',  # this seems to include current state
    'worker.threads_per_node=14',
    # 'worker=sequential',
    # # make sure to enable parallelization during training and simulation, because the scenarios enabling parallelization differ from those disabling parallelization
    f'ego_controller={EGO_CONTROLLER}',
    f'observation={OBSERVATION}',
    f'hydra.searchpath=[{COMMON_DIR}, {EXPERIMENT_DIR}, {ZIGNED_COMMON_DIR}, {ZIGNED_SIMULATION_DIR}]',
    'output_dir=${group}/${experiment}',
    *DATASET_PARAMS,
])


if __name__ == '__main__':
    # Run the simulation loop (real-time visualization not yet supported)
    main_simulation(cfg)