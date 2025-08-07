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

MODELTRAINING_TIME = '2024.12.19.13.22.29-100000'  # TODO
CHECKPOINT_PATH = f"../../../nuplan/exp/ritp_planner/training_ritp_planner_experiment/train_motionformer/2024.12.19.13.22.29/in_epoch_checkpoints/epoch\=0-step\=53166-rl_step\=100000.ckpt"  # TODO trained MotionFormer direction

# Create a directory to store the simulation artifacts
SAVE_DIR = f'/home/{USER_NAME}/nuplan/exp/ritp_planner'
EXPERIMENT = 'simulating_ritp_planner_experiment'

# Select simulation parameters
EGO_CONTROLLER = 'two_stage_controller'  # [log_play_back_controller, perfect_tracking_controller, two_stage_controller]
OBSERVATION = 'idm_agents_observation'  # [box_observation, idm_agents_observation, ego_centric_ml_agents_observation, lidar_pc_observation]
CHALLENGE = 'closed_loop_reactive_agents'  # [open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents]
METRIC = 'simulation_closed_loop_reactive_agents'  # [simulation_open_loop_boxes, simulation_closed_loop_nonreactive_agents, simulation_closed_loop_reactive_agents]
# SPLIT = "val"
SPLIT = "test"
NUM_MODES_FOR_EVAL = 2
SAMPLE_IDX = '0'
# SAMPLE_IDX = '1'
# SAMPLE_IDX = '2'
PARAMS = f'hybrid-num_plans\=1-num_modes_for_eval\={str(NUM_MODES_FOR_EVAL)}-use_rule_based_refine-{SPLIT}-{SAMPLE_IDX}'
JOB_NAME = f'model_{MODELTRAINING_TIME}-{CHALLENGE}-{PARAMS}'

DATASET_PARAMS = [
    'scenario_builder=nuplan_mini',  # use nuplan mini database (2.5h of 8 autolabeled logs in Las Vegas)
    'scenario_filter=mini_split',
    'scenario_filter.limit_total_scenarios=10000',
    '+scenario_filter_num_batches=3',  # divide scenarios into num_batches batches to save RAM.
    f'+scenario_filter_sample_idx={SAMPLE_IDX}',
    f'+val_test_scenarios={SPLIT}',
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
    'model=ritp_planner_model',
    'model.ritp_planner_model_params.use_rule_based_refine=True',
    'model.ritp_planner_model_params.hybrid_driven=True',
    'model.ritp_planner_model_params.num_plans=1',
    f'model.ritp_planner_model_params.num_modes_for_eval={NUM_MODES_FOR_EVAL}',
    'planner=ritp_planner',
    f'planner.ritp_planner.checkpoint_path={CHECKPOINT_PATH}',
    f'+simulation={CHALLENGE}',
    f'simulation_metric={METRIC}',
    'simulation_history_buffer_duration=5.0',  # this seems to include current state
    'worker.threads_per_node=40',
    # 'worker=sequential',
    # make sure to enable parallelization during training and simulation, because the scenarios enabling parallelization differ from those disabling parallelization
    f'ego_controller={EGO_CONTROLLER}',
    f'observation={OBSERVATION}',
    f'hydra.searchpath=[{COMMON_DIR}, {EXPERIMENT_DIR}, {ZIGNED_COMMON_DIR}, {ZIGNED_SIMULATION_DIR}]',
    'output_dir=${group}/${experiment}',
    *DATASET_PARAMS,
])


if __name__ == '__main__':
    # Run the simulation loop (real-time visualization not yet supported)
    main_simulation(cfg)