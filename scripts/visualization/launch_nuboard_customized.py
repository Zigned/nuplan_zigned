import os
import hydra
from pathlib import Path

from nuplan_zigned.script.run_nuboard import main as main_nuboard

USER_NAME = os.getlogin()
os.chdir(f'/home/{USER_NAME}/nuplan_zigned/scripts/simulation')  # set working directory

CONFIG_PATH = '../../../nuplan-devkit/nuplan/planning/script/config/nuboard'
CONFIG_NAME = 'default_nuboard'
COMMON_DIR = 'pkg://nuplan.planning.script.config.common'
EXPERIMENT_DIR = 'pkg://nuplan.planning.script.experiments'
ZIGNED_COMMON_DIR = 'pkg://nuplan_zigned.config.common'

METHOD = 'RITP'
# SPLIT = 'val'
SPLIT = 'test'
BATCH_IDX = 0
# BATCH_IDX = 1
# BATCH_IDX = 2
OUTPUT_DIR = '../../../nuplan/exp/ritp_planner/simulating_ritp_planner_experiment/model_2024.12.19.13.22.29-100000-closed_loop_nonreactive_agents-test-0'  # TODO simulation output direction
PORT_NUMBER = 7000
simulation_file = [str(file) for file in Path(OUTPUT_DIR).iterdir() if file.is_file() and file.suffix == '.nuboard']

VISUALIZATION_OPTIONS = [
    '+ego_trajectory_plotted="simulated_trajectory"',  # ["predicted_trajectory", "simulated_trajectory"]
    '+num_modes_plotted=1',  # [1, 2, 3, 4, 5, 6]
    # '+max_num_step_trajectory=null',  # [null, 81]
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
    'scenario_builder=nuplan_mini',  # set the database (same as simulation) used to fetch data for visualization
    f'simulation_path={simulation_file}',  # nuboard file path, if left empty the user can open the file inside nuBoard
    f'hydra.searchpath=[{COMMON_DIR}, {EXPERIMENT_DIR}, {ZIGNED_COMMON_DIR}]',
    f'port_number={PORT_NUMBER}',
    *VISUALIZATION_OPTIONS,
])

if __name__ == '__main__':
    # Run nuBoard
    main_nuboard(cfg)