import os
import hydra
from nuplan_zigned.script.avrl_run_training import main as main_train

USER_NAME = os.getlogin()
os.chdir(f'/home/{USER_NAME}/nuplan_zigned/scripts/training')  # set working directory
os.environ['NUPLAN_DATA_ROOT'] = f'/home/{USER_NAME}/nuplan/dataset'
os.environ['NUPLAN_MAPS_ROOT'] = f'/home/{USER_NAME}/nuplan/dataset/maps'
os.environ['NUPLAN_DB_FILES'] = f'/home/{USER_NAME}/nuplan/dataset/nuplan-v1.1/splits/mini'
os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'

# If set, use the env. variable to overwrite the Hydra config
CONFIG_PATH = '../../../nuplan-devkit/nuplan/planning/script/config/training'
CONFIG_NAME = 'default_training'

# Create a directory to store the cache and experiment artifacts
SAVE_DIR = f'/home/{USER_NAME}/nuplan/exp/avrl'  # optionally replace with persistent dir
EXPERIMENT = 'training_avrl_experiment'
PYFUNC = 'cache'  # TODO uncomment in feature caching stage
# PYFUNC = 'train'  # TODO uncomment in training stage
if PYFUNC == 'cache':
    JOB_NAME = 'cache_avrl_vector'
    NUM_WORKERS = '20'
    THREADS_PER_NODE = '20'
else:
    JOB_NAME = 'train_avrl_vector'
    NUM_WORKERS = '1'
    THREADS_PER_NODE = '1'
LOG_DIR = os.path.join(SAVE_DIR, EXPERIMENT, JOB_NAME)

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'group={SAVE_DIR}',
    f'cache.cache_path={SAVE_DIR}/cache_vector',
    f'experiment_name={EXPERIMENT}',
    f'job_name={JOB_NAME}',
    f'py_func={PYFUNC}',
    '+training=training_avrl_vector_model',
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_filter.limit_total_scenarios=10000',  # Choose 10000 scenarios to train with. Needs to be consistent with model_params.num_training_scenarios
    'lightning.trainer.checkpoint.monitor=metrics/val_avg_displacement_error',
    'lightning.trainer.params.accelerator=ddp',
    'lightning.trainer.params.max_epochs=15',
    'lightning.trainer.params.max_time=30:00:00:00',
    'lightning.trainer.params.precision=32',
    'lightning.trainer.params.gpus="0,"',
    # 'lightning.trainer.params.gpus="1,"',
    # 'lightning.trainer.params.gpus="2,"',
    # 'lightning.trainer.params.gpus="3,"',
    'data_loader.params.batch_size=1',
    f'data_loader.params.num_workers={NUM_WORKERS}',
    f'worker.threads_per_node={THREADS_PER_NODE}',
    'optimizer.lr=3e-7',
    'optimizer.weight_decay=0.',
    # 'worker=sequential',  # disable parallelization and run everything single-threaded (e.g. for debugging). Remove this arg to enable parallelization.
    #     # make sure to enable parallelization during training and simulation, because the scenarios enabling parallelization differ from those disabling parallelization
    'hydra.searchpath=["pkg://nuplan.planning.script.config.common", "pkg://nuplan_zigned.config", "pkg://nuplan_zigned.config/common", "pkg://nuplan_zigned.config.training"]',
])


if __name__ == '__main__':
    main_train(cfg)