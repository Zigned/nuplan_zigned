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
JOB_NAME = 'estimate_u_r_stats'
LOG_DIR = os.path.join(SAVE_DIR, EXPERIMENT, JOB_NAME)
REWARDFORMER_DIR = f'/home/{USER_NAME}/nuplan/exp/avrl/training_avrl_experiment/train_avrl_vector/2024.12.19.15.57.03/best_model/epoch\=5-step\=36191.ckpt'  # TODO trained RewardFormer direction
U_R_STATS_DIR = f'/home/{USER_NAME}/nuplan/exp/avrl/training_avrl_experiment/train_avrl_vector/2024.12.19.15.57.03/best_model/'  # TODO where stats to be saved

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'group={SAVE_DIR}',
    f'cache.cache_path={SAVE_DIR}/cache_vector',
    f'experiment_name={EXPERIMENT}',
    f'job_name={JOB_NAME}',
    'py_func=train',
    '+training=training_avrl_vector_model',
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_filter.limit_total_scenarios=10000',  # Choose 10000 scenarios to train with. Needs to be consistent with model_params.num_training_scenarios
    'lightning.trainer.checkpoint.monitor=metrics/val_PseudoMetric',
    'lightning.trainer.params.accelerator=ddp',
    'lightning.trainer.params.max_epochs=1',
    'lightning.trainer.params.max_time=30:00:00:00',
    'lightning.trainer.params.precision=32',
    'lightning.trainer.params.gpus="0,"',
    'data_loader.params.batch_size=1',  # has to be 1
    'data_loader.params.num_workers=4',
    'worker.threads_per_node=4',
    'optimizer.lr=3e-7',
    'optimizer.weight_decay=0.',
    # 'worker=sequential',  # disable parallelization and run everything single-threaded (e.g. for debugging). Remove this arg to enable parallelization.
    # # make sure to enable parallelization during training and simulation, because the scenarios enabling parallelization differ from those disabling parallelization
    'training_metric=[avrl_pseudo_metric]',
    f'model.model_params.estimate_u_r_stats=true',
    f'model.model_params.model_dir={REWARDFORMER_DIR}',
    f'model.model_params.u_r_stats_dir={U_R_STATS_DIR}',
    'callbacks=estimate_u_r_stats_callbacks',
    'hydra.searchpath=["pkg://nuplan.planning.script.config.common", "pkg://nuplan_zigned.config", "pkg://nuplan_zigned.config/common", "pkg://nuplan_zigned.config.training"]',
])


if __name__ == '__main__':
    main_train(cfg)
