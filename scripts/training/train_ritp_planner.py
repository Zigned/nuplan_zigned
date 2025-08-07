import os
import hydra

from nuplan_zigned.script.ritp_run_training import main as main_train

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
SAVE_DIR = f'/home/{USER_NAME}/nuplan/exp/ritp_planner'  # optionally replace with persistent dir
EXPERIMENT = 'training_ritp_planner_experiment'
PYFUNC = 'cache'  # TODO uncomment in feature caching stage
# PYFUNC = 'train'  # TODO uncomment in training stage
if PYFUNC == 'cache':
    JOB_NAME = 'cache_vector'
else:
    JOB_NAME = 'train_motionformer'
LOG_DIR = os.path.join(SAVE_DIR, EXPERIMENT, JOB_NAME)

# model dir
QCMAE_DIR = f'/home/{USER_NAME}/nuplan/exp/qcmae/training_qcmae_experiment/finetune_qcmae_with_sd/2024.04.26.10.35.16/best_model/epoch\=58-step\=88971.ckpt'  # TODO finetuned QCMAE model direction
REWARDFORMER_DIR = f'/home/{USER_NAME}/nuplan/exp/avrl/training_avrl_experiment/train_avrl_vector/2024.12.18.12.08.45/best_model/epoch\=8-step\=54287.ckpt'  # TODO trained RewardFormer
U_R_STATS_DIR = f'/home/{USER_NAME}/nuplan/exp/avrl/training_avrl_experiment/train_avrl_vector/2024.12.18.12.08.45/best_model/u_r_stats.npy'  # TODO stats of the trained RewardFormer

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
    '+training=training_ritp_planner',
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_filter=mini_split',
    'scenario_filter.limit_total_scenarios=10000',  # Choose 10000 scenarios to train with
    'lightning.trainer.checkpoint.monitor=metrics/val_minFDE',
    'lightning.trainer.params.accelerator=ddp',
    'lightning.trainer.params.max_epochs=60',
    'lightning.trainer.params.max_time=30:00:00:00',
    'lightning.trainer.params.precision=32',
    'lightning.trainer.params.gpus="0,"',
    # 'lightning.trainer.params.gpus="1,"',
    # 'lightning.trainer.params.gpus="2,"',
    # 'lightning.trainer.params.gpus="3,"',
    'callbacks=ritp_callbacks',
    'model.ritp_planner_model_params.num_rl_workers=${data_loader.params.batch_size}',  # need to be consistent with batch size (one sample for each worker)
    'model.ritp_planner_model_params.start_timesteps=2000',  # start_timesteps in TD3
    'model.ritp_planner_model_params.batch_size=1',  # batch size in TD3
    'model.rewardformer_model_params.lambda_u_r=1.5',  # uncertainty-penalization strength
    'data_loader.params.batch_size=4',
    'data_loader.params.num_workers=0',  # setting num_workers > 0 will cause "RuntimeError: unable to open shared memory object"
    'worker.threads_per_node=1',
    # 'worker=sequential',  # disable parallelization and run everything single-threaded (e.g. for debugging). Remove this arg to enable parallelization.
    # make sure to enable parallelization during training and simulation, because the scenarios enabling parallelization differ from those disabling parallelization
    'optimizer=adamw',
    'optimizer.lr=5e-5',  # for batch_size 1. lr_scheduler.max_lr is set to optimizer.lr by default.
    'optimizer.weight_decay=1e-4',
    f'model.qcmae_model_params.pretrained_model_dir={QCMAE_DIR}',
    f'model.rewardformer_model_params.model_dir={REWARDFORMER_DIR}',
    f'model.rewardformer_model_params.u_r_stats_dir={U_R_STATS_DIR}',
    'hydra.searchpath=["pkg://nuplan.planning.script.config.common", "pkg://nuplan.planning.script.config.simulation", "pkg://nuplan_zigned.config", "pkg://nuplan_zigned.config/common", "pkg://nuplan_zigned.config.training"]',
])


if __name__ == '__main__':
    main_train(cfg)
