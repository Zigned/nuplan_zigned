import os
import hydra
from nuplan_zigned.script.qcmae_run_training import main as main_train

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
SAVE_DIR = f'/home/{USER_NAME}/nuplan/exp/qcmae'  # optionally replace with persistent dir
EXPERIMENT = 'training_qcmae_experiment'
STAGE = 'cache'  # TODO uncomment in feature caching stage
# STAGE = 'pretrain'  # TODO uncomment in pretraining stage
# STAGE = 'finetune'  # TODO uncomment in finetuning stage
if STAGE == 'cache':
    PYFUNC = 'cache'
    JOB_NAME = 'cache_vector'
    NUM_WORKERS = '4'
    THREADS_PER_NODE = '4'
    MONITOR = 'metrics/val_minFDE'
    MAX_EPOCH = '30'
    BS = '4'
    LR = '5e-4'
    PRETRAIN = 'False'
    SELF_DISTILLATION = 'False'
    OTHER_PARAMS = []
elif STAGE == 'pretrain':
    PYFUNC = 'train'
    JOB_NAME = 'pretrain_qcmae'
    NUM_WORKERS = '1'
    THREADS_PER_NODE = '1'
    MONITOR = 'metrics/val_PseudoMetric'
    MAX_EPOCH = '30'
    BS = '1'
    LR = '2.5e-4'
    PRETRAIN = 'True'
    SELF_DISTILLATION = 'False'
    OTHER_PARAMS = [
        'training_metric=[qcmae_pseudo_metric]',
    ]
else:
    PYFUNC = 'train'
    JOB_NAME = 'finetune_qcmae_with_sd'
    NUM_WORKERS = '1'
    THREADS_PER_NODE = '1'
    MONITOR = 'metrics/val_minFDE'
    MAX_EPOCH = '60'
    BS = '1'
    LR = '2.5e-4'
    PRETRAIN = 'False'
    SELF_DISTILLATION = 'True'
    OTHER_PARAMS = [
        f'model.model_params.pretrained_model_dir="/home/{USER_NAME}/nuplan/exp/qcmae/training_qcmae_experiment/pretrain_qcmae/2024.04.25.10.22.46/checkpoints/epoch=29.ckpt"',  # TODO pretrained QCMAE model direction
    ]
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
    '+training=training_qcmae',
    'scenario_builder=nuplan_mini',  # use nuplan mini database
    'scenario_filter.limit_total_scenarios=10000',  # Choose 10000 scenarios to train with
    f'lightning.trainer.checkpoint.monitor={MONITOR}',
    'lightning.trainer.params.accelerator=ddp',
    f'lightning.trainer.params.max_epochs={MAX_EPOCH}',
    'lightning.trainer.params.max_time=30:00:00:00',
    'lightning.trainer.params.precision=32',
    'lightning.trainer.params.gpus="0,"',
    # 'lightning.trainer.params.gpus="1,"',
    # 'lightning.trainer.params.gpus="2,"',
    # 'lightning.trainer.params.gpus="3,"',
    f'data_loader.params.batch_size={BS}',
    f'data_loader.params.num_workers=4',
    'worker.threads_per_node=4',
    # 'worker=sequential',  # disable parallelization and run everything single-threaded (e.g. for debugging). Remove this arg to enable parallelization.
    # # make sure to enable parallelization during training and simulation, because the scenarios enabling parallelization differ from those disabling parallelization
    'optimizer=adamw',
    f'optimizer.lr={LR}',
    'optimizer.weight_decay=1e-4',
    'lr_scheduler=one_cycle_lr',
    f'model.model_params.self_distillation={SELF_DISTILLATION}',
    f'model.model_params.pretrain={PRETRAIN}',
    'hydra.searchpath=["pkg://nuplan.planning.script.config.common", "pkg://nuplan_zigned.config", "pkg://nuplan_zigned.config/common", "pkg://nuplan_zigned.config.training"]',
    *OTHER_PARAMS,
])


if __name__ == '__main__':
    main_train(cfg)
