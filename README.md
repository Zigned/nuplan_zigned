# nuplan_zigned
Author's PyTorch implementation of [Reinforced Imitative Trajectory Planning for Urban Automated Driving](http://arxiv.org/abs/2410.15607).

## Getting Started

### 1. Installation

- Setup the nuPlan devkit and dataset following the [nuPlan user guide](https://github.com/motional/nuplan-devkit?tab=readme-ov-file).

- If you encounter package incompatibility issues when installing the nuPlan devkit, replace the `~/nuplan-devkit/requirements.txt` and `~/nuplan-devkit/requirements_torch.txt` files with the `~/nuplan_zigned/nuplan_requirements/requirements.txt` and `~/nuplan_zigned/nuplan_requirements/requirements_torch.txt` files after cloning this repository.

- Clone this repository:

```bash
git clone https://github.com/Zigned/nuplan_zigned.git && cd nuplan_zigned
```



- Make sure the environment you created when installing the nuPlan devkit is activated:

```bash
conda activate nuplan
```

- Install dependencies for nuplan_zigned:

```bash
pip install -r requirements.txt
```

```bash
pip install -r requirements_torch.txt
```



- Install the local nuplan_zigned as a pip package:

```bash
pip install -e .
```



### 2. Training

#### 2.1 Training Bayesian RewardFormer

- Set `PYFUNC` to `'cache'` or `'train'` in `train_avrl_vector_rewardformer.py`, then run:

  ```bash
  python ~/nuplan_zigned/scripts/training/train_avrl_vector_rewardformer.py
  ```

  to cache features or train Bayesian RewardFormer, respectively.

- Run:

  ```bash
  python ~/nuplan_zigned/scripts/training/estimate_avrl_vector_rewardformer.py
  ```

  to estimate the mean and variance of the uncertainty of Bayesian RewardFormer following the AVRL method.

#### 2.2 Training QCMAE

- Set `PYFUNC` to `'cache'`, `'pretrain'`, or `'finetune'` in `train_qcmae.py`, then run:

  ```bash
  python ~/nuplan_zigned/scripts/training/train_qcmae.py
  ```

  to cache features, pretrain QCMAE, or finetune QCMAE, respectively.

#### 2.3 Training MotionFormer

- Set `PYFUNC` to `'cache'` or `'train'` in `train_ritp_planner.py`, then run:

  ```bash
  python ~/nuplan_zigned/scripts/training/train_ritp_planner.py
  ```

  to cache features or train MotionFormer, respectively.

### 3. Evaluation

- Run:

  ```bash
  python ~/nuplan_zigned/scripts/simulation/simulate_ritp_planner.py
  ```

  to conduct closed-loop simulation.

### 4. Visualization

- Run:

  ```bash
  python ~/nuplan_zigned/scripts/visualization/launch_nuboard_customized.py
  ```

  to launch our customized nuboard, where the simulated and ground truth trajectories of the ego vehicle are plotted.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{zeng2025reinforcedimitativetrajectoryplanning,
  title        = {Reinforced Imitative Trajectory Planning for Urban Automated Driving}, 
  author       = {Di Zeng and Ling Zheng and Xiantong Yang and Yinong Li},
  year         = {2025},
  eprint       = {2410.15607},
  archivePrefix= {arXiv},
  primaryClass = {cs.RO},
  url          = {https://arxiv.org/abs/2410.15607},
  note         = {Accepted by Automotive Innovation}
}
```

