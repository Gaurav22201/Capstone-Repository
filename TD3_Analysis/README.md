# TD3 Analysis

This repository contains code for training and analyzing Twin Delayed Deep Deterministic Policy Gradient (TD3) agents on various continuous control environments.

## Features

- Training TD3 agents on multiple environments
- TensorBoard logging for training metrics
- Video recording of best performing episodes
- Learning curve visualization
- Performance metrics tracking and saving
- Model checkpointing

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a TD3 agent:

```bash
python main.py --EnvIdex 0  # For Pendulum-v1
```

Available environments (EnvIdex):
- 0: Pendulum-v1
- 1: LunarLanderContinuous-v2
- 2: Humanoid-v4
- 3: HalfCheetah-v4
- 4: BipedalWalker-v3
- 5: BipedalWalkerHardcore-v3

### Command Line Arguments

- `--dvc`: Running device ('cuda' or 'cpu')
- `--EnvIdex`: Environment index (0-5)
- `--write`: Enable TensorBoard logging (default: True)
- `--render`: Enable environment rendering (default: False)
- `--Loadmodel`: Load pretrained model (default: False)
- `--ModelIdex`: Model checkpoint to load
- `--record_video`: Record evaluation videos (default: True)
- `--video_folder`: Folder to save evaluation videos
- `--seed`: Random seed
- `--Max_train_steps`: Maximum training steps
- `--save_interval`: Model saving interval
- `--eval_interval`: Evaluation interval
- `--gamma`: Discount factor
- `--net_width`: Network width
- `--a_lr`: Actor learning rate
- `--c_lr`: Critic learning rate
- `--batch_size`: Training batch size
- `--random_steps`: Random exploration steps
- `--noise`: Exploration noise
- `--policy_noise`: Noise added to target policy during critic update
- `--noise_clip`: Range to clip target policy noise
- `--policy_freq`: Frequency of delayed policy updates

### TD3 Specific Parameters

TD3 introduces several improvements over DDPG:

1. **Twin Critics**: Uses two Q-networks to reduce overestimation bias in the value function
2. **Delayed Policy Updates**: Updates the policy less frequently than the critics
3. **Target Policy Smoothing**: Adds noise to the target action to make it harder for the policy to exploit Q-function errors
4. **Clipped Double Q-learning**: Uses the minimum of the two Q-values to form the targets in the Bellman error loss function

These improvements help address common issues in DDPG like overestimation bias and sensitivity to hyperparameters.

### Results

The training process generates:
1. TensorBoard logs in the `runs/` directory
2. Model checkpoints in the `model/` directory
3. Evaluation videos in the `videos/` directory
4. Learning curves and metrics in the `results/` directory

### Visualization

To view training progress:
```bash
tensorboard --logdir runs
```

## Analysis

The code automatically generates:
1. Learning curves showing episode rewards over time
2. Performance metrics saved as JSON files
3. Videos of the best performing episodes
4. TensorBoard logs with various training metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details. 