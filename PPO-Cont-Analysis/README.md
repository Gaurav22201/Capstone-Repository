# PPO Continuous Control Analysis

This repository contains code for training and analyzing Proximal Policy Optimization (PPO) agents on various continuous control environments.

## Features

- Training PPO agents on multiple environments
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

To train a PPO agent:

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

### PPO Specific Features

PPO (Proximal Policy Optimization) is a policy gradient method that includes several key improvements:

1. **Clipped Surrogate Objective**: Prevents too large policy updates by clipping the objective
2. **Generalized Advantage Estimation (GAE)**: Reduces variance in policy gradients
3. **Multiple Epochs**: Updates the policy multiple times using the same data
4. **Value Function Clipping**: Prevents value function from having too large updates
5. **Entropy Bonus**: Encourages exploration by adding an entropy term to the objective

These features make PPO more stable and sample efficient compared to many other policy gradient methods.

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