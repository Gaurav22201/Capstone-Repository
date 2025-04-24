# RQ-1: Reinforcement Learning Algorithm Comparison

This directory contains scripts to analyze and compare different reinforcement learning algorithms (PPO, DDPG, TD3, SAC) on continuous control tasks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the parent directory containing the algorithm implementations is in your Python path.

## Running the Analysis

Run the analysis script:
```bash
python run_analysis.py
```

This will:
1. Run each algorithm (PPO, DDPG, TD3, SAC) on both environments (LunarLanderContinuous-v2, HalfCheetah-v2)
2. Use 10 different random seeds for each combination
3. Train for 1M timesteps per run
4. Collect metrics including rewards, system usage, and crash events

## Output

The script generates the following outputs in the `logs` and `plots` directories:

### Logs
- Individual CSV files for each run with columns:
  - step
  - episode_reward
  - seed
  - algorithm
  - environment
  - time_elapsed
  - memory_usage
  - cpu_usage
- A summary CSV of crash events across all runs

### Plots
1. Learning Curves:
   - Mean reward over time with ±1 standard deviation bands
   - One plot per environment

2. Final Performance Box Plots:
   - Distribution of final rewards across algorithms
   - Side-by-side comparison for both environments

3. Resource Usage vs Performance:
   - Memory usage vs reward
   - CPU usage vs reward
   - Training time vs reward
   - One plot per environment

## Metrics Tracked

1. Performance:
   - Final cumulative reward
   - Reward standard deviation
   - Reward variance over time

2. Stability:
   - Number of crash events (defined as 50% drop in reward)
   - NaN outputs

3. Resource Usage:
   - Memory usage (GB)
   - CPU usage (%)
   - Training time

## Notes

- Crash events are detected when there's a 50% drop in reward over a 10-episode window
- Each algorithm uses its standard hyperparameters from the original implementations
- System metrics are collected every 1000 timesteps
- The analysis automatically uses GPU if available, otherwise falls back to CPU 