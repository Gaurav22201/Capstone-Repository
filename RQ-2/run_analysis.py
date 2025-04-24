import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psutil
import time
from collections import defaultdict
from scipy import stats

# Import the algorithms
from PPO_Cont_Analysis import PPO
from DDPG_analysis import DDPG
from TD3_Analysis import TD3
from SAC_Cont_Analysis import SAC
from PPO_Cont_Analysis.PPO import PPO_agent
from DDPG_analysis.DDPG import DDPG_agent
from TD3_Analysis.TD3 import TD3_agent
from SAC_Cont_Analysis.SAC import SAC_agent

# Constants
ENVS = ['Pendulum-v1', 'BipedalWalker-v3']
ALGORITHMS = ['PPO', 'DDPG', 'TD3', 'SAC']
MAX_STEPS = 20_000  # Reduced for faster experiments
EVAL_INTERVAL = 50  # More frequent evaluation for shorter runs
INCREMENTAL_INTERVAL = 500  # Reduced for shorter runs
LOG_DIR = 'logs'
PLOT_DIR = 'plots'

# Performance thresholds for each environment
PERFORMANCE_THRESHOLDS = {
    'Pendulum-v1': -800,  # Adjusted for shorter training
    'BipedalWalker-v3': 15  # Adjusted for shorter training
}

def get_system_metrics():
    """Get current system metrics."""
    process = psutil.Process(os.getpid())
    return {
        'memory_usage': process.memory_info().rss / 1024 / 1024 / 1024,  # GB
        'cpu_usage': process.cpu_percent()
    }

def create_agent(algo_name, env, device='cpu'):
    """Create an agent based on algorithm name."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    common_params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'net_width': 256,
        'dvc': device,
        'gamma': 0.99,
        'batch_size': 256
    }
    
    if algo_name == 'PPO':
        return PPO_agent(a_lr=3e-4, c_lr=3e-4, **common_params)
    elif algo_name == 'DDPG':
        return DDPG_agent(a_lr=1e-4, c_lr=1e-3, noise=0.1, **common_params)
    elif algo_name == 'TD3':
        return TD3_agent(a_lr=1e-4, c_lr=1e-3, noise=0.1, 
                        policy_noise=0.2, noise_clip=0.5, policy_freq=2, **common_params)
    elif algo_name == 'SAC':
        return SAC_agent(a_lr=1e-4, c_lr=1e-3, **common_params)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def evaluate_policy(env, agent, eval_episodes=10):
    """Evaluate the policy without exploration noise."""
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            avg_reward += reward
    return avg_reward / eval_episodes

def calculate_learning_slope(rewards, start_step=100_000, end_step=200_000):
    """Calculate the learning curve slope between specified timesteps."""
    if len(rewards) == 0:
        return 0.0
        
    # Convert to numpy arrays for calculation
    rewards = np.array(rewards)
    steps = np.arange(len(rewards)) * EVAL_INTERVAL  # Convert indices to actual steps
    
    # Get indices for the range we want
    start_idx = start_step // EVAL_INTERVAL
    end_idx = min(end_step // EVAL_INTERVAL, len(rewards))
    
    # Get rewards between start_step and end_step
    if start_idx >= end_idx or start_idx >= len(rewards):
        return 0.0
        
    rewards_slice = rewards[start_idx:end_idx]
    steps_slice = steps[start_idx:end_idx]
    
    # Calculate slope using linear regression
    if len(steps_slice) > 1 and len(rewards_slice) > 1:
        slope, _, _, _, _ = stats.linregress(steps_slice, rewards_slice)
        return slope
    return 0.0

def run_experiment(algo_name, env_name):
    """Run a single experiment with given algorithm and environment."""
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create environment
    env = gym.make(env_name)
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = create_agent(algo_name, env, device)
    
    # Initialize metrics
    metrics = {
        'step': [],
        'episode_reward': [],
        'algorithm': [],
        'environment': [],
        'incremental_reward': None,  # Will be converted to list later
        'threshold_reached': None
    }
    
    total_steps = 0
    episode = 0
    last_incremental_reward = None
    incremental_rewards = []
    performance_threshold = PERFORMANCE_THRESHOLDS[env_name]
    
    while total_steps < MAX_STEPS:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done and total_steps < MAX_STEPS:
            # Select and take action
            if algo_name == 'PPO':
                action_tuple = agent.select_action(state, deterministic=False)
                action = action_tuple[0]
            else:
                action = agent.select_action(state, deterministic=False)
            
            # Ensure action is a numpy array
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.float32)
            elif isinstance(action, torch.Tensor):
                action = action.cpu().numpy().astype(np.float32)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition and train
            if algo_name == 'PPO':
                agent.buffer.add(state, action, reward, action_tuple[1], action_tuple[2], terminated)
                if agent.buffer.count == agent.buffer.size:
                    agent.train()
            else:
                agent.replay_buffer.add(state, action, reward, next_state, terminated)
                if total_steps >= 10000:
                    agent.train()
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Evaluate at regular intervals
            if total_steps % EVAL_INTERVAL == 0:
                eval_reward = evaluate_policy(env, agent)
                metrics['step'].append(total_steps)
                metrics['episode_reward'].append(eval_reward)
                metrics['algorithm'].append(algo_name)
                metrics['environment'].append(env_name)
                
                # Check if performance threshold is reached
                if metrics['threshold_reached'] is None and eval_reward >= performance_threshold:
                    metrics['threshold_reached'] = total_steps
            
            # Record incremental reward
            if total_steps % INCREMENTAL_INTERVAL == 0:
                eval_reward = evaluate_policy(env, agent)
                if last_incremental_reward is not None:
                    incremental_rewards.append(eval_reward - last_incremental_reward)
                last_incremental_reward = eval_reward
        
        episode += 1
    
    # Calculate learning curve slope
    slope = calculate_learning_slope(
        metrics['episode_reward'][100_000//EVAL_INTERVAL:200_000//EVAL_INTERVAL]
    )
    
    # Add incremental rewards to metrics
    metrics['incremental_reward'] = incremental_rewards
    
    # Create DataFrame only for step-based metrics
    # Ensure all arrays have the same length by padding incremental rewards if needed
    n_steps = len(metrics['step'])
    if len(incremental_rewards) < n_steps:
        # Pad with None or the last value to match length
        padding = [None] * (n_steps - len(incremental_rewards))
        incremental_rewards.extend(padding)
    elif len(incremental_rewards) > n_steps:
        # Truncate to match length
        incremental_rewards = incremental_rewards[:n_steps]
    
    df = pd.DataFrame({
        'step': metrics['step'],
        'episode_reward': metrics['episode_reward'],
        'algorithm': metrics['algorithm'],
        'environment': metrics['environment'],
        'incremental_reward': incremental_rewards,
        'threshold_reached': [metrics['threshold_reached']] * len(metrics['step'])  # Add threshold_reached column
    })
    
    # Save metrics to CSV
    filename = f"{LOG_DIR}/{algo_name}_{env_name}_rq2.csv"
    df.to_csv(filename, index=False)
    
    return {
        'threshold_steps': metrics['threshold_reached'],
        'learning_slope': slope,
        'incremental_rewards': incremental_rewards
    }

def analyze_results():
    """Analyze and plot results focusing on early sample efficiency metrics."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Load all CSV files
    all_data = []
    for filename in os.listdir(LOG_DIR):
        if filename.endswith('_rq2.csv'):
            df = pd.read_csv(os.path.join(LOG_DIR, filename))
            all_data.append(df)
    
    if not all_data:
        print("No data files found to analyze!")
        return
        
    data = pd.concat(all_data, ignore_index=True)
    
    # 1. Incremental Reward Analysis
    for env_name in ENVS:
        plt.figure(figsize=(12, 6))
        for algo_name in ALGORITHMS:
            algo_data = data[(data['environment'] == env_name) & 
                           (data['algorithm'] == algo_name)]
            
            if not algo_data.empty:
                incremental_rewards = algo_data.groupby('step')['incremental_reward'].mean()
                plt.plot(incremental_rewards.index, incremental_rewards.values,
                        label=f"{algo_name}")
        
        plt.xlabel('Timesteps')
        plt.ylabel('Incremental Reward (per 10k steps)')
        plt.title(f'Incremental Reward Analysis - {env_name}')
        plt.legend()
        plt.savefig(f'{PLOT_DIR}/incremental_rewards_{env_name}.png')
        plt.close()
    
    # 2. Learning Curve Analysis
    for env_name in ENVS:
        plt.figure(figsize=(12, 6))
        for algo_name in ALGORITHMS:
            algo_data = data[(data['environment'] == env_name) & 
                           (data['algorithm'] == algo_name)]
            
            if not algo_data.empty:
                rewards = algo_data.groupby('step')['episode_reward'].mean()
                plt.plot(rewards.index, rewards.values, label=f"{algo_name}")
        
        plt.xlabel('Timesteps')
        plt.ylabel('Average Episode Reward')
        plt.title(f'Learning Curve Analysis - {env_name}')
        plt.legend()
        plt.savefig(f'{PLOT_DIR}/learning_curves_{env_name}.png')
        plt.close()
    
    # 3. Learning Curve Slope Analysis
    plt.figure(figsize=(10, 6))
    slope_data = []
    for env_name in ENVS:
        for algo_name in ALGORITHMS:
            algo_data = data[(data['environment'] == env_name) & 
                           (data['algorithm'] == algo_name)]
            if not algo_data.empty:
                # Calculate slope using the episode rewards
                rewards = algo_data['episode_reward'].values
                slope = calculate_learning_slope(rewards)
                print(f"Calculated slope for {algo_name} on {env_name}: {slope}")  # Debug print
                slope_data.append({
                    'Environment': env_name,
                    'Algorithm': algo_name,
                    'Slope': slope
                })
    
    slope_df = pd.DataFrame(slope_data)
    if not slope_df.empty:  # Only plot if we have data
        plt.figure(figsize=(12, 6))
        sns.barplot(data=slope_df, x='Algorithm', y='Slope', hue='Environment')
        plt.title('Learning Curve Slope Analysis (100k-200k timesteps)')
        plt.xticks(rotation=45)
        plt.ylabel('Learning Rate (reward/timestep)')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/learning_slopes.png')
        print("Slope data:")
        print(slope_df)
    else:
        print("Warning: No slope data available for plotting")
    plt.close()
    
    # 4. Time to Threshold Analysis
    plt.figure(figsize=(10, 6))
    threshold_data = []
    for env_name in ENVS:
        for algo_name in ALGORITHMS:
            algo_data = data[(data['environment'] == env_name) & 
                           (data['algorithm'] == algo_name)]
            if not algo_data.empty:
                # Get threshold value, default to MAX_STEPS if threshold wasn't reached
                threshold_steps = algo_data['threshold_reached'].iloc[0]
                if pd.isna(threshold_steps):  # If threshold wasn't reached
                    threshold_steps = MAX_STEPS
                threshold_data.append({
                    'Environment': env_name,
                    'Algorithm': algo_name,
                    'Steps': threshold_steps
                })
    
    threshold_df = pd.DataFrame(threshold_data)
    sns.barplot(data=threshold_df, x='Algorithm', y='Steps', hue='Environment')
    plt.title('Timesteps to Reach Performance Threshold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/threshold_analysis.png')
    plt.close()
    
    print("Analysis completed! Check the plots directory for results.")

def main():
    """Main function to run all experiments and analysis."""
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Track metrics across all experiments
    results = defaultdict(dict)
    
    # Run all experiments
    for env_name in ENVS:
        for algo_name in ALGORITHMS:
            print(f"Running {algo_name} on {env_name}")
            exp_results = run_experiment(algo_name, env_name)
            
            # Store results
            results[env_name][algo_name] = {
                'threshold_steps': exp_results['threshold_steps'],
                'learning_slope': exp_results['learning_slope'],
                'incremental_rewards': exp_results['incremental_rewards']
            }
    
    # Save aggregate results
    aggregate_results = {
        'Environment': [],
        'Algorithm': [],
        'Threshold_Steps': [],
        'Learning_Slope': []
    }
    
    for env_name in results:
        for algo_name in results[env_name]:
            aggregate_results['Environment'].append(env_name)
            aggregate_results['Algorithm'].append(algo_name)
            aggregate_results['Threshold_Steps'].append(results[env_name][algo_name]['threshold_steps'])
            aggregate_results['Learning_Slope'].append(results[env_name][algo_name]['learning_slope'])
    
    # Save aggregate results to CSV
    pd.DataFrame(aggregate_results).to_csv(f'{LOG_DIR}/aggregate_results_rq2.csv', index=False)
    
    # Run analysis
    analyze_results()
    
    print("RQ-2 experiments and analysis completed!")

if __name__ == "__main__":
    main() 