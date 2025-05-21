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
SEEDS = list(range(10))  # Just 2 seeds for testing
ENVS = ['Pendulum-v1', 'BipedalWalker-v3']  # Using BipedalWalker instead of HalfCheetah
ALGORITHMS = ['PPO', 'DDPG', 'TD3', 'SAC']
MAX_STEPS = 100_000  # Reduced steps for testing
EVAL_INTERVAL = 1000
LOG_DIR = 'logs'
PLOT_DIR = 'plots'

def get_system_metrics():
    """Get current system metrics."""
    process = psutil.Process(os.getpid())
    return {
        'memory_usage': process.memory_info().rss / 1024 / 1024 / 1024,  # GB
        'cpu_usage': process.cpu_percent()
    }

def detect_crash(reward_history, window=10):
    """Detect crash events based on sudden reward drops."""
    if len(reward_history) < 2*window:  # Need at least 2 windows worth of data
        return False
    
    current_window = reward_history[-window:]
    previous_window = reward_history[-2*window:-window]
    
    if not current_window or not previous_window:  # Check for empty windows
        return False
        
    current_avg = np.mean(current_window)
    previous_avg = np.mean(previous_window)
    
    # Avoid division by zero or comparing with zero
    if abs(previous_avg) < 1e-10:  # If previous average is very close to zero
        return current_avg < -1.0  # Consider it a crash if current reward is significantly negative
    
    return current_avg < previous_avg * 0.5  # 50% drop threshold

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

def run_experiment(algo_name, env_name, seed):
    """Run a single experiment with given algorithm, environment and seed."""
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = create_agent(algo_name, env, device)
    
    # Initialize metrics
    metrics = {
        'step': [],
        'episode_reward': [],
        'seed': [],
        'algorithm': [],
        'environment': [],
        'time_elapsed': [],
        'memory_usage': [],
        'cpu_usage': []
    }
    
    start_time = time.time()
    total_steps = 0
    episode = 0
    crash_events = 0
    reward_history = []
    
    while total_steps < MAX_STEPS:
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done and total_steps < MAX_STEPS:
            # Select action
            if algo_name == 'PPO':
                action_tuple = agent.select_action(state, deterministic=False)
                action = action_tuple[0]  # First element is the action
                value = action_tuple[1]
                log_prob = action_tuple[2]
            else:
                action = agent.select_action(state, deterministic=False)
            
            # Ensure action is a numpy array with float32 dtype
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.float32)
            elif isinstance(action, torch.Tensor):
                action = action.cpu().numpy().astype(np.float32)
            else:
                action = np.array(action, dtype=np.float32)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition and train
            if algo_name == 'PPO':
                agent.buffer.add(state, action, reward, value, log_prob, terminated)
                if agent.buffer.count == agent.buffer.size:
                    agent.train()
            else:
                agent.replay_buffer.add(state, action, reward, next_state, terminated)
                if total_steps >= 10000:  # Start training after random steps
                    agent.train()
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Record metrics every EVAL_INTERVAL steps
            if total_steps % EVAL_INTERVAL == 0:
                eval_reward = evaluate_policy(env, agent)
                sys_metrics = get_system_metrics()
                
                metrics['step'].append(total_steps)
                metrics['episode_reward'].append(eval_reward)
                metrics['seed'].append(seed)
                metrics['algorithm'].append(algo_name)
                metrics['environment'].append(env_name)
                metrics['time_elapsed'].append(time.time() - start_time)
                metrics['memory_usage'].append(sys_metrics['memory_usage'])
                metrics['cpu_usage'].append(sys_metrics['cpu_usage'])
                
                # Check for crash events
                reward_history.append(eval_reward)
                if detect_crash(reward_history):
                    crash_events += 1
        
        episode += 1
    
    # Save metrics to CSV
    df = pd.DataFrame(metrics)
    filename = f"{LOG_DIR}/{algo_name}_{env_name}_seed{seed}.csv"
    df.to_csv(filename, index=False)
    
    return crash_events

def analyze_results():
    """Analyze and plot results from all experiments."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Load all CSV files
    all_data = []
    for filename in os.listdir(LOG_DIR):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(LOG_DIR, filename))
            all_data.append(df)
    
    if not all_data:  # Check if we have any data to analyze
        print("No data files found to analyze!")
        return
        
    data = pd.concat(all_data, ignore_index=True)
    
    # 1. Learning Curves with Standard Deviation
    for env_name in ENVS:
        plt.figure(figsize=(12, 8))
        for algo_name in ALGORITHMS:
            algo_data = data[(data['environment'] == env_name) & 
                           (data['algorithm'] == algo_name)]
            
            if algo_data.empty:  # Skip if no data for this algorithm
                continue
                
            # Calculate mean and std of rewards
            grouped = algo_data.groupby('step')['episode_reward']
            mean_reward = grouped.mean()
            std_reward = grouped.std().fillna(0)  # Replace NaN with 0 for std
            
            if not mean_reward.empty:  # Only plot if we have data
                plt.plot(mean_reward.index, mean_reward.values, label=algo_name)
                plt.fill_between(mean_reward.index,
                               mean_reward.values - std_reward.values,
                               mean_reward.values + std_reward.values,
                               alpha=0.2)
        
        plt.xlabel('Timesteps')
        plt.ylabel('Average Reward')
        plt.title(f'Learning Curves - {env_name}')
        plt.legend()
        plt.savefig(f'{PLOT_DIR}/learning_curves_{env_name}.png')
        plt.close()
    
    # 2. Final Performance Box Plots
    plt.figure(figsize=(12, 6))
    for i, env_name in enumerate(ENVS):
        plt.subplot(1, 2, i+1)
        env_data = data[data['environment'] == env_name]
        if not env_data.empty:  # Only create plot if we have data
            sns.boxplot(data=env_data, x='algorithm', y='episode_reward')
            plt.title(f'Final Performance - {env_name}')
            plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/final_performance_boxplot.png')
    plt.close()
    
    # 3. Resource Usage vs Performance
    for env_name in ENVS:
        plt.figure(figsize=(15, 5))
        
        # Memory vs Performance
        plt.subplot(1, 3, 1)
        for algo_name in ALGORITHMS:
            algo_data = data[(data['environment'] == env_name) & 
                           (data['algorithm'] == algo_name)]
            plt.scatter(algo_data['memory_usage'], algo_data['episode_reward'],
                       label=algo_name, alpha=0.5)
        plt.xlabel('Memory Usage (GB)')
        plt.ylabel('Reward')
        plt.title('Memory vs Performance')
        plt.legend()
        
        # CPU vs Performance
        plt.subplot(1, 3, 2)
        for algo_name in ALGORITHMS:
            algo_data = data[(data['environment'] == env_name) & 
                           (data['algorithm'] == algo_name)]
            plt.scatter(algo_data['cpu_usage'], algo_data['episode_reward'],
                       label=algo_name, alpha=0.5)
        plt.xlabel('CPU Usage (%)')
        plt.ylabel('Reward')
        plt.title('CPU vs Performance')
        plt.legend()
        
        # Time vs Performance
        plt.subplot(1, 3, 3)
        for algo_name in ALGORITHMS:
            algo_data = data[(data['environment'] == env_name) & 
                           (data['algorithm'] == algo_name)]
            plt.scatter(algo_data['time_elapsed'], algo_data['episode_reward'],
                       label=algo_name, alpha=0.5)
        plt.xlabel('Time Elapsed (s)')
        plt.ylabel('Reward')
        plt.title('Time vs Performance')
        plt.legend()
        
        plt.suptitle(f'Resource Usage vs Performance - {env_name}')
        plt.tight_layout()
        plt.savefig(f'{PLOT_DIR}/resource_performance_{env_name}.png')
        plt.close()

def main():
    """Main function to run all experiments and analysis."""
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Track crash events
    crash_events = defaultdict(lambda: defaultdict(int))
    
    # Run all experiments
    for env_name in ENVS:
        for algo_name in ALGORITHMS:
            for seed in SEEDS:
                print(f"Running {algo_name} on {env_name} with seed {seed}")
                crashes = run_experiment(algo_name, env_name, seed)
                crash_events[env_name][algo_name] += crashes
    
    # Save crash events to CSV
    crash_df = pd.DataFrame(crash_events).reset_index()
    crash_df.columns = ['Algorithm'] + list(ENVS)
    crash_df.to_csv(f'{LOG_DIR}/crash_events.csv', index=False)
    
    # Run analysis
    analyze_results()
    
    print("Experiments and analysis completed!")

if __name__ == "__main__":
    main() 
