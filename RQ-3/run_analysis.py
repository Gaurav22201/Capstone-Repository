import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the algorithms
from PPO_Cont_Analysis import PPO
from DDPG_analysis import DDPG
from TD3_Analysis import TD3
from SAC_Cont_Analysis import SAC
from PPO_Cont_Analysis.PPO import PPO_agent
from DDPG_analysis.DDPG import DDPG_agent
from TD3_Analysis.TD3 import TD3_agent
from SAC_Cont_Analysis.SAC import SAC_agent

import gymnasium as gym

# Constants
ENVS = ['Pendulum-v1', 'BipedalWalker-v3']
ALGORITHMS = ['PPO', 'DDPG', 'TD3', 'SAC']
MAX_STEPS = 20_000  # Reduced for faster experiments
EVAL_INTERVAL = 50  # More frequent evaluation for shorter runs
LOG_DIR = 'RQ-3/logs'
PLOT_DIR = 'RQ-3/plots'

class MetricsTracker:
    def __init__(self):
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.cpu_usages = []
        self.rewards = []
        
    def update(self, reward):
        """Update metrics after each episode"""
        self.rewards.append(reward)
        
        # Update CPU and memory metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = self.process.cpu_percent()
        
        self.peak_memory = max(self.peak_memory, memory_usage)
        self.cpu_usages.append(cpu_usage)
    
    def get_metrics(self):
        """Get final metrics"""
        training_time = time.time() - self.start_time
        return {
            'final_cumulative_reward': sum(self.rewards),
            'reward_variance': np.var(self.rewards) if self.rewards else 0,
            'training_time': training_time,
            'peak_memory_mb': self.peak_memory,
            'avg_cpu_usage': np.mean(self.cpu_usages) if self.cpu_usages else 0,
            'peak_cpu_usage': max(self.cpu_usages) if self.cpu_usages else 0
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
    rewards = []
    for _ in range(eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)
        avg_reward += episode_reward
    
    return avg_reward / eval_episodes, np.var(rewards)

def run_experiment(algo_name, env_name):
    """Run a single experiment with given algorithm and environment."""
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create environment and agent
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = create_agent(algo_name, env, device)
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    total_steps = 0
    episode = 0
    
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
            
            # Update metrics
            if total_steps % EVAL_INTERVAL == 0:
                eval_reward, reward_var = evaluate_policy(env, agent)
                metrics_tracker.update(eval_reward)
        
        episode += 1
    
    # Get final metrics
    final_metrics = metrics_tracker.get_metrics()
    final_metrics.update({
        'algorithm': algo_name,
        'environment': env_name,
        'total_steps': total_steps,
        'total_episodes': episode
    })
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([final_metrics])
    filename = f"{LOG_DIR}/{algo_name}_{env_name}_metrics.csv"
    metrics_df.to_csv(filename, index=False)
    
    return final_metrics

def analyze_results():
    """Analyze and plot the performance metrics."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Load all CSV files
    all_data = []
    for filename in os.listdir(LOG_DIR):
        if filename.endswith('_metrics.csv'):
            df = pd.read_csv(os.path.join(LOG_DIR, filename))
            all_data.append(df)
    
    if not all_data:
        print("No data files found to analyze!")
        return
    
    data = pd.concat(all_data, ignore_index=True)
    
    # 1. Final Cumulative Reward Analysis
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='algorithm', y='final_cumulative_reward', hue='environment')
    plt.title('Final Cumulative Reward by Algorithm and Environment')
    plt.xticks(range(len(ALGORITHMS)), ALGORITHMS, rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/final_rewards.png')
    plt.close()
    
    # 2. Reward Variance Analysis
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='algorithm', y='reward_variance', hue='environment')
    plt.title('Reward Variance by Algorithm and Environment')
    plt.xticks(range(len(ALGORITHMS)), ALGORITHMS, rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/reward_variance.png')
    plt.close()
    
    # 3. Training Time Analysis
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x='algorithm', y='training_time', hue='environment')
    plt.title('Training Time by Algorithm and Environment')
    plt.xticks(range(len(ALGORITHMS)), ALGORITHMS, rotation=45)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/training_time.png')
    plt.close()
    
    # 4. Resource Usage Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Memory usage plot
    sns.barplot(data=data, x='algorithm', y='peak_memory_mb', hue='environment', ax=ax1)
    ax1.set_title('Peak Memory Usage by Algorithm and Environment')
    # Fix tick labels
    ax1.set_xticks(range(len(ALGORITHMS)))
    ax1.set_xticklabels(ALGORITHMS, rotation=45)
    
    # CPU usage plot
    sns.barplot(data=data, x='algorithm', y='avg_cpu_usage', hue='environment', ax=ax2)
    ax2.set_title('Average CPU Usage by Algorithm and Environment')
    # Fix tick labels
    ax2.set_xticks(range(len(ALGORITHMS)))
    ax2.set_xticklabels(ALGORITHMS, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/resource_usage.png')
    plt.close()
    
    # Save summary statistics
    summary = data.groupby(['environment', 'algorithm']).agg({
        'final_cumulative_reward': ['mean', 'std'],
        'reward_variance': 'mean',
        'training_time': 'mean',
        'peak_memory_mb': 'mean',
        'avg_cpu_usage': 'mean',
        'peak_cpu_usage': 'mean'
    }).round(2)
    
    summary.to_csv(f'{LOG_DIR}/summary_statistics.csv')
    print("Analysis completed! Check the plots directory for results.")

def main():
    """Main function to run all experiments and analysis."""
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Run all experiments
    results = []
    for env_name in ENVS:
        for algo_name in ALGORITHMS:
            print(f"Running {algo_name} on {env_name}")
            metrics = run_experiment(algo_name, env_name)
            results.append(metrics)
    
    # Save aggregate results
    pd.DataFrame(results).to_csv(f'{LOG_DIR}/aggregate_results.csv', index=False)
    
    # Run analysis
    analyze_results()
    
    print("RQ-3 experiments and analysis completed!")

if __name__ == "__main__":
    main() 