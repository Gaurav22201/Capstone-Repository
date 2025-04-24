#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from scipy import stats

def plot_memory_usage(metrics, env_name, save_dir):
    """Plot memory usage over time"""
    plt.figure(figsize=(10, 6))
    if 'gpu_memory_gb' in metrics:
        memory_data = metrics['gpu_memory_gb']
        if isinstance(memory_data, (int, float)):
            plt.axhline(y=memory_data, color='b', linestyle='-')
        else:
            plt.plot(memory_data)
    plt.title(f'Memory Usage - {env_name}')
    plt.xlabel('Time')
    plt.ylabel('GPU Memory (GB)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'memory_usage.png'))
    plt.close()

def plot_reward_distribution(rewards, env_name, save_dir):
    """Plot reward distribution histogram"""
    plt.figure(figsize=(10, 6))
    
    # Check if we have enough data points
    if len(rewards) < 2:
        print(f"Warning: Not enough data points for {env_name} to plot distribution. Skipping distribution plot.")
        plt.close()
        return
    
    sns.histplot(rewards, kde=True)
    plt.title(f'Reward Distribution - {env_name}')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'reward_distribution.png'))
    plt.close()

def plot_reward_stability(rewards, env_name, save_dir, window=100):
    """Plot reward stability (rolling mean and std)"""
    plt.figure(figsize=(10, 6))
    
    # Check if we have enough data points
    if len(rewards) < window:
        print(f"Warning: Not enough data points for {env_name} to calculate rolling statistics. Skipping stability plot.")
        plt.close()
        return
    
    rewards_array = np.array(rewards)
    rolling_mean = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
    
    # Calculate rolling std with same length as rolling_mean
    rolling_std = []
    for i in range(len(rolling_mean)):
        start_idx = max(0, i)
        end_idx = min(len(rewards_array), i + window)
        if end_idx - start_idx > 1:  # Need at least 2 points for std
            rolling_std.append(np.std(rewards_array[start_idx:end_idx]))
        else:
            rolling_std.append(0)
    rolling_std = np.array(rolling_std)
    
    plt.plot(rolling_mean, label='Rolling Mean')
    plt.fill_between(range(len(rolling_mean)), 
                    rolling_mean - rolling_std,
                    rolling_mean + rolling_std,
                    alpha=0.2, label='±1 std')
    plt.title(f'Reward Stability - {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'reward_stability.png'))
    plt.close()

def plot_training_efficiency(metrics, env_name, save_dir):
    """Plot training efficiency metrics"""
    plt.figure(figsize=(10, 6))
    if 'sample_efficiency' in metrics:
        efficiency_data = metrics['sample_efficiency']
        plt.plot(efficiency_data if isinstance(efficiency_data, list) else [efficiency_data])
    plt.title(f'Training Efficiency - {env_name}')
    plt.xlabel('Training Steps')
    plt.ylabel('Sample Efficiency')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_efficiency.png'))
    plt.close()

def plot_training_progress(rewards, env_name, save_dir, window=100):
    """Plot smoothed training progress"""
    plt.figure(figsize=(10, 6))
    if len(rewards) > window:
        smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed_rewards)
    else:
        plt.plot(rewards)
    plt.title(f'Training Progress - {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()

def plot_learning_curve(rewards, env_name, save_dir):
    """Plot learning curve for a single environment"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(f'Learning Curve - {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Create environment-specific directory for plots
    env_dir = os.path.join('results', env_name)
    plots_dir = os.path.join(env_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, f'{env_name}_learning_curve.png'))
    plt.close()

def plot_ram_usage(metrics, env_name, save_dir):
    """Plot RAM usage over time"""
    plt.figure(figsize=(10, 6))
    if 'ram_usage_gb' in metrics:
        ram_data = metrics['ram_usage_gb']
        if isinstance(ram_data, (int, float)):
            plt.axhline(y=ram_data, color='r', linestyle='-')
        else:
            plt.plot(ram_data)
    plt.title(f'RAM Usage - {env_name}')
    plt.xlabel('Time')
    plt.ylabel('RAM Usage (GB)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'ram_usage.png'))
    plt.close()

def plot_cpu_usage(metrics, env_name, save_dir):
    """Plot CPU usage over time"""
    plt.figure(figsize=(10, 6))
    if 'cpu_usage_percent' in metrics:
        cpu_data = metrics['cpu_usage_percent']
        if isinstance(cpu_data, (int, float)):
            plt.axhline(y=cpu_data, color='g', linestyle='-')
        else:
            plt.plot(cpu_data)
    plt.title(f'CPU Usage - {env_name}')
    plt.xlabel('Time')
    plt.ylabel('CPU Usage (%)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'cpu_usage.png'))
    plt.close()

def analyze_environment(env_name, metrics_file):
    """Analyze training results for a single environment"""
    if not os.path.exists(metrics_file):
        print(f"No metrics found for {env_name}")
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract metrics
    total_steps = metrics.get('total_steps', 0)
    episode_rewards = metrics.get('episode_rewards', [])
    ram_usage = metrics.get('ram_usage_gb', [])
    cpu_usage = metrics.get('cpu_usage_percent', [])
    eval_rewards = metrics.get('eval_rewards', [])
    
    # Check if we have any rewards
    if not episode_rewards:
        print(f"No reward data found for {env_name}")
        return None
    
    final_reward = episode_rewards[-1] if episode_rewards else 0
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    max_reward = max(episode_rewards) if episode_rewards else 0
    reward_std = np.std(episode_rewards) if len(episode_rewards) > 1 else 0
    
    # Calculate RAM usage statistics
    avg_ram = np.mean(ram_usage) if ram_usage else 0
    max_ram = max(ram_usage) if ram_usage else 0
    min_ram = min(ram_usage) if ram_usage else 0
    
    # Calculate CPU usage statistics
    avg_cpu = np.mean(cpu_usage) if cpu_usage else 0
    max_cpu = max(cpu_usage) if cpu_usage else 0
    min_cpu = min(cpu_usage) if cpu_usage else 0
    
    # Get convergence metrics
    steps_to_convergence = metrics.get('steps_to_convergence', None)
    time_to_convergence = metrics.get('time_to_convergence', None)
    
    # Get other metrics
    collapse_events = metrics.get('collapse_events', 0)
    sample_efficiency = metrics.get('sample_efficiency', [])
    robustness_drop_percent = metrics.get('robustness_drop_percent', 0.0)
    
    # Calculate final sample efficiency
    final_efficiency = sample_efficiency[-1] if sample_efficiency else 0
    
    # Create plots directory
    plots_dir = os.path.join('results', env_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate all plots
    plot_learning_curve(episode_rewards, env_name, plots_dir)
    plot_ram_usage(metrics, env_name, plots_dir)
    plot_cpu_usage(metrics, env_name, plots_dir)
    plot_reward_distribution(episode_rewards, env_name, plots_dir)
    plot_reward_stability(episode_rewards, env_name, plots_dir)
    plot_training_efficiency(metrics, env_name, plots_dir)
    plot_training_progress(episode_rewards, env_name, plots_dir)
    
    # Return metrics
    return {
        'env_name': env_name,
        'total_steps': total_steps,
        'num_episodes': len(episode_rewards),
        'final_reward': final_reward,
        'average_reward': avg_reward,
        'max_reward': max_reward,
        'reward_std': reward_std,
        'training_time_min': metrics.get('training_time_min', 0),
        'ram_usage_gb': avg_ram,
        'max_ram_usage_gb': max_ram,
        'min_ram_usage_gb': min_ram,
        'cpu_usage_percent': avg_cpu,
        'max_cpu_usage_percent': max_cpu,
        'min_cpu_usage_percent': min_cpu,
        'steps_to_convergence': steps_to_convergence,
        'time_to_convergence': time_to_convergence,
        'collapse_events': collapse_events,
        'sample_efficiency': final_efficiency,
        'robustness_drop_percent': robustness_drop_percent
    }

def main():
    # Define environment names
    env_names = [
        'Pendulum-v1',
        'LunarLanderContinuous-v3',
        'Humanoid-v5',
        'HalfCheetah-v4',
        'BipedalWalker-v3',
        'BipedalWalkerHardcore-v3'
    ]
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Analyze each environment
    all_metrics = {}
    for env_name in env_names:
        metrics_file = os.path.join('results', f'metrics_{env_name}.json')
        if not os.path.exists(metrics_file):
            print(f"No metrics found for {env_name}")
            continue
        
        metrics = analyze_environment(env_name, metrics_file)
        if metrics:
            all_metrics[env_name] = metrics
    
    # Save combined metrics to text file
    with open(os.path.join('results', 'all_environments_metrics.txt'), 'w') as f:
        for env_name, metrics in all_metrics.items():
            f.write(f"Performance Metrics for {env_name}:\n")
            f.write(f"  Total Steps: {metrics['total_steps']}\n")
            f.write(f"  Number of Episodes: {metrics['num_episodes']}\n")
            f.write(f"  Final Reward: {metrics['final_reward']:.2f}\n")
            f.write(f"  Average Reward: {metrics['average_reward']:.2f}\n")
            f.write(f"  Max Reward: {metrics['max_reward']:.2f}\n")
            f.write(f"  Reward Std Dev: {metrics['reward_std']:.2f}\n")
            f.write(f"  Steps to Convergence: {metrics['steps_to_convergence'] or 'Not converged'}\n")
            f.write(f"  Time to Convergence (min): {f'{metrics['time_to_convergence']:.2f}' if metrics['time_to_convergence'] is not None else 'Not converged'}\n")
            f.write(f"  Collapse Events: {metrics['collapse_events']}\n")
            f.write(f"  Sample Efficiency: {metrics['sample_efficiency']:.2e}\n")
            f.write(f"  Robustness Drop (%): {metrics['robustness_drop_percent']:.2f}\n")
            f.write(f"  Average RAM Usage (GB): {metrics['ram_usage_gb']:.2f}\n")
            f.write(f"  Max RAM Usage (GB): {metrics['max_ram_usage_gb']:.2f}\n")
            f.write(f"  Min RAM Usage (GB): {metrics['min_ram_usage_gb']:.2f}\n")
            f.write(f"  Average CPU Usage (%): {metrics['cpu_usage_percent']:.2f}\n")
            f.write(f"  Max CPU Usage (%): {metrics['max_cpu_usage_percent']:.2f}\n")
            f.write(f"  Min CPU Usage (%): {metrics['min_cpu_usage_percent']:.2f}\n")
            f.write(f"  Training Time (min): {metrics['training_time_min']:.2f}\n")
            f.write("\n")
    
    print("\nAnalysis complete! Results organized by environment in the results directory.")

if __name__ == "__main__":
    main() 