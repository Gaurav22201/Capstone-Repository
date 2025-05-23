"""
DRL Model Benchmark: DQN vs. A2C on LunarLander-v2

This script simulates training and evaluation of two reinforcement learning models
(DQN and A2C) on the LunarLander-v2 environment.

Features:
- Simulated training with CPU/GPU usage tracking.
- Randomized reward generation for evaluation.
- Performance comparison via matplotlib plot.

Usage:
    Run this script to benchmark DQN and A2C over 50 simulated episodes.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import psutil


def get_gpu_stats():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e6  # MB
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1e6
            return gpu_memory_used, gpu_max_memory
    except ModuleNotFoundError:
        return None, None
    return None, None


def get_cpu_usage():
    return psutil.cpu_percent(interval=1)


def train_model(env_name, model_type, timesteps=10000):
    if model_type not in ['DQN', 'A2C']:
        raise ValueError("Invalid model type. Choose 'DQN' or 'A2C'.")

    print(f"\nTraining {model_type} on {env_name} for {timesteps} timesteps...")
    start_time = time.time()
    cpu_start = get_cpu_usage()
    time.sleep(2)  # Simulate training
    cpu_end = get_cpu_usage()
    end_time = time.time()

    gpu_usage, gpu_max_usage = get_gpu_stats()
    print(f"Training time: {end_time - start_time:.2f} sec")
    print(f"CPU usage: {cpu_start}% → {cpu_end}%")
    if gpu_usage is not None:
        print(f"GPU used: {gpu_usage:.2f} MB (Max: {gpu_max_usage:.2f} MB)")

    return {
        'model': f"{model_type} Model",
        'environment': f"{env_name} Environment"
    }


def evaluate_model(model, num_episodes=50):
    print(f"\nEvaluating {model['model']} in {model['environment']} over {num_episodes} episodes...")

    start_time = time.time()
    cpu_start = get_cpu_usage()

    if 'DQN' in model['model']:
        rewards = np.random.randint(50, 250, num_episodes).tolist()
    else:
        rewards = np.random.randint(30, 220, num_episodes).tolist()

    avg_reward = np.mean(rewards)

    cpu_end = get_cpu_usage()
    end_time = time.time()

    for i, reward in enumerate(rewards):
        print(f"Episode {i + 1}: Reward = {reward}")

    print(f"Evaluation time: {end_time - start_time:.2f} sec")
    print(f"CPU usage: {cpu_start}% → {cpu_end}%")
    return avg_reward, rewards


def plot_performance(model_rewards, model_names):
    print("\nPerformance Comparison:")
    for rewards, name in zip(model_rewards, model_names):
        print(f"{name} Avg Reward: {np.mean(rewards):.2f}")

    plt.figure(figsize=(10, 5))
    for rewards, name in zip(model_rewards, model_names):
        plt.plot(rewards, label=name)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("DRL Model Performance on LunarLander-v2")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    try:
        env_name = "LunarLander-v2"

        # Train models
        dqn_model = train_model(env_name, "DQN", timesteps=10000)
        a2c_model = train_model(env_name, "A2C", timesteps=10000)

        # Evaluate models
        dqn_reward, dqn_rewards = evaluate_model(dqn_model)
        a2c_reward, a2c_rewards = evaluate_model(a2c_model)

        print(f"\nDQN Avg Reward: {dqn_reward:.2f}")
        print(f"A2C Avg Reward: {a2c_reward:.2f}")

        # Plot results
        plot_performance([dqn_rewards, a2c_rewards], ["DQN", "A2C"])

    except ValueError as e:
        print(f"Error: {e}")
