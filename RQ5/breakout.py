"""
DRL Simulation: DQN vs. A2C on Breakout-v0

This script compares two Deep Reinforcement Learning algorithms — DQN and A2C —
on the Atari environment `Breakout-v0` using simulated training and evaluation.

Key Features:
- Simulates training time and CPU/GPU usage tracking.
- Generates random rewards to mimic evaluation behavior.
- Visualizes and compares performance using matplotlib.

Usage:
    python breakout_drl_sim.py

Outputs:
- Printed training/evaluation logs with resource usage
- Reward plot comparing both models

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
    time.sleep(2)  # Simulated training duration
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

    # Simulated evaluation rewards
    if 'DQN' in model['model']:
        rewards = np.random.randint(20, 80, num_episodes).tolist()
    else:
        rewards = np.random.randint(15, 70, num_episodes).tolist()

    avg_reward = np.mean(rewards)

    cpu_end = get_cpu_usage()
    end_time = time.time()

    for i, reward in enumerate(rewards):
        print(f"Episode {i + 1}: Reward = {reward}")

    print(f"Evaluation time: {end_time - start_time:.2f} sec")
    print(f"CPU usage: {cpu_start}% → {cpu_end}%")
    return avg_reward, rewards


def plot_performance(model_rewards, model_names):
    print("\nComparing Performance:")
    for rewards, name in zip(model_rewards, model_names):
        print(f"{name} Avg Reward: {np.mean(rewards):.2f}")

    plt.figure(figsize=(10, 5))
    for rewards, name in zip(model_rewards, model_names):
        plt.plot(rewards, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DRL Model Performance on Breakout")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    try:
        env_name = "Breakout-v0"

        # Train both models (simulated)
        dqn_model = train_model(env_name, "DQN", timesteps=10000)
        a2c_model = train_model(env_name, "A2C", timesteps=10000)

        # Evaluate both models (simulated)
        dqn_reward, dqn_rewards = evaluate_model(dqn_model)
        a2c_reward, a2c_rewards = evaluate_model(a2c_model)

        # Final results
        print(f"\nDQN Avg Reward: {dqn_reward:.2f}")
        print(f"A2C Avg Reward: {a2c_reward:.2f}")

        # Plot
        plot_performance([dqn_rewards, a2c_rewards], ["DQN", "A2C"])

    except ValueError as e:
        print(f"Error: {e}")
