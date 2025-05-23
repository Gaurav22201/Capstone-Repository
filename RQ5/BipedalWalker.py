"""
DRL Model Comparison on BipedalWalker-v3

This script simulates training and evaluation of two Deep Reinforcement Learning (DRL) models
(PPO and SAC) on the continuous control environment `BipedalWalker-v3`. The script includes:
- Simulated training time and CPU/GPU usage profiling
- Pseudo evaluation with randomly generated rewards
- Reward comparison via matplotlib plots
"""



import time
import numpy as np
import matplotlib.pyplot as plt
import psutil


def get_gpu_stats():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e6
            gpu_max_memory = torch.cuda.max_memory_allocated() / 1e6
            return gpu_memory_used, gpu_max_memory
    except ModuleNotFoundError:
        return None, None
    return None, None


def get_cpu_usage():
    return psutil.cpu_percent(interval=1)


def train_model(env_name, model_type, timesteps=10000):
    if model_type not in ['PPO', 'SAC']:
        raise ValueError("Use PPO or SAC for continuous action envs like BipedalWalker.")

    print(f"\nTraining {model_type} on {env_name} for {timesteps} timesteps...")
    start_time = time.time()
    cpu_start = get_cpu_usage()
    time.sleep(2.5)  # Simulate training
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


def evaluate_model(model, num_episodes=40):
    print(f"\nEvaluating {model['model']} in {model['environment']} over {num_episodes} episodes...")

    start_time = time.time()
    cpu_start = get_cpu_usage()

    if 'PPO' in model['model']:
        rewards = np.random.randint(100, 300, num_episodes).tolist()
    else:
        rewards = np.random.randint(80, 250, num_episodes).tolist()

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
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DRL Model Performance on BipedalWalker-v3")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    try:
        env_name = "BipedalWalker-v3"

        # Train simulated models
        ppo_model = train_model(env_name, "PPO", timesteps=10000)
        sac_model = train_model(env_name, "SAC", timesteps=10000)

        # Evaluate
        ppo_reward, ppo_rewards = evaluate_model(ppo_model)
        sac_reward, sac_rewards = evaluate_model(sac_model)

        print(f"\nPPO Avg Reward: {ppo_reward:.2f}")
        print(f"SAC Avg Reward: {sac_reward:.2f}")

        # Plot comparison
        plot_performance([ppo_rewards, sac_rewards], ["PPO", "SAC"])

    except ValueError as e:
        print(f"Error: {e}")
