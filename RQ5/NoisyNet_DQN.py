"""
NoisyNet DQN on LunarLander-v2

This script trains a NoisyNet-enhanced Deep Q-Network on the OpenAI Gym environment
`LunarLander-v2` using PyTorch.

Main Features:
- NoisyLinear layers for better exploration (NoisyNet).
- Replay buffer and target network updates.
- CPU/GPU usage tracking via psutil and GPUtil.
- Logs training summary to CSV and plots rewards.

Run:
    python noisynet_dqn_lunarlander.py

Outputs:
- noisetnet_dqn_log.csv
- Reward plot (matplotlib)
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import time
import psutil
import csv
import os

if torch.cuda.is_available():
    import GPUtil

# Parameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 10000
TARGET_UPDATE = 10
EPISODES = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Helper function: Print CPU/GPU usage
def print_resource_usage(start_cpu):
    end_cpu = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()
    print(f"CPU usage: {start_cpu:.1f}% → {end_cpu:.1f}% | Memory: {memory.percent:.1f}%")
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            print(f"GPU {i} — Load: {gpu.load * 100:.1f}% | Mem: {gpu.memoryUsed:.1f}MB / {gpu.memoryTotal:.1f}MB")
    else:
        print("No GPU available or not being used.")

# Helper function: Log results to CSV
def log_to_csv(filename, row, header=False):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists and header:
            writer.writeheader()
        writer.writerow(row)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-mu_range, mu_range)
        self.mu_bias.data.uniform_(-mu_range, mu_range)
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.epsilon_weight = epsilon_out.ger(epsilon_in)
        self.epsilon_bias = epsilon_out
    def forward(self, x):
        if self.training:
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return torch.nn.functional.linear(x, weight, bias)
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

# Noisy DQN Network
class NoisyDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(NoisyDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = NoisyLinear(128, 128)
        self.fc3 = NoisyLinear(128, action_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    def reset_noise(self):
        self.fc2.reset_noise()
        self.fc3.reset_noise()

# Noisy DQN Agent
class NoisyDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.policy_net = NoisyDQN(state_dim, action_dim).to(device)
        self.target_net = NoisyDQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.gamma = GAMMA
        self.action_dim = action_dim
        self.update_target()
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.policy_net(state).argmax(1).item()
    def remember(self, *args):
        self.memory.push(*args)
    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*batch))
        states = torch.FloatTensor(batch.state).to(device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(batch.reward).to(device)
        next_states = torch.FloatTensor(batch.next_state).to(device)
        dones = torch.FloatTensor(batch.done).to(device)
        current_q = self.policy_net(states).gather(1, actions).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

# Training loop
def train_noisynet():
    env = gym.make("LunarLander-v2")
    #env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = NoisyDQNAgent(state_dim, action_dim)
    rewards = []
    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
        if ep % TARGET_UPDATE == 0:
            agent.update_target()
        rewards.append(total_reward)
        print(f"Episode {ep+1}/{EPISODES} — Reward: {total_reward:.2f}")
    env.close()
    return rewards

# Plotting
def plot_rewards(rewards):
    plt.plot(rewards, label="NoisyNet DQN")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("NoisyNet DQN on LunarLander-v2")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    env_name = "LunarLander-v2"
    model_name = "NoisyNet DQN"

    print(f"Training {model_name} model in {env_name} for {EPISODES} episodes...")

    start_cpu = psutil.cpu_percent(interval=None)
    start_time = time.time()

    rewards = train_noisynet()

    elapsed_time = time.time() - start_time
    avg_reward = np.mean(rewards)
    end_cpu = psutil.cpu_percent(interval=None)

    print(f"Training time: {elapsed_time:.2f} seconds")
    print_resource_usage(start_cpu)
    print(f"{model_name} Average Reward: {avg_reward:.2f}")

    # Log to CSV
    log_to_csv(
        filename="noisynet_dqn_log.csv",
        row={
            "Model": model_name,
            "Environment": env_name,
            "Training Time (s)": round(elapsed_time, 2),
            "CPU Start (%)": round(start_cpu, 2),
            "CPU End (%)": round(end_cpu, 2),
            "Average Reward": round(avg_reward, 2)
        },
        header=True
    )

    plot_rewards(rewards)
