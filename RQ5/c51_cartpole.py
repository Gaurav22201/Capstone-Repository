"""
C51 (Categorical DQN) on LunarLander-v2

This script implements the Categorical Deep Q-Network (C51) algorithm using PyTorch
to train an agent in the OpenAI Gym environment `LunarLander-v2`.

Features:
- C51 distributional Q-learning with 51 atoms (discrete value bins).
- Epsilon-greedy policy with decay.
- Experience replay and target network updates.
- Resource monitoring (CPU/GPU) and logging to CSV.
- Reward plotting over episodes.

Usage:
    python c51_lunarlander.py

Output:
- Printed training logs and stats.
- CSV file: `c51_log.csv`
- Reward plot using matplotlib.

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

# Hyperparameters
V_MIN = 0
V_MAX = 200
ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (ATOMS - 1)
Z = torch.linspace(V_MIN, V_MAX, ATOMS)

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
EPISODES = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Helpers
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

def log_to_csv(filename, row, header=False):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists and header:
            writer.writeheader()
        writer.writerow(row)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, *args):
        self.buffer.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# Categorical DQN Model
class CategoricalDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CategoricalDQN, self).__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim * ATOMS)
        )
    def forward(self, x):
        batch_size = x.size(0)
        x = self.net(x).view(batch_size, self.action_dim, ATOMS)
        return torch.softmax(x, dim=2)

# Agent
class C51Agent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.policy_net = CategoricalDQN(state_dim, action_dim).to(device)
        self.target_net = CategoricalDQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.update_target()
        self.epsilon = EPSILON_START

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            dist = self.policy_net(state) * Z.to(device)
            q_values = dist.sum(2)
            return q_values.argmax(1).item()

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

        with torch.no_grad():
            next_dist = self.target_net(next_states)
            next_q = (next_dist * Z.to(device)).sum(2)
            next_actions = next_q.argmax(1)
            next_dist = next_dist[range(BATCH_SIZE), next_actions]

            t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * GAMMA * Z.unsqueeze(0)
            t_z = t_z.clamp(V_MIN, V_MAX)
            b = (t_z - V_MIN) / DELTA_Z
            l = b.floor().long()
            u = b.ceil().long()

            proj_dist = torch.zeros_like(next_dist)
            for i in range(BATCH_SIZE):
                for j in range(ATOMS):
                    l_idx = l[i][j]
                    u_idx = u[i][j]
                    if l_idx == u_idx:
                        proj_dist[i][l_idx] += next_dist[i][j]
                    else:
                        proj_dist[i][l_idx] += next_dist[i][j] * (u[i][j] - b[i][j])
                        proj_dist[i][u_idx] += next_dist[i][j] * (b[i][j] - l[i][j])

        dist = self.policy_net(states)
        log_p = torch.log(dist[range(BATCH_SIZE), actions.squeeze()])
        loss = -(proj_dist * log_p).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_END)

# Training
def train_c51():
    env = gym.make("LunarLander-v2")
    #env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = C51Agent(state_dim, action_dim)

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
        print(f"Episode {ep+1}/{EPISODES} — Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    env.close()
    return rewards

# Plotting
def plot_rewards(rewards):
    plt.plot(rewards, label="C51")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("C51 (Categorical DQN) on LunarLander-v2")
    plt.legend()
    plt.grid()
    plt.show()

# Main
if __name__ == "__main__":
    env_name = "LunarLander-v2"
    model_name = "C51 (Categorical DQN)"

    print(f"Training {model_name} model in {env_name} for {EPISODES} episodes...")

    start_cpu = psutil.cpu_percent(interval=None)
    start_time = time.time()

    rewards = train_c51()

    elapsed_time = time.time() - start_time
    avg_reward = np.mean(rewards)
    end_cpu = psutil.cpu_percent(interval=None)

    print(f"Training time: {elapsed_time:.2f} seconds")
    print_resource_usage(start_cpu)
    print(f"{model_name} Average Reward: {avg_reward:.2f}")

    log_to_csv(
        filename="c51_log.csv",
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
