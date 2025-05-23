"""
Double DQN with Prioritized Experience Replay (PER) on LunarLander-v2

This script implements a Deep Reinforcement Learning agent using Double DQN with
Prioritized Experience Replay in the OpenAI Gym environment `LunarLander-v2`.

Main Features:
- Double DQN architecture to reduce overestimation bias.
- Prioritized Experience Replay to sample important transitions more frequently.
- Epsilon-greedy exploration with decay.
- Training resource tracking (CPU/GPU) and logging to CSV.
- Reward visualization using matplotlib.

Usage:
    python ddqn_per_lunarlander.py

Outputs:
- Console logs of training rewards and epsilon values.
- CSV log file: `ddqn_per_log.csv`
- Reward plot showing agent performance over episodes.
"""



import gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import csv
import os

def log_to_csv(filename, row, header=False):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists and header:
            writer.writeheader()
        writer.writerow(row)

if torch.cuda.is_available():
    import GPUtil

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Experience structure
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# Prioritized Experience Replay
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, transition):
        max_prio = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for i, prio in zip(indices, priorities):
            self.priorities[i] = prio

    def __len__(self):
        return len(self.buffer)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# DDQN Agent with PER
class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.memory = PrioritizedReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.beta = 0.4
        self.beta_increment = 1e-3

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.loss_fn = nn.SmoothL1Loss()
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).argmax().item()

    def remember(self, *args):
        self.memory.push(Transition(*args))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        transitions, indices, weights = self.memory.sample(self.batch_size, beta=self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(batch.action).unsqueeze(1)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(batch.done)
        weights = torch.FloatTensor(weights)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze()

        # Double DQN target Q values
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = (self.loss_fn(q_values, target_q) * weights).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        prios = (q_values - target_q).abs().detach().numpy() + 1e-5
        self.memory.update_priorities(indices, prios)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train DDQN with PER
def train_ddqn_per(episodes=300):
    #env = gym.make("CartPole-v1")
    env = gym.make("LunarLander-v2")

    agent = DDQNAgent(env.observation_space.shape[0], env.action_space.n)
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        agent.update_target()
        rewards.append(total_reward)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

    env.close()
    return rewards

# Plotting
def plot_rewards(rewards):
    plt.plot(rewards, label="DDQN + PER")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDQN with Prioritized Experience Replay on LunarLander-v2")
    plt.legend()
    plt.grid()
    plt.show()


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


if __name__ == "__main__":
    env_name = "LunarLander-v2"
    model_name = "DDQN + PER"

    print(f"Training {model_name} model in {env_name} for 300 episodes...")

    start_cpu = psutil.cpu_percent(interval=None)
    start_time = time.time()

    rewards = train_ddqn_per(episodes=300)

    elapsed_time = time.time() - start_time
    end_cpu = psutil.cpu_percent(interval=None)

    print(f"Training time: {elapsed_time:.2f} seconds")
    print_resource_usage(start_cpu)

    avg_reward = np.mean(rewards)
    print(f"{model_name} Average Reward: {avg_reward:.2f}")

    # Log to CSV
    log_to_csv(
        filename="ddqn_per_log.csv",
        row={
            "Model": model_name,
            "Environment": env_name,
            "Training Time (s)": round(elapsed_time, 2),
            "CPU Start (%)": round(start_cpu, 2),
            "CPU End (%)": round(end_cpu, 2),
            "Average Reward": round(avg_reward, 2)
        },
        header=True  # Write header only once
    )

    plot_rewards(rewards)

