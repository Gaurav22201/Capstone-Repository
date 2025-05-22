import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch
import time
from tqdm import tqdm

# Define environment and algorithm setup
envs = {
    'CartPole-v1': 'discrete',
    'Pendulum-v1': 'continuous',
    'MountainCarContinuous-v0': 'continuous'
}

algorithms = {
    'DQN': DQN,
    'REINFORCE': None,
    'PPO': PPO,
    'A2C': A2C
}

# Custom REINFORCE implementation
class REINFORCEAgent:
    def __init__(self, env, learning_rate=0.001, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else 1),
            torch.nn.Softmax(dim=-1) if isinstance(env.action_space, gym.spaces.Discrete) else torch.nn.Identity()
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy(state)
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            return action.item(), m.log_prob(action)
        else:
            action = probs.detach().numpy()
            return action, None

    def train(self, episodes=500, max_timesteps=200):
        all_rewards = []
        for ep in tqdm(range(episodes), desc="Training REINFORCE"):
            state, _ = self.env.reset()
            rewards = []
            log_probs = []
            for _ in range(max_timesteps):
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                rewards.append(reward)
                if log_prob is not None:
                    log_probs.append(log_prob)
                state = next_state
                if terminated or truncated:
                    break
            all_rewards.append(sum(rewards))
            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.tensor(discounted_rewards)
            if log_probs:
                loss = -torch.stack(log_probs) * discounted_rewards
                loss = loss.sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return all_rewards

# Train and evaluate
results = {}
training_times = {}
EPISODES = 500

for env_name, env_type in envs.items():
    for algo_name, algo_class in algorithms.items():
        print(f"Training {algo_name} on {env_name}")
        env = Monitor(gym.make(env_name))
        if algo_name == 'DQN' and env_type != 'discrete':
            print(f"Skipping {algo_name} on {env_name} (unsupported action space)")
            continue

        if env_name == 'CartPole-v1':
            timesteps = 100000
        else:
            timesteps = 300000

        start_time = time.time()

        if algo_name == 'REINFORCE':
            agent = REINFORCEAgent(env)
            rewards = agent.train(episodes=EPISODES)
        else:
            model = algo_class('MlpPolicy', env, verbose=0, learning_rate=0.0001)
            model.learn(total_timesteps=timesteps, progress_bar=True)
            eval_env = Monitor(gym.make(env_name))
            rewards, _ = evaluate_policy(model, eval_env, n_eval_episodes=EPISODES, return_episode_rewards=True)

        duration = time.time() - start_time
        training_times[(env_name, algo_name)] = duration / 60  # in minutes

        results[(env_name, algo_name)] = rewards

# Prepare results DataFrame
data = []
for (env, algo), rewards in results.items():
    for r in rewards:
        data.append((env, algo, r))
df = pd.DataFrame(data, columns=['Environment', 'Algorithm', 'Reward'])

# Plot Final Average Rewards
plt.figure(figsize=(10,6))
sns.barplot(data=df, x='Environment', y='Reward', hue='Algorithm', errorbar='sd')
plt.title('Final Average Rewards Across Environments')
plt.ylabel('Average Reward')
plt.grid(True)
plt.legend()
plt.show()

# Plot Reward vs Episodes (smoothed)
fig, axs = plt.subplots(1, len(envs), figsize=(20,5))
for i, env in enumerate(envs.keys()):
    for algo in algorithms.keys():
        if (env, algo) in results:
            rewards = results[(env, algo)]
            rewards = pd.Series(rewards).rolling(10).mean()
            axs[i].plot(rewards, label=algo)
    axs[i].set_title(env)
    axs[i].set_xlabel('Episodes')
    axs[i].set_ylabel('Smoothed Reward')
    axs[i].legend()
    axs[i].grid(True)
plt.tight_layout()
plt.show()

# Reward Mean ± Std plot
reward_summary = df.groupby(['Environment', 'Algorithm'])['Reward'].agg(['mean', 'std']).reset_index()
plt.figure(figsize=(12,6))
sns.barplot(data=reward_summary, x='Environment', y='mean', hue='Algorithm', errorbar=None)
for idx, row in reward_summary.iterrows():
    plt.errorbar(x=idx//len(algorithms), y=row['mean'], yerr=row['std'], fmt='none', c='black')
plt.title('Reward Mean ± Std Across Environments')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()
plt.show()

# Plot Training Times
training_df = pd.DataFrame(training_times.items(), columns=['(Environment, Algorithm)', 'Minutes'])
training_df[['Environment', 'Algorithm']] = pd.DataFrame(training_df['(Environment, Algorithm)'].tolist(), index=training_df.index)
plt.figure(figsize=(12,6))
sns.barplot(x='Environment', y='Minutes', hue='Algorithm', data=training_df)
plt.title('Training Time per Algorithm and Environment (minutes)')
plt.grid(True)
plt.show()