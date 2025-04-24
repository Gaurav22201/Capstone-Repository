import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import copy

class PPO_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action):
        super(PPO_Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width // 2)
        self.mean = nn.Linear(net_width // 2, action_dim)
        self.log_std = nn.Linear(net_width // 2, action_dim)
        
        self.ln1 = nn.LayerNorm(net_width)
        self.ln2 = nn.LayerNorm(net_width)
        self.ln3 = nn.LayerNorm(net_width // 2)
        
        self.max_action = max_action
        
    def forward(self, state):
        a = F.relu(self.ln1(self.l1(state)))
        a = F.relu(self.ln2(self.l2(a)))
        a = F.relu(self.ln3(self.l3(a)))
        
        mean = torch.tanh(self.mean(a)) * self.max_action
        log_std = torch.clamp(self.log_std(a), -20, 2)
        std = log_std.exp()
        
        return mean, std

    def get_dist(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        return dist
    
    def sample(self, state):
        dist = self.get_dist(state)
        action = dist.rsample()  # Use reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def evaluate(self, state, action):
        dist = self.get_dist(state)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().mean()
        return log_prob, entropy

class PPO_Critic(nn.Module):
    def __init__(self, state_dim, net_width):
        super(PPO_Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width // 2)
        self.l4 = nn.Linear(net_width // 2, 1)
        
        self.ln1 = nn.LayerNorm(net_width)
        self.ln2 = nn.LayerNorm(net_width)
        self.ln3 = nn.LayerNorm(net_width // 2)
        
    def forward(self, state):
        v = F.relu(self.ln1(self.l1(state)))
        v = F.relu(self.ln2(self.l2(v)))
        v = F.relu(self.ln3(self.l3(v)))
        v = self.l4(v)
        return v

class PPO_Buffer:
    def __init__(self, state_dim, action_dim, size, gamma=0.99, gae_lambda=0.95, device='cpu'):
        self.size = size
        self.count = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Preallocate buffers
        self.states = torch.zeros((size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminals = torch.zeros(size, dtype=torch.bool, device=device)
        
    def add(self, state, action, reward, value, log_prob, terminal):
        idx = self.count % self.size
        
        self.states[idx] = torch.as_tensor(state, device=self.device)
        self.actions[idx] = torch.as_tensor(action, device=self.device)
        self.rewards[idx] = torch.as_tensor(reward, device=self.device)
        self.values[idx] = torch.as_tensor(value, device=self.device).reshape(-1)
        self.log_probs[idx] = torch.as_tensor(log_prob, device=self.device)
        self.terminals[idx] = terminal
        
        self.count += 1
        
    def compute_advantages(self, last_value):
        size = min(self.count, self.size)
        
        # GAE calculation
        advantages = torch.zeros(size, device=self.device)
        last_advantage = 0
        last_value = last_value
        
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.terminals[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.terminals[t])
                
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
            
        self.returns = advantages + self.values[:size]
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
    def get(self):
        size = min(self.count, self.size)
        return (
            self.states[:size],
            self.actions[:size],
            self.log_probs[:size],
            self.returns[:size],
            self.advantages[:size]
        )
        
    def clear(self):
        self.count = 0

class PPO_agent():
    def __init__(self, **kwargs):
        # Initialize hyperparameters
        self.__dict__.update(kwargs)
        
        # PPO specific parameters
        self.clip_param = 0.2
        self.ppo_epochs = 10
        self.num_mini_batches = 4
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        
        # Initialize actor
        self.actor = PPO_Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a_lr)
        
        # Initialize critic
        self.critic = PPO_Critic(self.state_dim, self.net_width).to(self.dvc)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.c_lr)
        
        # Initialize buffer
        self.buffer = PPO_Buffer(
            self.state_dim,
            self.action_dim,
            size=2048,  # PPO buffer size
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.dvc
        )
        
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.dvc)
            if deterministic:
                mean, _ = self.actor(state)
                return mean.cpu().numpy()[0]
            else:
                action, log_prob = self.actor.sample(state)
                value = self.critic(state)
                return (
                    action.cpu().numpy()[0],
                    value.cpu().numpy()[0],
                    log_prob.cpu().numpy()
                )
    
    def train(self):
        # Get batch data
        states, actions, old_log_probs, returns, advantages = self.buffer.get()
        batch_size = len(states)
        mini_batch_size = batch_size // self.num_mini_batches
        
        # PPO update for K epochs
        for _ in range(self.ppo_epochs):
            # Generate random permutation of indices
            indices = torch.randperm(batch_size)
            
            # Update in mini-batches
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]
                
                # Get mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Get current log probs and values
                new_log_probs, entropy = self.actor.evaluate(mb_states, mb_actions)
                values = self.critic(mb_states).squeeze()
                
                # Calculate ratios and surrogate objectives
                ratios = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                
                # Calculate losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        
        # Clear buffer after updates
        self.buffer.clear()
    
    def save(self, env_name, timestep):
        torch.save(self.actor.state_dict(), f"./model/{env_name}_ppo_actor_{timestep}.pth")
        torch.save(self.critic.state_dict(), f"./model/{env_name}_ppo_critic_{timestep}.pth")
    
    def load(self, env_name, timestep):
        self.actor.load_state_dict(torch.load(f"./model/{env_name}_ppo_actor_{timestep}.pth"))
        self.critic.load_state_dict(torch.load(f"./model/{env_name}_ppo_critic_{timestep}.pth")) 