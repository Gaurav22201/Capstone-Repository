import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import Actor, Critic, ReplayBuffer

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, max_action):
        super(GaussianPolicy, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_width),
            nn.LayerNorm(net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.LayerNorm(net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width // 2),
            nn.LayerNorm(net_width // 2),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(net_width // 2, action_dim)
        self.log_std = nn.Linear(net_width // 2, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        features = self.net(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action * self.max_action, log_prob

class SAC_agent:
    def __init__(self, **kwargs):
        # Initialize hyperparameters
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.alpha = 0.2  # Temperature parameter
        self.target_entropy = -torch.prod(torch.Tensor([self.action_dim]).to(self.dvc)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.dvc)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.a_lr)
        
        # Initialize actor (policy) network
        self.actor = GaussianPolicy(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        
        # Initialize critic networks
        self.critic1 = Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        self.critic2 = Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.c_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.c_lr)
        
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), dvc=self.dvc)
        self.total_it = 0
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.dvc)
            if deterministic:
                mean, _ = self.actor(state)
                return torch.tanh(mean).cpu().data.numpy().flatten() * self.max_action
            else:
                action, _ = self.actor.sample(state)
                return action.cpu().data.numpy().flatten()
    
    def train(self):
        self.total_it += 1
        
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # Update critics
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (~done) * self.gamma * (target_Q - self.alpha * next_log_pi)
        
        # Critic 1 loss
        current_Q1 = self.critic1(state, action)
        critic1_loss = F.mse_loss(current_Q1, target_Q)
        
        # Critic 2 loss
        current_Q2 = self.critic2(state, action)
        critic2_loss = F.mse_loss(current_Q2, target_Q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        action_pi, log_pi = self.actor.sample(state)
        Q1_pi = self.critic1(state, action_pi)
        Q2_pi = self.critic2(state, action_pi)
        Q_pi = torch.min(Q1_pi, Q2_pi)
        
        actor_loss = (self.alpha * log_pi - Q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature parameter alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, env_name, timestep):
        torch.save(self.actor.state_dict(), f"./model/{env_name}_actor{timestep}.pth")
        torch.save(self.critic1.state_dict(), f"./model/{env_name}_critic1_{timestep}.pth")
        torch.save(self.critic2.state_dict(), f"./model/{env_name}_critic2_{timestep}.pth")
    
    def load(self, env_name, timestep):
        self.actor.load_state_dict(torch.load(f"./model/{env_name}_actor{timestep}.pth", map_location=self.dvc))
        self.critic1.load_state_dict(torch.load(f"./model/{env_name}_critic1_{timestep}.pth", map_location=self.dvc))
        self.critic2.load_state_dict(torch.load(f"./model/{env_name}_critic2_{timestep}.pth", map_location=self.dvc))
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2) 