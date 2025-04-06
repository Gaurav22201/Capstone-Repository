from utils import Actor, Critic
import torch.nn.functional as F
import numpy as np
import torch
import copy


class TD3_agent():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent
        self.__dict__.update(kwargs)
        self.tau = 0.005  # Target network update rate
        self.policy_noise = 0.2  # Noise added to target policy during critic update
        self.noise_clip = 0.5  # Range to clip target policy noise
        self.policy_freq = 2  # Frequency of delayed policy updates

        # Initialize actor networks
        self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        # Initialize critic networks
        self.critic = Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        self.total_it = 0  # Total training iterations
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), dvc=self.dvc)
        
    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)
            a = self.actor(state).cpu().numpy()[0]
            if deterministic:
                return a
            else:
                noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
                return (a + noise).clip(-self.max_action, self.max_action)

    def train(self):
        self.total_it += 1

        # Sample replay buffer 
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(s_next) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(s_next, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(s, a)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()

            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, EnvName, timestep):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, timestep))
        torch.save(self.critic.state_dict(), "./model/{}_critic{}.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
        self.critic.load_state_dict(torch.load("./model/{}_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, s, a, r, s_next, terminated):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind] 