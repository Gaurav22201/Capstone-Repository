import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width // 2)
        self.l4 = nn.Linear(net_width // 2, action_dim)
        
        self.ln1 = nn.LayerNorm(net_width)
        self.ln2 = nn.LayerNorm(net_width)
        self.ln3 = nn.LayerNorm(net_width // 2)
        
        self.maxaction = maxaction

    def forward(self, state):
        a = F.relu(self.ln1(self.l1(state)))
        a = F.relu(self.ln2(self.l2(a)))
        a = F.relu(self.ln3(self.l3(a)))
        a = torch.tanh(self.l4(a)) * self.maxaction
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width // 2)
        self.l4 = nn.Linear(net_width // 2, 1)
        
        self.ln1 = nn.LayerNorm(net_width)
        self.ln2 = nn.LayerNorm(net_width)
        self.ln3 = nn.LayerNorm(net_width // 2)

        # Q2 architecture
        self.l5 = nn.Linear(state_dim + action_dim, net_width)
        self.l6 = nn.Linear(net_width, net_width)
        self.l7 = nn.Linear(net_width, net_width // 2)
        self.l8 = nn.Linear(net_width // 2, 1)
        
        self.ln4 = nn.LayerNorm(net_width)
        self.ln5 = nn.LayerNorm(net_width)
        self.ln6 = nn.LayerNorm(net_width // 2)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.ln1(self.l1(sa)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = F.relu(self.ln3(self.l3(q1)))
        q1 = self.l4(q1)

        # Q2
        q2 = F.relu(self.ln4(self.l5(sa)))
        q2 = F.relu(self.ln5(self.l6(q2)))
        q2 = F.relu(self.ln6(self.l7(q2)))
        q2 = self.l8(q2)
        
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.ln1(self.l1(sa)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = F.relu(self.ln3(self.l3(q1)))
        q1 = self.l4(q1)
        return q1

class ReplayBuffer:
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

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)
        self.r[self.ptr] = torch.tensor([[float(r)]], dtype=torch.float, device=self.dvc)
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = bool(dw)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)


#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')