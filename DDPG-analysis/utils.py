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


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, net_width // 2)
        self.l4 = nn.Linear(net_width // 2, 1)
        
        self.ln1 = nn.LayerNorm(net_width)
        self.ln2 = nn.LayerNorm(net_width)
        self.ln3 = nn.LayerNorm(net_width // 2)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.ln1(self.l1(sa)))
        q = F.relu(self.ln2(self.l2(q)))
        q = F.relu(self.ln3(self.l3(q)))
        q = self.l4(q)
        return q

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