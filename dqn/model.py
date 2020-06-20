import copy
import quadprog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam

class DQN(nn.Module):
    def __init__(self,
                 dim_state,
                 dim_action,
                 dim_hidden):
        super(DQN, self).__init__()
        self.batch_size = 128
        self.device = "cuda"

        self.Q = nn.Sequential(
            nn.Linear(dim_state+dim_action, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1))

        self.Q_target = copy.deepcopy(self.Q)
        self.Q_target.requires_grad = False

        self.gamma = 0.95
        self.max_q = None
        self.opt = Adam(self.Q.parameters(), lr=1e-3)

    def forward(self, s, a, use_target=False):
        x = torch.cat([s, a], -1)
        q = self.Q_target if use_target else self.Q
        return q(x)

    def update_target(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def judge_prediction(self, state, action, threshold=0.1):
        s  = torch.FloatTensor(state).to(self.device)
        a  = torch.FloatTensor(action).to(self.device)
        q = self.forward(s, a)
        return q.gt(threshold).long()

    def update(self, state, action, reward, nstate):
        s  = torch.FloatTensor(state).to(self.device)
        a  = torch.FloatTensor(action).to(self.device)
        r  = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(nstate).to(self.device)

        q_sa = self.forward(s, a)

        na = a.unsqueeze(1).repeat(1, 100, 1)
        na = na + torch.randn_like(na) * na.abs().max() * 0.1
        ns = ns.unsqueeze(1).expand(-1, 100, -1)
        nq_sa = self.forward(ns, na, True)

        if self.max_q is None:
            self.max_q = q_sa.max().item()
        else:
            self.max_q = max(q_sa.max().item(), self.max_q)

        self.opt.zero_grad()
        td_error = F.mse_loss(q_sa, r + self.gamma * nq_sa.max(1)[0])
        td_error.backward()
        self.opt.step()
        return td_error.item()

    def save(self, path='./dqn.pt'):
        torch.save(self.state_dict(), path)

    def load(self, path='./dqn.pt'):
        self.load_state_dict(torch.load(path))
