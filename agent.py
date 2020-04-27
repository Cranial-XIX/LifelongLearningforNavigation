import numpy as np
import quadprog
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from torch.optim import *
from tqdm import tqdm


class Replay(data.Dataset):
    """
    A dataset wrapper used as a memory to store the data
    """
    def __init__(self):
        super(Replay, self).__init__()
        self.rb = []

    def __len__(self):
        return len(self.rb)

    def add(self, v, w, l, n_mem):
        i = 0
        while len(self.rb) < n_mem and i < v.shape[0]:
            self.rb.append((v[i], w[i], l[i]))
            i += 1

    def sample(self, n):
        batch = random.sample(self.rb, n)
        return map(torch.stack, zip(*batch))

    def reduce(self, m):
        self.rb = self.rb[:m]


class Model(nn.Module):
    def __init__(self, dim_input):
        super(Model, self).__init__()
        dim_hidden = 128
        self.base = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )
        self.v_head = nn.Linear(dim_hidden, 1)
        self.w_head = nn.Linear(dim_hidden, 1)

    def forward(self, x):
        h = self.base(x)
        v, w = self.v_head(h), self.w_head(h)
        v.clamp_(0, 2)
        w.clamp_(-1.4, 1.4)
        return v, w


class Agent(nn.Module):
    """
    using gradient episodic memory
    """
    def __init__(self,
                 dim_input,
                 cuda=True,
                 mem_budget=1000):
        super(Agent, self).__init__()
        self.task = -1
        self.memories = {} # per task replay
        self.n_mem = mem_budget
        self.tqdm = True

        self.net = Model(dim_input)
        self.cuda = cuda
        if cuda:
            self.net.cuda()
        self.create_optimizer()

    def save(self, path="./models/trained_agent.pt"):
        torch.save(self.net.state_dict(), path)

    def load(self, path="./models/trained_agent.pt"):
        self.net.load_state_dict(torch.load(path))

    def predict(self, x):
        with torch.no_grad():
            v, w = self.net(x)
        return v.item(), w.item()

    def count_params(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params

    def increment_task(self):
        self.task += 1
        self.memories[self.task] = Replay()
        self.create_optimizer()

    def create_optimizer(self):
        params = list(self.net.parameters())
        self.opt = Adam(params, lr=3e-4)

    def loss_fn(self, v, w, l):
        vv, ww = self.net(l)
        return F.mse_loss(vv, v) + F.mse_loss(ww, w)

    def grad_to_vector(self):
        vec = []
        for n, p in self.named_parameters():
            if 'weight' in n and p.grad is not None:
                vec.append(p.grad.data.view(-1).clone())
            else:
                vec.append(torch.zeros_like(p).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        pointer = 0
        for n, p in self.named_parameters():
            num_param = p.numel()
            if 'weight' in n and p.grad is not None:
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
            pointer += num_param

    def project2cone2(self, gradient, memories):
        memories_np = memories.cpu().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]

        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + 0.5
        v = quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        if self.cuda:
            x = torch.Tensor(x).view(-1, 1).cuda()
        return x

    def get_G(self):
        prev_grads = []
        for t, m in self.memories.items():
            if t < self.task:
                self.zero_grad()
                v, w, l = m.sample(min(64, self.n_mem))

                v = v.unsqueeze(-1)
                w = w.unsqueeze(-1)
                if self.cuda:
                    v = v.cuda(); w = w.cuda(); l = l.cuda()

                loss = self.loss_fn(v, w, l)
                loss.backward()
                prev_grads.append(self.grad_to_vector())
        return torch.stack(prev_grads)

    def test(self, dataloader):
        with torch.no_grad():
            L = 0
            for v, w, l in dataloader:
                v = v.unsqueeze(-1)
                w = w.unsqueeze(-1)
                if self.cuda:
                    v = v.cuda(); w = w.cuda(); l = l.cuda()
                loss = self.loss_fn(v, w, l)
                L += loss.item()
        print(f"[info] test loss is {L/len(dataloader)}")

    def learn(self, dataloader, n_batches=500, gem=True):
        curr_mem = self.memories[self.task]

        batch = 0
        done = False
        total_loss = 0

        if self.tqdm:
            pbar = tqdm(total=n_batches)

        while not done:
            for v, w, l in dataloader:
                batch += 1

                if len(curr_mem) < self.n_mem:
                    curr_mem.add(v, w, l, self.n_mem)

                v = v.unsqueeze(-1)
                w = w.unsqueeze(-1)
                if self.cuda:
                    v = v.cuda(); w = w.cuda(); l = l.cuda()

                if gem and self.task > 0:
                    G = self.get_G()

                self.opt.zero_grad()
                loss = self.loss_fn(v, w, l)
                loss.backward()
                total_loss += (loss.detach().cpu().item() - total_loss) * 1/batch

                if gem and self.task > 0:
                    curr_grad = self.grad_to_vector()
                    dotp = G.mm(curr_grad.view(-1, 1))
                    if (dotp < 0).sum() != 0:
                        new_grad = self.project2cone2(curr_grad, G)
                        self.vector_to_grad(new_grad)
                self.opt.step()

                if self.tqdm:
                    pbar.set_description("mse error: %7.4f" % (total_loss))
                    pbar.update(1)

                if batch >= n_batches:
                    done = True
                    break

        if self.tqdm:
            pbar.close()

        return total_loss
