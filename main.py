import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from agent import *

def make_data():
    data = []
    for task in range(1, 2+1):
        csvf = f'./data/task{task}-lidar.csv'
        with open (csvf, 'r') as csvfile:
            reader = csv.reader(csvfile)
            T0, LIDAR = [], []
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                t = float(line[0])
                lidar = [float(x) for x in line[14][1:-1].split(",")]
                lidar = [(min(x, 10)-5)/5 for x in lidar]
                T0.append(t)
                LIDAR.append(lidar)

        csvf = f'./data/task{task}-cmd.csv'
        with open (csvf, 'r') as csvfile:
            reader = csv.reader(csvfile)
            T, V, W = [], [], []
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                t = line[0]
                v = line[2]
                w = line[8]
                t, v, w = map(float, [t, v, w])
                T.append(t)
                V.append(v)
                W.append(w)

        L = []
        i=0
        for j, (t, v, w) in enumerate(zip(T, V, W)):
            while T0[i] < t:
                i += 1
            L.append(LIDAR[i-1])
        V, W, L = map(np.array, [V, W, L])
        data.append((V, W, L))
    np.savez("./data/lifelong", data=data)

class NavDataset(Dataset):
    def __init__(self, V, W, L):
        self.V = V.astype(np.float32)
        self.W = W.astype(np.float32)
        self.L = L.astype(np.float32)
        self.n = V.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        v = self.V[idx]
        w = self.W[idx]
        l = self.L[idx]
        return v, w, l

def lifelong():
    data = np.load("./data/lifelong.npz", allow_pickle=True)['data']

    V1, W1, L1 = data[0]
    V2, W2, L2 = data[1]

    batch_size = 64
    loader1 = torch.utils.data.DataLoader(NavDataset(V1, W1, L1), batch_size=batch_size, num_workers=4, shuffle=True)
    loader2 = torch.utils.data.DataLoader(NavDataset(V2, W2, L2), batch_size=batch_size, num_workers=4, shuffle=True)

    agent = Agent(dim_input=L1.shape[1])

    print("[info] learning task 1 ...")
    agent.increment_task()
    agent.learn(loader1)
    agent.test(loader1)

    print("[info] learning task 2 ...")
    agent.increment_task()
    agent.learn(loader2, gem=True)
    agent.test(loader2)
    agent.test(loader1)

    agent.save()

def predict_example():
    agent = Agent(dim_input=720, cuda=False) # set cuda to False for prediction
    agent.load()

    lidar = torch.randn(720).clamp_(-1, 1).reshape(1, -1) # size [1, 720]
    v, w = agent.predict(lidar) # v, w are numbers
    print(v, w)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lifelong Robotics')
    parser.add_argument('--mode', help='which mode to run', choices=['make-data', 'lifelong-learn', 'predict-example'])
    if args.mode == 'make-data':
        make_data()
    elif args.mode == 'lifelong-learn':
        lifelong()
    elif args.mode == 'predict-example':
        predict_example()
    else:
        raise NotImplementedError
