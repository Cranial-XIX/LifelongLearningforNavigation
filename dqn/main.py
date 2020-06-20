import numpy as np
from tqdm import tqdm
from model import *
from torch.utils.data import Dataset

class OffPolicyQDataset(Dataset):
    def __init__(self, S, A, R, NS, I):
        self.S = S.astype(np.float32)
        self.A = A.astype(np.float32)
        self.R = R.astype(np.float32)
        self.NS= NS.astype(np.float32)
        self.I = I.astype(np.int)
        self.n  = S.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.S[i], self.A[i], self.R[i], self.NS[i], self.I[i]

def test_judgement(dqn, loader):
    fn_tp = 0.
    tp = 0.
    tp_fp = 0.
    for s, a, _, _, i in loader:
        i = torch.LongTensor(i)
        pred = dqn.judge_prediction(s, a, 0.05).cpu().squeeze()
        tp += (pred.eq(1).long() * i).sum().item()
        tp_fp += pred.eq(1).sum().item()
        fn_tp += i.eq(1).sum().item()
    p = tp/(tp_fp + 1e-4) + 1e-4
    r = tp/(fn_tp + 1e-4) + 1e-4
    print("precision {} recall {} f1 {}".format(p, r, 2*p*r/(p+r)))

def main():
    data = np.load("Q_data.npz", encoding="latin1", allow_pickle=True)
    S, A, R, NS, I = data['S'], data['A'], data['R'], data['N'], data['I']
    S = np.array([np.concatenate([x[0], np.array(x[1]).reshape(-1), np.array(x[2]).reshape(-1)]) for x in S])
    NS = np.array([np.concatenate([x[0], np.array(x[1]).reshape(-1), np.array(x[2]).reshape(-1)]) for x in NS])
    S = np.clip(S, 0, 2)
    NS = np.clip(NS, 0, 2)
    dim_state = S.shape[-1]
    dim_action = A.shape[-1]
    dataset = OffPolicyQDataset(S, A, R, NS, I)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True)

    dqn = DQN(dim_state, dim_action, 128)
    dqn.cuda()

    nepoch = 5
    pbar = tqdm(total=nepoch*len(loader))
    t = 0
    for epoch in range(nepoch):
        td_loss = 0.
        accuracy = 0.
        k = 0
        for s, a, r, ns, i in loader:
            t += 1
            k += 1
            pbar.update(1)
            td = dqn.update(s, a, r, ns)
            td_loss += td
            if t % 100 == 0:
                dqn.update_target()
            pbar.set_description("td error {}".format(td_loss/k))
        dqn.save()
    test_judgement(dqn, loader)

if __name__ == "__main__":
    main()



