import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
from net.st_gcn_ssl import Model
from net.losses import BBFC_loss, QRcost
from metsk.data.meta_data_utils import calc_populationA, get_hcp_aal

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Argument parser
parser = argparse.ArgumentParser(description='ST-GCN Contrastive Learning')
parser.add_argument('--epochs', type=int, default=30001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--W', type=int, default=128)
parser.add_argument('--num_nodes', type=int, default=116)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--LR', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--QR', type=float, default=0.0)
parser.add_argument('--tau', type=float, default=30.0)
args = parser.parse_args()

# Model and training params
W = args.W
batch_size = args.batch_size

# Contrastive loss function
def contrastive_loss(X, tau):
    X1 = X[:batch_size//2]
    X2 = X[batch_size//2:]
    inner_prod = torch.matmul(X1, X2.T / tau)
    diag = torch.diagonal(inner_prod, 0)
    nume = torch.exp(diag)
    denoms = torch.zeros_like(nume)
    for i in range(nume.shape[0]):
        denoms[i] = torch.sum(torch.exp(inner_prod[inner_prod != diag[i]]))
    loss = -torch.log(nume / denoms).mean()
    return loss

# Load data
data_all, label_all = get_hcp_aal()
A = calc_populationA(data_all)

# Initialize model
net = Model(1, args.num_classes, args.num_nodes, None, True, device).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.LR, weight_decay=0.001)

# Training loop
training_loss = 0.0
for epoch in range(args.epochs):
    net.train()
    idx_batch = np.random.permutation(data_all.shape[0])[:batch_size]
    train_data_batch = np.zeros((batch_size, 1, W, args.num_nodes, 1))
    k = batch_size // 2
    for i in range(k):
        r1 = random.randint(0, data_all.shape[2] - W)
        train_data_batch[i] = data_all[idx_batch[i], :, r1:r1+W, :, :]
        r2 = random.randint(0, data_all.shape[2] - W)
        train_data_batch[i + k] = data_all[idx_batch[i], :, r2:r2+W, :, :]

    inputs = torch.from_numpy(train_data_batch).float().to(device)
    optimizer.zero_grad()
    outputs = net(inputs, A)
    loss = contrastive_loss(outputs, args.tau)
    loss.backward()
    optimizer.step()
    training_loss += loss.item()

    if epoch % 1000 == 0:
        print(f"[Epoch {epoch+1}] Training Loss: {training_loss/1000:.6f}")
        training_loss = 0.0

        if epoch > 15000:
            torch.save(net.state_dict(), f"../meta_saved_models/hcp_aal_pret_con_{epoch}.pth")
