import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from net.st_gcn import Model
from net.losses import BBFC_loss, QRcost
from metsk.data.meta_data_utils import get_public_data

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Argument parser
parser = argparse.ArgumentParser(description='ST-GCN Hyper-parameters')
parser.add_argument('--epochs', type=int, default=4001)
parser.add_argument('--start_test_epoch', type=int, default=1500)
parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--nfold', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--W', type=int, default=128)
parser.add_argument('--TS', type=int, default=64)
parser.add_argument('--num_nodes', type=int, default=116)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--LR', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=0.0)
parser.add_argument('--QR', type=float, default=0)
parser.add_argument('--finetune', type=bool, default=False)
parser.add_argument('--load_model', type=str, default='hcp_pretrain.pth')
parser.add_argument('--fp_dis', type=str, default="../data_npz/ADHD_AAL.npz")
args = parser.parse_args()

# Load data
data_all, label_all = get_public_data(args.fp_dis)

acc_iters, auc_iters = [], []

for _ in range(args.iterations):
    kfold = StratifiedKFold(n_splits=args.nfold, shuffle=True, random_state=1211)
    acc_folds, auc_folds = [], []

    for train_ind, test_ind in kfold.split(data_all, label_all):
        train_data = data_all[train_ind]
        train_label = label_all[train_ind]
        test_data = data_all[test_ind]
        test_label = label_all[test_ind]

        net = Model(1, args.num_classes, args.num_nodes, None, True, device).to(device)

        if args.finetune:
            state_dict = torch.load(args.load_model)
            for key in list(state_dict.keys()):
                if key.startswith('fcn.') or key.startswith('st_gcn_networks.3') or key.startswith('edge_importance'):
                    del state_dict[key]
            net.load_state_dict(state_dict, strict=False)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=args.LR, weight_decay=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-5)

        # Build population adjacency
        sequence_all = np.concatenate([train_data[i].squeeze().T for i in range(train_data.shape[0])], axis=1)
        A = np.corrcoef(sequence_all)
        A[~np.isfinite(A)] = 0

        best_acc, best_auc = 0.0, 0.0

        for epoch in range(args.epochs):
            net.train()
            idx_batch = np.random.choice(train_data.shape[0], args.batch_size, replace=False)
            r1 = random.randint(0, train_data.shape[2] - args.W)

            train_batch = np.array([train_data[i, :, r1:r1+args.W, :, :] for i in idx_batch])
            label_batch = train_label[idx_batch]

            inputs = torch.from_numpy(train_batch).float().to(device)
            targets = torch.from_numpy(label_batch).float().to(device)

            optimizer.zero_grad()
            outputs = net(inputs, A)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            scheduler.step(loss)

            if epoch % 100 == 0 and epoch > args.start_test_epoch:
                net.eval()
                prediction = np.zeros(test_data.shape[0])
                voter = np.zeros(test_data.shape[0])

                for _ in range(args.TS):
                    idx = np.random.permutation(test_data.shape[0])
                    for k in range(test_data.shape[0] // args.batch_size):
                        batch_idx = idx[k * args.batch_size: (k + 1) * args.batch_size]
                        test_batch = np.array([
                            test_data[i, :, random.randint(0, test_data.shape[2] - args.W):][:, :, :args.W, :, :] for i in batch_idx
                        ])
                        outputs = net(torch.from_numpy(test_batch).float().to(device), A).detach().cpu().numpy()
                        prediction[batch_idx] += outputs[:, 0]
                        voter[batch_idx] += 1

                prediction /= voter
                pred_label = (prediction > 0.5).astype(int)
                acc = np.mean(pred_label == test_label[:, 0])
                auc = roc_auc_score(test_label[:, 0], prediction)


        acc_folds.append(acc)
        auc_folds.append(auc)

    acc_iters.append(acc_folds)
    auc_iters.append(auc_folds)

print("Final Results")
print("ACC", np.mean(acc_iters), np.std(acc_iters))
print("AUC", np.mean(auc_iters), np.std(auc_iters))
