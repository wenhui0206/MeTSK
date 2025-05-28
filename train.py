import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
import logging
from copy import deepcopy

from metsk.net.st_gcn_meta import CLS_Head, Model
from metsk.data.meta_data_utils import generate_episodes, get_hcp_aal, get_public_data, calc_populationA
from metsk.meta_learning import Meta

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Argument parser
parser = argparse.ArgumentParser(description='ST-GCN Hyper-parameters')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--start_test_epoch', type=int, default=20)
parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--W', type=int, default=128)
parser.add_argument('--TS', type=int, default=64)
parser.add_argument('--num_nodes', type=int, default=116)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--update_lr', type=float, default=0.01)
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--update_step', type=int, default=25)
parser.add_argument('--num_episodes', type=int, default=30)
parser.add_argument('--num_samples', type=int, default=8)
parser.add_argument('--num_src_tasks', type=int, default=1)
parser.add_argument('--ft', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--alpha', type=float, default=15)
parser.add_argument('--tau', type=float, default=30)
parser.add_argument('--warmup_step', type=int, default=20)
parser.add_argument('--multi_task', type=bool, default=False)
parser.add_argument('--nosrc', type=bool, default=False)
parser.add_argument('--fp_dis', type=str, default='../data_npz/ADHD_AAL.npz')
args = parser.parse_args()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s ===> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
for arg in vars(args):
    logging.info(f"{arg} = {getattr(args, arg)}")

# Load data and adjacency
data_src, labels_src = get_hcp_aal()
A_src = calc_populationA(data_src)
data_tgt_all, labels_tgt_all = get_public_data(args.fp_dis)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1211)
auc_folds, acc_folds = [], []

for train_ind, test_ind in kfold.split(data_tgt_all, labels_tgt_all):
    data_tgt = data_tgt_all[train_ind]
    labels_tgt = labels_tgt_all[train_ind]
    data_tgt_test = data_tgt_all[test_ind]
    labels_tgt_test = labels_tgt_all[test_ind]
    A_tgt = calc_populationA(data_tgt)

    net = Model(1, args.num_classes, args.num_nodes, None, True, device).to(device)
    headS = CLS_Head(64, args.num_classes, args.num_nodes, None, True, device, 'Source').to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(list(net.parameters()) + list(headS.parameters()), lr=args.meta_lr, weight_decay=0.001)

    meta_learner = Meta(net, optimizer, criterion, A_src, A_tgt, headS, None, args, device)
    test_tgt = torch.from_numpy(data_tgt_test).float().to(device)
    test_y_tgt = torch.from_numpy(labels_tgt_test).float().to(device)

    for epoch_i in range(1, args.epochs):
        kfold_inner = StratifiedKFold(n_splits=5, shuffle=True)
        headT = CLS_Head(64, args.num_classes, args.num_nodes, None, True, device, 'Target').to(device)

        for train_idx, val_idx in kfold_inner.split(data_tgt, labels_tgt):
            data_tgt_tr, labels_tgt_tr = data_tgt[train_idx], labels_tgt[train_idx]
            data_tgt_val, labels_tgt_val = data_tgt[val_idx], labels_tgt[val_idx]
            break

        batch_data_src, batch_y_src = generate_episodes(args.num_episodes, data_src, labels_src, args.num_samples, 2)
        batch_data_tgt, batch_y_tgt = generate_episodes(args.num_episodes, data_tgt_tr, labels_tgt_tr, args.num_samples, 2)
        batch_data_tgt_val, batch_y_tgt_val = generate_episodes(args.num_episodes, data_tgt_val, labels_tgt_val, args.num_samples, 2)

        batch_data_src = torch.from_numpy(batch_data_src).float().to(device)
        batch_y_src = torch.from_numpy(batch_y_src).float().to(device)
        batch_data_tgt = torch.from_numpy(batch_data_tgt).float().to(device)
        batch_y_tgt = torch.from_numpy(batch_y_tgt).float().to(device)
        batch_data_tgt_val = torch.from_numpy(batch_data_tgt_val).float().to(device)
        batch_y_tgt_val = torch.from_numpy(batch_y_tgt_val).float().to(device)

        meta_learner.headT = headT
        res = meta_learner(batch_data_src, batch_y_src, batch_data_tgt, batch_y_tgt, batch_data_tgt_val, batch_y_tgt_val, epoch_i)
        
    test_acc, test_auc = meta_learner.test_stgcn(net, meta_learner.fheadT, test_tgt, test_y_tgt[:, 0].long(), A_tgt, args.num_samples * 2)
    logging.info(f"Test ACC: {test_acc:.4f}, AUC: {test_auc:.4f}")

    auc_folds.append(test_auc)
    acc_folds.append(test_acc)
    del net, headS, headT

logging.info(f"CV Mean ACC: {np.mean(acc_folds):.4f}, Mean AUC: {np.mean(auc_folds):.4f}")
print('AUC', auc_folds, np.std(auc_folds))
print('ACC', acc_folds, np.std(acc_folds))
np.save(f'./evals/5f_auc_{args.model_name}.npy', np.array(auc_folds))
np.save(f'./evals/5f_acc_{args.model_name}.npy', np.array(acc_folds))
