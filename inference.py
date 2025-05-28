import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
from metsk.net.st_gcn_meta import Model, CLS_Head
from net.losses import BBFC_loss, QRcost
from metsk.data.meta_data_utils import get_AD_data, calc_populationA, get_pd_data
from metsk.meta_learning import Meta

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Argument parser
parser = argparse.ArgumentParser(description='ST-GCN Hyper-parameters')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--start_test_epoch', type=int, default=20)
parser.add_argument('--W', type=int, default=128)
parser.add_argument('--TS', type=int, default=64)
parser.add_argument('--num_nodes', type=int, default=116)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--update_lr', type=float, default=0.01)
parser.add_argument('--meta_lr', type=float, default=0.001)
parser.add_argument('--update_step', type=int, default=10)
parser.add_argument('--update_step_test', type=int, default=10)
parser.add_argument('--num_episodes', type=int, default=20)
parser.add_argument('--num_samples', type=int, default=8)
parser.add_argument('--num_src_tasks', type=int, default=1)
parser.add_argument('--ft', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--alpha', type=float, default=30.0)
parser.add_argument('--tau', type=float, default=30.0)
parser.add_argument('--warmup_step', type=int, default=30)
args = parser.parse_args()

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s ===> %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
for arg in vars(args):
    logging.info(f"{arg} = {getattr(args, arg)}")

# Load data and adjacency
X, y = get_pd_data(filename="PD_neurocon_AAL.npz")
A = calc_populationA(X)
A[~np.isfinite(A)] = 0

# Build model
net = Model(1, args.num_classes, args.num_nodes, None, True, device).to(device)
headT = CLS_Head(64, args.num_classes, args.num_nodes, None, True, device, 'Target').to(device)
headS = CLS_Head(64, args.num_classes, args.num_nodes, None, True, device, 'Source').to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.meta_lr, weight_decay=0.001)

# Load pre-trained weights
net.load_state_dict(torch.load('../meta_saved_models/meta_model_meta_abide296_a7_w15.pth', map_location='cpu'), strict=False)
headT.load_state_dict(torch.load('../meta_saved_models/target_head_meta_abide296_a7_w15.pth', map_location='cpu'), strict=False)

# Run meta-learner
meta_learner = Meta(net, optimizer, criterion, A, A, headS, headT, args, device)
X_tensor = torch.from_numpy(X).float().to(device)
y_tensor = torch.from_numpy(y[:, 0]).long().to(device)

# Extract features and evaluate
graph_model_feat, head_feat, fc_feat = meta_learner.extract_features(net, headT, X_tensor, A, args.num_samples * 2)

np.savez("metsk_feat.npz", graph_model=graph_model_feat)