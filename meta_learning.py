import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import random

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, model, optimizer, criterion, A_src, A_tgt, source_head, target_head, args, device):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.alpha = args.alpha
        self.tau = args.tau
        self.warmup_step = args.warmup_step
        self.args = args

        self.net = model
        self.meta_optim = optimizer
        self.criterion = criterion
        self.A_src = A_src
        self.A_tgt = A_tgt
        self.window_size = args.W
        self.TS = args.TS
        self.device = device
        self.headS = source_head
        self.headT = target_head
        self.fheadT = None

    def clip_grad_by_norm_(self, grad, max_norm):
        total_norm = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / len(grad)

    def get_diff_optimizer_and_functional_model(self, model, inner_opt):
        from higher import monkeypatch, optim
        fmodel = monkeypatch(
            model,
            self.device,
            copy_initial_weights=False,
            track_higher_grads=True
        )
        diffopt = optim.get_diff_optim(
            inner_opt,
            model.parameters(),
            fmodel=fmodel,
            device=self.device,
            override=None,
            track_higher_grads=True
        )
        return fmodel, diffopt

    def construct_input_stgcn(self, train_data, W, num_nodes):
        batch_size = train_data.size(0)
        train_data_batch = torch.zeros((batch_size, 1, W, num_nodes, 1))
        total_len = train_data.shape[2]
        for i in range(batch_size):
            r1 = random.randint(0, total_len - W)
            train_data_batch[i] = train_data[i, :, r1:r1+W, :, :]
        return train_data_batch.to(self.device)

    def construct_input_stgcn_con(self, train_data, W, num_nodes):
        batch_size = train_data.size(0)
        train_data_batch = torch.zeros((batch_size * 2, 1, W, num_nodes, 1))
        for i in range(batch_size):
            r1 = random.randint(0, train_data.shape[2] - W)
            train_data_batch[i] = train_data[i, :, r1:r1+W, :, :]
            r2 = random.randint(0, train_data.shape[2] - W)
            train_data_batch[i + batch_size] = train_data[i, :, r2:r2+W, :, :]
        return train_data_batch.to(self.device)

    def contrastive_loss(self, X, tau):
        batch_size = X.size(0)
        X1, X2 = X[:batch_size // 2], X[batch_size // 2:]
        inner_prod = torch.matmul(X1, X2.T / tau)
        diag = torch.diagonal(inner_prod, 0)
        nume = torch.exp(diag)
        denoms = torch.stack([torch.sum(torch.exp(inner_prod[i][torch.arange(inner_prod.size(1)) != i])) for i in range(inner_prod.size(0))])
        return -torch.log(nume / denoms).mean()

    def forward(self, x_src, y_src, x_tgt, y_tgt, x_tgt_val, y_tgt_val, epoch):
        num_episodes, num_samples, _, time_len, num_nodes, _ = x_src.size()
        losses_tgt, losses_src = 0.0, 0.0
        corrects = 0
        self.net.train()

        inner_opt = torch.optim.SGD(self.headT.parameters(), lr=self.update_lr)
        self.headT.train()

        for i in range(num_episodes):
            idx = np.random.permutation(num_samples)
            inputs_src = self.construct_input_stgcn_con(x_src[i], self.window_size, num_nodes)
            labels_src = y_src[i, idx, :]
            self.meta_optim.zero_grad()

            if epoch < self.warmup_step:
                featureS = self.net(inputs_src, self.A_src)
                logits_src = self.headS(featureS, self.A_src)
                loss_con = self.contrastive_loss(logits_src, self.tau)
                losses_src += loss_con
                loss_con.backward()
                self.meta_optim.step()
                continue

            train_tgt = self.construct_input_stgcn(x_tgt[i], self.window_size, num_nodes)[idx]
            val_tgt = self.construct_input_stgcn(x_tgt_val[i], self.window_size, num_nodes)[idx]
            fheadT, diffopt = self.get_diff_optimizer_and_functional_model(self.headT, inner_opt)

            labels_tgt_tr = y_tgt[i, idx, :]
            labels_tgt_val = y_tgt_val[i, idx, :]

            with torch.no_grad():
                featureT = self.net(train_tgt, self.A_tgt)

            logits = None
            for _ in range(self.update_step):
                logits = fheadT(featureT, self.A_tgt)
                loss = self.criterion(logits, labels_tgt_tr)
                diffopt.step(loss)

            featureS = self.net(inputs_src, self.A_src)
            featureT_val = self.net(val_tgt, self.A_tgt)

            logits_src = self.headS(featureS, self.A_src)
            logits_tgt = fheadT(featureT_val, self.A_tgt)

            if self.args.nosrc:
                loss_q = self.criterion(logits_tgt, labels_tgt_val)
            else:
                loss_meta_val = self.criterion(logits_tgt, labels_tgt_val)
                losses_tgt += loss_meta_val.item()
                loss_q = self.contrastive_loss(logits_src, self.tau) + self.alpha * loss_meta_val

            losses_src += loss_q - loss_meta_val.item()
            self.fheadT = fheadT

            with torch.no_grad():
                if logits is not None:
                    pred_t = logits_tgt.cpu().numpy() > 0.5
                    corrects += np.sum(pred_t[:, 0] == labels_tgt_val[:, 0].cpu().numpy())

            loss_q.backward()
            self.meta_optim.step()
            self.headT.load_state_dict(fheadT.state_dict(), strict=False)
            del diffopt
            torch.cuda.empty_cache()

        if epoch < self.warmup_step:
            return losses_src.item() / num_episodes, 0.0, 0.0, 0.0

        accs = corrects / (num_samples * num_episodes)
        return losses_src.item() / num_episodes, losses_tgt / num_episodes, 0.0, accs

    def meta_test(self, x_src, y_src, x_tgt, y_tgt, A_test, batch_size):
        return self.test_stgcn(self.net, self.headT, x_tgt, y_tgt[:, 0].long(), A_test, batch_size)
