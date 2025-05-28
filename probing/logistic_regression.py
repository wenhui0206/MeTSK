import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from input_utils import call_input


# Logistic Regression Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))

# Training function
def train_model(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data).view(-1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            output = model(data).view(-1)
            predictions.extend(output.tolist())
            targets.extend(target.tolist())
    return roc_auc_score(targets, predictions)

def main():
    X, y = call_input()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1211)

    fold_aucs = []
    for train_idx, test_idx in kf.split(X, y):
        model = LogisticRegression(input_size=X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor[train_idx], y_tensor[train_idx]),
            batch_size=6, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tensor[test_idx], y_tensor[test_idx]),
            batch_size=6)

        for epoch in range(5):
            loss = train_model(model, train_loader, criterion, optimizer)
            print(f"Epoch {epoch} - Loss: {loss:.4f}")

        auc = evaluate_model(model, test_loader)
        fold_aucs.append(auc)
        print(f"Fold AUC: {auc:.4f}")

    print(f"\nMean AUC: {np.mean(fold_aucs):.4f}, Std AUC: {np.std(fold_aucs):.4f}")

if __name__ == "__main__":
    main()
