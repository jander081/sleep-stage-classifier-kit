# ---- Imports: Data, ML, and Paths ----
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

# ---- Custom Dataset for loading spectrograms ----
class SleepSpectrogramDataset(Dataset):
    def __init__(self, df, root_dir):
        # DataFrame with patient, path, label info
        self.paths = df['path'].tolist()
        self.labels = df['stage'].tolist()
        self.patient_ids = df['patient_id'].tolist()
        self.root_dir = Path(root_dir)

        # Label encoding
        label_set = sorted(list(set(self.labels)))
        self.label_to_idx = {lab: idx for idx, lab in enumerate(label_set)}
        self.idx_to_label = {idx: lab for lab, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Full path includes patient folder
        full_path = self.root_dir / self.patient_ids[idx] / self.paths[idx]
        x = np.load(full_path)
        x = np.expand_dims(x, axis=0)
        x = torch.tensor(x, dtype=torch.float32)
        y = self.label_to_idx[self.labels[idx]]
        return x, y

# ---- Simple CNN (unchanged) ----
class SimpleSleepCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 32 * 14, 64)  # adjust to match your data!
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total > 0 else 0

def main():
    # ---- Configs ----
    index_csv = "../dataset/master_index.csv"
    root_dir = "../dataset"
    batch_size = 16
    epochs = 10
    lr = 1e-3
    test_ratio = 0.3   # 30% of patients for test

    # ---- Load and split by patient ----
    df = pd.read_csv(index_csv)
    unique_patients = sorted(df['patient_id'].unique())
    random.shuffle(unique_patients)
    n_test = max(1, int(len(unique_patients) * test_ratio))
    test_patients = unique_patients[:n_test]
    train_patients = unique_patients[n_test:]

    train_df = df[df['patient_id'].isin(train_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]

    print(f"Train patients: {train_patients}")
    print(f"Test patients: {test_patients}")
    print(f"Train epochs: {len(train_df)}, Test epochs: {len(test_df)}")

    # ---- Datasets and loaders ----
    train_dataset = SleepSpectrogramDataset(train_df, root_dir)
    test_dataset = SleepSpectrogramDataset(test_df, root_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_classes = len(set(df['stage']))
    model = SimpleSleepCNN(n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- Training loop ----
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_acc = eval_accuracy(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Test accuracy: {test_acc:.3f}")

    # ---- Save model weights ----
    Path("../models").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), "../models/sleep_cnn.pth")
    print("Training done, model saved to models/.")

if __name__ == "__main__":
    main()