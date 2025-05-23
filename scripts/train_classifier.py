# scripts/train_classifier.py
"""
Sleep Stage Classification Training Script

This script trains a simple convolutional neural network (CNN) to classify sleep stages 
based on spectrogram data extracted from EEG or related signals. The input data consists 
of spectrograms saved as .npy files (NumPy arrays), each representing one epoch (time segment) 
of sleep data.

Data Structure:
- A CSV file (index_csv) is used to index the dataset. It contains at least two columns:
  1. 'path': relative file paths to the .npy spectrogram files for each epoch.
  2. 'stage': the sleep stage label for each epoch (e.g., 'Wake', 'N1', 'N2', 'REM', etc.).
- The root_dir is the base directory where these spectrogram files are stored.

Training Pipeline:
- The dataset is loaded and wrapped in a PyTorch Dataset class to enable easy batching.
- A simple CNN with two convolutional layers and max pooling is defined to extract features.
- The network outputs logits for each sleep stage class, which are trained using cross-entropy loss.
- The training loop iterates over the dataset for a fixed number of epochs.
- After training, the model weights are saved to disk.

Note:
- This script does not explicitly split data into train/test sets; it assumes the CSV 
  provided is for training. For evaluation, a separate CSV and dataset loader should be used.
- Labels are automatically encoded as integers for classification.
"""
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
    """
    PyTorch Dataset for loading sleep spectrograms and their associated sleep stage labels.

    Args:
        index_csv (str or Path): Path to CSV file listing spectrogram file paths and labels.
        root_dir (str or Path): Root directory where spectrogram .npy files are stored.

    The CSV file is expected to have at least two columns:
        - 'path': relative path to the .npy spectrogram file for each epoch.
        - 'stage': sleep stage label string for each epoch.

    Attributes:
        paths (list of str): List of relative file paths to spectrograms.
        labels (list of str): Corresponding list of sleep stage labels.
        label_to_idx (dict): Mapping from label string to integer index.
        idx_to_label (dict): Reverse mapping from integer index to label string.
    """
    def __init__(self, df, root_dir):
        # DataFrame with patient, path, label info
        self.paths = df['path'].tolist()
        self.labels = df['stage'].tolist()
        self.patient_ids = df['patient_id'].tolist()
        self.root_dir = Path(root_dir)

        # Label encoding
        label_set = sorted(list(set(self.labels)))
         # unique labels sorted for consistent mapping
        self.label_to_idx = {lab: idx for idx, lab in enumerate(label_set)}
        self.idx_to_label = {idx: lab for lab, idx in self.label_to_idx.items()}
        # {0: 'Sleep stage N1', 1: 'Sleep stage N2', 2: 'Sleep stage N3', 3: 'Sleep stage R', 4: 'Sleep stage W'}

    def __len__(self):
        # Total number of spectrograms (epochs) in dataset
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