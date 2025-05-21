# ---- Imports: Data, ML, and Paths ----
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ---- Custom Dataset for loading spectrograms ----
class SleepSpectrogramDataset(Dataset):
    def __init__(self, index_csv, root_dir):
        # Load CSV with epoch info and paths to .npy spectrograms
        df = pd.read_csv(index_csv)
        self.paths = df['path'].tolist()     # list of .npy files (per-epoch spectrograms)
        self.labels = df['stage'].tolist()   # list of corresponding stage labels
        self.root_dir = Path(root_dir)

        # Label encoding: convert stage names to integer indices (needed for PyTorch)
        label_set = sorted(list(set(self.labels)))
        self.label_to_idx = {lab: idx for idx, lab in enumerate(label_set)}
        self.idx_to_label = {idx: lab for lab, idx in self.label_to_idx.items()}

        # DEBUG: Print mapping for your reference
        print("Label to index mapping:", self.label_to_idx)

    def __len__(self):
        # Number of epochs in dataset
        return len(self.paths)

    def __getitem__(self, idx):
        # Load one spectrogram and label
        x = np.load(self.root_dir / self.paths[idx])   # shape: (freq, time)
        x = np.expand_dims(x, axis=0)                  # add channel dim: (1, freq, time)
        x = torch.tensor(x, dtype=torch.float32)
        y = self.label_to_idx[self.labels[idx]]        # convert label to int
        return x, y

# ---- Simple CNN for classifying sleep stages from spectrograms ----
class SimpleSleepCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # Conv layer 1: (in_channels=1, out_channels=16, kernel=3x3)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # Conv layer 2: (in_channels=16, out_channels=32, kernel=3x3)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # MaxPool layer (reduces spatial dims by half each time)
        self.pool = nn.MaxPool2d(2)

        # Flattening will depend on input shape; adjust if shape mismatch
        # Example: input spectrogram (1, 128, 59) --> pool --> (32, 32, 14) after 2 pools
        self.fc1 = nn.Linear(32 * 32 * 14, 64)   # adjust these numbers to match your actual output shape!
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (batch, channels, freq, time)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        # Debug: print output shape after conv/pool
        # print("Shape before flatten:", x.shape)
        x = x.view(x.size(0), -1)    # flatten all but batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---- Training function for one epoch ----
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()   # set to training mode
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)           # predictions (logits)
        loss = loss_fn(out, yb)   # compute loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

# ---- Main script ----
def main():
    # Configs
    index_csv = "../dataset/SN001/index.csv"   # CSV listing spectrograms and labels
    root_dir = "../dataset/SN001"
    batch_size = 16
    epochs = 10
    lr = 1e-3

    # Load dataset and preview first item
    dataset = SleepSpectrogramDataset(index_csv, root_dir)
    x0, y0 = dataset[0]
    print("First spectrogram shape:", x0.shape)   # should be (1, freq, time)
    print("First label index:", y0, "label:", dataset.idx_to_label[y0])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    n_classes = len(set(dataset.labels))
    model = SimpleSleepCNN(n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Training setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        loss = train_one_epoch(model, loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    # Save model weights
    torch.save(model.state_dict(), "sleep_cnn.pth")
    print("Training done, model saved.")

if __name__ == "__main__":
    main()