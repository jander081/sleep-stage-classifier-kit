import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ------------- Custom Dataset for loading spectrograms -------------
class SleepSpectrogramDataset(Dataset):
    def __init__(self, index_csv, root_dir):
        df = pd.read_csv(index_csv)
        self.paths = df['path'].tolist()
        self.labels = df['stage'].tolist()
        self.root_dir = Path(root_dir)

        # Label encoding (simple example; adjust as needed)
        label_set = sorted(list(set(self.labels)))
        self.label_to_idx = {lab: idx for idx, lab in enumerate(label_set)}
        self.idx_to_label = {idx: lab for lab, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = np.load(self.root_dir / self.paths[idx])  # (freq, time)
        x = np.expand_dims(x, axis=0)  # add channel dim: (1, freq, time)
        x = torch.tensor(x, dtype=torch.float32)
        y = self.label_to_idx[self.labels[idx]]
        return x, y