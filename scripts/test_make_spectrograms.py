import numpy as np
from pyedflib import highlevel
from scipy.signal import spectrogram
from pathlib import Path

# ---- Config ----
RAW_EDF = "../data/raw_edf/SN001.edf"
SCORING_EDF = "../data/scoring_edf/SN001_sleepscoring.edf"
OUT_DIR = Path("../dataset/SN001")
CHAN_IDX = 0           # F4-C4 or your favorite EEG channel
SR = 256               # sample rate Hz (double-check your file!)
EPOCH_SEC = 30
NFFT = 512             # FFT length for 0.5 Hz freq bins
KEEP_MAX_HZ = 64       # Only keep 0-64 Hz (more than enough for sleep)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load signals and scoring ----
signals, signal_headers, _ = highlevel.read_edf(RAW_EDF)
_, _, header_score = highlevel.read_edf(SCORING_EDF)

epoch_len = int(EPOCH_SEC * SR)
n_epochs = len(signals[CHAN_IDX]) // epoch_len

# ---- Clean labels ----
header_score['annotations'] = [
    (onset, dur, desc)
    for onset, dur, desc in header_score['annotations']
    if "Lights off@@EEG" not in desc
]

labels = np.array([desc for _, _, desc in header_score['annotations']], dtype=object)
labels[-1] = "Sleep stage W"
assert len(labels) == n_epochs

# ---- Precompute frequency mask for 0-64 Hz ----
dummy_f = np.fft.rfftfreq(NFFT, 1/SR)
freq_mask = dummy_f <= KEEP_MAX_HZ

# ---- Save spectrograms per epoch ----
for ep in range(n_epochs):
    start = ep * epoch_len
    epoch_signal = signals[CHAN_IDX][start:start+epoch_len]

    f, t, Sx = spectrogram(epoch_signal,
                           fs=SR,
                           nperseg=256,
                           noverlap=128,
                           nfft=NFFT)

    Sx = Sx[freq_mask, :]      # keep 0-64 Hz
    Sx = Sx.astype("float32")  # save space

    np.save(OUT_DIR / f"ep{ep:04d}.npy", Sx)

# ---- Write CSV index ----
import pandas as pd
index = pd.DataFrame({
    "epoch": np.arange(n_epochs),
    "stage": labels,
    "path": [f"ep{ep:04d}.npy" for ep in range(n_epochs)]
})
index.to_csv(OUT_DIR / "index.csv", index=False)
print("Done. Saved spectrograms and index for", RAW_EDF)