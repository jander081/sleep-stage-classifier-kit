import numpy as np
import pandas as pd
from pyedflib import highlevel
from scipy.signal import spectrogram
from pathlib import Path

# ---- Config ----
RAW_EDF_DIR = Path("../data/raw_edf")
SCORING_EDF_DIR = Path("../data/scoring_edf")
OUT_ROOT = Path("../dataset")
CHAN_IDX = 0           # Change if you want a different EEG channel
SR = 256
EPOCH_SEC = 30
NFFT = 512
KEEP_MAX_HZ = 64

def process_patient(raw_edf_path, scoring_edf_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load signals and scoring
    signals, signal_headers, _ = highlevel.read_edf(str(raw_edf_path))
    _, _, header_score = highlevel.read_edf(str(scoring_edf_path))

    epoch_len = int(EPOCH_SEC * SR)
    n_epochs = len(signals[CHAN_IDX]) // epoch_len

    # Clean labels
    header_score['annotations'] = [
        (onset, dur, desc)
        for onset, dur, desc in header_score['annotations']
        if "Lights off@@EEG" not in desc
    ]

    # labels = np.array([desc for _, _, desc in header_score['annotations']], dtype=object)
    # labels[-1] = "Sleep stage W"
    # assert len(labels) == n_epochs

    labels = np.array([desc for _, _, desc in header_score['annotations']], dtype=object)
    labels[-1] = "Sleep stage W"
    if len(labels) != n_epochs:
        print(f"[ERROR] {raw_edf_path.name}: {len(labels)} labels vs {n_epochs} epochs")
        # Optionally print some details:
        print("First 10 labels:", labels[:10])
        print("Last 10 labels:", labels[-10:])
        print("Check the annotations or scoring file for this patient.")
        return  # skip this patient for now
    # assert len(labels) == n_epochs  # (remove or comment this out)

    # Precompute frequency mask
    dummy_f = np.fft.rfftfreq(NFFT, 1/SR)
    freq_mask = dummy_f <= KEEP_MAX_HZ

    # Save spectrograms per epoch
    for ep in range(n_epochs):
        start = ep * epoch_len
        epoch_signal = signals[CHAN_IDX][start:start+epoch_len]
        f, t, Sx = spectrogram(epoch_signal,
                               fs=SR,
                               nperseg=256,
                               noverlap=128,
                               nfft=NFFT)
        Sx = Sx[freq_mask, :]
        Sx = Sx.astype("float32")
        np.save(out_dir / f"ep{ep:04d}.npy", Sx)

    # Write CSV index
    index = pd.DataFrame({
        "epoch": np.arange(n_epochs),
        "stage": labels,
        "path": [f"ep{ep:04d}.npy" for ep in range(n_epochs)]
    })
    index.to_csv(out_dir / "index.csv", index=False)
    print(f"Processed {raw_edf_path.name} -> {out_dir}")

def main():
    # Get all patient IDs from raw EDF directory (assumes filename like SN001.edf)
    raw_edfs = list(RAW_EDF_DIR.glob("*.edf"))
    print(f"Found {len(raw_edfs)} patients.")

    for raw_path in raw_edfs:
        patient_id = raw_path.stem  # SN001
        scoring_path = SCORING_EDF_DIR / f"{patient_id}_sleepscoring.edf"
        out_dir = OUT_ROOT / patient_id

        if scoring_path.exists():
            process_patient(raw_path, scoring_path, out_dir)
        else:
            print(f"Missing scoring EDF for {patient_id}, skipping.")

    print("Done. All available patients processed.")

if __name__ == "__main__":
    main()