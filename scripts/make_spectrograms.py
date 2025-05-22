"""
This script processes raw EEG EDF files and their corresponding sleep scoring EDF files to generate
spectrograms for each 30-second epoch of EEG data. It extracts a specified EEG channel, computes 
spectrograms limited to a maximum frequency, and saves these spectrograms along with labels indicating 
sleep stages. The output is organized per patient in a dataset directory, with an index CSV summarizing 
epochs and their sleep stages.

Usage:
- Place raw EDF files in the configured RAW_EDF_DIR.
- Place corresponding scoring EDF files in SCORING_EDF_DIR.
- Run the script to generate spectrograms and labels saved under OUT_ROOT.
"""
# ---- Imports ----
import numpy as np
import pandas as pd
from pyedflib import highlevel
from scipy.signal import spectrogram
from pathlib import Path
import glob
import os

# ---- Config ----
RAW_EDF_DIR = Path("../data/raw_edf")
SCORING_EDF_DIR = Path("../data/scoring_edf")
OUT_ROOT = Path("../dataset")
CHAN_IDX = 0           # Change if you want a different EEG channel
SR = 256
EPOCH_SEC = 30
NFFT = 512
KEEP_MAX_HZ = 64

def create_labels(h_ann, n_epochs):
    """
    Create labels for each epoch.
    :param h_ann: header_score['annotations'] is a list of tuples (onset, duration, description)
    :param n_epochs: number of epochs
    :return: list of labels
    """
    labels = np.array([desc for _, _, desc in h_ann], dtype=object)
    labels[-1] = 'Sleep stage W'  # change lights on label to W; just in case it isn't truncated
    labels = labels[:n_epochs]
    return labels

def get_frequency_mask(nfft, sr, max_hz):
    """
    Generate a boolean mask to select frequency bins up to max_hz from the FFT output.

    :param nfft: number of FFT points used in spectrogram calculation
    :param sr: sampling rate of the EEG signal
    :param max_hz: maximum frequency to keep in the spectrogram
    :return: boolean numpy array indicating which frequency bins to keep
    """
    # Compute the frequencies corresponding to FFT bins (real FFT frequencies)
    dummy_f = np.fft.rfftfreq(nfft, 1/sr) # shape (257,)
    # Create a (T or F) boolean mask for frequencies less than or equal to max_hz
    # T -> [0:128] then F -> [129:256]
    return dummy_f <= max_hz

def process_signal_epoch(signal, sr, nfft, freq_mask):
    """
    Compute the spectrogram of a single EEG epoch and apply frequency masking.

    :param signal: 1D numpy array containing EEG data for one epoch
    :param sr: sampling rate of the EEG signal
    :param nfft: number of FFT points for spectrogram calculation
    :param freq_mask: boolean mask to filter frequency bins
    :return: 2D numpy array of the spectrogram limited to desired frequencies
    """
    # Compute spectrogram using a 256-sample window with 128-sample overlap
    f, t, Sx = spectrogram(signal,
                           fs=sr,
                           nperseg=256,
                           noverlap=128,
                           nfft=nfft)
    # Apply frequency mask to keep only frequencies up to max_hz 64 Hz
    Sx = Sx[freq_mask, :]
    # Sx.shape == (129, 59) # ==> (64, 30) for 0.5 Hz resolution
    # Convert spectrogram data to float32 for memory efficiency
    return Sx.astype("float32")

def process_patient(raw_edf_path, scoring_edf_path, out_dir):
    """
    Process a single patient's raw EEG and scoring EDF files to generate spectrograms per epoch.

    :param raw_edf_path: Path object pointing to the raw EEG EDF file
    :param scoring_edf_path: Path object pointing to the scoring EDF file (annotations)
    :param out_dir: Path object where output spectrograms and index CSV will be saved
    """
    # Ensure the output directory exists, create if necessary
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load signals and scoring
    # Load raw EEG signals and headers from EDF
    signals, signal_headers, _ = highlevel.read_edf(str(raw_edf_path))
    # Load scoring header (contains annotations) from scoring EDF
    _, _, header_score = highlevel.read_edf(str(scoring_edf_path))
    # Calculate number of samples per epoch based on sampling rate (256) and epoch duration (30 seconds)
    epoch_len = int(EPOCH_SEC * SR) # == 7680 samples
    # Calculate number of epochs based on the length of the EEG signal and epoch length
    n_epochs = len(signals[CHAN_IDX]) // epoch_len

    # Clean annotations
    # Filter out any annotations labeled as 'Lights off@@' from the scoring annotations
    header_score['annotations'] = [
        (onset, dur, desc)
        for onset, dur, desc in header_score['annotations']
        if "Lights off@@" not in desc
    ]

    # Create labels using helper
    labels = create_labels(header_score['annotations'], n_epochs)

    # Precompute frequency mask
    freq_mask = get_frequency_mask(NFFT, SR, KEEP_MAX_HZ)

    # Save spectrograms per epoch
    # Iterate over each epoch to compute and save its spectrogram
    for ep in range(n_epochs):
        start = ep * epoch_len
        # Extract EEG data segment for the current epoch
        epoch_signal = signals[CHAN_IDX][start:start+epoch_len]
        # Compute the spectrogram for the current epoch
        Sx = process_signal_epoch(epoch_signal, SR, NFFT, freq_mask)
        # Save the spectrogram to a .npy file (numpy binary file) in the output directory
        np.save(out_dir / f"ep{ep:04d}.npy", Sx)

    # Write CSV index
    # Create a DataFrame to index epochs with their sleep stage labels and file paths
    index = pd.DataFrame({
        "epoch": np.arange(n_epochs),
        "stage": labels,
        "path": [f"ep{ep:04d}.npy" for ep in range(n_epochs)]
    })
    # Save the index DataFrame to a CSV file in the output directory
    index.to_csv(out_dir / "index.csv", index=False)
    print(f"Processed {raw_edf_path.name} -> {out_dir}")


def update_master_index(dataset_dir):
    """
    Combines all per-patient index.csv files into a single master_index.csv in the dataset directory.
    """
    master_records = []
    # Find all index.csv files in subfolders
    for index_path in dataset_dir.glob("*/index.csv"):
        patient_id = index_path.parent.name
        df = pd.read_csv(index_path)
        df["patient_id"] = patient_id  # Add patient column
        master_records.append(df)
    # Concatenate all per-patient dataframes
    if master_records:
        master_df = pd.concat(master_records, ignore_index=True)
        master_df.to_csv(dataset_dir / "master_index.csv", index=False)
        print(f"master_index.csv updated with {len(master_df)} rows.")
    else:
        print("No per-patient index.csv files found.")

def main():
    """
    Main function to process all patients' raw and scoring EDF files.

    Workflow:
    - Finds all raw EDF files in RAW_EDF_DIR.
    - For each raw EDF, constructs the corresponding scoring EDF path.
    - Checks if the scoring EDF file exists; if not, skips that patient.
    - Processes each patient by generating spectrograms and labels.
    - Prints summary messages indicating progress and completion.
    """
    # List all EDF files in the raw EDF directory
    raw_edfs = list(RAW_EDF_DIR.glob("*.edf"))
    print(f"Found {len(raw_edfs)} patients.")
    # Iterate over each raw EDF file
    for raw_path in raw_edfs:
        # extract patient ID from the raw EDF filename
        patient_id = raw_path.stem  
        # compose the expected scoring EDF path based on the patient ID
        scoring_path = SCORING_EDF_DIR / f"{patient_id}_sleepscoring.edf" 
        # define the output directory for this patient
        out_dir = OUT_ROOT / patient_id
        # only process if the scoring EDF file exists
        if scoring_path.exists():
            process_patient(raw_path, scoring_path, out_dir)
        else:
            # notify if the scoring EDF file is missing
            print(f"Missing scoring EDF for {patient_id}, skipping.")

    print("Done. All available patients processed.")
    # Update the master index CSV file with all processed patients
    update_master_index(OUT_ROOT)
    print("Master index updated.")

if __name__ == "__main__":
    main()