# src/sleepkit/utils.py

import yaml
import numpy as np
from scipy.signal import spectrogram
import os
from sleepkit.logging_utils import setup_logger

logger = setup_logger()

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_frequency_mask(nfft, sr, max_hz):
    """
    Returns boolean mask for frequency bins <= max_hz.
    :param nfft: FFT size
    :param sr: sample rate
    :param max_hz: max frequency to keep
    :return: numpy boolean mask, shape (n_freqs,)
    """
    # If SR = 256, then 1/SR = 0.00390625 seconds per sample
    freqs = np.fft.rfftfreq(nfft, 1 / sr) # shape == (257,); == [0. 0.5 1. 1.5... 129]
    return freqs <= max_hz  # True/false array. true == 0:128 then false == (129:256)


def process_epoch_signal(signal, sr, nfft, max_hz):
    """
    Generates a 2D spectrogram for one channel/epoch.
    :param signal: 1D numpy array, length = sr*30 (30s epoch)
    :return: 2D numpy array, (n_freqs, n_windows) dB-scaled, up to max_hz
    """
    logger.info(f"Processing epoch signal with shape {signal.shape}")
    f, t, Sxx = spectrogram(signal, 
                            fs=sr, 
                            window='hann', 
                            nperseg=256, # 1‑s window
                            noverlap=128, # 0.5‑s hop
                            nfft=nfft,  # 512 → 0.5 Hz resolution
                            scaling='density', 
                            mode='psd')
    # mask = f <= max_hz # shape == (257,); == [0. 0.5 1. 1.5... 129]
    mask = get_frequency_mask(nfft, sr, max_hz)
    # convert power to decibels (dB); compresses a huge numeric range st can be seen on color scale; +10 dB ≈ 10× more power
    return 10 * np.log10(Sxx[mask] + 1e-8)


def extract_labels(annotations, n_epochs):
    """
    Convert a list of EDF annotations to an array of epoch labels.
    :param annotations: list of (onset, duration, label) tuples
    :param n_epochs: number of 30s epochs (typically len(signal) // (30*sr))
    :return: numpy array of labels, length n_epochs
    """
    logger.info(f"Extracting labels for {n_epochs} epochs, {len(annotations)} annotation events")
    labels = np.array([desc for _, _, desc in annotations], dtype=object)
    # Truncate or pad as needed for edge cases
    if len(labels) > n_epochs:
        logger.warning(f"More labels ({len(labels)}) than epochs ({n_epochs}), truncating.")
        labels = labels[:n_epochs]
    elif len(labels) < n_epochs:
        logger.warning(f"Fewer labels ({len(labels)}) than epochs ({n_epochs}), padding with 'UNK'.")
        labels = np.pad(labels, (0, n_epochs - len(labels)), 'constant', constant_values='UNK')
    return labels


def save_epoch_spectrogram(spectrogram, out_dir, patient_id, epoch_idx, channel_name):
    """
    Saves a single epoch's spectrogram to disk.
    :param spectrogram: 2D numpy array (freqs x windows)
    :param out_dir: output directory for patient data
    :param patient_id: string or int patient identifier
    :param epoch_idx: int, epoch number
    :param channel_name: str, e.g., 'EEG_F4_M1'
    """
    patient_dir = os.path.join(out_dir, f"{patient_id}")
    os.makedirs(patient_dir, exist_ok=True)
    fname = f"epoch_{epoch_idx:04d}_{channel_name}.npy"
    fpath = os.path.join(patient_dir, fname)
    np.save(fpath, spectrogram)
    logger.info(f"Saved {fpath}")
    return fpath