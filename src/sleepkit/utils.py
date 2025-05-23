# src/sleepkit/utils.py

import yaml
import numpy as np
from scipy.signal import spectrogram

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
    f, t, Sxx = spectrogram(signal, 
                            fs=sr, 
                            window='hann', 
                            nperseg=256, # 1‑s window
                            noverlap=128, # 0.5‑s hop
                            nfft=nfft,  # 512 → 0.5 Hz resolution
                            scaling='density', 
                            mode='psd')
    mask = f <= max_hz # shape == (257,); == [0. 0.5 1. 1.5... 129]
    # convert power to decibels (dB); compresses a huge numeric range st can be seen on color scale; +10 dB ≈ 10× more power
    return 10 * np.log10(Sxx[mask] + 1e-8)