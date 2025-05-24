# tsts/test_utils.py

# use %python -m pytest to run 
import numpy as np
import os
from sleepkit.utils import load_config, get_frequency_mask, process_epoch_signal, \
extract_labels, save_epoch_spectrogram


def test_load_config():
    config = load_config("config.yaml")
    assert isinstance(config, dict)
    assert "raw_dir" in config
    assert "channels_of_interest" in config
   

def test_import_utils():
    from sleepkit import utils
    assert hasattr(utils, 'load_config')

def test_get_frequency_mask():
    mask = get_frequency_mask(nfft=512, sr=256, max_hz=64)
    assert mask.any()           # At least some True values
    assert mask[0] == True      # DC (0 Hz freq) always included
    assert mask.sum() < 512     # Only a fraction of bins


def test_process_epoch_signal():
    sr = 256
    nfft = 512
    max_hz = 64
    # Simulate random noise for 30s
    signal = np.random.randn(sr * 30)
    spec = process_epoch_signal(signal, sr, nfft, max_hz)
    assert spec.shape[0] > 0
    assert spec.shape[1] > 0
    assert np.isfinite(spec).all()

def test_extract_labels_truncate():
    anns = [(0, 30, "Stage1")] * 12
    labels = extract_labels(anns, 10) # 10 epochs, but 12 annotations (labels)
    assert len(labels) == 10
    assert labels[-1] == "Stage1"

def test_extract_labels_pad():
    anns = [(0, 30, "Stage1")] * 8
    labels = extract_labels(anns, 10)  # 10 epochs, but only 8 annotations (labels)
    assert len(labels) == 10
    assert labels[-1] == "UNK"


def test_save_epoch_spectrogram(tmp_path):
    # Setup: create dummy data
    spectrogram = np.random.randn(129, 59)  # e.g., 129 freq bins x 59 windows
    out_dir = tmp_path / "test_patient_out"
    patient_id = "TEST_PATIENT"
    epoch_idx = 7
    channel_name = "EEG_F4_M1"

    # Call function
    fpath = save_epoch_spectrogram(
        spectrogram,
        str(out_dir),
        patient_id,
        epoch_idx,
        channel_name
    )

    # Check file exists
    assert os.path.isfile(fpath)

    # Load and check shape
    arr = np.load(fpath)
    assert arr.shape == spectrogram.shape

    # Clean up (pytest's tmp_path auto-deletes the directory)



if __name__ == "__main__":
    # For quick manual test
    test_load_config()
    print("All tests passed.")