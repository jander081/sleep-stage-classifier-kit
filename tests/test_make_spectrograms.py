# tests/test_make_spectrograms.py
"""	
pytest — it runs as a script, which doesn’t always set up PYTHONPATH to include your project root.
python -m pytest — it runs pytest as a module, which correctly includes your current directory (project root) in the import path.
"""

import numpy as np
from scripts.make_spectrograms import create_labels, get_frequency_mask, process_signal_epoch
# run with python -m pytest or else the import will fail

def test_create_labels_truncates():
    # 5 annotations, 3 epochs; should truncate to 3
    ann = [(0, 30, "A"), (30, 30, "B"), (60, 30, "C"), (90, 30, "D"), (120, 30, "E")]
    n_epochs = 3
    labels = create_labels(ann, n_epochs)
    assert len(labels) == n_epochs
    assert labels[0] == "A"
    assert labels[-1] == "C"

def test_create_labels_last_label_is_w():
    ann = [(0, 30, "A"), (30, 30, "B"), (60, 30, "Lights on@@EEG")]
    n_epochs = 3
    labels = create_labels(ann, n_epochs)
    assert labels[-1] == "Sleep stage W"

def test_get_frequency_mask_shape():
    nfft = 512
    sr = 256
    max_hz = 64
    mask = get_frequency_mask(nfft, sr, max_hz)
    # There should be 129 True values (0.0 Hz to 64.0 Hz by 0.5)
    assert mask.sum() == 129

def test_process_signal_epoch_shape():
    signal = np.random.randn(256*30)  # 30 seconds at 256 Hz
    sr = 256
    nfft = 512
    freq_mask = get_frequency_mask(nfft, sr, 64)
    Sx = process_signal_epoch(signal, sr, nfft, freq_mask)
    # 129 frequency bins by 59 time windows
    assert Sx.shape[0] == 129
    assert Sx.shape[1] > 0

# Add more tests as desired for other helpers