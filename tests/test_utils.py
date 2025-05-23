# tsts/test_utils.py

# use %python -m pytest to run 
import numpy as np
import os
from sleepkit.utils import load_config

def test_load_config():
    config = load_config("config.yaml")
    assert isinstance(config, dict)
    assert "raw_dir" in config
    assert "channels_of_interest" in config
   

def test_import_utils():
    from sleepkit import utils
    assert hasattr(utils, 'load_config')


from sleepkit.utils import get_frequency_mask

def test_get_frequency_mask():
    mask = get_frequency_mask(nfft=512, sr=256, max_hz=64)
    assert mask.any()           # At least some True values
    assert mask[0] == True      # DC (0 Hz freq) always included
    assert mask.sum() < 512     # Only a fraction of bins


import numpy as np
from sleepkit.utils import process_epoch_signal

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


if __name__ == "__main__":
    # For quick manual test
    test_load_config()
    print("All tests passed.")