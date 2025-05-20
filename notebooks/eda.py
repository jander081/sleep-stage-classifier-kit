

# %% 
from pyedflib import highlevel
import pyedflib as plib
import numpy as np
import matplotlib.pyplot as plt


# %%
path = "data/raw_edf/SN001.edf"
signals, signal_headers, header = highlevel.read_edf(path) # len(signal_headers) == len(signals) == 8
# len(header) == len(header_score) == 12
# len(signals[0]) == 6,566,400; len(header_score['annotations']) == 856
path2 = "data/scoring_edf/SN001_sleepscoring.edf"
_, _, header_score = highlevel.read_edf(path2)
# %%

chan_idx = 0  
sr = signal_headers[chan_idx]['sample_frequency'] # e.g. 256.0 Hz
epoch_len = int(30 * sr) # 7680.0 samples per epoch
# 30 seconds per epoch
# 6_566_400 samples / 7680.0 samples per epoch == 855 epochs
# 6_566_400 samples / 256.0 samples per second == 25_600 seconds
# 25_600 seconds / 60 seconds per minute == 426.6666666666667 minutes
# 426.6666666666667 minutes / 60 minutes per hour == 7.111111111111111 hours
n_epochs = int(len(signals[0]) // epoch_len) # 6_566_400 // 7680.0 == 855# %%
# signal_headers is a list of dictionaries, each dictionary corresponds to a signal header
# signal_headers
len(signal_headers) # == 8
signal_headers[2]



# %%
# Strip any “Lights off@@EEG” annotations
header_score['annotations'] = [
    (onset, dur, desc)
    for onset, dur, desc in header_score['annotations']
    if "Lights off@@EEG" not in desc          # substring match, case‑sensitive
]

print(len(header_score['annotations'])) 

# %%

# after you’ve removed the “Lights off” annotation:
labels = np.array([desc for _, _, desc in header_score['annotations']], dtype=object)
assert len(labels) == n_epochs    # n_epochs from the raw‑signal length
labels[-1] = 'Sleep stage W'   # change lights on label to W


# %%


# %%