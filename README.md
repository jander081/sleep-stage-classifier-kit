# SleepKit: Multi-Channel Sleep Stage Classifier Toolkit

**SleepKit** is a modular, research-grade Python toolkit for preprocessing polysomnography (PSG) data, generating multi-channel spectrograms, and building deep learning models to classify sleep stages.

---

## Features

- **Handles full-night PSG data** from 8+ channels (EEG, EMG, EOG, ECG)
- **Generates per-epoch, multi-channel spectrograms** for all patient files
- **Flexible config (`config.yaml`)** to set channels, sampling rate, NFFT, frequency mask, and more
- **Batch processing** of 150+ patients (scalable, memory-aware)
- **Train/test split and classifier training** (deep learning-ready)
- **Unit tested** (`pytest`) and modular codebase
- **Logging** for easy debugging and reproducibility
- **Easy export** for downstream analysis in Python or R

---

<pre>
## Project Structure

```text
sleep_study/
│
├── src/
│   └── sleepkit/
│       ├── __init__.py
│       ├── utils.py
│       └── (more modules...)
│
├── scripts/
│   ├── make_spectrograms.py
│   ├── train_classifier.py
│   └── (other scripts...)
│
├── tests/
│   └── test_utils.py
│
├── data/
│   ├── raw_edf/
│   └── scoring_edf/
│
├── dataset/
│   └── (per-patient spectrograms and indexes)
│
├── config.yaml
├── pyproject.toml
├── README.md
└── notes.txt
```
</pre>

---

## ⚡ Quick Start

1. **Clone this repo and create your environment:**
    ```bash
    git clone https://github.com/your-username/sleepkit.git
    cd sleepkit
    pip install -e .
    ```

2. **Edit `config.yaml`:**  
   Set your desired channels, frequency range, data paths, and options.

3. **Prepare data:**  
   - Place raw EDF files in `data/raw_edf/`
   - Place sleep scoring files in `data/scoring_edf/`

4. **Build spectrograms:**
    ```bash
    python scripts/make_spectrograms.py
    ```

5. **Train classifier:**
    ```bash
    python scripts/train_classifier.py
    ```

6. **Run tests (optional but recommended):**
    ```bash
    python -m pytest
    ```

---

## Requirements

- Python 3.9+
- See `requirements.txt` for Python dependencies (or use `pyproject.toml`)

---

## Logging

- All scripts write INFO-level logs to `sleepkit.log` (in the project root).
- Check this file for batch processing status, errors, and progress.

---

## Development Notes

- All importable code is under `src/sleepkit/`
- Uses a modern Python packaging layout (`pyproject.toml`, editable install)
- Jupyter Notebooks:  
  Add this to your first cell for imports to work:
    ```python
    import sys, os
    sys.path.append(os.path.abspath('../src'))
    ```

---

## Dataset & Privacy

- **No patient data is included in this repo.**
- Place your own EDF/scoring files in `data/` as described above.

---

## ✔️ Current Progress

- [x] Multi-channel preprocessing pipeline, modularized and refactored
- [x] Helper functions with docstrings, comments, and logging
- [x] Unit tests with pytest
- [x] Works for at least 10 patients, scalable to 150+
- [ ] Multi-channel classifier in progress
- [ ] Export scripts for R (upcoming)
- [ ] Colab demo notebook (planned)

---

## Next Steps

- [ ] Finish multi-channel classifier and model evaluation
- [ ] Add advanced visualizations and performance metrics
- [ ] Add export for R and other statistical workflows
- [ ] Expand PyTest coverage as codebase grows

---

## Contributing

Pull requests and issue reports are welcome!  
For significant changes, open an issue to discuss your ideas.

---

## License

[MIT License] or your chosen license

---

## Acknowledgments

Developed by Jake (and collaborators).  
Thanks to the open-source PSG, EEG, and neuroscience communities.