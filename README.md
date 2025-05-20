# Sleep Stage Classifier Kit

A toolkit for preprocessing polysomnography EEG data, building spectrograms, and training deep learning models to classify sleep stages.

## Project Structure

- `scripts/`: Python scripts for data prep and model training
- `notebooks/`: Jupyter notebooks for exploration and demo
- `data/`: (Not tracked) Raw EDF and scoring files
- `dataset/`: (Not tracked) Output spectrograms and index files

## Quick Start

1. Install dependencies: 

% pip install -r requirements.txt

2. Prepare data:

Place raw EDF files and scoring EDF files in `data/raw_edf/` and `data/scoring_edf/` respectively.

3. Build spectrograms (example):

% python scripts/make_spectrograms.py

4. (Coming soon) Train classifier:

% python scripts/train_classifier.py

## Requirements

- Python 3.8+
- See `requirements.txt` for Python package dependencies

## Notes

- Data and dataset folders are excluded from git tracking.
- Example data not included due to size/privacy.

## TODO

- Add model training script
- Add Colab demo notebook
