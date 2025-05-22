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

- Add Colab demo notebook

## Current Progress

- [x] Preprocessing pipeline: Generates spectrograms and per-patient index.csv files for all patients in `data/raw_edf` and `data/scoring_edf`
- [x] Master index creation: Combines all patient index files into `master_index.csv` for full-dataset analysis
- [x] Helper functions refactored and commented for clarity
- [x] Unit tests (pytest) written for helper functions
- [x] Successful run for 5 patients, ready to add more

## Next Steps

- [x] Import 5 more patients (aiming for 10+ in dataset)
- [ ] Run/update spectrogram pipeline on new patients
- [ ] Update/check `master_index.csv`
- [ ] (Optional) Do exploratory data analysis in Jupyter
- [ ] Begin training the classifier using the preprocessed dataset
    - Prepare data splits for train/test
    - Train, evaluate, and visualize model performance
- [ ] Continue refactoring and testing as pipeline expands