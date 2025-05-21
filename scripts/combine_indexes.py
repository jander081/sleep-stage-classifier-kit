import pandas as pd
from pathlib import Path

# Path to dataset folder containing patient subfolders
dataset_root = Path("../dataset")   # adjust path if needed

# Prepare a list to hold DataFrames from all patients
dfs = []

# Loop over each patient folder
for patient_folder in dataset_root.iterdir():
    if patient_folder.is_dir():
        index_file = patient_folder / "index.csv"
        if index_file.exists():
            df = pd.read_csv(index_file)
            # Add patient ID (from folder name)
            df["patient_id"] = patient_folder.name
            dfs.append(df)
        else:
            print(f"No index.csv in {patient_folder}")

# Concatenate all patient DataFrames into one
if dfs:
    master_df = pd.concat(dfs, ignore_index=True)
    # Save to a new master index file
    master_df.to_csv(dataset_root / "master_index.csv", index=False)
    print("Combined index written to", dataset_root / "master_index.csv")
else:
    print("No index.csv files found in any patient folders!")