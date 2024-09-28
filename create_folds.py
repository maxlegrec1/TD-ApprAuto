"""
!! WARNING !!

DO NOT RUN THIS CODE AGAIN UNLESS YOU KNOW WHAT YOU ARE DOING


"""

import os

import pandas as pd
from sklearn.model_selection import KFold

output_dir = "folds"
db_path = "welddb/welddb.data"
N_FOLDS = 6

# Read the space-separated file
df = pd.read_csv(db_path, sep="\s+", header=None)

# Create the KFold object
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Get the indices for each fold
fold_indices = list(kf.split(df))


# Function to get a specific fold
def get_fold(df, fold_indices, fold_number):
    train_index, test_index = fold_indices[fold_number]
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    return train_df, test_df


# Create a directory to store the fold files
os.makedirs(output_dir, exist_ok=True)

# Save each fold as separate train and test CSV files
for i in range(N_FOLDS):
    train_df, test_df = get_fold(df, fold_indices, i)

    # Save train dataset
    train_filename = os.path.join(output_dir, f"{i+1}_train.csv")
    train_df.to_csv(train_filename, index=False)

    # Save test dataset
    test_filename = os.path.join(output_dir, f"{i+1}_test.csv")
    test_df.to_csv(test_filename, index=False)

    print(f"Fold {i+1} saved:")
    print(f"  Train set saved as: {train_filename}")
    print(f"  Test set saved as: {test_filename}")
    print(f"  Train set shape: {train_df.shape}")
    print(f"  Test set shape: {test_df.shape}")
    print()

print(f"All folds have been saved in the '{output_dir}' directory.")
