import os

import numpy as np
import pandas as pd
import streamlit as sl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


def create_folds(output_dir, db_path, n_folds):
    # Read the space-separated file
    df = pd.read_csv(db_path, sep="\s+", header=None)

    def preprocess(df):
        # drop the Weld ID Collumn
        df = df.drop(43, axis=1)

        # Onehot encode the collumn 27
        onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
        onehot_encoded = onehot.fit_transform(df[[27]])
        onehot_columns = [f"27_{cat}" for cat in onehot.categories_[0]]
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_columns, index=df.index)
        df = df.drop(27, axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        # convert all collumn names to strings
        df.columns = df.columns.astype(str)

        df.loc[df["24"] != "N", "23"] = df["23"].replace("N", "DC")

        onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
        onehot_encoded = onehot.fit_transform(df[["23"]])
        onehot_columns = [f"23_{cat}" for cat in onehot.categories_[0]]
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_columns, index=df.index)
        df = df.drop("23", axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
        onehot_encoded = onehot.fit_transform(df[["24"]])
        onehot_columns = [f"24_{cat}" for cat in onehot.categories_[0]]
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_columns, index=df.index)
        df = df.drop("24", axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        print(df)
        # Identify columns to be imputed
        columns_to_impute = [col for col in df.columns if col not in ["30", "31"]]
        columns_stay = ["30", "31"]

        # Replace 'N' with NaN for numeric conversion
        df[columns_to_impute] = df[columns_to_impute].replace("N", np.nan)
        df[columns_stay] = df[columns_stay].replace("N", -1)

        # Convert columns to numeric
        for col in columns_to_impute:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Create an imputer

        df = df.fillna(df.median())
        df[columns_stay] = df[columns_stay].replace(-1, "N")
        return df

    df = preprocess(df)

    # Create the KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

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
    for i in range(n_folds):
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


if __name__ == "__main__":
    output_dir = "folds"
    db_path = "welddb/welddb.data"
    N_FOLDS = 6

    create_folds(output_dir, db_path, N_FOLDS)
