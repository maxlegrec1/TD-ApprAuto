import os

import numpy as np
import pandas as pd
import streamlit as sl
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

COLUMNS = [
    "Carbon concentration",
    "Silicon concentration",
    "Manganese concentration",
    "Sulphur concentration",
    "Phosphorus concentration",
    "Nickel concentration",
    "Chromium concentration",
    "Molybdenum concentration",
    "Vanadium concentration",
    "Copper concentration",
    "Cobalt concentration",
    "Tungsten concentration",
    "Oxygen concentration",
    "Titanium concentration",
    "Nitrogen concentration",
    "Aluminium concentration",
    "Boron concentration",
    "Niobium concentration",
    "Tin concentration",
    "Arsenic concentration",
    "Antimony concentration",
    "Current",
    "Voltage",
    "AC or DC",
    "Electrode positive or negative",
    "Heat input",
    "Interpass temperature",
    "Type of weld",
    "Post weld heat treatment temperature",
    "Post weld heat treatment time",
    "Yield strength",
    "Ultimate tensile strength",
    "Elongation",
    "Reduction of Area",
    "Charpy temperature",
    "Charpy impact toughness",
    "Hardness",
    r"50 % FATT",
    "Primary ferrite in microstructure",
    "Ferrite with second phase",
    "Acicular ferrite",
    "Martensite",
    "Ferrite with carbide aggreagate",
    "Weld ID",
]


def create_folds(output_dir, db_path, n_folds):
    # Read the space-separated file
    df = pd.read_csv(db_path, sep="\s+", header=None)

    def preprocess(df):

        TO_KEEP = [
            "Carbon concentration",
            "Silicon concentration",
            "Manganese concentration",
            "Sulphur concentration",
            "Phosphorus concentration",
            "Nickel concentration",
            "Chromium concentration",
            "Molybdenum concentration",
            "Vanadium concentration",
            "Copper concentration",
            "Cobalt concentration",
            "Tungsten concentration",
            "Oxygen concentration",
            "Titanium concentration",
            "Nitrogen concentration",
            "Aluminium concentration",
            "Boron concentration",
            "Niobium concentration",
            "Tin concentration",
            "Arsenic concentration",
            "Antimony concentration",
            "Current",
            "Voltage",
            "Heat input",
            "Interpass temperature",
            "Post weld heat treatment temperature",
            "Post weld heat treatment time",
            "Yield strength",
            "Ultimate tensile strength",
        ]
        # convert all collumn names to strings
        # df.columns = df.columns.astype(str)
        column_mapping = {i: COLUMNS[i] for i in range(len(COLUMNS))}
        # print(column_mapping)
        df = df.rename(columns=column_mapping)
        print(df)
        # drop the Weld ID Column
        df = df.drop("Weld ID", axis=1)

        # Onehot encode the column 27
        onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
        onehot_encoded = onehot.fit_transform(df[["Type of weld"]])
        onehot_columns = [f"Type of weld_{cat}" for cat in onehot.categories_[0]]
        TO_KEEP += onehot_columns
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_columns, index=df.index)
        df = df.drop("Type of weld", axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        df.loc[df["Electrode positive or negative"] != "N", "AC or DC"] = df[
            "AC or DC"
        ].replace("N", "DC")

        # Onehot encode the collumn 23
        onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
        onehot_encoded = onehot.fit_transform(df[["AC or DC"]])
        onehot_columns = [f"AC or DC_{cat}" for cat in onehot.categories_[0]]
        TO_KEEP += onehot_columns
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_columns, index=df.index)
        df = df.drop("AC or DC", axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        # Onehot encode the collumn 24
        onehot = OneHotEncoder(sparse=False, handle_unknown="ignore")
        onehot_encoded = onehot.fit_transform(df[["Electrode positive or negative"]])
        onehot_columns = [
            f"Electrode positive or negative_{cat}" for cat in onehot.categories_[0]
        ]
        TO_KEEP += onehot_columns
        onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_columns, index=df.index)
        df = df.drop("Electrode positive or negative", axis=1)
        df = pd.concat([df, onehot_df], axis=1)

        # Identify columns to fill the Ns (everything except Y)
        columns_to_impute = [
            col
            for col in df.columns
            if col not in ["Yield strength", "Ultimate tensile strength"]
        ]
        columns_remain = ["Yield strength", "Ultimate tensile strength"]

        # Replace 'N' with NaN for numeric conversion
        df[columns_to_impute] = df[columns_to_impute].replace("N", np.nan)
        df[columns_remain] = df[columns_remain].replace("N", -1)

        # Convert columns to numeric
        for col in columns_to_impute:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(df.median())
        df[columns_remain] = df[columns_remain].replace(-1, "N")

        df = df[TO_KEEP]
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
