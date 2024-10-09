from typing import Generator, List, Literal, Optional, Tuple, Union

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from pca import pca

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

COLUMNS_STRING = [
    "AC or DC",
    "Electrode positive or negative",
    "Type of weld",
    "Weld ID",
]

COLUMNS_FLOAT = [
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
]

TARGET_FEATURES = [
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
]

FEATURES = [
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
]

IMPURITIES = [
    "Sulphur concentration",
    "Phosphorus concentration",
    "Oxygen concentration",
    "Titanium concentration",
    "Nitrogen concentration",
    "Aluminium concentration",
    "Boron concentration",
    "Niobium concentration",
    "Tin concentration",
    "Arsenic concentration",
    "Antimony concentration",
]

CORE_MATERIALS = [
    "Carbon concentration",
    "Silicon concentration",
    "Manganese concentration",
    "Nickel concentration",
    "Chromium concentration",
    "Molybdenum concentration",
    "Vanadium concentration",
    "Copper concentration",
    "Cobalt concentration",
    "Tungsten concentration",
]

WELD_TYPE = ["MMA", "SA", "FCA", "TSA", "ShMA", "NGSAW", "NGGMA", "SAA", "GTAA", "GMAA"]
ELECTRODE_TYPE = ["+", "-"]
AC_DC = ["AC", "DC"]


def get_data(
    target_features: Union[str, List[str]] = TARGET_FEATURES,
    features: List[str] = FEATURES,
    filename: str = "welddb/welddb.data",
    drop_y_nan_values: bool = True,
    nan_values: Optional[Literal["Gaussian", "Mean", "Median", "Zero", "Remove"]] = None,
    test_size: Optional[float] = None,
    random_state: int = 42,
    n_pca: Optional[int] = None,
    one_hot_encode: bool = True,
    normalize: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    """If test_size is None, returns X, y.\n
    Otherwise, returns X_train, X_test, y_train, y_test.

    @param target_features: The target features (or labels) to predict.
    @param features: The features to use.
    @param filename: The filename of the data (should not need to be changed).
    @param drop_y_nan_values: If True, drops the rows with NaN values in the target features.
    @param nan_values: The method to replace the features NaN values.
    @param test_size: The proportion of the dataset to include in the test split.
    @param random_state: The seed used by the random number generator.
    @param n_pca: The number of components to use in PCA. If None, PCA is not used.
    @param one_hot_encode: If True, one hot encode the categorical columns.
    @param normalize: If True, normalize the data.

    for "Custom1" nan_values, we assume that the core materials are not used if the value is NaN, so we replace it with 0.
    """
    if isinstance(target_features, str):
        target_features = [target_features]

    columns = set(features).union(target_features)

    assert columns.issubset(
        COLUMNS
    ), f"These columns are not in the data: {', '.join([col for col in columns if col not in COLUMNS])}"
    
    assert normalize or n_pca is None, "PCA can only be used with normalization"

    columns = list(columns)

    data = pd.read_csv(filename, delim_whitespace=True, header=None, names=COLUMNS)
    data = data[columns]

    remove_anomalies(data)
    data.replace("N", pd.NA, inplace=True)
    if one_hot_encode:
        data = one_hot_encode_all(data, columns)

    data = convert_to_float(
        data, columns, errors="ignore" if one_hot_encode else "raise"
    )

    if drop_y_nan_values:
        data.dropna(subset=target_features, inplace=True)

    X = data.drop(target_features, axis=1)  # Features
    y = data[target_features]  # Target

    if test_size is None:
        X = replace_nan(X, method=nan_values)
        y = replace_nan(y, method=nan_values)
        
        if normalize:
            X = scale(X)
            X = pca(X, n_components=n_pca)
        
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    X_train, X_test = replace_nan(X_train, X_test, method=nan_values)
    y_train, y_test = replace_nan(y_train, y_test, method=nan_values)
    
    if normalize:
        X_train, X_test = scale(X_train, X_test)
        X_train, X_test = pca(X_train, X_test, n_components=n_pca)

    return X_train, X_test, y_train, y_test


def get_data_information(
    columns: List[str] = COLUMNS_FLOAT,
    filename: str = "welddb/welddb.data",
    output_filename: str = "readme_table.md",
) -> pd.DataFrame:
    """Returns a dataframe with the mean, std, median, min and max of the float columns."""

    data = pd.read_csv(filename, delim_whitespace=True, header=None, names=COLUMNS)
    data = data[columns]
    remove_anomalies(data)
    data.replace("N", pd.NA, inplace=True)
    data = convert_to_float(data, columns, errors="ignore")
    data_information = pd.DataFrame(
        np.zeros((5, len(columns)), dtype=float), columns=columns
    )
    data_information.iloc[0] = data.mean()
    data_information.iloc[1] = data.std()
    data_information.iloc[2] = data.median()
    data_information.iloc[3] = data.min()
    data_information.iloc[4] = data.max()

    # Convert to Markdown format
    markdown_table = data_information.to_markdown(index=False)

    # Save the Markdown table to a .md file
    with open(output_filename, "w") as f:
        f.write(markdown_table)

    return data_information


def get_cross_validation_data(
    X: pd.DataFrame, y: pd.DataFrame, n_folds: int = 5, random_state: int = 42
) -> Generator[
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None
]:
    """Returns a generator of X_train, X_val, y_train, y_val."""

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_index, val_index in kf.split(X):
        yield X.iloc[train_index], X.iloc[val_index], y.iloc[train_index], y.iloc[
            val_index
        ]


def scale(
    data_train: pd.DataFrame,
    data_test: Optional[pd.DataFrame] = None,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """Normalizes the data using the StandardScaler."""
    
    scaler = StandardScaler()
    data_train = pd.DataFrame(scaler.fit_transform(data_train), columns=data_train.columns)
    if data_test is None:
        return data_train
    
    data_test = pd.DataFrame(scaler.transform(data_test), columns=data_test.columns)
    return data_train, data_test


def replace_nan(
    data_train: pd.DataFrame,
    data_test: Optional[pd.DataFrame] = None,
    method: Literal[
        "Gaussian", "Mean", "Median", "Zero", "Remove", "Custom1", None
    ] = None,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """The mean and std are calculated only with the training data."""
    if method is None:
        return data_train if data_test is None else data_train, data_test

    if method == "Remove":
        data_train = data_train.dropna()
        if data_test is None:
            return data_train

        data_test = data_test.dropna()
        return data_train, data_test

    for column in data_train.columns:
        if column in COLUMNS_STRING:
            continue

        if (
            data_train[column].isna().all() or method == "Zero"
        ):  # If all values are NaN, fill with 0
            func = lambda: 0
        elif method == "Gaussian":
            mean, std = get_mean_std(data_train, column)
            func = lambda: np.random.normal(mean, std)
        elif method == "Mean":
            mean = data_train[column].mean()
            func = lambda: mean
        elif method == "Median":
            median = data_train[column].median()
            func = lambda: median
        elif method == "Custom1":
            if (
                column in CORE_MATERIALS
            ):  # If it is a core material and the value is NaN, we assume the material is not used
                func = lambda: 0
            else:
                median = data_train[column].median()
                func = lambda: median

        data_train[column] = data_train[column].apply(
            lambda x: func() if pd.isna(x) else x
        )
        if data_test is not None:
            data_test[column] = data_test[column].apply(
                lambda x: func() if pd.isna(x) else x
            )

    if data_test is None:
        return data_train
    return data_train, data_test


def get_defined(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[data[column] != "N"][column]


def get_numeric(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.to_numeric(data[column], errors="coerce").dropna()


def get_proportions_enum(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[column].value_counts(normalize=True)


def get_mean_std(data: pd.DataFrame, column: str) -> Tuple[float, float]:
    values = get_defined(data, column)
    return float(values.mean()), float(values.std())


def get_min_max(data: pd.DataFrame, column: str) -> Tuple[float, float]:
    values = get_defined(data, column)
    return float(values.min()), float(values.max())


def one_hot_encode(data: pd.DataFrame, column: str, prefix: str = None) -> pd.DataFrame:
    encoded_columns = pd.get_dummies(data[column], prefix=prefix)
    data.drop(column, axis=1, inplace=True)
    return pd.concat([data, encoded_columns], axis=1)


def remove_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        data[column].replace(
            r"<(\d+)", r"\1", regex=True, inplace=True
        )  # Replace <99 by 99
    if "Nitrogen concentration" in data.columns:
        data["Nitrogen concentration"].replace(
            r"(\d+)tot(\d+|nd)res", r"\1", regex=True, inplace=True
        )  # Replace 99tot55res by 99
    if "Electrode positive or negative" in data.columns:
        data["Electrode positive or negative"].replace(
            r"\d+", "N", regex=True, inplace=True
        )  # Replace 0 by N
    if "Interpass temperature" in data.columns:
        data["Interpass temperature"].replace(
            r"\d+-\d+", "175", regex=True, inplace=True
        )  # Replace 150-200 by 175
    if "Hardness" in data.columns:
        data["Hardness"].replace(
            r"(\d+)\(?Hv\d+\)?", r"\1", regex=True, inplace=True
        )  # Replace 99(Hv30) by 99
    if "Type of weld" in data.columns:
        data["Type of Weld"].replace("ShMA","MMA",inplace=True)
    return data


def one_hot_encode_all(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in set(COLUMNS_STRING).intersection(columns):
        data = one_hot_encode(data, column)
    return data


def convert_to_float(
    data: pd.DataFrame,
    columns: List[str],
    errors: Literal["raise", "coerce", "ignore"] = "raise",
) -> pd.DataFrame:
    for column in set(COLUMNS_FLOAT).intersection(columns):
        data[column] = pd.to_numeric(data[column], errors=errors)
    return data


if __name__ == "__main__":
    print("Testing data.py...")
    X, y = get_data("Yield strength")
    print(X)
    print(y)
