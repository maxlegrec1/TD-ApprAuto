from typing import List, Tuple, Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


COLUMNS = ['Carbon concentration', 'Silicon concentration', 'Manganese concentration', 'Sulphur concentration','Phosphorus concentration', 'Nickel concentration', 'Chromium concentration', 'Molybdenum concentration', 'Vanadium concentration', 'Copper concentration', 'Cobalt concentration', 'Tungsten concentration', 'Oxygen concentration', 'Titanium concentration', 'Nitrogen concentration', 'Aluminium concentration', 'Boron concentration', 'Niobium concentration', 'Tin concentration', 'Arsenic concentration', 'Antimony concentration', 'Current', 'Voltage', 'AC or DC', 'Electrode positive or negative', 'Heat input', 'Interpass temperature', 'Type of weld', 'Post weld heat treatment temperature', 'Post weld heat treatment time', 'Yield strength', 'Ultimate tensile strength', 'Elongation', 'Reduction of Area', 'Charpy temperature', 'Charpy impact toughness', 'Hardness', r'50 % FATT', 'Primary ferrite in microstructure', 'Ferrite with second phase', 'Acicular ferrite', 'Martensite', 'Ferrite with carbide aggreagate', 'Weld ID']

COLUMNS_STRING = ['AC or DC', 'Electrode positive or negative', 'Type of weld', 'Weld ID']
COLUMNS_FLOAT = ['Carbon concentration', 'Silicon concentration', 'Manganese concentration', 'Sulphur concentration','Phosphorus concentration', 'Nickel concentration', 'Chromium concentration', 'Molybdenum concentration', 'Vanadium concentration', 'Copper concentration', 'Cobalt concentration', 'Tungsten concentration', 'Oxygen concentration', 'Titanium concentration', 'Nitrogen concentration', 'Aluminium concentration', 'Boron concentration', 'Niobium concentration', 'Tin concentration', 'Arsenic concentration', 'Antimony concentration', 'Current', 'Voltage', 'Heat input', 'Interpass temperature', 'Post weld heat treatment temperature', 'Post weld heat treatment time', 'Yield strength', 'Ultimate tensile strength', 'Elongation', 'Reduction of Area', 'Charpy temperature', 'Charpy impact toughness', 'Hardness', r'50 % FATT', 'Primary ferrite in microstructure', 'Ferrite with second phase', 'Acicular ferrite', 'Martensite', 'Ferrite with carbide aggreagate']

TARGET_FEATURES = ['Yield strength', 'Ultimate tensile strength', 'Elongation', 'Reduction of Area', 'Charpy temperature', 'Charpy impact toughness', 'Hardness', r'50 % FATT', 'Primary ferrite in microstructure', 'Ferrite with second phase', 'Acicular ferrite', 'Martensite', 'Ferrite with carbide aggreagate']
FEATURES = ['Carbon concentration', 'Silicon concentration', 'Manganese concentration', 'Sulphur concentration','Phosphorus concentration', 'Nickel concentration', 'Chromium concentration', 'Molybdenum concentration', 'Vanadium concentration', 'Copper concentration', 'Cobalt concentration', 'Tungsten concentration', 'Oxygen concentration', 'Titanium concentration', 'Nitrogen concentration', 'Aluminium concentration', 'Boron concentration', 'Niobium concentration', 'Tin concentration', 'Arsenic concentration', 'Antimony concentration', 'Current', 'Voltage', 'AC or DC', 'Electrode positive or negative', 'Heat input', 'Interpass temperature', 'Type of weld', 'Post weld heat treatment temperature', 'Post weld heat treatment time']

WELD_TYPE = ['MMA', 'SA', 'FCA', 'TSA', 'ShMA', 'NGSAW', 'NGGMA', 'SAA', 'GTAA', 'GMAA']
ELECTRODE_TYPE = ['+', '-']
AC_DC = ['AC', 'DC']


def get_data(
    target_features: str | List[str] = TARGET_FEATURES,
    features: List[str] = FEATURES,
    filename: str = 'welddb/welddb.data',
    drop_y_nan_values: bool = False,
    nan_values: Literal['Gaussian', 'Mean', 'Median', 'Zero', 'Remove'] = 'Gaussian',
    test_size: None | float = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """If test_size is None, returns X, y.\n
    Otherwise, returns X_train, X_test, y_train, y_test."""
    if isinstance(target_features, str):
        target_features = [target_features]
    
    columns = features + target_features
    
    assert set(columns).issubset(COLUMNS), f"These columns are not in the data: {', '.join([col for col in columns if col not in COLUMNS])}"
    
    data = pd.read_csv(filename, delim_whitespace=True, header=None, names=COLUMNS)
    data = data[columns]

    remove_anomalies(data)
    data.replace('N', pd.NA, inplace=True)
    
    data = one_hot_encode_all(data, columns)
    data = convert_to_float(data, columns)
    
    if drop_y_nan_values:
        data.dropna(subset=target_features, inplace=True)
    
    X = data.drop(target_features, axis=1)  # Features
    y = data[target_features]               # Target
    
    if test_size is None:
        X: pd.DataFrame = replace_nan(X, method=nan_values)
        y: pd.DataFrame = replace_nan(y, method=nan_values)
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_test = replace_nan(X_train, X_test, method=nan_values)
    y_train, y_test = replace_nan(y_train, y_test, method=nan_values)

    return X_train, X_test, y_train, y_test


def replace_nan(data_train: pd.DataFrame, data_test: pd.DataFrame | None = None, method: Literal['Gaussian', 'Mean', 'Median', 'Zero', 'Remove'] = 'Gaussian') -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    """The mean and std are calculated only with the training data."""
    if method == 'Remove':
        data_train = data_train.dropna()
        if data_test is None:
            return data_train
            
        data_test = data_test.dropna()
        return data_train, data_test
    
    for column in data_train.columns:
        if data_train[column].isna().all() or method == 'Zero': # If all values are NaN, fill with 0
            func = lambda: 0
        elif method == 'Gaussian':
            mean, std = get_mean_std(data_train, column)
            func = lambda: np.random.normal(mean, std)
        elif method == 'Mean':
            mean = data_train[column].mean()
            func = lambda: mean
        elif method == 'Median':
            median = data_train[column].median()
            func = lambda: median

        data_train[column] = data_train[column].apply(lambda x: func() if pd.isna(x) else x)
        if data_test is not None:
            data_test[column] = data_test[column].apply(lambda x: func() if pd.isna(x) else x)
    
    if data_test is None:
        return data_train
    return data_train, data_test


def get_defined(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[data[column] != 'N'][column]


def get_numeric(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return pd.to_numeric(data[column], errors='coerce').dropna()


def get_proportions_enum(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[column].value_counts(normalize=True)


def get_mean_std(data: pd.DataFrame, column: str) -> Tuple[float, float]:
    values = get_defined(data, column)
    return float(values.mean()), float(values.std())


def one_hot_encode(data: pd.DataFrame, column: str, prefix: str = None) -> pd.DataFrame:
    encoded_columns = pd.get_dummies(data[column], prefix=prefix)
    data.drop(column, axis=1, inplace=True)
    return pd.concat([data, encoded_columns], axis=1)


def remove_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        data[column] = data[column].replace(r'<(\d+)', r'\1', regex=True) # Replace <99 by 99
    if 'Nitrogen concentration' in data.columns:
        data['Nitrogen concentration'] = data['Nitrogen concentration'].replace(r'\d+tot(\d+|nd)res', 'N', regex=True) # Replace 99tot99res by N
    if 'Electrode positive or negative' in data.columns:
        data['Electrode positive or negative'] = data['Electrode positive or negative'].replace(r'\d+', 'N', regex=True) # Replace 0 by N
    if 'Interpass temperature' in data.columns:
        data['Interpass temperature'] = data['Interpass temperature'].replace(r'\d+-\d+', '175', regex=True) # Replace 150-200 by 175
    if 'Hardness' in data.columns:
        data['Hardness'] = data['Hardness'].replace(r'(\d+)\(?Hv\d+\)?', r'\1', regex=True) # Replace 99(Hv30) by 99
    return data


def one_hot_encode_all(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in set(COLUMNS_STRING).intersection(columns):
        data = one_hot_encode(data, column)
    return data


def convert_to_float(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for column in set(COLUMNS_FLOAT).intersection(columns):
        data[column] = pd.to_numeric(data[column])
    return data


if __name__ == "__main__":
    print("Testing data.py...")
    X, y = get_data("Yield strength")
    print(X)
    print(y)