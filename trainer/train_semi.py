import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import r2_score

from trainer.train import AverageModel, train


def train_semi(
    model_type,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    threshold=50,
    metric: callable = r2_score,
    n_folds: int = 10,
    print_results: bool = False,
    **model_params: dict,
) -> AverageModel:

    X_train_supervised = X_train.copy()
    y_train_supervised = y_train.copy()

    valid_indices = ~y_train_supervised.isna().any(axis=1)
    X_train_supervised = X_train_supervised[valid_indices]
    y_train_supervised = y_train_supervised[valid_indices]

    model_supervised = train(
        RandomForestQuantileRegressor,
        X_train_supervised,
        y_train_supervised,
        **model_params,
    )
    X_train = X_train[model_supervised.models[0].feature_names_in_]

    ######################################## SECOND TRAINING BEGINS ##########################################################

    predictions = model_supervised.models[0].predict(
        X_train, quantiles=[0.025, 0.5, 0.975]
    )

    y_interval = predictions[:, :, 2] - predictions[:, :, 0]
    mean = predictions[:, :, 0].mean(axis=0)
    # print(mean.shape, mean)
    small_interval = y_interval <= threshold
    small_interval = pd.DataFrame(
        small_interval, index=y_train.index, columns=y_train.columns
    )

    # Create a DataFrame from the predictions
    pred_df = pd.DataFrame(
        predictions[:, :, 1], index=y_train.index, columns=y_train.columns
    )

    condition = ~(y_train.isna() & small_interval)

    # print(
    #    "Number of values replaces using semi supervised :",
    #    (y_train.isna() & small_interval).sum(),
    # )

    # Use pandas' where method to replace NaN values in y_train with predicted values
    y_train = y_train.where(condition, pred_df)

    valid_indices = ~y_train.isna().any(axis=1)
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]

    model_semi = train(
        model_type,
        X_train,
        y_train,
        **model_params,
    )

    return model_semi
