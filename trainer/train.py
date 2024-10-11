import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

from utils.data import get_cross_validation_data


class AverageModel(BaseEstimator, RegressorMixin):
    def __init__(self, models: list, mode: str = "median"):
        self.models = models
        self.mode = mode

    def predict(self, X):
        if self.mode == "mean":
            preds = np.mean([model.predict(X) for model in self.models], axis=0)
        elif self.mode == "median":
            preds = np.median([model.predict(X) for model in self.models], axis=0)
        elif self.mode == "single":
            preds = np.array([self.models[0].predict(X)])
        else:
            raise ValueError("Incorrect Mode entered")
        return preds


def train(
    model_type,
    X: pd.DataFrame,
    y: pd.DataFrame,
    metric: callable = r2_score,
    n_folds: int = 10,
    print_results: bool = False,
    **model_params: dict,
) -> AverageModel:
    """
    Train a model using cross-validation and return an AverageModel object.

    @param model_type: The model class to use.
    @param X: The input data.
    @param y: The target data.
    @param metric: The metric to evaluate the model.
    @param n_folds: The number of cross-validation folds.
    @param print_results: Whether to print intermediate results or not.
    @param model_params: The parameters to pass to the model constructor.
    """
    train_metrics = []
    val_metrics = []
    models = []

    for X_train, X_val, y_train, y_val in get_cross_validation_data(
        X, y, n_folds=n_folds, random_state=1
    ):
        model = model_type(**model_params)
        model.fit(X_train, y_train)
        models.append(model)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_metrics.append(metric(y_train, y_train_pred))
        val_metrics.append(metric(y_val, y_val_pred))

    if print_results:
        train_median = np.median(train_metrics)
        val_median = np.median(val_metrics)

        print("Results:")
        print(
            f"{train_metrics = }\n    mean = {sum(train_metrics) / len(train_metrics)}\n    median = {train_median}"
        )
        print(
            f"{val_metrics   = }\n    mean = {sum(val_metrics) / len(val_metrics)}\n    median = {val_median}"
        )

    return AverageModel(models)


def print_scores(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metric: callable = r2_score,
):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_score = metric(y_train, y_train_pred)
    test_score = metric(y_test, y_test_pred)

    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")

    return test_score


def calculate_scores(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    metrics={"r2_score": r2_score, "mse": mean_squared_error},
):
    # makes predictions
    y_train_pred = model.predict(X_train)
    y_train_pred = y_train_pred.reshape(y_train_pred.shape[0], y_train_pred.shape[1] if len(y_train_pred.shape) > 1 else 1)
    y_test_pred = model.predict(X_test)
    y_test_pred = y_test_pred.reshape(y_test_pred.shape[0], y_test_pred.shape[1] if len(y_test_pred.shape) > 1 else 1)

    result_dir = {}

    # calculate global metrics
    for metric_name, metric in metrics.items():

        train_score = metric(y_train, y_train_pred)
        test_score = metric(y_test, y_test_pred)
        result_dir[f"{metric_name}_train"] = train_score
        result_dir[f"{metric_name}_test"] = test_score
    # calculate label-wise metrics
    for metric_name, metric in metrics.items():
        for column in y_train.columns:
            train_score = metric(
                y_train[column], y_train_pred[:, y_train.columns.get_loc(column)]
            )
            test_score = metric(
                y_test[column], y_test_pred[:, y_test.columns.get_loc(column)]
            )
            result_dir[f"{metric_name}_{column}_train"] = train_score
            result_dir[f"{metric_name}_{column}_test"] = test_score

    return result_dir
