from data import get_cross_validation_data
import pandas as pd
from sklearn.metrics import r2_score
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class AverageModel(BaseEstimator, RegressorMixin):
    def __init__(self, models: list):
        self.models = models
    
    def predict(self, X):
        preds = np.mean([model.predict(X) for model in self.models], axis=0)
        return preds


def train(model_type, X: pd.DataFrame, y: pd.DataFrame, metric: callable = r2_score, n_folds: int = 10, print_results: bool = False, **model_params: dict) -> AverageModel:
    train_metrics = []
    val_metrics = []
    models = []
    
    for X_train, X_val, y_train, y_val in get_cross_validation_data(X, y, n_folds=n_folds, random_state=1):
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
        print(f"{train_metrics = }\n    mean = {sum(train_metrics) / len(train_metrics)}\n    median = {train_median}")
        print(f"{val_metrics   = }\n    mean = {sum(val_metrics) / len(val_metrics)}\n    median = {val_median}")
    
    return AverageModel(models)


def print_scores(model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, metric: callable = r2_score):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_score = metric(y_train, y_train_pred)
    test_score = metric(y_test, y_test_pred)
    
    print(f"Train score: {train_score}")
    print(f"Test score: {test_score}")