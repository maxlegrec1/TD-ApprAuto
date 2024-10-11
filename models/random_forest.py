import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from trainer.train import print_scores, train
from utils.data import COLUMNS_STRING, FEATURES, get_data

target_features = ["Yield strength", "Ultimate tensile strength"]

test_scores_model1 = []
test_scores_model2 = []
# for random_state in tqdm(range(0, 50, 5)):
for random_state in tqdm([1]):
    X_train, X_test, y_train, y_test = get_data(
        target_features,
        test_size=0.2,
        drop_y_nan_values=True,
        nan_values="Custom1",
        random_state=random_state,
    )
    model1 = train(
        RandomForestRegressor,
        X_train,
        y_train,
        n_estimators=100,
        random_state=random_state,
    )

    test_scores_model1.append(print_scores(model1, X_train, X_test, y_train, y_test))

    ######################################## SECOND TRAINING BEGINS ##########################################################

    X_train, X_test, y_train, y_test = get_data(
        target_features,
        test_size=0.2,
        drop_y_nan_values=False,
        nan_values="Custom1",
        random_state=random_state,
    )
    # drop rows from x_test and y_test where there is a nan in y_test
    valid_indices = ~y_test.isna().any(axis=1)
    # Use these indices to filter both X_test and y_test
    X_test = X_test[valid_indices]
    y_test = y_test[valid_indices]

    # reorder columns so that the order match the model's expectation
    X_train = X_train[model1.models[0].feature_names_in_]
    X_test = X_test[model1.models[0].feature_names_in_]

    predictions = model1.predict(X_train)

    # Create a DataFrame from the predictions
    pred_df = pd.DataFrame(predictions, index=y_train.index, columns=y_train.columns)

    # Use pandas' where method to replace NaN values in y_train with predicted values
    y_train = y_train.where(y_train.notna(), pred_df)

    model2 = train(
        RandomForestRegressor,
        X_train,
        y_train,
        n_estimators=100,
        random_state=random_state,
    )

    test_scores_model2.append(print_scores(model2, X_train, X_test, y_train, y_test))


plt.figure(figsize=(10, 6))  # Increase figure size for better visibility

plt.plot(test_scores_model1, label="Supervised", color="r", linewidth=2)
plt.plot(test_scores_model2, label="Semi Supervised", color="b", linewidth=2)

plt.title("Supervised vs Semi Supervised", fontsize=16)
plt.xlabel("Random State", fontsize=12)
plt.ylabel("R2 Score", fontsize=12)

plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
