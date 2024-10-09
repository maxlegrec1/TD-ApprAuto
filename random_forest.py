import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from data import COLUMNS_STRING, FEATURES, get_data
from train import print_scores, train

target_features = ["Yield strength", "Ultimate tensile strength"]

test_scores_model1 = []
test_scores_model2 = []
for random_state in tqdm(range(0, 50, 5)):
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

    with open("model.pkl", "wb") as file:
        pickle.dump(model1, file, pickle.HIGHEST_PROTOCOL)
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

    # load the supervised model
    with open("model.pkl", "rb") as f:
        model1 = pickle.load(f)

    # reorder columns so that the order match the model's expectation
    X_train = X_train[model1.models[0].feature_names_in_]
    X_test = X_test[model1.models[0].feature_names_in_]
    """
    for i in range(X_train.shape[0]):
        pred = model1.predict(X_train.iloc[i : i + 1])[0]
        if np.isnan(np.sum(y_train.iloc[i, 0])):
            y_train.iloc[i, 0] = pred[0]
        if np.isnan(np.sum(y_train.iloc[i, 1])):
            y_train.iloc[i, 1] = pred[1]
    """
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
