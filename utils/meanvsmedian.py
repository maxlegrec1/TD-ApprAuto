import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils.data import COLUMNS_STRING, FEATURES, get_data
from utils.train import print_scores, train

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

    model1.mode = "median"

    test_scores_model2.append(print_scores(model1, X_train, X_test, y_train, y_test))


plt.plot(test_scores_model1, label="Supervised", color="r")

plt.plot(test_scores_model2, label="Semi Supervised", color="b")

plt.title("Supervised vs Semi Supervised")

plt.show()
