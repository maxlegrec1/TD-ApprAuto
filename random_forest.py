import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data import COLUMNS_STRING, FEATURES, get_data
from train import print_scores, train

target_features = ["Yield strength", "Ultimate tensile strength", "Elongation", "Reduction of Area"]#, "Charpy impact toughness"]



def quality(row: pd.Series) -> float:
    if row.isna().all():
        return np.nan
    
    inv_materials = set(["Martensite", "Ferrite with carbide aggregate", r"50 % FATT"])
    s = 0.0
    n = 0
    for material in row.index:
        if not np.isnan(row[material]):
            s += (-1.0 if row[material] in inv_materials else 1.0) * row[material]
            n += 1
    
    return s / n


X_train, X_test, y_train, y_test = get_data(
    target_features,
    # set(FEATURES) - set(COLUMNS_STRING),
    test_size=0.2,
    drop_y_nan_values=True,
    nan_values="Custom1",
    # n_pca=15,
    random_state=1,
    # quality=quality,
)

model = train(
    RandomForestRegressor,
    X_train,
    y_train,
    n_estimators=100,
    random_state=42,
    print_results=True,
)

print_scores(model, X_train, X_test, y_train, y_test)

feature_importances = model.models[0].feature_importances_
features = X_train.columns
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), feature_importances[indices], color="b", align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.title("Feature Importance in Random Forest")
plt.show()
