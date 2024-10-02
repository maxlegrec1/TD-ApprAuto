from sklearn.ensemble import RandomForestRegressor

from data import get_data
from train import print_scores, train


target_features = ["Yield strength", "Ultimate tensile strength"]

X_train, X_test, y_train, y_test = get_data(
    target_features, test_size=0.2, drop_y_nan_values=True, nan_values="Median", n_pca=None
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

# feature_importances = model.feature_importances_
# features = X_train.columns
# indices = np.argsort(feature_importances)

# plt.figure(figsize=(10, 6))
# plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.title('Feature Importance in Random Forest')
# plt.show()
