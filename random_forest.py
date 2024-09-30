from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
from data import get_data
from sklearn.metrics import r2_score


target_features = ['Yield strength', 'Ultimate tensile strength']

X_train, X_test, y_train, y_test = get_data(target_features, test_size=0.2, drop_y_nan_values=True)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nLinear Regression Results:")
print(f"Training R2 Score: {train_r2:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")

feature_importances = model.feature_importances_
features = X_train.columns
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Feature Importance in Random Forest')
plt.show()
