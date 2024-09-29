from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from data import get_data

# import utils

# (X_train, y_train), (X_test, y_test) = utils.load_dataset(fold=1)


# X_train, y_train = utils.keep_known_Y(X_train, y_train)
# X_test, y_test = utils.keep_known_Y(X_test, y_test)
# Y_train, y_test = y_train["31"], y_test["31"]

X_train, X_test, y_train, y_test = get_data(["Yield strength", "Ultimate tensile strength"], test_size=0.2, drop_y_nan_values=True)


def train_linear_regression(X_train, X_test, y_train, y_test):
    # Initialize the linear regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on both training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return model, train_mse, test_mse, train_r2, test_r2


model, train_mse, test_mse, train_r2, test_r2 = train_linear_regression(
    X_train, X_test, y_train, y_test
)


print("\nLinear Regression Results:")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Training R2 Score: {train_r2:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")

# If you want to see the coefficients and intercept of the model
# print("\nModel Coefficients:")
# for feature, coef in zip(X_train.columns, model.coef_):
#     print(f"{feature}: {coef:.4f}")
# print(f"Intercept: {model.intercept_[0]:.4f}")
