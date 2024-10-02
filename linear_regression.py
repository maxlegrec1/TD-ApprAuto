from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from data import get_data
from train import print_scores, train


def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return model, train_mse, test_mse, train_r2, test_r2


X_train, X_test, y_train, y_test = get_data(
    target_features=["Yield strength", "Ultimate tensile strength"],
    test_size=0.2,
    drop_y_nan_values=True,
    nan_values="Median",
    random_state=1,
)

model = train(LinearRegression, X_train, y_train, print_results=True)
print_scores(model, X_train, X_test, y_train, y_test)
