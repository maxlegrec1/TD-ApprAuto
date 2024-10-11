from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from utils.data import get_data
from utils.train import print_scores, train

X_train, X_test, y_train, y_test = get_data(
    target_features=["Yield strength", "Ultimate tensile strength"],
    test_size=0.2,
    drop_y_nan_values=True,
    nan_values="Median",
    random_state=1,
)

model = train(LinearRegression, X_train, y_train, print_results=True)
print_scores(model, X_train, X_test, y_train, y_test)
