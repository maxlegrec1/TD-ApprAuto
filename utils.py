import pandas as pd


def load_dataset(fold=1):

    train_path = f"folds/{fold}_train.csv"
    test_path = f"folds/{fold}_test.csv"

    train_ds = pd.read_csv(train_path)
    test_ds = pd.read_csv(test_path)

    # Yield strength / MPa is collumn 30 and Ultimate tensile strength / MPa is collumn 31

    X_train = pd.concat([train_ds.iloc[:, :27], train_ds.iloc[:, 29:]], axis=1)
    Y_train = train_ds.iloc[:, 27:29]

    X_test = pd.concat([test_ds.iloc[:, :27], test_ds.iloc[:, 29:]], axis=1)
    Y_test = test_ds.iloc[:, 27:29]

    return (X_train, Y_train), (X_test, Y_test)


# (X_train, Y_train), (X_test, Y_test) = load_dataset()


def keep_known_Y(X, Y):

    mask = ~Y.isin(["N"]).any(axis=1)
    X_filtered = X[mask]
    Y_filtered = Y[mask]

    return X_filtered, Y_filtered


# X_f, Y_f = keep_known_Y(X_train, Y_train)
