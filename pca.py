from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


def pca(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None, n_components: int | None = 10) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
    if n_components is None:
        if X_test is None:
            return X_train
        return X_train, X_test
    
    # scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
    if X_test is None:
        return X_train_pca

    X_test_pca = pd.DataFrame(pca.transform(X_test))
    return X_train_pca, X_test_pca
