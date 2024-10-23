from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    n_components: Optional[int] = 10,
    plot: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    if n_components is None:
        if X_test is None:
            return X_train
        return X_train, X_test

    pca = PCA(n_components=n_components)

    X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca.iloc[:, 0], X_train_pca.iloc[:, 1], alpha=0.25)
        for ax in range(X_train.shape[1]):
            important_names = {"Molybdenum concentration", "Manganese concentration", "Chromium concentration"}
            name: str = X_train.columns[ax]
            arrow_color = '#00ff00' if name in important_names else 'red'
            plt.arrow(0, 0, 10*pca.components_[0, ax], 10*pca.components_[1, ax], head_width=0.1, head_length=0.1, fc=arrow_color, ec=arrow_color)
            plt.text(10*pca.components_[0, ax], 10*pca.components_[1, ax], name, color='black', ha='right', fontsize=6)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Training Data')
        plt.grid(True)
        plt.show()
    
    if X_test is None:
        return X_train_pca

    X_test_pca = pd.DataFrame(pca.transform(X_test))
    return X_train_pca, X_test_pca
