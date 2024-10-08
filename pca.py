from typing import Optional, Tuple, Union

import pandas as pd
from sklearn.decomposition import PCA
import pandas as pd


def pca(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    n_components: Optional[int] = 10,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    if n_components is None:
        if X_test is None:
            return X_train
        return X_train, X_test
    
    pca = PCA(n_components=n_components)

    X_train_pca = pd.DataFrame(pca.fit_transform(X_train))
    if X_test is None:
        return X_train_pca

    X_test_pca = pd.DataFrame(pca.transform(X_test))
    return X_train_pca, X_test_pca
