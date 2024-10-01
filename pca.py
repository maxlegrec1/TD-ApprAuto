from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from data import get_data

X_train, X_test, y_train, y_test = get_data(
    target_features=["Yield strength", "Ultimate tensile strength"],
    test_size=0.2,
    drop_y_nan_values=True,
    nan_values='Median'
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
            c=y_train["Yield strength"],
            cmap='viridis')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.title('PCA des données d\'entraînement')
plt.colorbar(label='Résistance à la traction (Yield strength)')
plt.show()

explained_variance = pca.explained_variance_ratio_
print(f'Variance expliquée par les premières composantes: {explained_variance}')
