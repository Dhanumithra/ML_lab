# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)

# 2. Display the first five records
print(df.head())

# 3. Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Display the transformed dataset
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
print(pca_df)

# 6. Find the explained variance ratio
print(pca.explained_variance_ratio_)


# 7. Calculate total variance retained
total_variance = pca.explained_variance_ratio_.sum()
print("Total Variance Retained:", total_variance * 100)

# 9. Compare dataset shape before and after PCA
print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)

# 8. Plot a scatter graph of PC1 versus PC2
plt.figure(figsize=(8, 6))

colors = ['red', 'green', 'blue']

for i in range(3):
    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        color=colors[i],
        label=iris.target_names[i]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.grid(True)

plt.show()