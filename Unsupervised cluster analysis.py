import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load Excel file, assuming data is in the first sheet
data = pd.read_excel('C:/Users/PlasticZzz/Desktop/ear/data1_shuffled.xlsx', sheet_name=0)

# Save column names
columns = data.columns

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Perform PCA dimensionality reduction
pca = PCA(n_components=8)
data_pca = pca.fit_transform(X_scaled)

# Plot cumulative explained variance
plt.figure(figsize=(10, 7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of Components', fontsize=14)
plt.ylabel('Cumulative Explained Variance', fontsize=14)
plt.title('PCA Cumulative Explained Variance', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig('C:/Users/PlasticZzz/Desktop/pca_cumulative_variance.tif', format='tif', dpi=1200)  # Increase resolution to 1200 DPI
plt.show()

# Display PCA components
print('PCA Components:')
print(pca.components_)

# Display the top 5 features contributing most to each principal component
for i, component in enumerate(pca.components_, start=1):
    print(f"\nTop 5 features contributing most to Principal Component {i}:")
    top_features = pd.Series(component, index=columns).nlargest(5)
    print(top_features)

# Find the optimal number of clusters
best_score = -1
best_k = -1
for k in range(6, 20):
    kmeans = KMeans(n_clusters=k, random_state=42)  # Adding random_state for reproducibility
    labels = kmeans.fit_predict(data_pca)
    score = silhouette_score(data_pca, labels)
    if score > best_score:
        best_score = score
        best_k = k
print(f'The optimal number of clusters is {best_k}, with a silhouette score of {best_score:.2f}')

# Perform clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)  # Adding random_state for reproducibility
kmeans.fit(data_pca)

# Visualize clustering results
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with transparency alpha=0.7 to reduce point density
scatter = ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2],
                     c=kmeans.labels_, cmap='viridis', alpha=0.7, edgecolors='w', s=60)

# Set axis labels and title
ax.set_xlabel('PC 1', fontsize=14)
ax.set_ylabel('PC 2', fontsize=14)
ax.set_zlabel('PC 3', fontsize=14)
ax.set_title('3D Scatter Plot of Clusters after PCA', fontsize=16)

# Display color bar
cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('Cluster Label', fontsize=14)

# Customize grid style
ax.grid(True, linestyle='--', alpha=0.6)

# Optionally customize axis ranges
ax.set_xlim(np.min(data_pca[:, 0]), np.max(data_pca[:, 0]))
ax.set_ylim(np.min(data_pca[:, 1]), np.max(data_pca[:, 1]))
ax.set_zlim(np.min(data_pca[:, 2]), np.max(data_pca[:, 2]))

plt.savefig('C:/Users/PlasticZzz/Desktop/3D_scatter_plot_clusters.tif', format='tif', dpi=1200)  # Increase resolution to 1200 DPI
plt.show()