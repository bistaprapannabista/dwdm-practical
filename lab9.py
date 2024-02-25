import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn_extra.cluster import KMedoids

# Generate synthetic data with 3 clusters
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Initialize K-medoids with 3 clusters
kmedoids = KMedoids(n_clusters=3, random_state=0)

# Fit K-medoids to the data
kmedoids.fit(X)

# Get the cluster medoids and labels
medoids = kmedoids.cluster_centers_
labels = kmedoids.labels_

# Plot the data points and cluster medoids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5, edgecolors='k')
plt.scatter(medoids[:, 0], medoids[:, 1], marker='X', s=200, c='red', label='Medoids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-medoids Clustering')
plt.legend()
plt.grid(True)
plt.show()
