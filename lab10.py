import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate synthetic data with two moons
X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)

# Initialize DBSCAN with epsilon=0.2 and min_samples=5
dbscan = DBSCAN(eps=0.2, min_samples=5)

# Fit DBSCAN to the data
dbscan.fit(X)

# Get the cluster labels (-1 for noise points)
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Plot the data points and cluster assignments
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5, edgecolors='k')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

print("Number of clusters:", n_clusters_)
print("Number of noise points:", n_noise_)
