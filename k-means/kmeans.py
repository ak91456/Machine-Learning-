import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("kmeans - kmeans_blobs.csv")

# Implementing K-Means from scratch
class KMeans:
    def __init__(self, k, iterations=100):
        self.k = k
        self.iteration = iteration
        self.centroids = None

    def centroid_initialize(self, X):
        np.random.seed(42)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices]

    def distance(self, X, centroids):
        dist = np.zeros((X.shape[0], self.k))
        for i in range(self.k):
            dist[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
        return dist

    def clusters(self, dist):
        return np.argmin(dist, axis=1)

    def new_centroids(self, X, labels):
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        return new_centroids

    def fit(self, X):
        self.centroids = self.centroid_initialize(X)
        for _ in range(self.iteration):
            dist = self.distance(X, self.centroids)
            labels = self.clusters(dist)
            new_centroids = self.new_centroids(X, labels)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        return labels

# Run K-Means with k=3 to assign pre-clustering labels
kmeans_initial = KMeans(k=3)
X = data[['x1', 'x2']].values
initial_labels = kmeans_initial.fit(X)  # Assign labels based on k=3 clustering

# Define colors matching the reference figure
pre_cluster_colors = ['purple', 'yellow', 'teal']
point_colors = [pre_cluster_colors[label] for label in initial_labels]

# Plot dataset before clustering with correct cluster-based colors
plt.figure(figsize=(6, 5))
for i in range(len(X)):
    plt.scatter(X[i, 0], X[i, 1], color=point_colors[i], alpha=0.6)

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dataset Before Clustering (Cluster-Based Coloring)")
plt.show()

# Running K-Means for k=2 and k=3 with correct colors
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Define color maps matching the reference figure
cluster_colors_k2 = ['purple', 'yellow']  # Colors for k=2
cluster_colors_k3 = ['purple', 'yellow', 'teal']  # Colors for k=3

for i, k in enumerate([2, 3]):
    kmeans = KMeans(k=k)
    labels = kmeans.fit(X)

    # Assign colors based on cluster labels
    cluster_colors = cluster_colors_k2 if k == 2 else cluster_colors_k3
    point_colors = [cluster_colors[label] for label in labels]

    # Plot clustered data with correct colors
    for j in range(len(X)):
        axes[i].scatter(X[j, 0], X[j, 1], color=point_colors[j], alpha=0.6, label="Data Points" if j == 0 else "")

    # Plot centroids in red
    axes[i].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

    axes[i].set_xlabel("x1")
    axes[i].set_ylabel("x2")
    axes[i].set_title(f"K-Means Clustering (k={k})")
    axes[i].legend()

plt.tight_layout()
plt.show()
