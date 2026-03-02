"""
K-Means Clustering
==================
Partitions data into k clusters by iteratively assigning each point to the nearest
centroid and then recomputing centroids as cluster means.
"""

import numpy as np


class KMeans:
    """K-Means clustering using Euclidean distance."""

    def __init__(self, k: int = 3, max_iterations: int = 100, random_state: int = 42):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.k, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iterations):
            labels = self._assign(X)
            new_centroids = np.array(
                [
                    X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                    for j in range(self.k)
                ]
            )
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign(X)

    def _assign(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)


def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Sum of squared distances from each point to its assigned centroid."""
    return float(
        sum(
            np.sum((X[labels == j] - centroids[j]) ** 2)
            for j in range(len(centroids))
        )
    )


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic dataset with 3 distinct clusters
    centers = [(1, 1), (5, 5), (9, 1)]
    X = np.vstack(
        [rng.normal(loc=c, scale=0.8, size=(50, 2)) for c in centers]
    )

    model = KMeans(k=3, max_iterations=100, random_state=42)
    model.fit(X)
    labels = model.predict(X)

    score = inertia(X, labels, model.centroids)
    print("Centroids found:")
    for i, c in enumerate(model.centroids):
        print(f"  Cluster {i}: ({c[0]:.2f}, {c[1]:.2f})")
    print(f"Inertia: {score:.2f}")
