"""
Logistic Regression
===================
Binary classifier trained with gradient descent using the log-loss (cross-entropy)
cost function and the sigmoid activation function.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    """Binary logistic regression trained with gradient descent."""

    def __init__(self, learning_rate: float = 0.1, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = sigmoid(X @ self.weights + self.bias)
            error = y_pred - y

            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic linearly-separable dataset
    n = 200
    X0 = rng.normal(loc=[2, 2], scale=1.0, size=(n // 2, 2))
    X1 = rng.normal(loc=[5, 5], scale=1.0, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)])

    # Shuffle and split
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print(f"Test accuracy: {acc * 100:.1f}%")
