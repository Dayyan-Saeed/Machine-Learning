"""
Linear Regression
=================
Fits a linear model  y = X @ w + b  using the closed-form ordinary-least-squares
solution and gradient descent, then evaluates mean-squared error on synthetic data.
"""

import numpy as np


class LinearRegression:
    """Ordinary least-squares linear regression trained with gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic dataset: y = 3x + 5 + noise
    X = rng.uniform(0, 10, size=(100, 1))
    y = 3 * X.squeeze() + 5 + rng.normal(0, 1, size=100)

    split = 80
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Learned weight : {model.weights[0]:.4f}  (true: 3.0)")
    print(f"Learned bias   : {model.bias:.4f}  (true: 5.0)")
    print(f"Test MSE       : {mse:.4f}")
