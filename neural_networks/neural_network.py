"""
Feedforward Neural Network
==========================
A fully-connected (dense) neural network with configurable hidden layers trained
via backpropagation and stochastic gradient descent.

Architecture:  Input → [Hidden layers with ReLU] → Output layer with sigmoid
Loss function: Binary cross-entropy
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


class NeuralNetwork:
    """Feedforward neural network with one or more hidden layers."""

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
    ):
        """
        Parameters
        ----------
        layer_sizes : list of ints
            Number of units in each layer including input and output.
            Example: [2, 4, 1] → 2 inputs, 4 hidden units, 1 output.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._init_params()

    def _init_params(self) -> None:
        rng = np.random.default_rng(42)
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            # He initialisation for ReLU layers
            scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def _forward(self, X: np.ndarray) -> tuple[list, list]:
        activations = [X]
        pre_activations = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            pre_activations.append(z)
            # Use sigmoid for the output layer, ReLU for hidden layers
            a = sigmoid(z) if i == len(self.weights) - 1 else relu(z)
            activations.append(a)
        return activations, pre_activations

    def _backward(
        self,
        activations: list,
        pre_activations: list,
        y: np.ndarray,
    ) -> tuple[list, list]:
        n = y.shape[0]
        dw_list: list[np.ndarray] = [None] * len(self.weights)  # type: ignore[list-item]
        db_list: list[np.ndarray] = [None] * len(self.biases)  # type: ignore[list-item]

        # Output layer gradient (binary cross-entropy + sigmoid)
        delta = activations[-1] - y
        for i in reversed(range(len(self.weights))):
            dw_list[i] = (1 / n) * activations[i].T @ delta
            db_list[i] = (1 / n) * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(
                    pre_activations[i - 1]
                )
        return dw_list, db_list

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetwork":
        y = y.reshape(-1, 1)
        for _ in range(self.n_iterations):
            activations, pre_activations = self._forward(X)
            dw_list, db_list = self._backward(activations, pre_activations, y)
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dw_list[i]
                self.biases[i] -= self.learning_rate * db_list[i]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X)
        return activations[-1].squeeze()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic XOR-like dataset (non-linearly separable)
    n = 400
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float)

    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Network: 2 inputs → 8 hidden → 8 hidden → 1 output
    model = NeuralNetwork(
        layer_sizes=[2, 8, 8, 1],
        learning_rate=0.05,
        n_iterations=2000,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(f"Test accuracy on XOR dataset: {acc * 100:.1f}%")
