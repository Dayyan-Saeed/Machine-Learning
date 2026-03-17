# Machine Learning

A repository of machine learning concepts, algorithms, and implementations in Python.

## Contents

| Topic | Description |
|-------|-------------|
| [Supervised Learning](supervised_learning/) | Algorithms that learn from labelled training data |
| [Unsupervised Learning](unsupervised_learning/) | Algorithms that discover patterns in unlabelled data |
| [Neural Networks](neural_networks/) | Feedforward and multi-layer neural network implementations |
| [Data](data/) | Sample datasets and data-loading utilities |

## Algorithms Covered

### Supervised Learning
- **Linear Regression** – predict continuous values by fitting a line to training data
- **Logistic Regression** – binary classification using the sigmoid function

### Unsupervised Learning
- **K-Means Clustering** – partition data into *k* clusters by iteratively updating centroids

### Neural Networks
- **Feedforward Neural Network** – fully-connected network with configurable hidden layers trained with backpropagation

## Requirements

```
numpy
```

Install with:

```bash
pip install numpy
```

## Usage

Each subdirectory contains a standalone Python script. Run any of them directly, for example:

```bash
python supervised_learning/linear_regression.py
python supervised_learning/logistic_regression.py
python unsupervised_learning/k_means.py
python neural_networks/neural_network.py
```
