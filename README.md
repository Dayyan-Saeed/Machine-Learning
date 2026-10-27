# Machine Learning Foundations & Feature Engineering

Welcome to my comprehensive **Machine Learning** repository! This project serves as a structured, hands-on guide covering essential machine learning algorithms, critical data preprocessing techniques, and robust feature engineering workflows using Python and scikit-learn.

---

## 📂 Repository Structure

The repository is organized into distinct categories, tracking the journey from raw data exploration to advanced predictive modeling.

### 🔍 1. Data Exploration & Descriptive Statistics
Before modeling, understanding data distributions and behaviors is essential:
* `understanding-your-data-descriptive-stats` — Initial data profiling, shape analysis, and basic metrics.
* `univariate-analysis` — Exploring individual variables independently.
* `bivariate-analysis` — Analyzing relationships between pairs of variables.
* `pandas-profiling` — Automated exploratory data analysis reports.

### 🛠️ 2. Data Ingestion & API Pipelines
Real-world data lives in various formats and across networks:
* `Api-to-dataframe` — Fetching live data via APIs and converting it into structured DataFrames.
* `Pandas-dataframe-using-web-scraping` — Gathering tabular data directly from web scraping workflows.
* `Working with csv files` — Standard file operations and optimization tricks.
* `Working-with-json-and-sql` — Parsing unstructured nested JSON data and executing relational SQL queries.

### 🧼 3. Feature Engineering & Preprocessing
Data preparation determines model quality. This section handles missing data, encoding, scaling, and distribution transformations:
* **Missing Value Imputation**:
  * `complete-case-analysis` — Handling missing data through listwise deletion.
  * `imputing-numerical-data` — Statistical measures (Mean/Median) for numeric gaps.
  * `handling-missing-categorical-data` — Mode and frequent category replacements.
  * `missing-indicator` — Flagging missing values explicitly to retain variance information.
  * `knn-imputer` — K-Nearest Neighbors multivariate imputation.
  * `iterative-imputer` — Advanced multivariate imputation by modeling each feature as a function of others.
* **Categorical Encoding**:
  * `ordinal-encoding` — Converting ordered categories into rank integers.
  * `one-hot-encoding` — Transforming nominal fields into binary columns.
* **Numerical Scaling & Transformation**:
  * `standardization` — Centering distributions around zero with unit variance.
  * `normalization` — Scaling ranges directly between a fixed bound (0 and 1).
  * `binning-and-binarization` — Discretizing continuous data into categorical buckets or binary thresholds.
  * `function-transformer` — Applying custom log, exponential, or reciprocal mathematical edits.
  * `power-transformer` — Stabilizing variance using Box-Cox or Yeo-Johnson transforms to yield normal distributions.
* **Complex Feature Interactions**:
  * `handling-mixed-variables` — Parsing fields containing both numeric strings and text markers.
  * `handling-date-and-time` — Extracting operational features like day, month, hour, and weekend statuses.
  * `feature-construction-and-feature-splitting` — Engineering completely new indicators or splitting text composites.

### ⚡ 4. Advanced Scikit-Learn Workflows
* `column-transformer` — Applying entirely different preprocessing pipelines to distinct columns simultaneously.
* `sklearn-pipelines` — Chaining preprocessing blocks cleanly with estimator classes for leakage-free cross-validation.

### 🚨 5. Outlier Engineering
Methods for detecting, trimming, or capping distribution extremes:
* `outlier-detection-using-percentiles` — Quantile-based filtering strategies.
* `outlier-removal-using-zscore` — Identifying data extremes based on standard deviation distances.
* `outlier-removal-using-iqr-method` — Leveraging the Interquartile Range to build non-parametric box boundaries.

### 📉 6. Regression Models & Optimizers
* `simple-linear-regression` — Single-variable continuous targets.
* `multiple-linear-regression` — Multi-variable linear combinations.
* `polynomial-regression` — Engineering non-linear feature degrees for curved trends.
* `gradient-descent` & `types-of-gradient-descent` — Deep dive into optimization routines (Batch, Stochastic, and Mini-Batch).
* **Regularized Models**:
  * `lasso-regression` — L1 Regularization for embedded feature selection.
  * `lasso-regression` — L2 Regularization to combat multicollinearity.
  * `elasticnet-regression` — Combining L1 and L2 penalties linearly.
  * `regularized-linear-models` — Comprehensive comparative suite.

### 📊 7. Classification Models
* `logistic-regression` & `logistic-regression-contd` — Foundational probabilistic binary and multi-class classification.

### 🌲 8. Ensemble Methods & Dimensionality Reduction
* `adaboost` — Adaptive boosting algorithms focusing sequentially on hard-to-classify samples.
* `random-forest` — Bootstrap aggregating (bagging) over decorative decision tree branches.
* `pca` — Principal Component Analysis for orthogonal dimensionality reduction and noise extraction.

### 📐 9. Validation & Evaluation Metrics
* `regression-metrics` — Measuring continuous error (MAE, MSE, RMSE, R²).
* `classification-metrics` — Gauging discrete performance (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

---

## 🛠️ Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your computer.

### Installation
Clone this repository and enter the directory:
```bash
git clone https://github.com
cd Machine-Learning
```

Install the core dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Usage
Launch Jupyter Notebook or JupyterLab to interactively explore and run any of the topic directories:
```bash
jupyter notebook
```
