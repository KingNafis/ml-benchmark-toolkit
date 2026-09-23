# ML Benchmark Toolkit (`mlbenchmark`)

[![PyPI version](https://img.shields.io/pypi/v/ml-benchmark-toolkit.svg)](https://pypi.org/project/ml-benchmark-toolkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/ml-benchmark-toolkit.svg)](https://pypi.org/project/ml-benchmark-toolkit/)

A lightweight Python library to automate classification model evaluation, side-by-side benchmarking, and visual reporting.

---

## Features
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score (macro and weighted), Log-Loss, and ROC-AUC.
- **Model Comparison Table**: Compare multiple models at once, ranked by any metric into a Pandas DataFrame.
- **Visuals**: One-line annotated confusion matrix heatmaps and multi-model ROC curves (binary and multiclass OvR).
- **Zero-Friction API**: Minimal dependencies (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`).

---

## Installation

### From PyPI
```bash
pip install ml-benchmark-toolkit
```

### From Source (Editable Mode)
```bash
git clone [https://github.com/KingNafis/ml-benchmark-toolkit.git](https://github.com/KingNafis/ml-benchmark-toolkit.git)
cd ml-benchmark-toolkit
pip install -e .
```

---

## Quickstart

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlbenchmark import BenchmarkSuite

# 1. Prepare data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train candidate models
rf = RandomForestClassifier().fit(X_train, y_train)
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# 3. Initialize suite with test ground truth
suite = BenchmarkSuite(y_true=y_test)
suite.add_model("Random Forest", y_pred=rf.predict(X_test), y_proba=rf.predict_proba(X_test))
suite.add_model("Logistic Regression", y_pred=lr.predict(X_test), y_proba=lr.predict_proba(X_test))

# 4. Generate comparison table ranked by F1 Score
print(suite.summary_table(sort_by="f1_macro"))

# 5. Save visual comparison charts
suite.plot_roc_curves(save_path="roc_curve.png")
suite.plot_confusion_matrices(save_path="confusion_matrices.png")
```

---

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
