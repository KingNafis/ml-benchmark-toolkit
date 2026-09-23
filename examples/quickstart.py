"""End-to-end quickstart: train two models, benchmark them, export a report.

Run with:
    python examples/quickstart.py
"""

from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlbenchmark import ModelComparator, Report

# 1. Synthetic multi-class dataset
X, y = make_classification(
    n_samples=1500,
    n_features=20,
    n_informative=10,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Train a couple of candidate models
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# A fabricated training history, e.g. for a neural-net-style model, to
# demonstrate the training-curve plot on one of the models.
fake_history = {
    "loss": [1.10, 0.85, 0.67, 0.52, 0.41, 0.34, 0.29, 0.26],
    "val_loss": [1.15, 0.95, 0.80, 0.70, 0.66, 0.64, 0.63, 0.64],
    "accuracy": [0.45, 0.58, 0.68, 0.75, 0.80, 0.84, 0.87, 0.89],
    "val_accuracy": [0.42, 0.55, 0.63, 0.68, 0.71, 0.72, 0.73, 0.72],
}

# 3. Register results with the comparator
comparator = ModelComparator()
comparator.add_model(
    name="random_forest",
    y_true=y_test,
    y_pred=rf.predict(X_test),
    y_proba=rf.predict_proba(X_test),
)
comparator.add_model(
    name="logistic_regression",
    y_true=y_test,
    y_pred=logreg.predict(X_test),
    y_proba=logreg.predict_proba(X_test),
    history=fake_history,
)

# 4. Ranked comparison table (DataFrame + Markdown)
table = comparator.comparison_table(rank_by="f1_macro")
print(table)
print()
print(comparator.comparison_markdown(rank_by="f1_macro"))

# 5. Standalone HTML dashboard with embedded confusion matrices, ROC
#    curves, and training curves
report = Report(comparator, title="Synthetic Classification Benchmark", rank_by="f1_macro")
html_path = report.to_html("benchmark_report.html")
report.to_markdown("benchmark_report.md")  # also writes the file to disk

print(f"\nHTML report written to: {html_path.resolve()}")
print("Markdown summary written to: benchmark_report.md")
