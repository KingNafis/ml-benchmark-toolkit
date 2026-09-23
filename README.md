# ml-benchmark-toolkit

Lightweight, dependency-minimal Python toolkit (`mlbenchmark`) for evaluating,
comparing, and visually reporting on classification models — single-model or
side-by-side multi-model benchmarks.

## Features

- **Metrics**: accuracy, precision/recall/F1 (macro & weighted), log-loss, ROC-AUC
  (binary and multi-class One-vs-Rest), computed with graceful fallback when
  `y_proba` is unavailable.
- **Visualizations**: annotated confusion-matrix heatmaps (raw or normalized),
  ROC curves (multi-class OvR with per-class + micro-average AUC), and
  train-vs-validation training curves from a Keras-style `history` dict.
- **Multi-model comparison**: register any number of models and get a ranked
  `pandas.DataFrame` / Markdown comparison table.
- **One-line reporting**: `Report.to_html()` compiles all tables and figures
  into a single, standalone, self-contained HTML dashboard (figures are
  base64-embedded, no external assets required).

## Installation

```bash
pip install ml-benchmark-toolkit
```

Or, for local development:

```bash
git clone https://github.com/yourorg/ml-benchmark-toolkit.git
cd ml-benchmark-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

```python
from mlbenchmark import ModelComparator, Report

comparator = ModelComparator()
comparator.add_model("random_forest", y_true, y_pred_rf, y_proba_rf)
comparator.add_model("logistic_regression", y_true, y_pred_lr, y_proba_lr)

# Ranked comparison table
table = comparator.comparison_table(rank_by="f1_macro")
print(table)

# Standalone HTML dashboard: tables + confusion matrices + ROC curves
report = Report(comparator, title="My Benchmark")
report.to_html("report.html")
report.to_markdown("report.md")
```

See [`examples/quickstart.py`](examples/quickstart.py) for a full,
runnable end-to-end example (including training curves).

## API overview

| Module | Purpose |
|---|---|
| `mlbenchmark.metrics` | `compute_metrics(y_true, y_pred, y_proba=None)` -> dict |
| `mlbenchmark.plots` | `plot_confusion_matrix`, `plot_roc_curve`, `plot_training_curves` -> `matplotlib.figure.Figure` |
| `mlbenchmark.comparator` | `ModelComparator` — register models, build ranked comparison tables and per-model figure dicts |
| `mlbenchmark.report` | `Report` — compile a `ModelComparator` into a standalone HTML/Markdown dashboard |

## Development & Distribution

### 1. Local editable install & test

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python examples/quickstart.py
pytest
```

### 2. Build distribution artifacts

```bash
pip install --upgrade build
python -m build                  # produces dist/*.whl and dist/*.tar.gz
```

### 3. Publish to TestPyPI, then PyPI

```bash
pip install --upgrade twine

# TestPyPI first (use an API token, not your password)
twine upload --repository testpypi dist/* \
  -u __token__ -p "$TEST_PYPI_API_TOKEN"

# Verify install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ ml-benchmark-toolkit

# Production PyPI
twine upload dist/* -u __token__ -p "$PYPI_API_TOKEN"
```

Store tokens as environment variables or in `~/.pypirc` — never commit them.

### 4. Automated releases via GitHub Actions

See [`.github/workflows/publish.yml`](.github/workflows/publish.yml): the
workflow builds and publishes to PyPI automatically whenever a tag matching
`v*.*.*` is pushed, using PyPI's [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC — no long-lived API token stored in the repo).

```bash
git tag v0.1.0
git push origin v0.1.0
```

## License

MIT
