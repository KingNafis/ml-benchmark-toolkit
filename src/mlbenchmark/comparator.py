"""Multi-model registration and side-by-side comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .metrics import compute_metrics
from .plots import plot_confusion_matrix, plot_roc_curve, plot_training_curves

ArrayLike = Sequence[Any]


@dataclass
class ModelResult:
    """Container for a single model's evaluation inputs and computed metrics."""

    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None
    history: Optional[Dict[str, List[float]]] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y_true = np.asarray(self.y_true)
        self.y_pred = np.asarray(self.y_pred)
        if self.y_proba is not None:
            self.y_proba = np.asarray(self.y_proba)
        self.metrics = compute_metrics(self.y_true, self.y_pred, self.y_proba)


class ModelComparator:
    """Registers evaluation results for multiple models and compares them.

    Examples
    --------
    >>> comparator = ModelComparator()
    >>> comparator.add_model("random_forest", y_true, y_pred, y_proba)
    >>> comparator.add_model("xgboost", y_true, y_pred2, y_proba2)
    >>> table = comparator.comparison_table(rank_by="f1_macro")
    """

    def __init__(self) -> None:
        self._results: Dict[str, ModelResult] = {}

    def add_model(
        self,
        name: str,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        y_proba: Optional[ArrayLike] = None,
        history: Optional[Dict[str, List[float]]] = None,
    ) -> "ModelComparator":
        """Register a model's predictions for comparison. Returns self for chaining."""
        if name in self._results:
            raise ValueError(f"A model named '{name}' has already been added.")
        self._results[name] = ModelResult(
            name=name, y_true=y_true, y_pred=y_pred, y_proba=y_proba, history=history
        )
        return self

    @property
    def model_names(self) -> List[str]:
        return list(self._results.keys())

    def get_result(self, name: str) -> ModelResult:
        if name not in self._results:
            raise KeyError(f"No model named '{name}' has been registered.")
        return self._results[name]

    def comparison_table(
        self,
        rank_by: str = "f1_macro",
        ascending: bool = False,
        metrics: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Build a ranked Pandas DataFrame comparing all registered models.

        Parameters
        ----------
        rank_by : str, default "f1_macro"
            Metric column name to sort the table by.
        ascending : bool, default False
            Sort order; metrics like log_loss are usually best ascending.
        metrics : sequence of str, optional
            Subset of metric columns to include. Defaults to all computed
            metrics.
        """
        if not self._results:
            raise ValueError("No models have been added yet. Call add_model() first.")

        rows = []
        for name, result in self._results.items():
            row = {"model": name}
            row.update(result.metrics)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("model")

        if metrics is not None:
            missing = set(metrics) - set(df.columns)
            if missing:
                raise ValueError(f"Unknown metric(s) requested: {sorted(missing)}")
            df = df[list(metrics)]

        if rank_by not in df.columns:
            raise ValueError(
                f"rank_by='{rank_by}' is not a computed metric. "
                f"Available metrics: {list(df.columns)}"
            )

        df = df.sort_values(by=rank_by, ascending=ascending)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df

    def comparison_markdown(
        self, rank_by: str = "f1_macro", ascending: bool = False, decimals: int = 4
    ) -> str:
        """Return the comparison table formatted as a Markdown string."""
        df = self.comparison_table(rank_by=rank_by, ascending=ascending)
        return df.round(decimals).to_markdown()

    def confusion_matrices(self, normalize: bool = False) -> Dict[str, Figure]:
        """Return a dict of model name -> confusion-matrix Figure."""
        return {
            name: plot_confusion_matrix(
                result.y_true,
                result.y_pred,
                normalize=normalize,
                title=f"Confusion Matrix — {name}",
            )
            for name, result in self._results.items()
        }

    def roc_curves(self) -> Dict[str, Figure]:
        """Return a dict of model name -> ROC-curve Figure (skips models without y_proba)."""
        figures = {}
        for name, result in self._results.items():
            if result.y_proba is None:
                continue
            figures[name] = plot_roc_curve(
                result.y_true, result.y_proba, title=f"ROC Curve — {name}"
            )
        return figures

    def training_curves(self) -> Dict[str, Figure]:
        """Return a dict of model name -> training-curve Figure (skips models without history)."""
        figures = {}
        for name, result in self._results.items():
            if result.history is None:
                continue
            figures[name] = plot_training_curves(
                result.history, title=f"Training History — {name}"
            )
        return figures
