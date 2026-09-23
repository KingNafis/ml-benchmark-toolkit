"""Classification metric computation utilities.

This module provides a single entry point, :func:`compute_metrics`, that
takes true labels, predicted labels and (optionally) predicted
probabilities and returns a flat dictionary of standard classification
metrics. It defensively handles binary vs. multi-class problems and
missing probability estimates.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


ArrayLike = Sequence[Any]


def _to_numpy(arr: Optional[ArrayLike]) -> Optional[np.ndarray]:
    """Convert list-like/pandas input to a numpy array, passing through None."""
    if arr is None:
        return None
    return np.asarray(arr)


def is_binary(y_true: ArrayLike) -> bool:
    """Return True if `y_true` contains exactly two distinct classes."""
    return len(np.unique(_to_numpy(y_true))) <= 2


def compute_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_proba: Optional[ArrayLike] = None,
    *,
    average: str = "macro",
    labels: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """Compute a standard suite of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth class labels.
    y_pred : array-like
        Predicted class labels.
    y_proba : array-like, optional
        Predicted probabilities. For binary problems this may be a 1D
        array of positive-class probabilities or a 2D array of shape
        (n_samples, 2). For multi-class problems it must be a 2D array
        of shape (n_samples, n_classes). If omitted, log-loss and
        ROC-AUC are reported as NaN.
    average : str, default "macro"
        Averaging strategy for the secondary precision/recall/F1 metric
        (in addition to the always-computed macro and weighted variants).
    labels : sequence, optional
        Explicit ordering of class labels. Inferred from `y_true`/`y_pred`
        when not provided.

    Returns
    -------
    dict
        Dictionary of metric name -> float value. Metrics that cannot be
        computed (e.g. ROC-AUC without probabilities) are set to NaN
        rather than raising, so downstream reporting degrades gracefully.
    """
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)
    y_proba_arr = _to_numpy(y_proba)

    if y_true_arr is None or y_pred_arr is None:
        raise ValueError("y_true and y_pred are required and cannot be None.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"y_true (n={len(y_true_arr)}) and y_pred (n={len(y_pred_arr)}) "
            "must have the same length."
        )

    resolved_labels = list(labels) if labels is not None else sorted(
        set(np.unique(y_true_arr).tolist()) | set(np.unique(y_pred_arr).tolist())
    )
    binary = len(resolved_labels) <= 2

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
    }

    if average not in ("macro", "weighted"):
        metrics[f"precision_{average}"] = float(
            precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0)
        )
        metrics[f"recall_{average}"] = float(
            recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0)
        )
        metrics[f"f1_{average}"] = float(
            f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0)
        )

    metrics["log_loss"] = np.nan
    metrics["roc_auc"] = np.nan

    if y_proba_arr is not None:
        try:
            proba_for_logloss = y_proba_arr
            if binary and proba_for_logloss.ndim == 1:
                proba_for_logloss = np.column_stack(
                    [1 - proba_for_logloss, proba_for_logloss]
                )
            metrics["log_loss"] = float(
                log_loss(y_true_arr, proba_for_logloss, labels=resolved_labels)
            )
        except (ValueError, IndexError):
            metrics["log_loss"] = np.nan

        try:
            if binary:
                pos_proba = (
                    y_proba_arr
                    if y_proba_arr.ndim == 1
                    else y_proba_arr[:, 1]
                )
                metrics["roc_auc"] = float(roc_auc_score(y_true_arr, pos_proba))
            else:
                y_bin = label_binarize(y_true_arr, classes=resolved_labels)
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_bin, y_proba_arr, average="macro", multi_class="ovr"
                    )
                )
        except (ValueError, IndexError):
            metrics["roc_auc"] = np.nan

    return metrics
