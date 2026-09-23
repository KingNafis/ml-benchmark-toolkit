"""Visualization utilities: confusion matrices, ROC curves, training curves.

All plotting functions return a `matplotlib.figure.Figure` instance rather
than calling `plt.show()`, so callers (including the reporting module) can
embed, save, or further customize the figures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe backend for servers/CI
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, roc_curve  # noqa: E402
from sklearn.preprocessing import label_binarize  # noqa: E402

ArrayLike = Sequence[Any]

sns.set_theme(style="whitegrid")


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[Sequence[Any]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple = (6, 5),
) -> Figure:
    """Plot an annotated confusion-matrix heatmap.

    Parameters
    ----------
    normalize : bool, default False
        If True, rows are normalized to show proportions instead of raw
        counts.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    resolved_labels = (
        list(labels)
        if labels is not None
        else sorted(set(np.unique(y_true_arr).tolist()) | set(np.unique(y_pred_arr).tolist()))
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=resolved_labels)
    fmt = "d"
    display_cm = cm
    if normalize:
        with np.errstate(all="ignore"):
            row_sums = cm.sum(axis=1, keepdims=True)
            display_cm = np.divide(
                cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
            )
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        display_cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=resolved_labels,
        yticklabels=resolved_labels,
        cbar=True,
        square=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_curve(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    *,
    labels: Optional[Sequence[Any]] = None,
    title: str = "ROC Curve",
    figsize: tuple = (6, 5),
) -> Figure:
    """Plot ROC curve(s) with AUC scores.

    For binary targets a single curve is drawn. For multi-class targets,
    a One-vs-Rest curve (with individual AUC) is drawn per class, plus a
    micro-average curve.
    """
    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)
    resolved_labels = (
        list(labels) if labels is not None else sorted(np.unique(y_true_arr).tolist())
    )

    fig, ax = plt.subplots(figsize=figsize)

    if len(resolved_labels) <= 2:
        pos_proba = y_proba_arr if y_proba_arr.ndim == 1 else y_proba_arr[:, 1]
        fpr, tpr, _ = roc_curve(y_true_arr, pos_proba, pos_label=resolved_labels[-1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    else:
        y_bin = label_binarize(y_true_arr, classes=resolved_labels)
        for i, class_label in enumerate(resolved_labels):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba_arr[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, label=f"Class {class_label} (AUC = {roc_auc:.3f})")

        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba_arr.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(
            fpr_micro,
            tpr_micro,
            lw=2.5,
            linestyle="--",
            color="black",
            label=f"Micro-average (AUC = {auc_micro:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    *,
    title: str = "Training History",
    figsize: tuple = (10, 4),
) -> Figure:
    """Plot train vs. validation loss/accuracy curves from a history dict.

    Parameters
    ----------
    history : dict
        Dictionary with any of the keys: "loss", "val_loss", "accuracy",
        "val_accuracy" (Keras-style naming is also accepted, e.g. "acc",
        "val_acc"). Each value is a sequence of per-epoch metric values.
    """
    normalized = dict(history)
    if "acc" in normalized and "accuracy" not in normalized:
        normalized["accuracy"] = normalized.pop("acc")
    if "val_acc" in normalized and "val_accuracy" not in normalized:
        normalized["val_accuracy"] = normalized.pop("val_acc")

    has_loss = "loss" in normalized or "val_loss" in normalized
    has_acc = "accuracy" in normalized or "val_accuracy" in normalized
    n_panels = max(sum([has_loss, has_acc]), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0
    if has_loss:
        ax = axes[panel_idx]
        if "loss" in normalized:
            epochs = range(1, len(normalized["loss"]) + 1)
            ax.plot(epochs, normalized["loss"], label="Train Loss", marker="o", markersize=3)
        if "val_loss" in normalized:
            epochs = range(1, len(normalized["val_loss"]) + 1)
            ax.plot(epochs, normalized["val_loss"], label="Val Loss", marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend(fontsize="small")
        panel_idx += 1

    if has_acc:
        ax = axes[panel_idx]
        if "accuracy" in normalized:
            epochs = range(1, len(normalized["accuracy"]) + 1)
            ax.plot(
                epochs, normalized["accuracy"], label="Train Accuracy", marker="o", markersize=3
            )
        if "val_accuracy" in normalized:
            epochs = range(1, len(normalized["val_accuracy"]) + 1)
            ax.plot(
                epochs,
                normalized["val_accuracy"],
                label="Val Accuracy",
                marker="o",
                markersize=3,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.legend(fontsize="small")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
