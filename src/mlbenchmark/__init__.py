"""ml-benchmark-toolkit: lightweight evaluation and reporting for classifiers.

Public API
----------
- compute_metrics: single-model metric computation
- plot_confusion_matrix, plot_roc_curve, plot_training_curves: figure builders
- ModelComparator: register multiple models and compare them side-by-side
- Report: compile a ModelComparator into a standalone HTML/Markdown dashboard
"""

from __future__ import annotations

__version__ = "0.1.0"

from .comparator import ModelComparator, ModelResult
from .metrics import compute_metrics, is_binary
from .plots import plot_confusion_matrix, plot_roc_curve, plot_training_curves
from .report import Report

__all__ = [
    "__version__",
    "compute_metrics",
    "is_binary",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_curves",
    "ModelComparator",
    "ModelResult",
    "Report",
]
