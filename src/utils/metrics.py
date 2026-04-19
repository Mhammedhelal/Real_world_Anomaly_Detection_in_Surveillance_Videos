"""
src/utils/metrics.py
--------------------
Evaluation metric utilities — pure NumPy, no scikit-learn dependency.

All functions operate on 1-D NumPy arrays and return plain Python floats
or dicts, so they can be used in any training / evaluation context without
additional dependencies.

Public API
----------
compute_auc(y_true, y_score)        → float   (AUC-ROC, trapezoidal)
compute_accuracy(y_true, y_pred)    → float
compute_per_class_accuracy(y_true, y_pred, num_classes, class_names)
                                    → dict[str, float]
compute_confusion_matrix(y_true, y_pred, num_classes)
                                    → np.ndarray  [num_classes, num_classes]
MetricsTracker                      → stateful running average for training
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AUC-ROC (frame-level or video-level)
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve using the trapezoidal rule.

    No scikit-learn dependency.  Works for binary labels (0 = normal,
    1 = anomalous) paired with continuous anomaly scores.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth binary labels.  Any positive value is treated as
        the positive class.
    y_score : np.ndarray, shape (N,)
        Predicted anomaly scores (higher → more anomalous).

    Returns
    -------
    float
        AUC-ROC in [0, 1].  0.5 = random, 1.0 = perfect.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )

    # Sort by score descending
    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_idx]

    # Cumulative TP and FP
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)

    n_pos = tp[-1]
    n_neg = fp[-1]

    if n_pos == 0 or n_neg == 0:
        logger.warning("AUC undefined: all samples have the same label.")
        return float("nan")

    tpr = tp / (n_pos + 1e-12)
    fpr = fp / (n_neg + 1e-12)

    # np.trapz was renamed to np.trapezoid in NumPy 2.0
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    auc = float(_trapz(tpr, fpr))
    return auc


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute overall classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth integer class labels.
    y_pred : np.ndarray, shape (N,)
        Predicted integer class labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_pred == y_true).mean())


# ---------------------------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------------------------

def compute_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Compute per-class accuracy (recall per class).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels in [0, num_classes).
    y_pred : np.ndarray
        Predicted integer labels in [0, num_classes).
    num_classes : int
        Total number of classes (including Normal = 0).
    class_names : sequence of str | None
        Human-readable names indexed by class id.  If None, uses
        ``"class_0"``, ``"class_1"``, … as keys.

    Returns
    -------
    dict[str, float]
        Mapping class_name → accuracy for classes that appear in y_true.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result: Dict[str, float] = {}
    for cls_idx in range(num_classes):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            continue  # class not present in this split

        cls_acc = float((y_pred[mask] == cls_idx).mean())
        name = (
            class_names[cls_idx]
            if class_names and cls_idx < len(class_names)
            else f"class_{cls_idx}"
        )
        result[name] = cls_acc

    return result


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute a confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Ground-truth integer labels.
    y_pred : np.ndarray, shape (N,)
        Predicted integer labels.
    num_classes : int
        Size of the square confusion matrix.

    Returns
    -------
    np.ndarray, shape (num_classes, num_classes)
        ``cm[i, j]`` = number of samples with true label *i* predicted as *j*.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Stateful tracker for training loops
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Lightweight running-average tracker for scalar metrics.

    Typical use inside a training loop::

        tracker = MetricsTracker()
        for features, labels in loader:
            loss = ...
            tracker.update("loss", loss.item())
            tracker.update("ranking_loss", ranking_loss.item())

        epoch_summary = tracker.summary()
        print(epoch_summary)  # {"loss": 0.312, "ranking_loss": 0.205}
        tracker.reset()

    Parameters
    ----------
    window : int | None
        If set, only the last *window* updates are averaged (sliding window).
        If None (default), a cumulative average is maintained.
    """

    def __init__(self, window: Optional[int] = None) -> None:
        self._window = window
        self._data: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------

    def update(self, name: str, value: float, n: int = 1) -> None:
        """
        Record *n* samples with mean *value*.

        Parameters
        ----------
        name : str
            Metric name.
        value : float
            Observed value (or batch mean).
        n : int
            Number of samples this value represents (for correct weighting).
        """
        self._data[name].extend([value] * n)
        if self._window and len(self._data[name]) > self._window:
            self._data[name] = self._data[name][-self._window :]

    def average(self, name: str) -> float:
        """Return current running average for *name*."""
        vals = self._data.get(name, [])
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self) -> Dict[str, float]:
        """Return dict of running averages for all tracked metrics."""
        return {name: self.average(name) for name in self._data}

    def reset(self) -> None:
        """Clear all accumulated values (call at the end of each epoch)."""
        self._data.clear()

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={self.average(k):.4f}" for k in self._data)
        return f"MetricsTracker({parts})"
