"""
src/utils/metrics.py
--------------------
Evaluation metric utilities — pure NumPy, no scikit-learn dependency.

Public API
----------
compute_auc_roc(y_true, y_score)    → float   AUC-ROC  (trapezoidal rule)
compute_auc_pr(y_true, y_score)     → float   AUC-PR   (average precision)
compute_accuracy(y_true, y_pred)    → float
compute_per_class_accuracy(...)     → dict[str, float]
compute_confusion_matrix(...)       → np.ndarray [C, C]
find_optimal_threshold(y_true, y_score, metric) → (threshold, metric_value)
MetricsTracker                      → stateful running-average accumulator

Naming note
-----------
``compute_auc`` was renamed to ``compute_auc_roc`` to be unambiguous
alongside the new ``compute_auc_pr``.  A deprecated alias is kept for
backward compatibility and will be removed in a future version.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# NumPy 2.0 renamed np.trapz → np.trapezoid
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)


# ---------------------------------------------------------------------------
# AUC-ROC
# ---------------------------------------------------------------------------

def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute the Area Under the ROC Curve (AUC-ROC) via the trapezoidal rule.

    A high AUC-ROC means the model ranks anomalous videos above normal ones
    regardless of threshold — the primary published metric for UCF-Crime
    (Sultani et al., 2018).

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Binary ground-truth labels: 0 = normal, 1 = anomalous.
        Any positive value is treated as the positive class.
    y_score : np.ndarray, shape (N,)
        Predicted anomaly scores (higher → more anomalous).

    Returns
    -------
    float
        AUC-ROC in [0, 1].  0.5 = random, 1.0 = perfect.
        Returns ``nan`` if only one class is present.

    Raises
    ------
    ValueError
        If y_true and y_score have different shapes.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )

    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_idx]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    n_pos = tp[-1]
    n_neg = fp[-1]

    if n_pos == 0 or n_neg == 0:
        logger.warning("AUC-ROC undefined: all labels are the same class.")
        return float("nan")

    tpr = tp / n_pos
    fpr = fp / n_neg

    # Prepend (0, 0) so the curve starts at the origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(_trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# AUC-PR  (Average Precision)
# ---------------------------------------------------------------------------

def compute_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute the Area Under the Precision-Recall Curve (AUC-PR).

    Also called *Average Precision* (AP).  AUC-PR is more informative than
    AUC-ROC on highly imbalanced datasets (e.g. UCF-Crime test set where
    anomalous segments are rare).  A random classifier achieves
    ``AUC-PR ≈ prevalence``, so higher is always better.

    This implementation uses the interpolated trapezoidal rule on the full
    precision-recall curve (one operating point per unique threshold), which
    matches scikit-learn's ``average_precision_score``.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Binary ground-truth labels: 0 = normal, 1 = anomalous.
    y_score : np.ndarray, shape (N,)
        Predicted anomaly scores (higher → more anomalous).

    Returns
    -------
    float
        AUC-PR in [0, 1].  Returns ``nan`` if there are no positive samples.

    Raises
    ------
    ValueError
        If y_true and y_score have different shapes.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
        )

    n_pos = y_true.sum()
    if n_pos == 0:
        logger.warning("AUC-PR undefined: no positive samples in y_true.")
        return float("nan")

    # Sort by descending score
    desc_idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_idx]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / n_pos

    # Append sentinel (recall=0, precision=1) at the start for a clean curve
    precision = np.concatenate([[1.0], precision])
    recall    = np.concatenate([[0.0], recall])

    return float(_trapz(precision, recall))


# ---------------------------------------------------------------------------
# Backward-compat alias (deprecated)
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Deprecated alias for :func:`compute_auc_roc`.

    .. deprecated::
        Use ``compute_auc_roc`` instead.
    """
    warnings.warn(
        "compute_auc() is deprecated and will be removed. "
        "Use compute_auc_roc() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_auc_roc(y_true, y_score)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: Literal["f1", "youden"] = "f1",
    n_thresholds: int = 200,
) -> Tuple[float, float]:
    """
    Find the anomaly-score threshold that maximises a validation metric.

    Rather than hard-coding 0.5, sweep candidate thresholds on a held-out
    validation set and return the one that maximises F1 or Youden's J.
    Store the returned threshold in your config for use at inference time.

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
        Binary labels (0 = normal, 1 = anomalous) from a **validation** set.
    y_score : np.ndarray, shape (N,)
        Corresponding anomaly scores from the model.
    metric : {"f1", "youden"}
        Optimisation objective:

        - ``"f1"``     : maximise F1 = 2·TP / (2·TP + FP + FN)
        - ``"youden"`` : maximise Youden's J = TPR − FPR
    n_thresholds : int
        Number of candidate thresholds (linearly spaced over score range).

    Returns
    -------
    (best_threshold, best_metric_value) : Tuple[float, float]

    Example
    -------
    >>> t, f1 = find_optimal_threshold(val_y, val_scores, metric="f1")
    >>> print(f"threshold={t:.3f}  val_F1={f1:.4f}")
    # → store t in configs/default.yaml as inference.anomaly_threshold
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    thresholds  = np.linspace(y_score.min(), y_score.max(), n_thresholds)
    best_thresh = float(thresholds[0])
    best_value  = -np.inf

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(float)
        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        tn = float(((y_pred == 0) & (y_true == 0)).sum())

        if metric == "f1":
            denom = 2 * tp + fp + fn
            value = (2 * tp / denom) if denom > 0 else 0.0
        elif metric == "youden":
            tpr   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            value = tpr - fpr
        else:
            raise ValueError(f"Unknown metric: {metric!r}. Choose 'f1' or 'youden'.")

        if value > best_value:
            best_value  = value
            best_thresh = float(thresh)

    logger.info(
        "find_optimal_threshold: metric=%s  threshold=%.4f  value=%.4f",
        metric, best_thresh, best_value,
    )
    return best_thresh, float(best_value)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall classification accuracy in [0, 1]."""
    return float((np.asarray(y_pred) == np.asarray(y_true)).mean())


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
    Per-class accuracy (recall per class).

    Only classes present in *y_true* are included in the output dict.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    result: Dict[str, float] = {}
    for cls_idx in range(num_classes):
        mask = y_true == cls_idx
        if mask.sum() == 0:
            continue
        name = (
            class_names[cls_idx]
            if class_names and cls_idx < len(class_names)
            else f"class_{cls_idx}"
        )
        result[name] = float((y_pred[mask] == cls_idx).mean())
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
    C×C confusion matrix.  ``cm[i, j]`` = samples true=i predicted=j.
    Out-of-range labels are silently ignored.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# MetricsTracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Lightweight stateful running-average accumulator for training loops.

    Usage
    -----
    ::

        tracker = MetricsTracker()
        for features, labels in loader:
            loss = criterion(...)
            tracker.update("loss", loss.item(), n=features.size(0))
        print(tracker.summary())   # {"loss": 0.312, ...}
        tracker.reset()

    Parameters
    ----------
    window : int | None
        Sliding-window size.  ``None`` (default) keeps a full-epoch average.
    """

    def __init__(self, window: Optional[int] = None) -> None:
        self._window = window
        self._data: Dict[str, List[float]] = defaultdict(list)

    def update(self, name: str, value: float, n: int = 1) -> None:
        """Record *n* observations with mean *value*."""
        self._data[name].extend([value] * n)
        if self._window and len(self._data[name]) > self._window:
            self._data[name] = self._data[name][-self._window:]

    def average(self, name: str) -> float:
        """Running average; returns ``nan`` if never updated."""
        vals = self._data.get(name, [])
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self) -> Dict[str, float]:
        """Dict of running averages for all tracked metrics."""
        return {name: self.average(name) for name in self._data}

    def reset(self) -> None:
        """Clear all data (call at epoch start)."""
        self._data.clear()

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={self.average(k):.4f}" for k in self._data)
        return f"MetricsTracker({parts})"
