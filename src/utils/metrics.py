import numpy as np


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn (pure numpy)."""
    # Sort by score descending
    desc_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_idx]

    # Cumulative TP and FP
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)

    tp_rate = tp / (tp[-1] + 1e-12)
    fp_rate = fp / (fp[-1] + 1e-12)

    # Trapezoidal AUC
    auc = np.trapz(tp_rate, fp_rate)
    return float(auc)