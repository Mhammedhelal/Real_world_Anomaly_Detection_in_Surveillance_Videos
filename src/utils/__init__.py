"""
src/utils/__init__.py
---------------------
Utility module public API.
"""

from src.utils.checkpointing import (
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from src.utils.logging import TrainingLogger, get_logger, setup_logging
from src.utils.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_auc,           # deprecated alias — kept for backward compat
    compute_auc_roc,
    compute_auc_pr,
    compute_confusion_matrix,
    compute_per_class_accuracy,
    find_optimal_threshold,
)
from src.utils.visualization import (
    ANOMALY_CLASSES,
    compare_anomaly_scores,
    plot_training_loss,
    visualize_anomaly,
)

__all__ = [
    # checkpointing
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
    # logging
    "setup_logging",
    "get_logger",
    "TrainingLogger",
    # metrics
    "compute_auc_roc",
    "compute_auc_pr",
    "compute_auc",             # deprecated alias
    "compute_accuracy",
    "compute_per_class_accuracy",
    "compute_confusion_matrix",
    "find_optimal_threshold",
    "MetricsTracker",
    # visualization
    "visualize_anomaly",
    "plot_training_loss",
    "compare_anomaly_scores",
    "ANOMALY_CLASSES",
]
