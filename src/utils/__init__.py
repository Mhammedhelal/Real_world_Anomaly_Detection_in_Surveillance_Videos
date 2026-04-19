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
    compute_auc,
    compute_confusion_matrix,
    compute_per_class_accuracy,
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
    "compute_auc",
    "compute_accuracy",
    "compute_per_class_accuracy",
    "compute_confusion_matrix",
    "MetricsTracker",
    # visualization
    "visualize_anomaly",
    "plot_training_loss",
    "compare_anomaly_scores",
    "ANOMALY_CLASSES",
]
