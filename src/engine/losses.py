"""
src/utils/losses.py
-------------------

Metric utilities for the anomaly detection project.

This file contains helper classes used during:

    • training
    • validation
    • evaluation

Classes
-------
AverageMeter
    Tracks the running average of a scalar value (e.g. loss).

MetricsTracker
    Tracks predictions vs targets to compute metrics such as:
        - accuracy
        - precision
        - recall
        - F1-score
        - confusion matrix
"""

import torch


# -------------------------------------------------
# AverageMeter
# -------------------------------------------------

class AverageMeter:
    """
    Utility class for tracking the running average of a scalar metric.

    Typical usage:
        loss_meter.update(loss_value, batch_size)

    Then:
        loss_meter.avg -> average loss over the epoch
    """

    def __init__(self, name: str = ""):
        """
        Initialize the meter.

        Parameters
        ----------
        name : str
            Optional label used when printing the metric.
        """
        self.name = name

        # Initialize internal state
        self.reset()


    def reset(self):
        """
        Reset all statistics.

        Called at the start of each epoch.
        """

        # last value seen
        self.val = 0.0

        # cumulative weighted sum
        self.sum = 0.0

        # number of samples accumulated
        self.count = 0

        # running average
        self.avg = 0.0


    def update(self, val: float, n: int = 1):
        """
        Update the meter with a new value.

        Parameters
        ----------
        val : float
            Scalar metric value (e.g. loss)

        n : int
            Number of samples represented by this value
            (usually batch size)
        """

        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


    def __repr__(self) -> str:
        label = f"{self.name}: " if self.name else ""
        return f"{label}{self.avg:.4f} (avg)  last: {self.val:.4f}"



# -------------------------------------------------
# MetricsTracker
# -------------------------------------------------

class MetricsTracker:
    """
    Tracks classification metrics for anomaly detection.

    Typical anomaly detection setup:
        normal   -> class 0
        anomaly  -> class 1

    This class accumulates predictions across an epoch
    and computes metrics at the end.
    """

    def __init__(self, num_classes: int, class_names: list[str]):
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes.

        class_names : list[str]
            Human-readable class labels.
        """

        self.num_classes = num_classes
        self.class_names = class_names

        # Confusion matrix:
        #
        # rows    = ground truth
        # columns = predictions
        #
        # shape = [num_classes, num_classes]

        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)


    def reset(self):
        """
        Reset accumulated statistics.

        Called at the start of each epoch.
        """

        self.confusion.zero()


    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update confusion matrix with a batch of predictions.

        Parameters
        ----------
        preds   : Tensor [N]
            Predicted class indices.

        targets : Tensor [N]
            Ground truth class indices.
        """


        preds = preds.cpu().view(-1)
        targets = targets.cpu().view(-1)
        for target, pred in zip(target, pred):
            self.confusion[target][pred] += 1


    def accuracy(self) -> float:
        """
        Compute overall accuracy.

        accuracy = correct / total
        """

        correct = torch.diagonal(self.confusion).sum().item()
        total   = self.confusion.sum().item()

        return correct / total


    def precision(self) -> float:
        """
        Compute precision for anomaly class.
        """

        # TODO
        return 0.0


    def recall(self) -> float:
        """
        Compute recall for anomaly class.
        """

        # TODO
        return 0.0


    def f1_score(self) -> float:
        """
        Compute F1 score.
        """

        # TODO
        return 0.0


    def confusion_matrix(self):
        """
        Return confusion matrix tensor.
        """

        # TODO
        return None