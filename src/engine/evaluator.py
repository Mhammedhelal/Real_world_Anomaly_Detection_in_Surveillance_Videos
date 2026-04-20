"""
src/engine/evaluator.py
-----------------------
Evaluation engine for the UCF-Crime Anomaly Detector.

Changes from original
---------------------
- Metric computation delegated to src.utils.metrics (no duplication).
- Console output uses structured logging (get_logger / TrainingLogger).
- load_model_from_checkpoint imported from src.utils.checkpointing.
- evaluate() convenience function remains backward-compatible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import VideoFeatureDataset, collate_fn
from src.models.anomaly_detector import AnomalyDetector
from src.utils.checkpointing import load_model_from_checkpoint
from src.utils.logging import TrainingLogger, get_logger
from src.utils.metrics import (
    MetricsTracker,
    compute_accuracy,
    compute_auc_roc,
    compute_auc_pr,
    compute_confusion_matrix,
    compute_per_class_accuracy,
)

logger = get_logger(__name__)

# UCF-Crime class names (index 0 = Normal)
ANOMALY_CLASSES: List[str] = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault",
    "Burglary", "Explosion", "Fighting", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "RoadAccidents",
]


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluation engine for anomaly detection models.

    Parameters
    ----------
    features_dir : str | Path
    checkpoint_path : str | Path
    config : Config | None
    device : str | None
    """

    def __init__(
        self,
        features_dir: str | Path,
        checkpoint_path: str | Path,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ) -> None:
        self.features_dir = Path(features_dir)
        self.checkpoint_path = Path(checkpoint_path)

        if config is None:
            config_path = (
                Path(__file__).resolve().parent.parent.parent
                / "configs"
                / "default.yaml"
            )
            config = Config.from_yaml(config_path)
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info("Evaluator | device=%s", self.device)

        self.model: Optional[AnomalyDetector] = None
        self.results: Dict = {}

    # ------------------------------------------------------------------

    def load_model(self) -> AnomalyDetector:
        if self.model is None:
            logger.info("Loading model from: %s", self.checkpoint_path)
            self.model = load_model_from_checkpoint(
                self.checkpoint_path, self.device
            )
        return self.model

    # ------------------------------------------------------------------

    def evaluate(
        self,
        split: str = "test",
        batch_size: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> Dict:
        """
        Run evaluation on *split*.

        Returns
        -------
        dict with keys:
            auc, accuracy, per_class_accuracy, confusion_matrix,
            anomaly_scores, binary_labels, true_labels, pred_labels
        """
        model = self.load_model()

        if batch_size is None:
            batch_size = self.config.training.batch_size
        if num_classes is None:
            num_classes = self.config.model.num_classes

        logger.info("Loading %s dataset from: %s", split, self.features_dir)
        dataset = VideoFeatureDataset(str(self.features_dir), split=split)

        if len(dataset) == 0:
            raise RuntimeError(
                f"No '{split}_*.npz' files found in {self.features_dir}."
            )

        num_workers = getattr(self.config.training, "num_workers", 4)
        logger.info(
            "Loaded %d %s samples  (%d skipped)  num_workers=%d",
            len(dataset), split, dataset.n_skipped, num_workers,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(num_workers > 0),
        )

        logger.info("Running inference …")
        results = self._run_inference(model, loader, num_classes)

        self.results = results
        self._log_results(results, split)
        return results

    # ------------------------------------------------------------------

    def _run_inference(
        self,
        model: AnomalyDetector,
        loader: DataLoader,
        num_classes: int,
    ) -> Dict:
        """Forward-pass over the loader; collect scores and labels."""
        model.eval()

        all_scores: List[float] = []
        all_binary: List[int] = []
        all_true: List[int] = []
        all_pred: List[int] = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels_np = labels.numpy()

                anomaly_scores, class_probs = model(features)

                # Video-level anomaly score: max over segments
                scores = anomaly_scores.squeeze(-1).cpu().numpy().max(axis=1)

                # Video-level class: mean prob over segments → argmax
                pred_cls = (
                    class_probs.mean(dim=1).cpu().numpy().argmax(axis=1)
                )

                all_scores.extend(scores.tolist())
                all_binary.extend((labels_np > 0).astype(int).tolist())
                all_true.extend(labels_np.tolist())
                all_pred.extend(pred_cls.tolist())

        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        y_score = np.array(all_scores)
        y_binary = np.array(all_binary)

        # Metrics — all from src.utils.metrics
        auc_roc = compute_auc_roc(y_binary, y_score)
        auc_pr  = compute_auc_pr(y_binary, y_score)
        acc     = compute_accuracy(y_true, y_pred)
        per_class = compute_per_class_accuracy(
            y_true, y_pred, num_classes, class_names=ANOMALY_CLASSES
        )
        cm = compute_confusion_matrix(y_true, y_pred, num_classes)

        return {
            "auc_roc": auc_roc,
            "auc_pr":  auc_pr,
            "auc":     auc_roc,   # backward-compat alias
            "accuracy": acc,
            "per_class_accuracy": per_class,
            "confusion_matrix": cm,
            "anomaly_scores": y_score,
            "binary_labels": y_binary,
            "true_labels": y_true,
            "pred_labels": y_pred,
        }

    # ------------------------------------------------------------------

    def _log_results(self, results: Dict, split: str) -> None:
        logger.info(
            "Eval [%s] AUC-ROC=%.4f  AUC-PR=%.4f  Accuracy=%.2f%%",
            split,
            results["auc_roc"],
            results["auc_pr"],
            results["accuracy"] * 100,
        )
        for cls_name, acc in results["per_class_accuracy"].items():
            logger.info("  %-15s %.1f%%", cls_name, acc * 100)

    # ------------------------------------------------------------------

    def plot_results(
        self,
        save_dir: Optional[str | Path] = None,
        show: bool = True,
    ) -> None:
        if not self.results:
            raise RuntimeError("No results. Call evaluate() first.")
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            self._plot_roc_curve(axes[0])
            self._plot_confusion_matrix(axes[1])
            plt.tight_layout()

            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig_path = save_dir / "evaluation_results.png"
                plt.savefig(fig_path, dpi=150)
                logger.info("Plots saved → %s", fig_path)

            if show:
                plt.show()
            else:
                plt.close()
        except ImportError:
            logger.warning("matplotlib not installed — skipping plots.")

    def _plot_roc_curve(self, ax) -> None:
        scores = self.results["anomaly_scores"]
        labels = self.results["binary_labels"]
        auc_roc = self.results["auc_roc"]
        auc_pr  = self.results["auc_pr"]

        desc_idx = np.argsort(scores)[::-1]
        y_s = labels[desc_idx]
        tp = np.cumsum(y_s)
        fp = np.cumsum(1 - y_s)
        tpr = tp / (tp[-1] + 1e-12)
        fpr = fp / (fp[-1] + 1e-12)

        ax.plot(fpr, tpr, color="crimson", linewidth=2,
                label=f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Video-Level)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_confusion_matrix(self, ax) -> None:
        cm = self.results["confusion_matrix"]
        true_labels = self.results["true_labels"]
        pred_labels = self.results["pred_labels"]

        present = sorted(set(true_labels) | set(pred_labels))
        sub_cm = cm[np.ix_(present, present)]
        names = [
            ANOMALY_CLASSES[i] if i < len(ANOMALY_CLASSES) else str(i)
            for i in present
        ]

        im = ax.imshow(sub_cm, cmap="Blues")
        ax.set_xticks(range(len(present)))
        ax.set_yticks(range(len(present)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        ax.figure.colorbar(im, ax=ax)

    def save_results(self, output_path: str | Path) -> None:
        if not self.results:
            raise RuntimeError("No results to save.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        serialisable = {
            "auc_roc": self.results["auc_roc"],
            "auc_pr":  self.results["auc_pr"],
            "accuracy": self.results["accuracy"],
            "per_class_accuracy": self.results["per_class_accuracy"],
            "confusion_matrix": self.results["confusion_matrix"].tolist(),
        }

        if output_path.suffix == ".json":
            import json
            with open(output_path, "w") as f:
                json.dump(serialisable, f, indent=2)
        else:
            import pickle
            with open(output_path, "wb") as f:
                pickle.dump(self.results, f)

        logger.info("Results saved → %s", output_path)


# ---------------------------------------------------------------------------
# Convenience function (backward-compatible)
# ---------------------------------------------------------------------------

def evaluate(
    features_dir: str,
    checkpoint_path: str,
    split: str = "test",
    batch_size: int = 16,
    num_classes: int = 14,
    device: Optional[str] = None,
    plot: bool = True,
    save_dir: Optional[str] = None,
    config: Optional[Config] = None,
) -> Dict:
    evaluator = Evaluator(
        features_dir=features_dir,
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    results = evaluator.evaluate(split=split, batch_size=batch_size, num_classes=num_classes)

    if plot or save_dir:
        evaluator.plot_results(save_dir=save_dir, show=plot)

    if save_dir:
        evaluator.save_results(Path(save_dir) / "evaluation_results.json")

    return results
