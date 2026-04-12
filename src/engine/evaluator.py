"""
src/engine/evaluator.py
-----------------------
Evaluation engine for the UCF-Crime Anomaly Detector.

Computes metrics:
  • Frame-level AUC-ROC (primary metric)
  • Video-level classification accuracy
  • Per-class performance
  • Confusion matrix

Separated from scripts for modularity and reusability.
"""

import os
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.anomaly_detector import AnomalyDetector
from src.data.dataset import VideoFeatureDataset, collate_fn
from src.utils.checkpointing import load_model_from_checkpoint
from src.utils.metrics import compute_auc
from src.config import Config


# UCF-Crime class names
ANOMALY_CLASSES = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault",
    "Burglary", "Explosion", "Fighting", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "RoadAccidents",
]


class Evaluator:
    """
    Evaluation engine for anomaly detection models.
    
    Handles:
      - Loading pretrained models
      - Computing evaluation metrics
      - Generating plots and reports
      - Saving results
    
    Parameters
    ----------
    features_dir : str | Path
        Directory containing extracted .npz feature files.
    checkpoint_path : str | Path
        Path to trained model checkpoint.
    config : Config | None
        Configuration object. If None, loads from default.yaml.
    device : str | None
        Device to use ('cuda', 'cpu', or None for auto-detect).
    """
    
    def __init__(
        self,
        features_dir: str | Path,
        checkpoint_path: str | Path,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        self.features_dir = Path(features_dir)
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load config
        if config is None:
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'default.yaml'
            config = Config.from_yaml(config_path)
        self.config = config
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"🖥️  Using device: {self.device}")
        
        # Model will be loaded when needed
        self.model: Optional[AnomalyDetector] = None
        
        # Results storage
        self.results: Dict = {}
    
    def load_model(self) -> AnomalyDetector:
        """Load model from checkpoint."""
        if self.model is not None:
            return self.model
        
        print(f"📦 Loading model from: {self.checkpoint_path}")
        self.model = load_model_from_checkpoint(
            str(self.checkpoint_path),
            self.device
        )
        return self.model
    
    def evaluate(
        self,
        split: str = "test",
        batch_size: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> Dict:
        """
        Run evaluation on the specified split.
        
        Parameters
        ----------
        split : str
            'test' or 'train'
        batch_size : int | None
            Batch size for DataLoader. Uses config default if None.
        num_classes : int | None
            Number of crime categories. Uses config default if None.
        
        Returns
        -------
        dict
            Results with keys: auc, accuracy, per_class_accuracy, confusion_matrix
        """
        # Load model
        model = self.load_model()
        
        # Get parameters from config if not provided
        if batch_size is None:
            batch_size = self.config.training.batch_size
        if num_classes is None:
            num_classes = self.config.model.num_classes
        
        # Load dataset
        print(f"\n📂 Loading {split} dataset from: {self.features_dir}")
        dataset = VideoFeatureDataset(str(self.features_dir), split=split)
        
        if len(dataset) == 0:
            raise RuntimeError(
                f"No '{split}_*.npz' files found in {self.features_dir}. "
                "Run extract_features.py first."
            )
        
        print(f"✅ Loaded {len(dataset)} {split} samples")
        
        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        
        # Run inference
        print(f"\n🔍 Running inference on {split} set...")
        metrics = self._compute_metrics(model, loader, num_classes)
        
        # Store results
        self.results = metrics
        
        # Print summary
        self._print_results(metrics, num_classes)
        
        return metrics
    
    def _compute_metrics(
        self,
        model: AnomalyDetector,
        loader: DataLoader,
        num_classes: int,
    ) -> Dict:
        """
        Compute evaluation metrics.
        
        Returns
        -------
        dict
            Metrics dictionary with auc, accuracy, per_class_accuracy, confusion_matrix
        """
        model.eval()
        
        all_anomaly_scores = []
        all_binary_labels = []
        all_true_labels = []
        all_pred_labels = []
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(loader):
                features = features.to(self.device)
                labels_np = labels.numpy()
                
                # Forward pass
                anomaly_scores, class_probs = model(features)
                
                # Anomaly scores: [B, S, 1] → max over segments
                scores = anomaly_scores.squeeze(-1).cpu().numpy()
                max_scores = scores.max(axis=1)
                
                # Video-level classification: mean class prob across segments
                mean_probs = class_probs.mean(dim=1).cpu().numpy()
                pred_classes = mean_probs.argmax(axis=1)
                
                # Collect results
                all_anomaly_scores.extend(max_scores.tolist())
                all_binary_labels.extend((labels_np > 0).astype(int).tolist())
                all_true_labels.extend(labels_np.tolist())
                all_pred_labels.extend(pred_classes.tolist())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * features.size(0)} samples...")
        
        # Convert to arrays
        all_anomaly_scores = np.array(all_anomaly_scores)
        all_binary_labels = np.array(all_binary_labels)
        all_true_labels = np.array(all_true_labels)
        all_pred_labels = np.array(all_pred_labels)
        
        # Compute metrics
        auc = compute_auc(all_binary_labels, all_anomaly_scores)
        accuracy = (all_pred_labels == all_true_labels).mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for cls_idx in range(num_classes):
            mask = all_true_labels == cls_idx
            if mask.sum() == 0:
                continue
            cls_acc = (all_pred_labels[mask] == cls_idx).mean()
            cls_name = ANOMALY_CLASSES[cls_idx] if cls_idx < len(ANOMALY_CLASSES) else str(cls_idx)
            per_class_acc[cls_name] = float(cls_acc)
        
        # Confusion matrix
        confusion = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(all_true_labels, all_pred_labels):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                confusion[t, p] += 1
        
        return {
            'auc': float(auc),
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': confusion,
            'anomaly_scores': all_anomaly_scores,
            'binary_labels': all_binary_labels,
            'true_labels': all_true_labels,
            'pred_labels': all_pred_labels,
        }
    
    def _print_results(self, metrics: Dict, num_classes: int) -> None:
        """Print evaluation results."""
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"📊 AUC-ROC (video-level): {metrics['auc']:.4f}")
        print(f"🎯 Classification Accuracy: {metrics['accuracy']*100:.2f}%")
        
        print("\n📋 Per-class accuracy:")
        for cls_name, cls_acc in metrics['per_class_accuracy'].items():
            count = (metrics['true_labels'] == 
                    list(metrics['per_class_accuracy'].keys()).index(cls_name) 
                    if cls_name in ANOMALY_CLASSES else 0)
            print(f"   {cls_name:<15} : {cls_acc*100:.1f}%")
    
    def plot_results(
        self,
        save_dir: Optional[str | Path] = None,
        show: bool = True,
    ) -> None:
        """
        Generate and display/save evaluation plots.
        
        Parameters
        ----------
        save_dir : str | Path | None
            Directory to save plots. If None, doesn't save.
        show : bool
            Whether to display plots interactively.
        """
        if not self.results:
            raise RuntimeError("No results to plot. Run evaluate() first.")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # ROC curve
            self._plot_roc_curve(axes[0])
            
            # Confusion matrix
            self._plot_confusion_matrix(axes[1])
            
            plt.tight_layout()
            
            if save_dir:
                save_dir = Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                fig_path = save_dir / "evaluation_results.png"
                plt.savefig(fig_path, dpi=150)
                print(f"💾 Plots saved → {fig_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
        
        except ImportError:
            print("⚠️  matplotlib not installed. Cannot generate plots.")
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
    
    def _plot_roc_curve(self, ax) -> None:
        """Plot ROC curve."""
        scores = self.results['anomaly_scores']
        labels = self.results['binary_labels']
        auc = self.results['auc']
        
        # Sort by score descending
        desc_idx = np.argsort(scores)[::-1]
        y_sorted = labels[desc_idx]
        
        # Compute TPR and FPR
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(1 - y_sorted)
        tpr = tp / (tp[-1] + 1e-12)
        fpr = fp / (fp[-1] + 1e-12)
        
        ax.plot(fpr, tpr, color="crimson", linewidth=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Video-Level)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrix(self, ax) -> None:
        """Plot confusion matrix."""
        confusion = self.results['confusion_matrix']
        true_labels = self.results['true_labels']
        pred_labels = self.results['pred_labels']
        
        # Get present classes
        present = sorted(set(true_labels) | set(pred_labels))
        sub_cm = confusion[np.ix_(present, present)]
        
        labels_present = [
            ANOMALY_CLASSES[i] if i < len(ANOMALY_CLASSES) else str(i)
            for i in present
        ]
        
        im = ax.imshow(sub_cm, cmap="Blues")
        ax.set_xticks(range(len(present)))
        ax.set_yticks(range(len(present)))
        ax.set_xticklabels(labels_present, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels_present, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        ax.figure.colorbar(im, ax=ax)
    
    def save_results(
        self,
        output_path: str | Path,
    ) -> None:
        """
        Save evaluation results to file.
        
        Parameters
        ----------
        output_path : str | Path
            Path to save results (JSON or pickle)
        """
        if not self.results:
            raise RuntimeError("No results to save. Run evaluate() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            'auc': self.results['auc'],
            'accuracy': self.results['accuracy'],
            'per_class_accuracy': self.results['per_class_accuracy'],
            'confusion_matrix': self.results['confusion_matrix'].tolist(),
        }
        
        if output_path.suffix == '.json':
            import json
            with open(output_path, 'w') as f:
                json.dump(results_serializable, f, indent=2)
        else:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(self.results, f)
        
        print(f"💾 Results saved → {output_path}")


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
    """
    Convenience function for evaluation.
    
    Parameters
    ----------
    features_dir : str
        Directory containing .npz feature files
    checkpoint_path : str
        Path to model checkpoint
    split : str
        'test' or 'train'
    batch_size : int
        Batch size
    num_classes : int
        Number of classes
    device : str | None
        Device ('cuda', 'cpu', or None)
    plot : bool
        Whether to show plots
    save_dir : str | None
        Directory to save plots and results
    config : Config | None
        Configuration object
    
    Returns
    -------
    dict
        Evaluation metrics
    """
    evaluator = Evaluator(
        features_dir=features_dir,
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    
    results = evaluator.evaluate(
        split=split,
        batch_size=batch_size,
        num_classes=num_classes,
    )
    
    if plot or save_dir:
        evaluator.plot_results(
            save_dir=save_dir,
            show=plot,
        )
    
    if save_dir:
        save_path = Path(save_dir) / "evaluation_results.json"
        evaluator.save_results(save_path)
    
    return results