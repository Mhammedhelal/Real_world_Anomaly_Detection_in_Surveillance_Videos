"""
Evaluation entrypoint for the UCF-Crime Anomaly Detector.

Computes:
  • Frame-level AUC-ROC  (primary metric for anomaly detection)
  • Video-level classification accuracy + per-class report
  • Confusion matrix

Usage (Colab):
    %run evaluate.py \
        --features-dir /content/drive/MyDrive/UCF_Crime/features \
        --checkpoint    /content/drive/MyDrive/UCF_Crime/checkpoints/anomaly_detector_epoch0100.pt

Or import and call directly:
    from evaluate import evaluate
    results = evaluate(features_dir=..., checkpoint_path=...)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.anomaly_detector import AnomalyDetector

# Reuse the dataset / collate from train.py
from train import VideoFeatureDataset, collate_fn

# UCF-Crime class names
ANOMALY_CLASSES = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault",
    "Burglary", "Explosion", "Fighting", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "RoadAccidents",
]


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> AnomalyDetector:
    """Load AnomalyDetector from a .pt checkpoint saved by train.py."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = ckpt.get("config", {})
    model = AnomalyDetector(
        input_size=cfg.get("input_size",  2131),
        hidden_size=cfg.get("hidden_size", 256),
        num_classes=cfg.get("num_classes", 14),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss",  float("nan"))
    print(f"✅ Loaded checkpoint: epoch {epoch}, train loss {loss:.4f}")
    return model


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


# ════════════════════════════════════════════════════════════════════════════
# Evaluation function
# ════════════════════════════════════════════════════════════════════════════

def evaluate(
    features_dir: str,
    checkpoint_path: str,
    split: str = "test",
    batch_size: int = 16,
    num_classes: int = 14,
    device: str = None,
    plot: bool = True,
    save_dir: str = None,
) -> dict:
    """
    Run evaluation on pre-extracted features.

    Args:
        features_dir    : Root directory containing .npz feature files.
        checkpoint_path : Path to .pt checkpoint saved by train.py.
        split           : 'test' or 'train'.
        batch_size      : Batch size for DataLoader.
        num_classes     : Number of crime categories.
        device          : 'cuda' | 'cpu' | None (auto).
        plot            : Whether to show AUC and confusion matrix plots.
        save_dir        : If set, saves plots here.

    Returns:
        Dictionary with keys: auc, accuracy, per_class_accuracy, confusion_matrix.
    """

    # ── Device ──────────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"🖥️  Using device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    dataset = VideoFeatureDataset(features_dir, split=split)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No '{split}_*.npz' files found in {features_dir}. "
            "Run extract_features.py first."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = load_model_from_checkpoint(checkpoint_path, device)

    # ── Inference ───────────────────────────────────────────────────────────
    all_anomaly_scores = []   # max anomaly score per video
    all_binary_labels  = []   # 0=normal, 1=anomalous  (for AUC)
    all_true_labels    = []   # full class index        (for accuracy)
    all_pred_labels    = []   # predicted class index

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels_np = labels.numpy()

            anomaly_scores, class_probs = model(features)
            # anomaly_scores: [B, S, 1]  →  max over segments
            scores = anomaly_scores.squeeze(-1).cpu().numpy()   # [B, S]
            max_scores = scores.max(axis=1)                     # [B]

            # Video-level classification: mean class prob across segments
            mean_probs = class_probs.mean(dim=1).cpu().numpy()  # [B, C]
            pred_classes = mean_probs.argmax(axis=1)            # [B]

            all_anomaly_scores.extend(max_scores.tolist())
            all_binary_labels.extend((labels_np > 0).astype(int).tolist())
            all_true_labels.extend(labels_np.tolist())
            all_pred_labels.extend(pred_classes.tolist())

    all_anomaly_scores = np.array(all_anomaly_scores)
    all_binary_labels  = np.array(all_binary_labels)
    all_true_labels    = np.array(all_true_labels)
    all_pred_labels    = np.array(all_pred_labels)

    # ── Metrics ─────────────────────────────────────────────────────────────

    # 1. Frame-level AUC (approximated at video level here)
    auc = compute_auc(all_binary_labels, all_anomaly_scores)
    print(f"\n📊 AUC-ROC (video-level): {auc:.4f}")

    # 2. Overall classification accuracy
    accuracy = (all_pred_labels == all_true_labels).mean()
    print(f"🎯 Classification Accuracy: {accuracy*100:.2f}%")

    # 3. Per-class accuracy
    per_class_acc = {}
    print("\n📋 Per-class accuracy:")
    for cls_idx in range(num_classes):
        mask = all_true_labels == cls_idx
        if mask.sum() == 0:
            continue
        cls_acc = (all_pred_labels[mask] == cls_idx).mean()
        cls_name = ANOMALY_CLASSES[cls_idx] if cls_idx < len(ANOMALY_CLASSES) else str(cls_idx)
        per_class_acc[cls_name] = float(cls_acc)
        print(f"   {cls_name:<15} : {cls_acc*100:.1f}%  ({mask.sum()} videos)")

    # 4. Confusion matrix (raw counts)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_true_labels, all_pred_labels):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    # ── Plots ───────────────────────────────────────────────────────────────
    if plot:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # — ROC curve —
            desc_idx = np.argsort(all_anomaly_scores)[::-1]
            y_sorted = all_binary_labels[desc_idx]
            tp = np.cumsum(y_sorted)
            fp = np.cumsum(1 - y_sorted)
            tpr = tp / (tp[-1] + 1e-12)
            fpr = fp / (fp[-1] + 1e-12)

            axes[0].plot(fpr, tpr, color="crimson", linewidth=2,
                         label=f"AUC = {auc:.4f}")
            axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].set_title("ROC Curve (Video-Level)")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # — Confusion matrix —
            present = sorted({t for t in all_true_labels} | {p for p in all_pred_labels})
            sub_cm   = confusion[np.ix_(present, present)]
            labels_present = [
                ANOMALY_CLASSES[i] if i < len(ANOMALY_CLASSES) else str(i)
                for i in present
            ]

            im = axes[1].imshow(sub_cm, cmap="Blues")
            axes[1].set_xticks(range(len(present)))
            axes[1].set_yticks(range(len(present)))
            axes[1].set_xticklabels(labels_present, rotation=45, ha="right", fontsize=8)
            axes[1].set_yticklabels(labels_present, fontsize=8)
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")
            axes[1].set_title("Confusion Matrix")
            fig.colorbar(im, ax=axes[1])

            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig_path = os.path.join(save_dir, "evaluation_results.png")
                plt.savefig(fig_path, dpi=150)
                print(f"💾 Evaluation plots saved → {fig_path}")

            plt.show()

        except Exception as e:
            print(f"⚠️  Could not produce plots: {e}")

    results = {
        "auc": auc,
        "accuracy": float(accuracy),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion,
    }
    return results


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the UCF-Crime anomaly detector.")
    p.add_argument("--features-dir",   type=str, required=True,
                   help="Root directory containing .npz feature files.")
    p.add_argument("--checkpoint",     type=str, required=True,
                   help="Path to .pt checkpoint file.")
    p.add_argument("--split",          type=str, default="test",
                   choices=["train", "test"])
    p.add_argument("--batch-size",     type=int, default=16)
    p.add_argument("--num-classes",    type=int, default=14)
    p.add_argument("--device",         type=str, default=None)
    p.add_argument("--save-dir",       type=str, default=None,
                   help="Directory to save evaluation plots.")
    p.add_argument("--no-plot", action="store_true",
                   help="Disable matplotlib plots.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        features_dir=args.features_dir,
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        device=args.device,
        plot=not args.no_plot,
        save_dir=args.save_dir,
    )