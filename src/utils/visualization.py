"""
src/utils/visualization.py
--------------------------
Visualization helpers for offline analysis, evaluation notebooks, and the
Evaluator.plot_results() method.

Plots provided
--------------
1. plot_roc_curve()              – styled ROC curve with AUC-ROC & AUC-PR
2. plot_confusion_matrix()       – full 14-class percentage confusion matrix
3. plot_per_class_accuracy()     – horizontal bar chart with Good/Excellent bands
4. plot_training_loss()          – loss curve with best-epoch marker
5. plot_evaluation_dashboard()   – 2×2 grid of all four plots in one figure
6. visualize_anomaly()           – per-segment anomaly score for a single video
7. compare_anomaly_scores()      – side-by-side anomaly scores for multiple videos

All functions accept an optional ``save_path`` / ``save_dir`` and return the
matplotlib Figure so callers can embed it in notebooks or close it themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)

# UCF-Crime class names (index 0 = Normal)
ANOMALY_CLASSES: List[str] = [
    "Normal", "Abuse", "Arrest", "Arson", "Assault",
    "Burglary", "Explosion", "Fighting", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism", "RoadAccidents",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        return plt, mticker
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualizations. "
            "Install it with: pip install matplotlib"
        )


def _savefig(fig, save_path: Optional[Union[str, Path]], dpi: int = 150) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved → %s", save_path)


# ---------------------------------------------------------------------------
# 1. ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    auc_roc: float,
    auc_pr: float,
    accuracy: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
) -> "plt.Figure":
    """
    Publication-quality ROC curve matching the style in the uploaded screenshots.

    Parameters
    ----------
    y_true  : binary ground-truth labels (0 / 1)
    y_score : anomaly scores
    auc_roc : pre-computed AUC-ROC (avoids recomputing inside this fn)
    auc_pr  : pre-computed AUC-PR
    accuracy: overall accuracy (optional, shown in legend box)
    save_path: if given, figure is saved here
    """
    plt, _ = _import_matplotlib()

    # Build ROC curve points
    desc_idx = np.argsort(y_score)[::-1]
    y_s = np.asarray(y_true, dtype=float)[desc_idx]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1.0 - y_s)
    n_pos = tp[-1]; n_neg = fp[-1]
    tpr = np.concatenate([[0.0], tp / max(n_pos, 1)])
    fpr = np.concatenate([[0.0], fp / max(n_neg, 1)])

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#f0f2f5")
    ax.set_facecolor("#eef0f5")

    # Shaded area under curve
    ax.fill_between(fpr, tpr, alpha=0.25, color="#3a6bc9")
    ax.plot(fpr, tpr, color="#1a3fa8", linewidth=2.5,
            label=f"Our Model (AUC = {auc_roc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, alpha=0.8,
            label="Random Classifier (AUC = 0.5)")

    # Legend info-box
    info_lines = [f"AUC-ROC: {auc_roc:.4f}", f"AUC-PR: {auc_pr:.4f}"]
    if accuracy is not None:
        info_lines.append(f"Accuracy: {accuracy * 100:.2f}%")
    ax.text(
        0.02, 0.98, "\n".join(info_lines),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5e9c8", alpha=0.85),
    )

    ax.set_xlabel("False Positive Rate (FPR)", fontsize=13)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.35, linestyle="--")
    fig.tight_layout()

    _savefig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Confusion Matrix - Model Predictions",
) -> "plt.Figure":
    """
    Full 14×14 (or C×C) percentage confusion matrix matching the uploaded style.

    Parameters
    ----------
    confusion_matrix : np.ndarray [C, C]  raw counts
    normalize        : show percentages instead of raw counts
    """
    plt, _ = _import_matplotlib()

    cm = confusion_matrix.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm / row_sums * 100.0
        fmt = "{:.0f}"
        cbar_label = "Percentage (%)"
    else:
        cm_display = cm
        fmt = "{:.0f}"
        cbar_label = "Count"

    n = cm_display.shape[0]
    names = class_names if class_names else ANOMALY_CLASSES[:n]

    fig, ax = plt.subplots(figsize=(max(10, n), max(9, n - 1)))
    im = ax.imshow(cm_display, cmap="Blues", vmin=0,
                   vmax=100 if normalize else None, aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label(cbar_label, fontsize=11)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Annotate cells
    thresh = cm_display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    color=color, fontsize=8)

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. Per-Class Accuracy
# ---------------------------------------------------------------------------

def plot_per_class_accuracy(
    per_class_accuracy: Dict[str, float],
    good_threshold: float = 0.60,
    excellent_threshold: float = 0.80,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Per-Class Detection Accuracy",
) -> "plt.Figure":
    """
    Horizontal bar chart with Good / Excellent threshold lines, matching the
    uploaded screenshot style (green = excellent, orange = below excellent).

    Parameters
    ----------
    per_class_accuracy : dict mapping class name → accuracy in [0, 1]
    good_threshold     : accuracy at which bar turns from bad to good
    excellent_threshold: accuracy at which bar turns green
    """
    plt, _ = _import_matplotlib()

    classes = list(per_class_accuracy.keys())
    values  = [per_class_accuracy[c] * 100 for c in classes]

    colors = [
        "#2ecc71" if v >= excellent_threshold * 100 else "#f39c12"
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(13, max(6, len(classes) * 0.55)))
    fig.patch.set_facecolor("#f4f4f4")
    ax.set_facecolor("#f4f4f4")

    bars = ax.barh(classes, values, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.65)

    # Threshold lines
    ax.axvline(good_threshold * 100, color="#e67e22", linestyle="--",
               linewidth=1.5, label=f"Good ({good_threshold*100:.0f}%)", alpha=0.85)
    ax.axvline(excellent_threshold * 100, color="#27ae60", linestyle="--",
               linewidth=1.5, label=f"Excellent ({excellent_threshold*100:.0f}%)", alpha=0.85)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_ylabel("Anomaly Class", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlim(0, 110)
    ax.legend(loc="lower right", fontsize=10)
    ax.invert_yaxis()   # top-to-bottom ordering
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Training Loss Curve
# ---------------------------------------------------------------------------

def plot_training_loss(
    losses: List[float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training Loss Over Time",
) -> "plt.Figure":
    """
    Loss curve with shaded area, best-epoch marker, and an info box —
    matching the style in the uploaded screenshot.
    """
    plt, _ = _import_matplotlib()

    epochs = list(range(1, len(losses) + 1))
    best_idx  = int(np.argmin(losses))
    best_loss = losses[best_idx]
    final_loss = losses[-1]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#eef0f5")
    ax.set_facecolor("#eef0f5")

    ax.fill_between(epochs, losses, alpha=0.30, color="#4a6cf0")
    ax.plot(epochs, losses, color="#2040c0", linewidth=2.2, label="Training Loss")
    ax.scatter(
        [best_idx + 1], [best_loss],
        color="red", s=80, zorder=5,
        label=f"Best Loss: {best_loss:.4f} (Epoch {best_idx + 1})",
    )

    # Info box
    info = (
        f"Final Loss: {final_loss:.4f}\n"
        f"Best Loss: {best_loss:.4f}\n"
        f"Total Epochs: {len(losses)}"
    )
    ax.text(
        0.72, 0.97, info,
        transform=ax.transAxes, verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5e9c8", alpha=0.85),
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(1, len(losses))
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _savefig(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Combined Dashboard (2×2)
# ---------------------------------------------------------------------------

def plot_evaluation_dashboard(
    losses: List[float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    confusion_matrix: np.ndarray,
    per_class_accuracy: Dict[str, float],
    auc_roc: float,
    auc_pr: float,
    accuracy: Optional[float] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Model Evaluation Dashboard",
) -> "plt.Figure":
    """
    2×2 grid combining training loss, ROC curve, confusion matrix,
    and per-class accuracy into a single figure.
    """
    plt, _ = _import_matplotlib()

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#f0f2f5")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    # ── Top-left: training loss ──────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    _draw_loss_on_ax(ax1, losses)

    # ── Top-right: ROC curve ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    _draw_roc_on_ax(ax2, y_true, y_score, auc_roc, auc_pr, accuracy)

    # ── Bottom-left: confusion matrix ────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    _draw_cm_on_ax(ax3, confusion_matrix, class_names)

    # ── Bottom-right: per-class accuracy ────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    _draw_per_class_on_ax(ax4, per_class_accuracy)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig(fig, save_path)
    return fig


# -- Axis-level drawing helpers (used by dashboard) --------------------------

def _draw_loss_on_ax(ax, losses):
    epochs = list(range(1, len(losses) + 1))
    best_idx = int(np.argmin(losses))
    ax.set_facecolor("#eef0f5")
    ax.fill_between(epochs, losses, alpha=0.25, color="#4a6cf0")
    ax.plot(epochs, losses, color="#2040c0", linewidth=2)
    ax.scatter([best_idx + 1], [losses[best_idx]], color="red", s=60, zorder=5)
    ax.set_title("Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_xlim(1, len(losses)); ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")


def _draw_roc_on_ax(ax, y_true, y_score, auc_roc, auc_pr, accuracy):
    desc_idx = np.argsort(y_score)[::-1]
    y_s = np.asarray(y_true, dtype=float)[desc_idx]
    tp = np.cumsum(y_s); fp = np.cumsum(1.0 - y_s)
    n_pos = tp[-1]; n_neg = fp[-1]
    tpr = np.concatenate([[0.0], tp / max(n_pos, 1)])
    fpr = np.concatenate([[0.0], fp / max(n_neg, 1)])

    ax.set_facecolor("#eef0f5")
    ax.fill_between(fpr, tpr, alpha=0.2, color="#3a6bc9")
    ax.plot(fpr, tpr, color="#1a3fa8", linewidth=2,
            label=f"AUC={auc_roc:.4f}")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.2, alpha=0.7)
    info = f"AUC-ROC: {auc_roc:.4f}\nAUC-PR: {auc_pr:.4f}"
    if accuracy is not None:
        info += f"\nAccuracy: {accuracy*100:.2f}%"
    ax.text(0.03, 0.97, info, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#f5e9c8", alpha=0.85))
    ax.set_title("ROC Curve", fontweight="bold")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")


def _draw_cm_on_ax(ax, cm_raw, class_names):
    cm = cm_raw.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100.0
    n = cm_pct.shape[0]
    names = class_names if class_names else ANOMALY_CLASSES[:n]

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("Confusion Matrix (%)", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thresh = cm_pct.max() / 2
    for i in range(n):
        for j in range(n):
            v = cm_pct[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > thresh else "black", fontsize=6)


def _draw_per_class_on_ax(ax, per_class_accuracy):
    classes = list(per_class_accuracy.keys())
    values  = [per_class_accuracy[c] * 100 for c in classes]
    colors  = ["#2ecc71" if v >= 80 else "#f39c12" for v in values]
    ax.set_facecolor("#f4f4f4")
    ax.barh(classes, values, color=colors, height=0.65, edgecolor="white")
    ax.axvline(60, color="#e67e22", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.axvline(80, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_title("Per-Class Accuracy", fontweight="bold")
    ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 110)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for val, patch in zip(values, ax.patches):
        ax.text(val + 0.5, patch.get_y() + patch.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=7, fontweight="bold")


# ---------------------------------------------------------------------------
# 6. Single-video anomaly score  (original, kept for backward compat)
# ---------------------------------------------------------------------------

def visualize_anomaly(model, video_features, video_name="Test Video", device="cuda"):
    """Predict and plot per-segment anomaly scores for a single video."""
    plt, _ = _import_matplotlib()

    model.eval()
    with torch.no_grad():
        if isinstance(video_features, list):
            video_features = torch.stack(video_features)
        if video_features.dim() == 2:
            input_tensor = video_features.unsqueeze(0).to(device)
        else:
            input_tensor = video_features.to(device)

        anomaly_scores, class_probs = model(input_tensor)
        scores = anomaly_scores.squeeze().cpu().numpy()
        mean_class_probs = class_probs.squeeze().mean(dim=0)
        pred_class_idx = torch.argmax(mean_class_probs).item()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(scores, label="Anomaly Score", color="red", linewidth=2)
    ax.fill_between(range(len(scores)), scores, color="red", alpha=0.2)
    ax.set_title(
        f"Anomaly Detection: {video_name}\n"
        f"Predicted Class: {ANOMALY_CLASSES[pred_class_idx]}"
    )
    ax.set_xlabel("Video Segments (Time)")
    ax.set_ylabel("Anomaly Probability (0–1)")
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    logger.info("visualize_anomaly: %s → %s", video_name, ANOMALY_CLASSES[pred_class_idx])
    return scores, pred_class_idx, ANOMALY_CLASSES[pred_class_idx]


# ---------------------------------------------------------------------------
# 7. Multi-video comparison  (original, kept for backward compat)
# ---------------------------------------------------------------------------

def compare_anomaly_scores(model, videos_dict, device="cuda"):
    """Compare anomaly scores across multiple videos."""
    plt, _ = _import_matplotlib()

    model.eval()
    num_videos = len(videos_dict)
    fig, axes = plt.subplots(num_videos, 1, figsize=(12, 3 * num_videos))
    if num_videos == 1:
        axes = [axes]

    with torch.no_grad():
        for idx, (video_name, video_features) in enumerate(videos_dict.items()):
            if isinstance(video_features, list):
                video_features = torch.stack(video_features)
            if video_features.dim() == 2:
                input_tensor = video_features.unsqueeze(0).to(device)
            else:
                input_tensor = video_features.to(device)

            anomaly_scores, class_probs = model(input_tensor)
            scores = anomaly_scores.squeeze().cpu().numpy()
            mean_probs = class_probs.squeeze().mean(dim=0)
            pred_class_idx = torch.argmax(mean_probs).item()

            axes[idx].plot(scores, color="red", linewidth=2, label="Anomaly Score")
            axes[idx].fill_between(range(len(scores)), scores, color="red", alpha=0.2)
            axes[idx].set_title(
                f"{video_name}  –  Predicted: {ANOMALY_CLASSES[pred_class_idx]}"
            )
            axes[idx].set_xlabel("Video Segments (Time)")
            axes[idx].set_ylabel("Anomaly Prob (0–1)")
            axes[idx].set_ylim(0, 1.1)
            axes[idx].grid(True, linestyle="--", alpha=0.6)
            axes[idx].legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8. Convenience: generate all plots from an evaluation results dict
# ---------------------------------------------------------------------------

def generate_all_evaluation_plots(
    results: dict,
    losses: Optional[List[float]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, "plt.Figure"]:
    """
    Generate and save all four evaluation plots given a results dict returned
    by ``Evaluator.evaluate()`` (and optionally a loss history list).

    Expected keys in *results*:
        auc_roc, auc_pr, accuracy, anomaly_scores, binary_labels,
        confusion_matrix, per_class_accuracy

    Returns a dict of figure objects keyed by plot name.
    """
    save_dir = Path(save_dir) if save_dir else None
    figs: Dict[str, "plt.Figure"] = {}

    # -- ROC curve ------------------------------------------------------------
    figs["roc_curve"] = plot_roc_curve(
        y_true=results["binary_labels"],
        y_score=results["anomaly_scores"],
        auc_roc=results["auc_roc"],
        auc_pr=results["auc_pr"],
        accuracy=results.get("accuracy"),
        save_path=save_dir / "roc_curve.png" if save_dir else None,
    )

    # -- Confusion matrix -----------------------------------------------------
    figs["confusion_matrix"] = plot_confusion_matrix(
        confusion_matrix=results["confusion_matrix"],
        class_names=class_names,
        normalize=True,
        save_path=save_dir / "confusion_matrix.png" if save_dir else None,
    )

    # -- Per-class accuracy ---------------------------------------------------
    figs["per_class_accuracy"] = plot_per_class_accuracy(
        per_class_accuracy=results["per_class_accuracy"],
        save_path=save_dir / "per_class_accuracy.png" if save_dir else None,
    )

    # -- Training loss --------------------------------------------------------
    if losses:
        figs["training_loss"] = plot_training_loss(
            losses=losses,
            save_path=save_dir / "training_loss.png" if save_dir else None,
        )

    # -- Dashboard ------------------------------------------------------------
    if losses:
        figs["dashboard"] = plot_evaluation_dashboard(
            losses=losses,
            y_true=results["binary_labels"],
            y_score=results["anomaly_scores"],
            confusion_matrix=results["confusion_matrix"],
            per_class_accuracy=results["per_class_accuracy"],
            auc_roc=results["auc_roc"],
            auc_pr=results["auc_pr"],
            accuracy=results.get("accuracy"),
            class_names=class_names,
            save_path=save_dir / "dashboard.png" if save_dir else None,
        )

    return figs