"""
src/utils/visualization.py
--------------------------
Visualization helpers for offline analysis, evaluation notebooks, and the
Evaluator.plot_results() method.

Scope
-----
These functions require ``matplotlib`` and are intended for use in:
  - Jupyter notebooks during model analysis
  - The Evaluator to render ROC + confusion matrix plots
  - ``scripts/evaluate.py --no-plot`` skips them entirely

They are NOT called from the real-time inference pipeline or the trainer.
If matplotlib is not installed in a deployment environment that does not need
plots, this file can be ignored — the rest of the codebase does not depend on
it at runtime.
"""

import torch
import matplotlib.pyplot as plt

from src.utils.logging import get_logger

logger = get_logger(__name__)

# UCF-Crime Class Mapping
ANOMALY_CLASSES = [
    'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault',
    'Burglary', 'Explosion', 'Fighting', 'Robbery',
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', 'RoadAccidents',
]


def visualize_anomaly(model, video_features, video_name="Test Video", device='cuda'):
    """
    Predicts and plots anomaly scores across all segments of a video.

    Args:
        model: Trained AnomalyDetector model
        video_features: Tensor [Segments, feature_dim] or list of tensors
        video_name: Name of the video for title
        device: Device to run inference on
    """
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

    plt.figure(figsize=(12, 5))
    plt.plot(scores, label='Anomaly Score', color='red', linewidth=2)
    plt.fill_between(range(len(scores)), scores, color='red', alpha=0.2)
    plt.title(
        f"Anomaly Detection: {video_name}\n"
        f"Predicted Class: {ANOMALY_CLASSES[pred_class_idx]}"
    )
    plt.xlabel("Video Segments (Time)")
    plt.ylabel("Anomaly Probability (0-1)")
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    logger.info(
        "visualize_anomaly: %s → class=%s", video_name, ANOMALY_CLASSES[pred_class_idx]
    )
    return scores, pred_class_idx, ANOMALY_CLASSES[pred_class_idx]


def plot_training_loss(losses, save_path=None):
    """Plot training loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Training loss plot saved → %s", save_path)

    plt.show()


def compare_anomaly_scores(model, videos_dict, device='cuda'):
    """
    Compare anomaly scores across multiple videos.

    Args:
        model: Trained AnomalyDetector model
        videos_dict: {video_name: features_tensor}
        device: Device to run inference on
    """
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

            mean_class_probs = class_probs.squeeze().mean(dim=0)
            pred_class_idx = torch.argmax(mean_class_probs).item()

            axes[idx].plot(scores, label='Anomaly Score', color='red', linewidth=2)
            axes[idx].fill_between(range(len(scores)), scores, color='red', alpha=0.2)
            axes[idx].set_title(
                f"{video_name}\nPredicted Class: {ANOMALY_CLASSES[pred_class_idx]}"
            )
            axes[idx].set_xlabel("Video Segments (Time)")
            axes[idx].set_ylabel("Anomaly Probability (0-1)")
            axes[idx].set_ylim(0, 1.1)
            axes[idx].grid(True, linestyle='--', alpha=0.6)
            axes[idx].legend()

    plt.tight_layout()
    plt.show()
