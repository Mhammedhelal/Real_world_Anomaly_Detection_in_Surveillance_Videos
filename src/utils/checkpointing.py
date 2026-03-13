

import torch

from src.models.anomaly_detector import AnomalyDetector


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