

from pathlib import Path

import torch

from src.models.anomaly_detector import AnomalyDetector

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, filepath: str) -> None:
    """Serialize *state* to *filepath*, creating parent directories as needed.

    The caller decides what goes into *state*.  A typical call looks like:

        save_checkpoint(
            state={
                "epoch":      epoch,
                "stage":      1,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            },
            filepath="models/stage1.pt",
        )

    Parameters
    ----------
    state : dict
        Arbitrary dictionary to persist (must be torch-serialisable).
    filepath : str
        Destination path, e.g. ``"models/stage1.pt"``.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"✔  Checkpoint saved → {path}")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------    
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