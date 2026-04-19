"""
src/utils/checkpointing.py
--------------------------
Checkpoint save / load utilities — fully decoupled from model construction.

Design principles
-----------------
- save_checkpoint()  : serialises ANY dict to disk; caller decides content.
- load_checkpoint()  : deserialises a checkpoint dict; does NOT build a model.
- load_model_from_checkpoint() : convenience wrapper that builds + loads the
  AnomalyDetector from a previously-saved checkpoint dict.

Keeping load_checkpoint() separate from model construction means
future models can reuse this module without modification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, filepath: str | Path) -> Path:
    """
    Serialise *state* to *filepath*, creating parent directories as needed.

    Parameters
    ----------
    state : dict
        Arbitrary torch-serialisable dict.  Recommended keys::

            {
                "epoch":               int,
                "model_state_dict":    OrderedDict,
                "optimizer_state_dict": OrderedDict,
                "loss":                float,
                "config":              dict,   # plain dict, not Config object
            }

    filepath : str | Path
        Destination path, e.g. ``"outputs/checkpoints/best_model.pt"``.

    Returns
    -------
    Path
        Resolved path where the file was written.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved → %s", path)
    print(f"✔  Checkpoint saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Load (raw dict)
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict:
    """
    Load a raw checkpoint dict from disk.

    Does **not** build a model — callers handle model construction
    themselves so this function stays model-agnostic.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a ``.pt`` file saved by :func:`save_checkpoint`.
    device : torch.device | str
        Map location for tensors (default: ``"cpu"`` so checkpoints saved
        on GPU load cleanly on CPU-only machines).

    Returns
    -------
    dict
        The raw checkpoint dictionary.

    Raises
    ------
    FileNotFoundError
        If *checkpoint_path* does not exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    logger.info("Checkpoint loaded ← %s  (epoch=%s)", path, ckpt.get("epoch", "?"))
    return ckpt


# ---------------------------------------------------------------------------
# Convenience: build AnomalyDetector + load weights
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> "AnomalyDetector":  # noqa: F821  (forward ref avoids circular import)
    """
    Build an :class:`AnomalyDetector` and populate its weights from a
    checkpoint produced by :func:`save_checkpoint`.

    The checkpoint must contain a ``"config"`` sub-dict with the keys
    ``input_size``, ``hidden_size``, and ``num_classes`` so the model can
    be reconstructed with the same architecture.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to ``.pt`` checkpoint file.
    device : torch.device | str
        Device on which to place the loaded model.

    Returns
    -------
    AnomalyDetector
        Model in eval mode with loaded weights.
    """
    # Imported here to avoid circular imports at module level
    from src.models.anomaly_detector import AnomalyDetector

    ckpt = load_checkpoint(checkpoint_path, device=device)

    cfg = ckpt.get("config", {})
    model = AnomalyDetector(
        input_size=cfg.get("input_size", 2131),
        hidden_size=cfg.get("hidden_size", 256),
        num_classes=cfg.get("num_classes", 14),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", float("nan"))
    logger.info("Model loaded: epoch=%s  loss=%.4f", epoch, loss)
    print(f"✅ Loaded model checkpoint: epoch {epoch},  train loss {loss:.4f}")
    return model
