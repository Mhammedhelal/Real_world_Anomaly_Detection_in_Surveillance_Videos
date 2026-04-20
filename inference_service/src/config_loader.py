"""
inference_service/src/config_loader.py
---------------------------------------
Configuration loader for the inference service.

Priority order (highest → lowest):
  1. Environment variables
  2. YAML config file (if provided)
  3. Hard-coded defaults

This keeps the service 12-factor-app compliant: all tuneable parameters
are exposed as env vars for Docker / Kubernetes deployments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, object] = {
    # Model
    "checkpoint_path": "/app/models/best_model.pt",
    "input_size":       2131,
    "hidden_size":      256,
    "num_classes":      14,

    # Preprocessing
    "frame_width":      224,
    "frame_height":     224,
    "segment_length":   16,

    # Inference
    "anomaly_threshold": 0.5,
    "device":            "cuda",

    # Localisation
    "localizer_strategy": "object",  # "object" or "region"

    # Storage (runtime)
    "features_dir": "/app/data/features",
    "metadata_dir": "/app/data/metadata",

    # Feature extraction
    "yolo_model":   "yolov8n.pt",
    "target_fps":   8,
}

# Mapping: config key → environment variable name
_ENV_MAP: Dict[str, str] = {
    "checkpoint_path":    "MODEL_CHECKPOINT",
    "input_size":         "INPUT_SIZE",
    "hidden_size":        "HIDDEN_SIZE",
    "num_classes":        "NUM_CLASSES",
    "frame_width":        "FRAME_WIDTH",
    "frame_height":       "FRAME_HEIGHT",
    "segment_length":     "SEGMENT_LENGTH",
    "anomaly_threshold":  "ANOMALY_THRESHOLD",
    "device":             "DEVICE",
    "localizer_strategy": "LOCALIZER_STRATEGY",
    "features_dir":       "FEATURES_DIR",
    "metadata_dir":       "METADATA_DIR",
    "yolo_model":         "YOLO_MODEL",
    "target_fps":         "TARGET_FPS",
}

# Keys that should be cast to int / float
_INT_KEYS   = {"input_size", "hidden_size", "num_classes",
               "frame_width", "frame_height", "segment_length", "target_fps"}
_FLOAT_KEYS = {"anomaly_threshold"}


def load_inference_config(
    config_path: Optional[str] = None,
) -> Dict[str, object]:
    """
    Return the merged inference configuration dictionary.

    Parameters
    ----------
    config_path : str | None
        Optional path to a YAML file whose values override defaults.
        Environment variables override the YAML file.

    Returns
    -------
    dict
        Keys correspond to ``_DEFAULTS`` above, values are correctly typed.
    """
    config = dict(_DEFAULTS)

    # Layer 1: YAML file
    if config_path:
        path = Path(config_path)
        if path.exists():
            with path.open() as fh:
                yaml_data = yaml.safe_load(fh) or {}
            config.update(yaml_data)

    # Layer 2: environment variables
    for key, env_name in _ENV_MAP.items():
        val = os.getenv(env_name)
        if val is not None:
            if key in _INT_KEYS:
                config[key] = int(val)
            elif key in _FLOAT_KEYS:
                config[key] = float(val)
            else:
                config[key] = val

    # Derived values
    config["frame_size"] = (int(config["frame_height"]), int(config["frame_width"]))

    return config


def get_model_config(config: Dict) -> Dict:
    """Extract model architecture parameters."""
    return {
        "input_size":  config["input_size"],
        "hidden_size": config["hidden_size"],
        "num_classes": config["num_classes"],
    }


def get_preprocessor_config(config: Dict) -> Dict:
    """Extract preprocessor parameters."""
    return {
        "frame_size":      config["frame_size"],
        "segment_length":  config["segment_length"],
    }