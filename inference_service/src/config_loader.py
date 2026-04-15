"""
inference_service/src/config_loader.py
--------------------
Configuration loader for inference service.

Loads config from environment variables and YAML files.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import yaml


def load_inference_config(config_path: Optional[str] = None) -> Dict:
    """
    Load inference configuration.
    
    Priority:
    1. Environment variables (highest)
    2. config_path YAML file
    3. Default values (lowest)
    
    Parameters
    ----------
    config_path : str | None
        Path to YAML config file
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    # Default configuration
    config = {
        # Model
        'checkpoint_path': os.getenv(
            'MODEL_CHECKPOINT',
            '/app/models/best_model.pt'
        ),
        'input_size': int(os.getenv('INPUT_SIZE', '2131')),
        'hidden_size': int(os.getenv('HIDDEN_SIZE', '256')),
        'num_classes': int(os.getenv('NUM_CLASSES', '14')),
        
        # Preprocessing
        'frame_width': int(os.getenv('FRAME_WIDTH', '224')),
        'frame_height': int(os.getenv('FRAME_HEIGHT', '224')),
        'segment_length': int(os.getenv('SEGMENT_LENGTH', '16')),
        
        # Inference
        'threshold': float(os.getenv('ANOMALY_THRESHOLD', '0.5')),
        'device': os.getenv('DEVICE', 'cuda'),
        
        # Storage
        'features_dir': os.getenv('FEATURES_DIR', '/app/data/features'),
        'metadata_dir': os.getenv('METADATA_DIR', '/app/data/metadata'),
        
        # Feature extraction
        'i3d_model': os.getenv('I3D_MODEL', 'i3d_r50'),
        'yolo_model': os.getenv('YOLO_MODEL', 'yolov8n.pt'),
    }
    
    # Load from YAML if provided
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                config.update(yaml_config)
    
    # Derived values
    config['frame_size'] = (config['frame_height'], config['frame_width'])
    
    return config


def get_model_config(config: Dict) -> Dict:
    """Extract model-specific config."""
    return {
        'input_size': config['input_size'],
        'hidden_size': config['hidden_size'],
        'num_classes': config['num_classes'],
    }


def get_preprocessor_config(config: Dict) -> Dict:
    """Extract preprocessor config."""
    return {
        'frame_size': config['frame_size'],
        'segment_length': config['segment_length'],
    }