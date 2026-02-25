"""
Configuration management for the project.

Loads hyperparameters from configs/default.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class that loads from YAML and provides dict-like access"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __getattr__(self, key):
        if key.startswith('_'):
            return super().__getattribute__(key)
        if key in self._config:
            value = self._config[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def to_dict(self):
        return self._config
    
    def __repr__(self):
        return f"Config({self._config})"


def load_config(config_path='configs/default.yaml'):
    """
    Load configuration from YAML files
    
    Args:
        config_path: Path to YAML configuration file (relative to project root)
    
    Returns:
        Config object with all settings
    """
    # Get absolute path
    if not os.path.isabs(config_path):
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


# Global config instance
_config = None


def get_config():
    """
    Get the global configuration object
    
    Returns:
        Config object with all settings
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config_path):
    """
    Set a new configuration file
    
    Args:
        config_path: Path to YAML configuration file
    """
    global _config
    _config = load_config(config_path)


def get_device():
    """Get the configured device"""
    config = get_config()
    return config['hardware']['device']


def get_model_config():
    """Get model configuration"""
    config = get_config()
    return {
        'input_size': config['model']['input_size'],
        'hidden_size': config['model']['hidden_size'],
        'num_classes': config['model']['num_classes'],
    }


def get_training_config():
    """Get training configuration"""
    config = get_config()
    return {
        'batch_size': config['training']['batch_size'],
        'num_epochs': config['training']['num_epochs'],
        'learning_rate': config['training']['learning_rate'],
        'rho': config['training']['rho'],
        'eps': config['training']['eps'],
    }


def get_loss_config():
    """Get loss function configuration"""
    config = get_config()
    return {
        'lambda_smoothness': config['loss']['lambda_smoothness'],
        'lambda_sparsity': config['loss']['lambda_sparsity'],
        'margin': config['loss']['margin'],
    }


def get_feature_extraction_config():
    """Get feature extraction configuration"""
    config = get_config()
    return {
        'motion_extractor': config['feature_extraction']['motion_extractor'],
        'target_fps': config['feature_extraction']['target_fps'],
        'i3d_pretrained': config['feature_extraction']['i3d_pretrained'],
        'i3d_frozen': config['feature_extraction']['i3d_frozen'],
        'yolo_model_name': config['feature_extraction']['yolo_model_name'],
    }
