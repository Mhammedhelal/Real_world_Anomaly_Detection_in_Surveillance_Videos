"""
Engine module containing training and feature extraction pipelines.
"""

from .pipeline import FeatureExtractionPipeline, process_in_batches, check_status
from .trainer import Trainer, train_model

__all__ = [
    'FeatureExtractionPipeline',
    'process_in_batches',
    'check_status',
    'Trainer',
    'train_model',
]
