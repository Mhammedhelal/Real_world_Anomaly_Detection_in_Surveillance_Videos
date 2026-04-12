"""
Data module containing dataset and preprocessing utilities.
"""

from src.data.dataset import VideoFeatureDataset, collate_fn
from src.data.labels import UCF_CRIME_CATEGORIES, get_class_name, get_label_from_name
from src.data.metadata import DatasetMetadata

# Backward compatibility aliases
VideoDataset = VideoFeatureDataset
collate_fn_variable_length = collate_fn
collate_fn_fixed_length = collate_fn

__all__ = [
    'VideoFeatureDataset',
    'VideoDataset',  # backward compatibility
    'collate_fn',
    'collate_fn_variable_length',  # backward compatibility
    'collate_fn_fixed_length',  # backward compatibility
    'DatasetMetadata',
    'UCF_CRIME_CATEGORIES',
    'get_class_name',
    'get_label_from_name',
]
