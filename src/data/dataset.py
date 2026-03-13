"""
Dataset classes and data loading utilities.

Extracted from: AnomalyDetector_helal_Feb_23.ipynb
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from pathlib import Path



class VideoDataset(Dataset):
    """
    Custom Dataset for handling pre-extracted video features.
    Note: Videos may have different numbers of segments.
    
    Args:
        features: List of feature tensors [Segments, 2131]
        labels: List of labels for each video
    """
    def __init__(self, features: List[torch.Tensor], labels: List[int]):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Returns a tuple: (Tensor[Segments, 2131], Label)
        return self.features[idx], self.labels[idx]


def collate_fn_variable_length(batch):
    """
    Collate function for handling variable sequence lengths in a single batch.
    Pads sequences to the length of the longest video in the batch.
    
    Args:
        batch: List of (features, label) tuples from VideoDataset
    
    Returns:
        Tuple of (padded_features, labels)
        - padded_features: [Batch, MaxSegments, 2131]
        - labels: [Batch]
    """
    features, labels = zip(*batch)
    # Pads sequences to the length of the longest video in the batch
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = torch.LongTensor(labels)
    return features_padded, labels


def collate_fn_fixed_length(batch, sequence_length=128):
    """
    Collate function for fixed-length sequences.
    Truncates or pads to a fixed length.
    
    Args:
        batch: List of (features, label) tuples
        sequence_length: Target sequence length
    
    Returns:
        Tuple of (fixed_features, labels)
    """
    features, labels = zip(*batch)
    
    fixed_features = []
    for feat in features:
        if feat.shape[0] > sequence_length:
            # Truncate
            fixed_features.append(feat[:sequence_length])
        else:
            # Pad
            padding = sequence_length - feat.shape[0]
            padded = torch.nn.functional.pad(feat, (0, 0, 0, padding))
            fixed_features.append(padded)
    
    features_stacked = torch.stack(fixed_features)
    labels = torch.LongTensor(labels)
    return features_stacked, labels
