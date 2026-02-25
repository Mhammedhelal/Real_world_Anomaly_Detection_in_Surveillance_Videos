"""
Dataset classes and data loading utilities.

Extracted from: AnomalyDetector_helal_Feb_23.ipynb
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional


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


def create_dataloaders(
    features_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    collate_type: str = 'variable_length',
    fixed_length: Optional[int] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and test dataloaders from extracted features.
    
    Args:
        features_dir: Directory containing .npz feature files
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle training data
        collate_type: 'variable_length' or 'fixed_length'
        fixed_length: Length for fixed-length collate (required if collate_type='fixed_length')
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    import os
    
    all_features = []
    all_labels = []
    
    # Load all features from directory
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
    
    for filename in sorted(feature_files):
        file_path = os.path.join(features_dir, filename)
        data = np.load(file_path, allow_pickle=True)
        
        # Load features: Shape [Segments, 2131]
        features = data['features']
        
        # Extract metadata to get label
        metadata = data['metadata'].item()
        label = int(metadata['label'])
        
        all_features.append(torch.FloatTensor(features))
        all_labels.append(label)
    
    if len(all_features) == 0:
        raise ValueError(f"No feature files found in {features_dir}")
    
    print(f"✅ Loaded {len(all_features)} video feature sets from {features_dir}")
    
    # Create dataset
    dataset = VideoDataset(all_features, all_labels)
    
    # Choose collate function
    if collate_type == 'fixed_length':
        if fixed_length is None:
            raise ValueError("fixed_length must be specified for fixed_length collate")
        collate_fn = lambda batch: collate_fn_fixed_length(batch, fixed_length)
    else:
        collate_fn = collate_fn_variable_length
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, None
