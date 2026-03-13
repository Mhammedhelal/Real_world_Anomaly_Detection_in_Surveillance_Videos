"""
Dataset classes and data loading utilities.

Extracted from: AnomalyDetector_helal_Feb_23.ipynb
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from pathlib import Path



# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class VideoFeatureDataset(Dataset):
    """
    Loads pre-extracted .npz feature files produced by extract_features.py.

    Each .npz file must contain:
        - 'features'  : np.ndarray  [num_segments, feature_dim]
        - 'metadata'  : dict-like   with key 'label' (int, 0 = normal)
    """

    def __init__(self, features_dir: str, split: str = "train"):
        """
        Args:
            features_dir: Root features directory (contains subfolders).
            split: 'train' or 'test' — files whose names start with this prefix.
        """
        self.samples = []   # list of (features_tensor, label)
        self.filenames = []

        # Walk all subfolders looking for .npz files matching the split
        for root, _, files in os.walk(features_dir):
            for fname in sorted(files):
                if not fname.endswith(".npz"):
                    continue
                if not fname.startswith(split + "_"):
                    continue

                path = os.path.join(root, fname)
                try:
                    data = np.load(path, allow_pickle=True)
                    features = data["features"].astype(np.float32)    # [S, D]
                    metadata = data["metadata"].item()
                    label = int(metadata["label"])

                    self.samples.append((torch.from_numpy(features), label))
                    self.filenames.append(fname)
                except Exception as e:
                    print(f"⚠️  Skipping {fname}: {e}")

        print(f"📦 Loaded {len(self.samples)} '{split}' feature files from {features_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        features, label = self.samples[idx]
        return features, label


def collate_fn(batch):
    """Pad variable-length segment sequences to the longest in the batch."""
    features, labels = zip(*batch)
    features_padded = nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels = torch.LongTensor(labels)
    return features_padded, labels