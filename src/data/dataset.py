"""
src/data/dataset.py
-------------------
Dataset classes and data loading utilities.

Extracted from: AnomalyDetector_helal_Feb_23.ipynb
"""

import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


class VideoFeatureDataset(Dataset):
    """
    Loads pre-extracted .npz feature files.

    Each .npz file must contain:
        - 'features'  : np.ndarray  [num_segments, feature_dim]
        - 'metadata'  : dict-like   with key 'label' (int, 0 = normal)
    """

    def __init__(self, features_dir: str, split: str = "train") -> None:
        self.samples: List[Tuple[torch.Tensor, int]] = []
        self.filenames: List[str] = []

        for root, _, files in os.walk(features_dir):
            for fname in sorted(files):
                if not fname.endswith(".npz"):
                    continue
                if not fname.startswith(split + "_"):
                    continue

                path = os.path.join(root, fname)
                try:
                    data = np.load(path, allow_pickle=True)
                    features = data["features"].astype(np.float32)
                    metadata = data["metadata"].item()
                    label = int(metadata["label"])
                    self.samples.append((torch.from_numpy(features), label))
                    self.filenames.append(fname)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", fname, exc)

        logger.info(
            "Loaded %d '%s' feature files from %s",
            len(self.samples), split, features_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        features, label = self.samples[idx]
        return features, label


def collate_fn(batch):
    """Pad variable-length segment sequences to the longest in the batch."""
    features, labels = zip(*batch)
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    labels_tensor = torch.LongTensor(labels)
    return features_padded, labels_tensor
