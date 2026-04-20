"""
src/data/dataset.py
-------------------
Dataset classes and data loading utilities.

Includes .npz validation at load time so corrupted or NaN-containing
feature files are caught before they silently corrupt training gradients.
"""

import os

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_features(features: np.ndarray, fname: str) -> str | None:
    """
    Validate a feature array loaded from an .npz file.

    Returns an error message string if invalid, else None.
    Checks performed:
      - Array is 2-D: [num_segments, feature_dim]
      - dtype is numeric (float32/float64)
      - No NaN values (silent gradient poison)
      - No Inf values
      - At least one segment present
      - Feature dim is positive
    """
    if features.ndim != 2:
        return f"expected 2-D array [S, D], got shape {features.shape}"
    if features.shape[0] == 0:
        return "zero segments (empty feature array)"
    if features.shape[1] == 0:
        return "zero feature dimensions"
    if not np.issubdtype(features.dtype, np.floating):
        return f"non-float dtype {features.dtype}"
    if np.isnan(features).any():
        n_nan = int(np.isnan(features).sum())
        return f"{n_nan} NaN value(s) detected — file is corrupted"
    if np.isinf(features).any():
        n_inf = int(np.isinf(features).sum())
        return f"{n_inf} Inf value(s) detected — file is corrupted"
    return None   # all checks passed


class VideoFeatureDataset(Dataset):
    """
    Loads pre-extracted .npz feature files produced by extract_features.py.

    Each .npz file must contain:
        - ``features``  : np.ndarray [num_segments, feature_dim], float32
        - ``metadata``  : dict-like  with key ``label`` (int, 0 = normal)

    Files that fail validation (wrong shape, NaN, Inf, bad dtype) are
    **skipped with a warning** rather than silently corrupting training.
    The count of skipped files is exposed as ``self.n_skipped``.

    Parameters
    ----------
    features_dir : str
        Root directory to scan for .npz files (recursive).
    split : str
        Only files whose name starts with ``<split>_`` are loaded.
    strict : bool
        If ``True``, raise ``ValueError`` on the first invalid file instead
        of skipping. Useful for CI / data-quality checks.
    """

    def __init__(
        self,
        features_dir: str,
        split: str = "train",
        strict: bool = False,
    ) -> None:
        self.features_dir = features_dir
        self.split = split
        self.strict = strict

        self.samples: List[Tuple[torch.Tensor, int]] = []
        self.filenames: List[str] = []
        self.n_skipped: int = 0

        self._load(features_dir, split)

    def _load(self, features_dir: str, split: str) -> None:
        prefix = split + "_"

        for root, _, files in os.walk(features_dir):
            for fname in sorted(files):
                if not fname.endswith(".npz"):
                    continue
                if not fname.startswith(prefix):
                    continue

                path = os.path.join(root, fname)
                self._load_one(path, fname)

        logger.info(
            "Loaded %d '%s' files from %s  (%d skipped due to validation errors)",
            len(self.samples), split, features_dir, self.n_skipped,
        )
        if self.n_skipped > 0:
            logger.warning(
                "%d file(s) were skipped — re-extract them with extract_features.py",
                self.n_skipped,
            )

    def _load_one(self, path: str, fname: str) -> None:
        try:
            data     = np.load(path, allow_pickle=True)
            features = data["features"].astype(np.float32)
            metadata = data["metadata"].item()
            label    = int(metadata["label"])
        except KeyError as exc:
            self._handle_invalid(fname, f"missing key {exc} in .npz file")
            return
        except Exception as exc:
            self._handle_invalid(fname, f"could not load: {exc}")
            return

        error = _validate_features(features, fname)
        if error:
            self._handle_invalid(fname, error)
            return

        self.samples.append((torch.from_numpy(features), label))
        self.filenames.append(fname)

    def _handle_invalid(self, fname: str, reason: str) -> None:
        msg = "Invalid feature file %s: %s"
        if self.strict:
            raise ValueError(msg % (fname, reason))
        logger.warning(msg, fname, reason)
        self.n_skipped += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


def collate_fn(batch):
    """Pad variable-length segment sequences to the longest in the batch."""
    features, labels = zip(*batch)
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    return features_padded, torch.LongTensor(labels)

