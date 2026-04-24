"""
tests/test_dataset_comprehensive.py
===================================

Comprehensive tests for VideoFeatureDataset and data loading utilities.

Coverage:
  - Dataset initialization
  - Dataset length and indexing
  - Feature loading and shapes
  - Label extraction
  - Collate function behavior (now returns features, labels, lengths)
  - Edge cases and error handling
  - Variable-length sequences
  - Batch independence
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import VideoFeatureDataset, collate_fn


# ============================================================================
# Fixtures for Dataset Testing
# ============================================================================

@pytest.fixture
def temp_features_dir(tmp_path):
    """Create temporary directory with synthetic feature files."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    return features_dir


@pytest.fixture
def sample_features_dir(temp_features_dir):
    """Create sample .npz feature files for testing."""
    # Create train samples
    for i in range(3):
        features = np.random.randn(10, 2131).astype(np.float32)
        metadata = {
            'label': i % 2,
            'class': f'class_{i}',
            'video_id': f'video_{i}'
        }
        filepath = temp_features_dir / f"train_video_{i}.npz"
        np.savez_compressed(filepath, features=features, metadata=metadata)

    # Create test samples
    for i in range(2):
        features = np.random.randn(8, 2131).astype(np.float32)
        metadata = {
            'label': (i + 1) % 2,
            'class': f'test_class_{i}',
            'video_id': f'test_video_{i}'
        }
        filepath = temp_features_dir / f"test_video_{i}.npz"
        np.savez_compressed(filepath, features=features, metadata=metadata)

    return temp_features_dir


# ============================================================================
# 1. Dataset Initialization Tests
# ============================================================================

class TestVideoFeatureDatasetInitialization:
    """Test dataset initialization."""

    def test_dataset_initialization(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        assert dataset is not None

    def test_dataset_loads_train_split(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        assert len(dataset) == 3

    def test_dataset_loads_test_split(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="test")
        assert len(dataset) == 2

    def test_dataset_with_nonexistent_directory(self):
        dataset = VideoFeatureDataset("/nonexistent/path", split="train")
        assert len(dataset) == 0


# ============================================================================
# 2. Dataset Length Tests
# ============================================================================

class TestVideoFeatureDatasetLength:

    def test_dataset_len(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        assert len(dataset) == 3

    def test_dataset_len_zero_for_empty(self, temp_features_dir):
        dataset = VideoFeatureDataset(str(temp_features_dir), split="train")
        assert len(dataset) == 0


# ============================================================================
# 3. Dataset Item Retrieval Tests
# ============================================================================

class TestVideoFeatureDatasetItemRetrieval:

    def test_getitem_returns_tuple(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_features_tensor(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        features, label = dataset[0]
        assert isinstance(features, torch.Tensor)

    def test_getitem_label_type(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        features, label = dataset[0]
        assert isinstance(label, (int, np.integer, torch.Tensor))

    def test_getitem_features_shape(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        features, label = dataset[0]
        assert features.ndim == 2
        assert features.shape[1] == 2131

    def test_getitem_features_dtype(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        features, label = dataset[0]
        assert features.dtype == torch.float32

    def test_getitem_all_indices(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        for idx in range(len(dataset)):
            features, label = dataset[idx]
            assert features is not None
            assert label is not None

    def test_getitem_out_of_bounds(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        with pytest.raises(IndexError):
            _ = dataset[1000]


# ============================================================================
# 4. Feature Validity Tests
# ============================================================================

class TestVideoFeatureDatasetFeatureValidity:

    def test_features_no_nan(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        for idx in range(len(dataset)):
            features, _ = dataset[idx]
            assert not torch.isnan(features).any()

    def test_features_no_inf(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        for idx in range(len(dataset)):
            features, _ = dataset[idx]
            assert not torch.isinf(features).any()

    def test_features_finite(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        for idx in range(len(dataset)):
            features, _ = dataset[idx]
            assert torch.isfinite(features).all()


# ============================================================================
# 5. Label Validity Tests
# ============================================================================

class TestVideoFeatureDatasetLabelValidity:

    def test_labels_in_valid_range(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            assert label >= 0

    def test_labels_are_integers(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            assert isinstance(label, (int, np.integer, torch.Tensor))


# ============================================================================
# 6. Collate Function Tests
# ============================================================================

class TestCollateFunction:
    """collate_fn now returns (features_padded, labels, lengths)."""

    def test_collate_fn_returns_three_values(self):
        batch = [
            (torch.randn(10, 2131), 0),
            (torch.randn(12, 2131), 1),
        ]
        result = collate_fn(batch)
        assert len(result) == 3, "collate_fn must return (features, labels, lengths)"

    def test_collate_fn_basic(self):
        batch = [
            (torch.randn(10, 2131), 0),
            (torch.randn(12, 2131), 1),
        ]
        features_padded, labels, lengths = collate_fn(batch)
        assert isinstance(features_padded, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(lengths, torch.Tensor)

    def test_collate_fn_padding(self):
        batch = [
            (torch.randn(8,  2131), 0),
            (torch.randn(12, 2131), 1),
            (torch.randn(10, 2131), 0),
        ]
        features_padded, labels, lengths = collate_fn(batch)
        # Padded to longest sequence
        assert features_padded.shape[1] == 12
        assert features_padded.shape[0] == 3

    def test_collate_fn_lengths_correct(self):
        """lengths must reflect the real (pre-padding) segment counts."""
        batch = [
            (torch.randn(8,  2131), 0),
            (torch.randn(12, 2131), 1),
            (torch.randn(10, 2131), 0),
        ]
        _, _, lengths = collate_fn(batch)
        assert lengths.tolist() == [8, 12, 10]

    def test_collate_fn_lengths_dtype(self):
        batch = [
            (torch.randn(10, 2131), 0),
            (torch.randn(12, 2131), 1),
        ]
        _, _, lengths = collate_fn(batch)
        assert lengths.dtype == torch.long

    def test_collate_fn_labels(self):
        batch = [
            (torch.randn(10, 2131), 0),
            (torch.randn(12, 2131), 1),
            (torch.randn(10, 2131), 2),
        ]
        _, labels, _ = collate_fn(batch)
        assert labels.shape == (3,)
        assert torch.all(labels == torch.LongTensor([0, 1, 2]))

    def test_collate_fn_with_single_sample(self):
        batch = [(torch.randn(10, 2131), 0)]
        features_padded, labels, lengths = collate_fn(batch)
        assert features_padded.shape == (1, 10, 2131)
        assert labels.shape == (1,)
        assert lengths.tolist() == [10]

    def test_collate_fn_equal_length_sequences(self):
        """No padding needed — lengths should equal the common length."""
        batch = [
            (torch.randn(10, 2131), 0),
            (torch.randn(10, 2131), 1),
            (torch.randn(10, 2131), 2),
        ]
        features_padded, labels, lengths = collate_fn(batch)
        assert features_padded.shape == (3, 10, 2131)
        assert lengths.tolist() == [10, 10, 10]

    def test_collate_fn_dtype_preservation(self):
        batch = [
            (torch.randn(10, 2131, dtype=torch.float32), 0),
            (torch.randn(12, 2131, dtype=torch.float32), 1),
        ]
        features_padded, _, _ = collate_fn(batch)
        assert features_padded.dtype == torch.float32

    def test_collate_fn_lengths_match_real_data(self):
        """Verify that real segments up to lengths[i] are unmodified."""
        f0 = torch.randn(8,  2131)
        f1 = torch.randn(12, 2131)
        batch = [(f0, 0), (f1, 1)]
        features_padded, _, lengths = collate_fn(batch)

        # First video: real segments 0..7
        assert torch.allclose(features_padded[0, :8, :], f0)
        # Padded positions (8..11) should be zeros
        assert torch.allclose(features_padded[0, 8:, :], torch.zeros(4, 2131))

        # Second video: all segments are real (longest, no padding)
        assert torch.allclose(features_padded[1, :12, :], f1)


# ============================================================================
# 7. DataLoader Integration Tests
# ============================================================================

class TestDatasetDataLoaderIntegration:

    def test_dataset_with_dataloader(self, sample_features_dir):
        from torch.utils.data import DataLoader

        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
        )

        for features, labels, lengths in dataloader:
            assert features.shape[0] in (1, 2)
            assert labels.shape[0] in (1, 2)
            assert lengths.shape[0] in (1, 2)
            # lengths must not exceed the padded time dimension
            assert lengths.max().item() <= features.shape[1]

    def test_dataloader_batch_size(self, sample_features_dir):
        from torch.utils.data import DataLoader

        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        dataloader = DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
        )

        batch_sizes = [features.shape[0] for features, _, _ in dataloader]
        assert batch_sizes == [2, 1]

    def test_dataloader_lengths_all_positive(self, sample_features_dir):
        from torch.utils.data import DataLoader

        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        dataloader = DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
        )

        for _, _, lengths in dataloader:
            assert (lengths > 0).all()

    def test_dataloader_shuffle(self, sample_features_dir):
        from torch.utils.data import DataLoader

        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        dataloader = DataLoader(
            dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
        )

        items = list(dataloader)
        assert len(items) == len(dataset)


# ============================================================================
# 8. Edge Cases Tests
# ============================================================================

class TestVideoFeatureDatasetEdgeCases:

    def test_dataset_with_single_sample(self, temp_features_dir):
        features = np.random.randn(10, 2131).astype(np.float32)
        metadata = {'label': 0, 'class': 'test'}
        filepath = temp_features_dir / "train_single.npz"
        np.savez_compressed(filepath, features=features, metadata=metadata)

        dataset = VideoFeatureDataset(str(temp_features_dir), split="train")
        assert len(dataset) == 1

    def test_dataset_with_zero_features(self, temp_features_dir):
        features = np.zeros((10, 2131), dtype=np.float32)
        metadata = {'label': 0, 'class': 'test'}
        filepath = temp_features_dir / "train_zeros.npz"
        np.savez_compressed(filepath, features=features, metadata=metadata)

        dataset = VideoFeatureDataset(str(temp_features_dir), split="train")
        loaded_features, _ = dataset[0]
        assert torch.allclose(loaded_features, torch.zeros(10, 2131))

    def test_dataset_with_very_long_sequence(self, temp_features_dir):
        features = np.random.randn(1000, 2131).astype(np.float32)
        metadata = {'label': 1, 'class': 'test'}
        filepath = temp_features_dir / "train_long.npz"
        np.savez_compressed(filepath, features=features, metadata=metadata)

        dataset = VideoFeatureDataset(str(temp_features_dir), split="train")
        loaded_features, _ = dataset[0]
        assert loaded_features.shape[0] == 1000

    def test_dataset_with_very_short_sequence(self, temp_features_dir):
        features = np.random.randn(1, 2131).astype(np.float32)
        metadata = {'label': 1, 'class': 'test'}
        filepath = temp_features_dir / "train_short.npz"
        np.savez_compressed(filepath, features=features, metadata=metadata)

        dataset = VideoFeatureDataset(str(temp_features_dir), split="train")
        loaded_features, _ = dataset[0]
        assert loaded_features.shape[0] == 1


# ============================================================================
# 9. Batch Independence Tests
# ============================================================================

class TestDatasetBatchIndependence:

    def test_batch_samples_independent(self, sample_features_dir):
        from torch.utils.data import DataLoader

        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        dataloader = DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=collate_fn
        )

        for features, labels, lengths in dataloader:
            original_features = features.clone()
            features[0, 0, 0] = 999.0

            features2, labels2, lengths2 = next(iter(dataloader))
            assert not torch.allclose(features, features2)


# ============================================================================
# 10. Reproducibility Tests
# ============================================================================

class TestVideoFeatureDatasetReproducibility:

    def test_dataset_deterministic_access(self, sample_features_dir):
        dataset = VideoFeatureDataset(str(sample_features_dir), split="train")
        features_1, label_1 = dataset[0]
        features_2, label_2 = dataset[0]
        assert torch.allclose(features_1, features_2)
        assert label_1 == label_2