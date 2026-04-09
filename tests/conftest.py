"""
Pytest fixtures and configuration for the test suite.

Provides reusable fixtures for model initialization, data generation,
and device handling across all tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def device():
    """Determine device for testing (CPU or GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Force CPU device for CPU-specific tests."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def seed_everything():
    """Set seeds for reproducibility in each test."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield


@pytest.fixture
def synthetic_video_features():
    """Generate synthetic video features for testing."""
    # [batch_size=2, num_segments=10, feature_dim=2131]
    return torch.randn(2, 10, 2131, dtype=torch.float32)


@pytest.fixture
def synthetic_video_features_variable():
    """Generate variable-length video features (simulating batch with different lengths)."""
    # Create features with different segment lengths
    features_list = [
        torch.randn(8, 2131, dtype=torch.float32),   # 8 segments
        torch.randn(12, 2131, dtype=torch.float32),  # 12 segments
        torch.randn(10, 2131, dtype=torch.float32),  # 10 segments
    ]
    return features_list


@pytest.fixture
def synthetic_short_features():
    """Generate short synthetic features for edge case testing."""
    # [batch_size=1, num_segments=1, feature_dim=2131]
    return torch.randn(1, 1, 2131, dtype=torch.float32)


@pytest.fixture
def synthetic_zero_features():
    """Generate all-zero features for edge case testing."""
    return torch.zeros(2, 10, 2131, dtype=torch.float32)


@pytest.fixture
def synthetic_labels():
    """Generate synthetic labels (0=normal, 1-13=anomaly types)."""
    return torch.LongTensor([0, 1, 2, 0])  # Mixed normal and anomalous


@pytest.fixture
def synthetic_binary_labels():
    """Generate synthetic binary labels (0=normal, 1=anomalous)."""
    return torch.LongTensor([0, 0, 1, 1])


@pytest.fixture
def synthetic_video_frames():
    """Generate synthetic video frames [T, C, H, W]."""
    # [16 frames, 3 channels, 224x224 resolution]
    return torch.randint(0, 256, (16, 3, 224, 224), dtype=torch.uint8)


@pytest.fixture
def synthetic_batch_video_frames():
    """Generate batch of synthetic video frames [B, T, C, H, W]."""
    # [batch_size=2, 16 frames, 3 channels, 112x112]
    return torch.randint(0, 256, (2, 16, 3, 112, 112), dtype=torch.uint8)


@pytest.fixture
def synthetic_3d_video():
    """Generate synthetic 3D video tensor [B, C, T, H, W] (I3D format)."""
    # [batch_size=1, 3 channels, 16 frames, 224x224]
    return torch.randn(1, 3, 16, 224, 224, dtype=torch.float32)


@pytest.fixture
def synthetic_3d_batch_video():
    """Generate batch of 3D video tensors [B, C, T, H, W]."""
    # [batch_size=2, 3 channels, 16 frames, 224x224]
    return torch.randn(2, 3, 16, 224, 224, dtype=torch.float32)


@pytest.fixture
def synthetic_yolo_detections():
    """Generate synthetic YOLO detection features [num_detections, feature_dim]."""
    # [10 detections, 83 features (YOLOv8n)]
    return torch.randn(10, 83, dtype=torch.float32)


@pytest.fixture
def model_config():
    """Minimal configuration for model testing."""
    return {
        'input_size': 2131,
        'hidden_size': 256,
        'num_classes': 14,
    }


@pytest.fixture
def loss_config():
    """Configuration for loss function testing."""
    return {
        'lambda1': 8e-5,  # Smoothness
        'lambda2': 8e-5,  # Sparsity
        'margin': 1.0,
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for file-based tests."""
    return tmp_path


def is_gpu_available():
    """Check if GPU is available."""
    return torch.cuda.is_available()


@pytest.fixture(params=["cpu", "cuda"], ids=["cpu", "cuda"])
def device_param(request):
    """Parametrized fixture to run tests on both CPU and GPU."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)
