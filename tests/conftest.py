"""
tests/conftest.py
-----------------
Pytest fixtures and configuration.
Updated: imports setup_logging so test output uses the structured logger.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Silence library loggers during tests; keep WARNING+
import logging
logging.getLogger("src").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device():
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def seed_everything():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    yield


@pytest.fixture
def synthetic_video_features():
    return torch.randn(2, 10, 2131, dtype=torch.float32)


@pytest.fixture
def synthetic_video_features_variable():
    return [
        torch.randn(8,  2131, dtype=torch.float32),
        torch.randn(12, 2131, dtype=torch.float32),
        torch.randn(10, 2131, dtype=torch.float32),
    ]


@pytest.fixture
def synthetic_short_features():
    return torch.randn(1, 1, 2131, dtype=torch.float32)


@pytest.fixture
def synthetic_zero_features():
    return torch.zeros(2, 10, 2131, dtype=torch.float32)


@pytest.fixture
def synthetic_labels():
    return torch.LongTensor([0, 1, 2, 0])


@pytest.fixture
def synthetic_binary_labels():
    return torch.LongTensor([0, 0, 1, 1])


@pytest.fixture
def synthetic_video_frames():
    return torch.randint(0, 256, (16, 3, 224, 224), dtype=torch.uint8)


@pytest.fixture
def synthetic_3d_video():
    return torch.randn(1, 3, 16, 224, 224, dtype=torch.float32)


@pytest.fixture
def synthetic_3d_batch_video():
    return torch.randn(2, 3, 16, 224, 224, dtype=torch.float32)


@pytest.fixture
def model_config():
    return {'input_size': 2131, 'hidden_size': 256, 'num_classes': 14}


@pytest.fixture
def loss_config():
    return {'lambda1': 8e-5, 'lambda2': 8e-5, 'margin': 1.0}


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture(params=["cpu"], ids=["cpu"])
def device_param(request):
    # Only parametrize CPU by default; add "cuda" if GPU available
    return torch.device(request.param)
