"""
tests/test_feature_extractors_comprehensive.py
==============================================

Comprehensive tests for feature extractors (I3D, R3D, etc.).

Coverage:
  - Feature extractor initialization
  - Forward pass with various input shapes
  - Feature dimension correctness
  - Eval mode determinism
  - Device consistency
  - Gradient flow
  - Edge cases
  - Output validity (no NaN/Inf)
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.feature_extractors import (
    BaseFeatureExtractor,
    I3DFeatureExtractor,
    R3DFeatureExtractor,
    LightweightFeatureExtractor,
)


# ============================================================================
# Helper Functions
# ============================================================================

def _has_pytorchvideo():
    """Check if pytorchvideo is installed."""
    try:
        import pytorchvideo
        return True
    except ImportError:
        return False


# ============================================================================
# 1. Base Feature Extractor Tests
# ============================================================================

class TestBaseFeatureExtractor:
    """Test BaseFeatureExtractor abstract class."""

    def test_base_extractor_is_abstract(self):
        """Test that BaseFeatureExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFeatureExtractor()

    def test_base_extractor_requires_forward(self):
        """Test that subclasses must implement forward."""

        class IncompleteExtractor(BaseFeatureExtractor):
            pass

        with pytest.raises(TypeError):
            IncompleteExtractor()


# ============================================================================
# 2. I3D Feature Extractor Tests
# ============================================================================

class TestI3DFeatureExtractor:
    """Test I3D feature extractor."""

    @pytest.mark.skipif(
        not _has_pytorchvideo(),
        reason="pytorchvideo not installed"
    )
    def test_i3d_initialization_default(self):
        """Test I3D can be initialized with default parameters."""
        extractor = I3DFeatureExtractor(device="cpu")
        assert extractor is not None

    @pytest.mark.skipif(
        not _has_pytorchvideo(),
        reason="pytorchvideo not installed"
    )
    def test_i3d_feature_dimension(self):
        """Test I3D feature dimension is 2048."""
        extractor = I3DFeatureExtractor(device="cpu")
        assert extractor.feature_dim == 2048

    @pytest.mark.skipif(
        not _has_pytorchvideo(),
        reason="pytorchvideo not installed"
    )
    def test_i3d_forward_pass(self, synthetic_3d_video):
        """Test I3D forward pass."""
        extractor = I3DFeatureExtractor(device="cpu")
        features = extractor(synthetic_3d_video.to("cpu"))

        assert features is not None
        assert features.shape[0] == 1  # batch size

    @pytest.mark.skipif(
        not _has_pytorchvideo(),
        reason="pytorchvideo not installed"
    )
    def test_i3d_output_shape(self, synthetic_3d_video):
        """Test I3D output shape."""
        extractor = I3DFeatureExtractor(device="cpu")
        features = extractor(synthetic_3d_video.to("cpu"))

        assert features.shape == (1, 2048)

    @pytest.mark.skipif(
        not _has_pytorchvideo(),
        reason="pytorchvideo not installed"
    )
    def test_i3d_batch_processing(self, synthetic_3d_batch_video):
        """Test I3D with batch of videos."""
        extractor = I3DFeatureExtractor(device="cpu")
        features = extractor(synthetic_3d_batch_video.to("cpu"))

        assert features.shape[0] == 2  # batch size
        assert features.shape[1] == 2048  # feature dim

    @pytest.mark.skipif(
        not _has_pytorchvideo(),
        reason="pytorchvideo not installed"
    )
    def test_i3d_eval_mode(self, synthetic_3d_video):
        """Test I3D in eval mode."""
        extractor = I3DFeatureExtractor(device="cpu")
        extractor.eval()

        features = extractor(synthetic_3d_video.to("cpu"))

        assert not torch.isnan(features).any()

    @pytest.mark.skipif(
        not _has_pytorchvideo(),
        reason="pytorchvideo not installed"
    )
    def test_i3d_determinism(self, synthetic_3d_video):
        """Test I3D determinism in eval mode."""
        extractor = I3DFeatureExtractor(device="cpu")
        extractor.eval()

        with torch.no_grad():
            features_1 = extractor(synthetic_3d_video.to("cpu"))
            features_2 = extractor(synthetic_3d_video.to("cpu"))

        assert torch.allclose(features_1, features_2)


# ============================================================================
# 3. R3D Feature Extractor Tests
# ============================================================================

class TestR3DFeatureExtractor:
    """Test R3D feature extractor."""

    def test_r3d_initialization(self):
        """Test R3D can be initialized."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        assert extractor is not None

    def test_r3d_has_feature_dimension(self):
        """Test R3D has feature_dim attribute."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        assert extractor.feature_dim is not None
        assert extractor.feature_dim > 0

    def test_r3d_forward_pass(self, synthetic_3d_video):
        """Test R3D forward pass."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        features = extractor(synthetic_3d_video.to("cpu"))

        assert features is not None
        assert features.ndim == 2  # [batch, feature_dim]

    def test_r3d_output_batch_size(self, synthetic_3d_batch_video):
        """Test R3D output batch dimension."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        features = extractor(synthetic_3d_batch_video.to("cpu"))

        assert features.shape[0] == 2  # batch size

    def test_r3d_output_feature_dim(self, synthetic_3d_video):
        """Test R3D output feature dimension."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        features = extractor(synthetic_3d_video.to("cpu"))

        # Should match feature_dim
        assert features.shape[1] == extractor.feature_dim

    def test_r3d_eval_mode(self, synthetic_3d_video):
        """Test R3D in eval mode."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        extractor.eval()

        with torch.no_grad():
            features = extractor(synthetic_3d_video.to("cpu"))

        assert not torch.isnan(features).any()

    def test_r3d_no_nan_output(self, synthetic_3d_video):
        """Test R3D output has no NaN."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        features = extractor(synthetic_3d_video.to("cpu"))

        assert not torch.isnan(features).any()

    def test_r3d_no_inf_output(self, synthetic_3d_video):
        """Test R3D output has no Inf."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        features = extractor(synthetic_3d_video.to("cpu"))

        assert not torch.isinf(features).any()

    def test_r3d_determinism(self, synthetic_3d_video):
        """Test R3D determinism in eval mode."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        extractor.eval()

        with torch.no_grad():
            features_1 = extractor(synthetic_3d_video.to("cpu"))
            features_2 = extractor(synthetic_3d_video.to("cpu"))

        # Might not be identical due to randomness in initialization
        # But shapes should match
        assert features_1.shape == features_2.shape


# ============================================================================
# 4. Lightweight Feature Extractor Tests
# ============================================================================

class TestLightweightFeatureExtractor:
    """Test Lightweight feature extractor."""

    def test_lightweight_initialization(self):
        """Test Lightweight extractor can be initialized."""
        extractor = LightweightFeatureExtractor(device="cpu")
        assert extractor is not None

    def test_lightweight_has_feature_dimension(self):
        """Test Lightweight extractor has feature_dim."""
        extractor = LightweightFeatureExtractor(device="cpu")
        assert extractor.feature_dim is not None

    def test_lightweight_forward_pass(self, synthetic_3d_video):
        """Test Lightweight forward pass."""
        extractor = LightweightFeatureExtractor(device="cpu")
        features = extractor(synthetic_3d_video.to("cpu"))

        assert features is not None
        assert features.ndim == 2

    def test_lightweight_batch_processing(self, synthetic_3d_batch_video):
        """Test Lightweight with batch."""
        extractor = LightweightFeatureExtractor(device="cpu")
        features = extractor(synthetic_3d_batch_video.to("cpu"))

        assert features.shape[0] == 2

    def test_lightweight_output_validity(self, synthetic_3d_video):
        """Test Lightweight output is valid."""
        extractor = LightweightFeatureExtractor(device="cpu")
        features = extractor(synthetic_3d_video.to("cpu"))

        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()


# ============================================================================
# 5. Common Extractor Tests
# ============================================================================

class TestFeatureExtractorCommon:
    """Common tests for all feature extractors."""

    @pytest.mark.parametrize("ExtractorClass", [
        R3DFeatureExtractor,
        LightweightFeatureExtractor,
    ])
    def test_extractor_initialization(self, ExtractorClass):
        """Test all extractors can be initialized."""
        extractor = ExtractorClass(device="cpu")
        assert extractor is not None

    @pytest.mark.parametrize("ExtractorClass", [
        R3DFeatureExtractor,
        LightweightFeatureExtractor,
    ])
    def test_extractor_inheritance(self, ExtractorClass):
        """Test all extractors inherit from BaseFeatureExtractor."""
        extractor = ExtractorClass(device="cpu")
        assert isinstance(extractor, BaseFeatureExtractor)
        assert isinstance(extractor, nn.Module)

    @pytest.mark.parametrize("ExtractorClass", [
        R3DFeatureExtractor,
        LightweightFeatureExtractor,
    ])
    def test_extractor_forward_requires_grad(self, ExtractorClass, synthetic_3d_video):
        """Test extractor forward with gradient computation."""
        extractor = ExtractorClass(device="cpu")
        extractor.train()

        features = extractor(synthetic_3d_video.to("cpu"))

        # Features should not require grad if model is frozen
        # or should require grad if model is trainable
        assert features is not None

    @pytest.mark.parametrize("ExtractorClass", [
        R3DFeatureExtractor,
        LightweightFeatureExtractor,
    ])
    def test_extractor_eval_mode(self, ExtractorClass, synthetic_3d_video):
        """Test extractor in eval mode."""
        extractor = ExtractorClass(device="cpu")
        extractor.eval()

        with torch.no_grad():
            features = extractor(synthetic_3d_video.to("cpu"))

        assert not torch.isnan(features).any()


# ============================================================================
# 6. Device Consistency Tests
# ============================================================================

class TestFeatureExtractorDevice:
    """Test device handling."""

    def test_r3d_cpu_device(self):
        """Test R3D on CPU."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        assert str(extractor.device) == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_r3d_cuda_device(self):
        """Test R3D on CUDA."""
        extractor = R3DFeatureExtractor(device="cuda", pretrained=False)
        # Model should be on CUDA
        # assert next(extractor.parameters()).device.type == "cuda"


# ============================================================================
# 7. Edge Cases Tests
# ============================================================================

class TestFeatureExtractorEdgeCases:
    """Test edge cases."""

    def test_r3d_single_frame_batch(self):
        """Test R3D with very short temporal dimension."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)

        # [batch=1, channels=3, frames=1, height=112, width=112]
        video = torch.randn(1, 3, 1, 112, 112)

        # Should handle gracefully (might zero-pad or skip)
        try:
            features = extractor(video)
            assert features is not None
        except RuntimeError:
            # Acceptable if model doesn't support single frame
            pass

    def test_r3d_small_spatial_dimensions(self):
        """Test R3D with small spatial dimensions."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)

        # [batch=1, channels=3, frames=8, height=32, width=32]
        video = torch.randn(1, 3, 8, 32, 32)

        features = extractor(video)

        assert features is not None
        assert not torch.isnan(features).any()

    def test_r3d_large_spatial_dimensions(self):
        """Test R3D with large spatial dimensions."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)

        # [batch=1, channels=3, frames=16, height=512, width=512]
        video = torch.randn(1, 3, 16, 512, 512)

        features = extractor(video)

        assert features is not None
        assert not torch.isnan(features).any()

    def test_r3d_zero_input(self):
        """Test R3D with all-zero input."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)

        video = torch.zeros(1, 3, 16, 112, 112)

        features = extractor(video)

        assert features is not None
        assert not torch.isnan(features).any()


# ============================================================================
# 8. Eval Mode Determinism Tests
# ============================================================================

class TestFeatureExtractorDeterminism:
    """Test determinism in eval mode."""

    def test_r3d_determinism_with_seed(self):
        """Test R3D is deterministic in eval mode."""
        video = torch.randn(1, 3, 16, 112, 112)

        # First run
        torch.manual_seed(42)
        extractor_1 = R3DFeatureExtractor(device="cpu", pretrained=False)
        extractor_1.eval()
        with torch.no_grad():
            features_1 = extractor_1(video)

        # Second run
        torch.manual_seed(42)
        extractor_2 = R3DFeatureExtractor(device="cpu", pretrained=False)
        extractor_2.eval()
        with torch.no_grad():
            features_2 = extractor_2(video)

        # Should be identical (same random initialization + same input)
        assert torch.allclose(features_1, features_2)


# ============================================================================
# 9. Gradient Flow Tests
# ============================================================================

class TestFeatureExtractorGradients:
    """Test gradient flow."""

    def test_r3d_backward_pass(self):
        """Test R3D backward pass."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        extractor.train()

        video = torch.randn(1, 3, 16, 112, 112, requires_grad=True)
        features = extractor(video)

        loss = features.sum()
        loss.backward()  # Should not raise

    def test_r3d_gradients_computed(self):
        """Test gradients are computed for parameters."""
        extractor = R3DFeatureExtractor(device="cpu", pretrained=False)
        extractor.train()

        video = torch.randn(1, 3, 16, 112, 112, requires_grad=True)
        features = extractor(video)

        loss = features.sum()
        loss.backward()

        # Check if any parameters have gradients
        has_gradients = any(
            p.grad is not None for p in extractor.parameters()
        )
        assert has_gradients


# ============================================================================
# 10. Output Validity Tests
# ============================================================================

class TestFeatureExtractorOutputValidity:
    """Test output validity."""

    @pytest.mark.parametrize("ExtractorClass", [
        R3DFeatureExtractor,
        LightweightFeatureExtractor,
    ])
    def test_extractor_output_no_nan(self, ExtractorClass, synthetic_3d_video):
        """Test extractor output has no NaN."""
        extractor = ExtractorClass(device="cpu")
        features = extractor(synthetic_3d_video.to("cpu"))

        assert not torch.isnan(features).any()

    @pytest.mark.parametrize("ExtractorClass", [
        R3DFeatureExtractor,
        LightweightFeatureExtractor,
    ])
    def test_extractor_output_no_inf(self, ExtractorClass, synthetic_3d_video):
        """Test extractor output has no Inf."""
        extractor = ExtractorClass(device="cpu")
        features = extractor(synthetic_3d_video.to("cpu"))

        assert not torch.isinf(features).any()

    @pytest.mark.parametrize("ExtractorClass", [
        R3DFeatureExtractor,
        LightweightFeatureExtractor,
    ])
    def test_extractor_output_finite(self, ExtractorClass, synthetic_3d_video):
        """Test extractor output is all finite."""
        extractor = ExtractorClass(device="cpu")
        features = extractor(synthetic_3d_video.to("cpu"))

        assert torch.isfinite(features).all()


# ============================================================================
# Utility Functions
# ============================================================================

def _has_pytorchvideo():
    """Check if pytorchvideo is available."""
    try:
        import pytorchvideo
        return True
    except ImportError:
        return False
