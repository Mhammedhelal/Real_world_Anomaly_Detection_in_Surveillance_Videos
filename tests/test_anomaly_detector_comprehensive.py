"""
tests/test_anomaly_detector_comprehensive.py
==============================

Comprehensive tests for the AnomalyDetector model.

Coverage:
  - Model initialization and architecture
  - Forward pass with various input shapes
  - Eval mode determinism
  - Train vs eval behavior
  - Device consistency (CPU/CUDA)
  - Gradient flow and backward pass
  - Edge cases (zero inputs, single segment, etc.)
  - Output validity (no NaN/Inf)
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.anomaly_detector import AnomalyDetector


# ============================================================================
# 1. Model Initialization Tests
# ============================================================================

class TestAnomalyDetectorInitialization:
    """Test model initialization and architecture."""

    def test_model_initialization_default(self):
        """Test model can be initialized with default parameters."""
        model = AnomalyDetector()
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_initialization_custom(self):
        """Test model initialization with custom parameters."""
        model = AnomalyDetector(input_size=2048, hidden_size=512, num_classes=10)
        assert model is not None

    def test_model_has_required_layers(self):
        """Verify all required layers exist in the model."""
        model = AnomalyDetector()
        assert hasattr(model, 'bigru')
        assert hasattr(model, 'anomaly_head')
        assert hasattr(model, 'class_head')

    def test_bigru_configuration(self):
        """Test GRU configuration."""
        input_size = 2131
        hidden_size = 256
        model = AnomalyDetector(input_size=input_size, hidden_size=hidden_size)

        assert model.bigru.input_size == input_size
        assert model.bigru.hidden_size == hidden_size
        assert model.bigru.bidirectional is True

    def test_model_is_trainable(self):
        """Verify model parameters have requires_grad=True."""
        model = AnomalyDetector()
        for param in model.parameters():
            assert param.requires_grad is True


# ============================================================================
# 2. Forward Pass Tests
# ============================================================================

class TestAnomalyDetectorForward:
    """Test forward pass behavior."""

    def test_forward_pass_basic(self, synthetic_video_features):
        """Test basic forward pass execution."""
        model = AnomalyDetector()
        anomaly_scores, class_probs = model(synthetic_video_features)

        assert anomaly_scores is not None
        assert class_probs is not None

    def test_forward_output_shapes(self, synthetic_video_features):
        """Verify output tensor shapes are correct."""
        batch_size = 2
        num_segments = 10
        num_classes = 14

        model = AnomalyDetector(num_classes=num_classes)
        anomaly_scores, class_probs = model(synthetic_video_features)

        # anomaly_scores: [batch_size, num_segments, 1]
        assert anomaly_scores.shape == (batch_size, num_segments, 1)

        # class_probs: [batch_size, num_segments, num_classes]
        assert class_probs.shape == (batch_size, num_segments, num_classes)

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = AnomalyDetector()

        for batch_size in [1, 2, 4, 8]:
            features = torch.randn(batch_size, 10, 2131)
            anomaly_scores, class_probs = model(features)

            assert anomaly_scores.shape[0] == batch_size
            assert class_probs.shape[0] == batch_size

    def test_forward_different_sequence_lengths(self):
        """Test forward pass with different sequence (segment) lengths."""
        model = AnomalyDetector()

        for seq_len in [1, 5, 10, 20, 100]:
            features = torch.randn(2, seq_len, 2131)
            anomaly_scores, class_probs = model(features)

            assert anomaly_scores.shape == (2, seq_len, 1)
            assert class_probs.shape == (2, seq_len, 14)

    def test_forward_single_segment(self):
        """Test forward pass with single segment (edge case)."""
        model = AnomalyDetector()
        features = torch.randn(1, 1, 2131)

        anomaly_scores, class_probs = model(features)

        assert anomaly_scores.shape == (1, 1, 1)
        assert class_probs.shape == (1, 1, 14)


# ============================================================================
# 3. Output Validity Tests
# ============================================================================

class TestAnomalyDetectorOutputValidity:
    """Test output value validity."""

    def test_anomaly_scores_range(self, synthetic_video_features):
        """Verify anomaly scores are in [0, 1] (sigmoid output)."""
        model = AnomalyDetector()
        anomaly_scores, _ = model(synthetic_video_features)

        assert anomaly_scores.min() >= 0.0
        assert anomaly_scores.max() <= 1.0

    def test_class_probs_sum_to_one(self, synthetic_video_features):
        """Verify class probabilities sum to 1 (softmax output)."""
        model = AnomalyDetector()
        _, class_probs = model(synthetic_video_features)

        # Sum over class dimension should be 1
        prob_sums = class_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_no_nan_in_output(self, synthetic_video_features):
        """Verify no NaN values in output."""
        model = AnomalyDetector()
        anomaly_scores, class_probs = model(synthetic_video_features)

        assert not torch.isnan(anomaly_scores).any()
        assert not torch.isnan(class_probs).any()

    def test_no_inf_in_output(self, synthetic_video_features):
        """Verify no Inf values in output."""
        model = AnomalyDetector()
        anomaly_scores, class_probs = model(synthetic_video_features)

        assert not torch.isinf(anomaly_scores).any()
        assert not torch.isinf(class_probs).any()


# ============================================================================
# 4. Eval Mode Determinism Tests
# ============================================================================

class TestAnomalyDetectorDeterminism:
    """Test determinism in eval mode."""

    def test_eval_mode_determinism(self, synthetic_video_features):
        """Same input → identical output in eval mode."""
        model = AnomalyDetector()
        model.eval()

        # Run multiple times
        outputs_1 = model(synthetic_video_features)
        outputs_2 = model(synthetic_video_features)
        outputs_3 = model(synthetic_video_features)

        # All outputs should be identical
        assert torch.allclose(outputs_1[0], outputs_2[0])
        assert torch.allclose(outputs_2[0], outputs_3[0])
        assert torch.allclose(outputs_1[1], outputs_2[1])
        assert torch.allclose(outputs_2[1], outputs_3[1])

    def test_different_inputs_different_outputs(self):
        """Different inputs should produce different outputs."""
        model = AnomalyDetector()
        model.eval()

        features_1 = torch.randn(2, 10, 2131)
        features_2 = torch.randn(2, 10, 2131)

        scores_1, probs_1 = model(features_1)
        scores_2, probs_2 = model(features_2)

        # Outputs should be different
        assert not torch.allclose(scores_1, scores_2)
        assert not torch.allclose(probs_1, probs_2)


# ============================================================================
# 5. Train vs Eval Mode Tests
# ============================================================================

class TestAnomalyDetectorTrainEvalBehavior:
    """Test differences between train and eval modes."""

    def test_train_eval_mode_switching(self, synthetic_video_features):
        """Test that train/eval mode switches work correctly."""
        model = AnomalyDetector()

        # Switch to eval
        model.eval()
        assert not model.training

        # Switch to train
        model.train()
        assert model.training

        # Switch back to eval
        model.eval()
        assert not model.training

    def test_eval_mode_no_grad(self, synthetic_video_features):
        """Verify gradients not computed in eval mode with no_grad."""
        model = AnomalyDetector()
        model.eval()

        with torch.no_grad():
            scores, probs = model(synthetic_video_features)

        assert scores.requires_grad is False
        assert probs.requires_grad is False

    def test_train_mode_gradient_computation(self, synthetic_video_features):
        """Verify gradients are computed in train mode."""
        model = AnomalyDetector()
        model.train()

        scores, probs = model(synthetic_video_features)

        assert scores.requires_grad is True
        assert probs.requires_grad is True


# ============================================================================
# 6. Device Consistency Tests
# ============================================================================

class TestAnomalyDetectorDeviceConsistency:
    """Test device handling and consistency."""

    def test_model_to_cpu(self, synthetic_video_features):
        """Test model can be moved to CPU."""
        model = AnomalyDetector()
        model = model.to("cpu")

        features = synthetic_video_features.to("cpu")
        anomaly_scores, class_probs = model(features)

        assert anomaly_scores.device.type == "cpu"
        assert class_probs.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_cuda(self, synthetic_video_features):
        """Test model can be moved to CUDA."""
        model = AnomalyDetector()
        model = model.to("cuda")

        features = synthetic_video_features.to("cuda")
        anomaly_scores, class_probs = model(features)

        assert anomaly_scores.device.type == "cuda"
        assert class_probs.device.type == "cuda"

    def test_device_mismatch_detection(self):
        """Verify error when model and input are on different devices."""
        model = AnomalyDetector()
        model = model.to("cpu")

        # Try forward pass with mismatched device (if CUDA available)
        if torch.cuda.is_available():
            features = torch.randn(2, 10, 2131).to("cuda")

            with pytest.raises(RuntimeError):
                model(features)


# ============================================================================
# 7. Gradient Flow Tests
# ============================================================================

class TestAnomalyDetectorGradientFlow:
    """Test backward pass and gradient flow."""

    def test_backward_pass_runs(self, synthetic_video_features):
        """Verify backward pass completes without error."""
        model = AnomalyDetector()
        model.train()

        anomaly_scores, class_probs = model(synthetic_video_features)
        loss = anomaly_scores.sum() + class_probs.sum()

        # Should not raise
        loss.backward()

    def test_gradients_computed_for_all_params(self, synthetic_video_features):
        """Verify gradients are computed for all trainable parameters."""
        model = AnomalyDetector()
        model.train()

        anomaly_scores, class_probs = model(synthetic_video_features)
        loss = anomaly_scores.sum() + class_probs.sum()
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_gradients_finite(self, synthetic_video_features):
        """Verify gradients are finite (no NaN or Inf)."""
        model = AnomalyDetector()
        model.train()

        anomaly_scores, class_probs = model(synthetic_video_features)
        loss = anomaly_scores.sum() + class_probs.sum()
        loss.backward()

        for param in model.parameters():
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_gradient_values_bounded(self, synthetic_video_features):
        """Verify gradients are reasonably bounded (not exploding)."""
        model = AnomalyDetector()
        model.train()

        anomaly_scores, class_probs = model(synthetic_video_features)
        loss = anomaly_scores.sum() + class_probs.sum()
        loss.backward()

        for param in model.parameters():
            # Gradients should be reasonably small (not explosive)
            assert param.grad.abs().max() < 100.0

    def test_zero_grad_clears_gradients(self, synthetic_video_features):
        """Verify zero_grad() clears gradients."""
        model = AnomalyDetector()
        model.train()

        # First backward pass
        scores_1, probs_1 = model(synthetic_video_features)
        loss_1 = scores_1.sum() + probs_1.sum()
        loss_1.backward()

        # Verify gradients exist
        for param in model.parameters():
            assert param.grad is not None

        # Zero gradients
        model.zero_grad()

        # Verify gradients are cleared
        for param in model.parameters():
            if param.grad is not None:
                assert (param.grad == 0).all() if param.grad.dim() > 0 else (param.grad == 0)


# ============================================================================
# 8. Edge Cases Tests
# ============================================================================

class TestAnomalyDetectorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_input_features(self):
        """Test with all-zero input features."""
        model = AnomalyDetector()
        features = torch.zeros(2, 10, 2131)

        anomaly_scores, class_probs = model(features)

        # Should still produce valid outputs
        assert not torch.isnan(anomaly_scores).any()
        assert not torch.isnan(class_probs).any()

    def test_very_large_input_values(self):
        """Test with very large input values."""
        model = AnomalyDetector()
        features = torch.randn(2, 10, 2131) * 1e6

        anomaly_scores, class_probs = model(features)

        # Sigmoid output should still be in [0, 1]
        assert anomaly_scores.min() >= 0.0
        assert anomaly_scores.max() <= 1.0

    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        model = AnomalyDetector()
        features = torch.randn(1, 10, 2131)

        anomaly_scores, class_probs = model(features)

        assert anomaly_scores.shape[0] == 1
        assert class_probs.shape[0] == 1

    def test_many_samples_batch(self):
        """Test with large batch size."""
        model = AnomalyDetector()
        features = torch.randn(64, 10, 2131)

        anomaly_scores, class_probs = model(features)

        assert anomaly_scores.shape[0] == 64
        assert class_probs.shape[0] == 64

    def test_identical_frames_across_time(self):
        """Test with identical feature vectors across time (edge case)."""
        model = AnomalyDetector()
        features = torch.randn(2, 1, 2131)
        features = features.expand(2, 10, 2131)  # Replicate across time

        anomaly_scores, class_probs = model(features)

        # Should still be valid
        assert not torch.isnan(anomaly_scores).any()
        assert not torch.isnan(class_probs).any()


# ============================================================================
# 9. Component Consistency Tests
# ============================================================================

class TestAnomalyDetectorComponentConsistency:
    """Test component-wise consistency."""

    def test_output_shapes_consistent_with_input(self):
        """Verify output shapes are consistent with input shapes."""
        model = AnomalyDetector(num_classes=14)

        for batch in [1, 4, 8]:
            for seq_len in [5, 10, 20]:
                features = torch.randn(batch, seq_len, 2131)
                anomaly_scores, class_probs = model(features)

                assert anomaly_scores.shape == (batch, seq_len, 1)
                assert class_probs.shape == (batch, seq_len, 14)

    def test_different_classes_produce_different_outputs(self):
        """Verify changing num_classes affects output shape."""
        features = torch.randn(2, 10, 2131)

        for num_classes in [10, 14, 20]:
            model = AnomalyDetector(num_classes=num_classes)
            _, class_probs = model(features)

            assert class_probs.shape[-1] == num_classes


# ============================================================================
# 10. Reproducibility Tests
# ============================================================================

class TestAnomalyDetectorReproducibility:
    """Test reproducibility with seed setting."""

    def test_deterministic_with_seed(self):
        """Verify outputs are deterministic with seed in eval mode."""
        features = torch.randn(2, 10, 2131)

        # First run
        torch.manual_seed(42)
        model_1 = AnomalyDetector()
        model_1.eval()
        scores_1, probs_1 = model_1(features)

        # Second run
        torch.manual_seed(42)
        model_2 = AnomalyDetector()
        model_2.eval()
        scores_2, probs_2 = model_2(features)

        assert torch.allclose(scores_1, scores_2)
        assert torch.allclose(probs_1, probs_2)
