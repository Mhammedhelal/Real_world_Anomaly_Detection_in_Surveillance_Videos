"""
tests/test_loss_comprehensive.py
================================

Comprehensive tests for MIL (Multiple Instance Learning) Ranking Loss.

Coverage:
  - Loss initialization
  - Forward pass computation
  - Loss value validity
  - Gradient flow
  - Edge cases (empty batches, all-normal, all-anomaly)
  - Backward pass stability
  - Numerical stability
  - Different batch configurations
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.losses import MILRankingLoss


# ============================================================================
# 1. Loss Initialization Tests
# ============================================================================

class TestMILRankingLossInitialization:
    """Test loss function initialization."""

    def test_loss_initialization_default(self):
        """Test loss can be initialized with default parameters."""
        loss_fn = MILRankingLoss()
        assert loss_fn is not None
        assert isinstance(loss_fn, nn.Module)

    def test_loss_initialization_custom(self):
        """Test loss initialization with custom lambda values."""
        loss_fn = MILRankingLoss(lambda1=1e-4, lambda2=1e-4)
        assert loss_fn.lambda1 == 1e-4
        assert loss_fn.lambda2 == 1e-4

    def test_loss_default_lambda_values(self):
        """Verify default lambda values."""
        loss_fn = MILRankingLoss()
        assert loss_fn.lambda1 == 8e-5
        assert loss_fn.lambda2 == 8e-5

    def test_loss_parameters_are_registered(self):
        """Verify lambda parameters are properly set."""
        loss_fn = MILRankingLoss(lambda1=1e-4, lambda2=2e-4)
        assert loss_fn.lambda1 == 1e-4
        assert loss_fn.lambda2 == 2e-4


# ============================================================================
# 2. Forward Pass Tests
# ============================================================================

class TestMILRankingLossForward:
    """Test loss forward pass."""

    def test_loss_forward_basic(self):
        """Test basic loss computation."""
        loss_fn = MILRankingLoss()

        # Anomaly scores: [batch_size=4, num_segments=10, 1]
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        # Labels: [batch_size=4] with mixed normal (0) and anomalous (>0)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_loss_returns_scalar(self):
        """Verify loss returns a scalar value."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert loss.shape == torch.Size([])
        assert loss.ndim == 0

    def test_loss_dtype_float(self):
        """Verify loss returns float dtype."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert loss.dtype == torch.float32 or loss.dtype == torch.float64


# ============================================================================
# 3. Loss Value Validity Tests
# ============================================================================

class TestMILRankingLossValidity:
    """Test validity of loss values."""

    def test_loss_is_non_negative(self):
        """Loss should be non-negative (sum of relu, smoothness, sparsity)."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert loss.item() >= -1e-5  # Allow small numerical errors

    def test_loss_no_nan(self):
        """Loss should not contain NaN."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)

    def test_loss_no_inf(self):
        """Loss should not contain Inf."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isinf(loss)


# ============================================================================
# 4. Gradient Flow Tests
# ============================================================================

class TestMILRankingLossGradients:
    """Test gradient computation and flow."""

    def test_backward_pass_runs(self):
        """Verify backward pass completes without error."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)
        loss.backward()  # Should not raise

    def test_gradients_computed(self):
        """Verify gradients are computed for input."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)
        loss.backward()

        assert anomaly_scores.grad is not None

    def test_gradients_are_finite(self):
        """Verify gradients are finite (no NaN or Inf)."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)
        loss.backward()

        assert not torch.isnan(anomaly_scores.grad).any()
        assert not torch.isinf(anomaly_scores.grad).any()

    def test_gradients_non_zero(self):
        """Verify gradients are non-zero (loss depends on input)."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)
        loss.backward()

        assert not (anomaly_scores.grad == 0).all()


# ============================================================================
# 5. Component Breakdown Tests
# ============================================================================

class TestMILRankingLossComponents:
    """Test individual loss components."""

    def test_ranking_loss_component(self):
        """Verify ranking loss component (hinge loss) works."""
        loss_fn = MILRankingLoss(lambda1=0, lambda2=0)  # Only ranking loss
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        # Loss should be non-negative (hinge loss)
        assert loss.item() >= -1e-5

    def test_smoothness_component(self):
        """Verify smoothness regularization component."""
        loss_fn = MILRankingLoss(lambda1=1e-4, lambda2=0)

        # Create scores with large jumps
        anomaly_scores = torch.tensor([
            [[0.0], [1.0], [0.0], [1.0], [0.0]],
            [[0.5], [0.5], [0.5], [0.5], [0.5]]
        ], requires_grad=True)
        labels = torch.LongTensor([1, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert loss.item() > 0  # Smoothness penalty should be positive

    def test_sparsity_component(self):
        """Verify sparsity regularization component."""
        loss_fn = MILRankingLoss(lambda1=0, lambda2=1e-4)

        # Create high anomaly scores (violates sparsity)
        anomaly_scores = torch.ones(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([1, 1, 0, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert loss.item() > 0  # Sparsity penalty should be positive


# ============================================================================
# 6. Edge Cases Tests
# ============================================================================

class TestMILRankingLossEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_loss_with_single_batch(self):
        """Test loss with batch size of 1."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(1, 10, 1, requires_grad=True)
        labels = torch.LongTensor([1])

        loss = loss_fn(anomaly_scores, labels)

        # Should return 0 (no pos/neg mix for ranking)
        assert loss.item() >= 0

    def test_loss_with_all_normal_batch(self):
        """Test loss when all samples are normal (edge case)."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 0, 0, 0])  # All normal

        loss = loss_fn(anomaly_scores, labels)

        # Should return 0 (no anomalous samples for MIL)
        assert loss.item() == 0.0

    def test_loss_with_all_anomalous_batch(self):
        """Test loss when all samples are anomalous (edge case)."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([1, 2, 3, 4])  # All anomalous

        loss = loss_fn(anomaly_scores, labels)

        # Should return 0 (no normal samples for MIL)
        assert loss.item() == 0.0

    def test_loss_with_zero_scores(self):
        """Test loss with all-zero anomaly scores."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.zeros(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        # Loss should be valid
        assert not torch.isnan(loss)

    def test_loss_with_one_scores(self):
        """Test loss with all-one anomaly scores."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.ones(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        # Loss should be valid
        assert not torch.isnan(loss)

    def test_loss_with_single_segment(self):
        """Test loss with single segment per sample."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 1, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        # Should still compute correctly
        assert not torch.isnan(loss)

    def test_loss_with_many_segments(self):
        """Test loss with many segments per sample."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 1000, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        # Should handle large sequences
        assert not torch.isnan(loss)


# ============================================================================
# 7. Different Label Configurations Tests
# ============================================================================

class TestMILRankingLossLabelConfigurations:
    """Test with different label configurations."""

    def test_loss_binary_labels(self):
        """Test with binary labels (0=normal, 1=anomalous)."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 0, 1, 1])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)

    def test_loss_multiclass_labels(self):
        """Test with multiclass anomaly labels (0=normal, 1-13=crime types)."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(8, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 3, 4, 0, 0, 13])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)

    def test_loss_unbalanced_labels(self):
        """Test with unbalanced label distribution."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(10, 10, 1, requires_grad=True)
        # Many normal, few anomalous
        labels = torch.LongTensor([0, 0, 0, 0, 0, 0, 0, 1, 2, 3])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)


# ============================================================================
# 8. Determinism Tests
# ============================================================================

class TestMILRankingLossDeterminism:
    """Test determinism of loss computation."""

    def test_loss_determinism_with_seed(self):
        """Same input → same loss output with seed."""
        loss_fn = MILRankingLoss()

        # First run
        torch.manual_seed(42)
        scores_1 = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])
        loss_1 = loss_fn(scores_1, labels)

        # Second run with same seed
        torch.manual_seed(42)
        scores_2 = torch.randn(4, 10, 1, requires_grad=True)
        loss_2 = loss_fn(scores_2, labels)

        assert torch.allclose(loss_1, loss_2)

    def test_loss_determinism_multiple_runs(self):
        """Multiple forward passes with same input → same loss."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss_1 = loss_fn(anomaly_scores.detach().requires_grad_(True), labels)
        loss_2 = loss_fn(anomaly_scores.detach().requires_grad_(True), labels)

        assert torch.allclose(loss_1, loss_2)


# ============================================================================
# 9. Numerical Stability Tests
# ============================================================================

class TestMILRankingLossNumericalStability:
    """Test numerical stability."""

    def test_loss_with_very_large_values(self):
        """Test loss stability with very large input values."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True) * 1e6
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_loss_with_very_small_values(self):
        """Test loss stability with very small input values."""
        loss_fn = MILRankingLoss()
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True) * 1e-6
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ============================================================================
# 10. Lambda Parameter Sensitivity Tests
# ============================================================================

class TestMILRankingLossLambdaSensitivity:
    """Test sensitivity to lambda parameters."""

    def test_lambda1_zero(self):
        """Test with zero smoothness lambda."""
        loss_fn = MILRankingLoss(lambda1=0, lambda2=8e-5)
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)

    def test_lambda2_zero(self):
        """Test with zero sparsity lambda."""
        loss_fn = MILRankingLoss(lambda1=8e-5, lambda2=0)
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)

    def test_both_lambdas_zero(self):
        """Test with both lambdas zero (only ranking loss)."""
        loss_fn = MILRankingLoss(lambda1=0, lambda2=0)
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)

    def test_large_lambda_values(self):
        """Test with large lambda values."""
        loss_fn = MILRankingLoss(lambda1=1e-2, lambda2=1e-2)
        anomaly_scores = torch.randn(4, 10, 1, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        loss = loss_fn(anomaly_scores, labels)

        assert not torch.isnan(loss)
