"""
tests/test_anomaly_detector_comprehensive.py
============================================

Comprehensive tests for the AnomalyDetector model.

AnomalyDetector.forward now accepts an optional ``lengths`` argument for
GRU packing.  All tests pass synthetic lengths so the packed path is
exercised; the None path is tested separately.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.anomaly_detector import AnomalyDetector


# ============================================================================
# Helpers
# ============================================================================

def _lengths_for(features: torch.Tensor) -> torch.Tensor:
    """Return a LongTensor of real lengths (all equal to seq_len)."""
    return torch.LongTensor([features.size(1)] * features.size(0))


# ============================================================================
# 1. Model Initialization Tests
# ============================================================================

class TestAnomalyDetectorInitialization:

    def test_model_initialization_default(self):
        model = AnomalyDetector()
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_model_initialization_custom(self):
        model = AnomalyDetector(input_size=2048, hidden_size=512, num_classes=10)
        assert model is not None

    def test_model_has_required_layers(self):
        model = AnomalyDetector()
        assert hasattr(model, 'bigru')
        assert hasattr(model, 'anomaly_head')
        assert hasattr(model, 'class_head')

    def test_bigru_configuration(self):
        model = AnomalyDetector(input_size=2131, hidden_size=256)
        assert model.bigru.input_size == 2131
        assert model.bigru.hidden_size == 256
        assert model.bigru.bidirectional is True

    def test_model_is_trainable(self):
        model = AnomalyDetector()
        for param in model.parameters():
            assert param.requires_grad is True


# ============================================================================
# 2. Forward Pass Tests
# ============================================================================

class TestAnomalyDetectorForward:

    def test_forward_pass_basic(self, synthetic_video_features):
        model = AnomalyDetector()
        lengths = _lengths_for(synthetic_video_features)
        anomaly_scores, class_probs = model(synthetic_video_features, lengths=lengths)
        assert anomaly_scores is not None
        assert class_probs is not None

    def test_forward_output_shapes(self, synthetic_video_features):
        batch_size, num_segments, num_classes = 2, 10, 14
        model = AnomalyDetector(num_classes=num_classes)
        lengths = _lengths_for(synthetic_video_features)
        anomaly_scores, class_probs = model(synthetic_video_features, lengths=lengths)
        assert anomaly_scores.shape == (batch_size, num_segments, 1)
        assert class_probs.shape    == (batch_size, num_segments, num_classes)

    def test_forward_different_batch_sizes(self):
        model = AnomalyDetector()
        for batch_size in [1, 2, 4, 8]:
            features = torch.randn(batch_size, 10, 2131)
            lengths  = _lengths_for(features)
            anomaly_scores, class_probs = model(features, lengths=lengths)
            assert anomaly_scores.shape[0] == batch_size
            assert class_probs.shape[0]    == batch_size

    def test_forward_different_sequence_lengths(self):
        model = AnomalyDetector()
        for seq_len in [1, 5, 10, 20, 100]:
            features = torch.randn(2, seq_len, 2131)
            lengths  = _lengths_for(features)
            anomaly_scores, class_probs = model(features, lengths=lengths)
            assert anomaly_scores.shape == (2, seq_len, 1)
            assert class_probs.shape    == (2, seq_len, 14)

    def test_forward_single_segment(self):
        model    = AnomalyDetector()
        features = torch.randn(1, 1, 2131)
        lengths  = _lengths_for(features)
        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert anomaly_scores.shape == (1, 1, 1)
        assert class_probs.shape    == (1, 1, 14)

    def test_forward_without_lengths(self):
        """lengths=None must still work (single-video / no-padding path)."""
        model    = AnomalyDetector()
        features = torch.randn(2, 10, 2131)
        anomaly_scores, class_probs = model(features)   # no lengths
        assert anomaly_scores.shape == (2, 10, 1)
        assert class_probs.shape    == (2, 10, 14)

    def test_forward_variable_lengths(self):
        """Pass genuinely different lengths — exercises the packing logic."""
        from src.data.dataset import collate_fn
        samples  = [(torch.randn(8, 2131), 0), (torch.randn(12, 2131), 1)]
        features, labels, lengths = collate_fn(samples)

        model = AnomalyDetector()
        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert anomaly_scores.shape == (2, 12, 1)   # padded to max=12
        assert class_probs.shape    == (2, 12, 14)


# ============================================================================
# 3. Output Validity Tests
# ============================================================================

class TestAnomalyDetectorOutputValidity:

    def test_anomaly_scores_range(self, synthetic_video_features):
        model   = AnomalyDetector()
        lengths = _lengths_for(synthetic_video_features)
        anomaly_scores, _ = model(synthetic_video_features, lengths=lengths)
        assert anomaly_scores.min() >= 0.0
        assert anomaly_scores.max() <= 1.0

    def test_class_probs_sum_to_one(self, synthetic_video_features):
        model   = AnomalyDetector()
        lengths = _lengths_for(synthetic_video_features)
        _, class_probs = model(synthetic_video_features, lengths=lengths)
        prob_sums = class_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_no_nan_in_output(self, synthetic_video_features):
        model   = AnomalyDetector()
        lengths = _lengths_for(synthetic_video_features)
        anomaly_scores, class_probs = model(synthetic_video_features, lengths=lengths)
        assert not torch.isnan(anomaly_scores).any()
        assert not torch.isnan(class_probs).any()

    def test_no_inf_in_output(self, synthetic_video_features):
        model   = AnomalyDetector()
        lengths = _lengths_for(synthetic_video_features)
        anomaly_scores, class_probs = model(synthetic_video_features, lengths=lengths)
        assert not torch.isinf(anomaly_scores).any()
        assert not torch.isinf(class_probs).any()


# ============================================================================
# 4. Eval Mode Determinism Tests
# ============================================================================

class TestAnomalyDetectorDeterminism:

    def test_eval_mode_determinism(self, synthetic_video_features):
        model   = AnomalyDetector()
        model.eval()
        lengths = _lengths_for(synthetic_video_features)

        out1 = model(synthetic_video_features, lengths=lengths)
        out2 = model(synthetic_video_features, lengths=lengths)
        out3 = model(synthetic_video_features, lengths=lengths)

        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out2[0], out3[0])
        assert torch.allclose(out1[1], out2[1])
        assert torch.allclose(out2[1], out3[1])

    def test_different_inputs_different_outputs(self):
        model = AnomalyDetector()
        model.eval()

        f1 = torch.randn(2, 10, 2131)
        f2 = torch.randn(2, 10, 2131)
        l1 = _lengths_for(f1)
        l2 = _lengths_for(f2)

        scores_1, probs_1 = model(f1, lengths=l1)
        scores_2, probs_2 = model(f2, lengths=l2)

        assert not torch.allclose(scores_1, scores_2)
        assert not torch.allclose(probs_1,  probs_2)


# ============================================================================
# 5. Train vs Eval Mode Tests
# ============================================================================

class TestAnomalyDetectorTrainEvalBehavior:

    def test_train_eval_mode_switching(self, synthetic_video_features):
        model = AnomalyDetector()
        model.eval();  assert not model.training
        model.train(); assert model.training
        model.eval();  assert not model.training

    def test_eval_mode_no_grad(self, synthetic_video_features):
        model   = AnomalyDetector()
        model.eval()
        lengths = _lengths_for(synthetic_video_features)

        with torch.no_grad():
            scores, probs = model(synthetic_video_features, lengths=lengths)

        assert scores.requires_grad is False
        assert probs.requires_grad  is False

    def test_train_mode_gradient_computation(self, synthetic_video_features):
        model   = AnomalyDetector()
        model.train()
        lengths = _lengths_for(synthetic_video_features)

        scores, probs = model(synthetic_video_features, lengths=lengths)

        assert scores.requires_grad is True
        assert probs.requires_grad  is True


# ============================================================================
# 6. Device Consistency Tests
# ============================================================================

class TestAnomalyDetectorDeviceConsistency:

    def test_model_to_cpu(self, synthetic_video_features):
        model   = AnomalyDetector().to("cpu")
        features = synthetic_video_features.to("cpu")
        lengths  = _lengths_for(features)

        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert anomaly_scores.device.type == "cpu"
        assert class_probs.device.type    == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_to_cuda(self, synthetic_video_features):
        model    = AnomalyDetector().to("cuda")
        features = synthetic_video_features.to("cuda")
        lengths  = _lengths_for(features)  # stays CPU — intentional

        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert anomaly_scores.device.type == "cuda"
        assert class_probs.device.type    == "cuda"

    def test_device_mismatch_detection(self):
        model = AnomalyDetector().to("cpu")
        if torch.cuda.is_available():
            features = torch.randn(2, 10, 2131).to("cuda")
            lengths  = _lengths_for(features)
            with pytest.raises(RuntimeError):
                model(features, lengths=lengths)


# ============================================================================
# 7. Gradient Flow Tests
# ============================================================================

class TestAnomalyDetectorGradientFlow:

    def test_backward_pass_runs(self, synthetic_video_features):
        model   = AnomalyDetector()
        model.train()
        lengths = _lengths_for(synthetic_video_features)

        anomaly_scores, class_probs = model(synthetic_video_features, lengths=lengths)
        loss = anomaly_scores.sum() + class_probs.sum()
        loss.backward()

    def test_gradients_computed_for_all_params(self, synthetic_video_features):
        model   = AnomalyDetector()
        model.train()
        lengths = _lengths_for(synthetic_video_features)

        anomaly_scores, class_probs = model(synthetic_video_features, lengths=lengths)
        (anomaly_scores.sum() + class_probs.sum()).backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_gradients_finite(self, synthetic_video_features):
        model   = AnomalyDetector()
        model.train()
        lengths = _lengths_for(synthetic_video_features)

        anomaly_scores, class_probs = model(synthetic_video_features, lengths=lengths)
        (anomaly_scores.sum() + class_probs.sum()).backward()

        for param in model.parameters():
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

    def test_gradient_values_bounded(self, synthetic_video_features):
        model   = AnomalyDetector()
        model.train()
        lengths = _lengths_for(synthetic_video_features)

        anomaly_scores, class_probs = model(synthetic_video_features, lengths=lengths)
        (anomaly_scores.sum() + class_probs.sum()).backward()

        for param in model.parameters():
            assert param.grad.abs().max() < 100.0

    def test_zero_grad_clears_gradients(self, synthetic_video_features):
        model   = AnomalyDetector()
        model.train()
        lengths = _lengths_for(synthetic_video_features)

        scores, probs = model(synthetic_video_features, lengths=lengths)
        (scores.sum() + probs.sum()).backward()

        for param in model.parameters():
            assert param.grad is not None

        model.zero_grad()

        for param in model.parameters():
            if param.grad is not None:
                assert (param.grad == 0).all()


# ============================================================================
# 8. Edge Cases Tests
# ============================================================================

class TestAnomalyDetectorEdgeCases:

    def test_zero_input_features(self):
        model    = AnomalyDetector()
        features = torch.zeros(2, 10, 2131)
        lengths  = _lengths_for(features)
        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert not torch.isnan(anomaly_scores).any()
        assert not torch.isnan(class_probs).any()

    def test_very_large_input_values(self):
        model    = AnomalyDetector()
        features = torch.randn(2, 10, 2131) * 1e6
        lengths  = _lengths_for(features)
        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert anomaly_scores.min() >= 0.0
        assert anomaly_scores.max() <= 1.0

    def test_single_sample_batch(self):
        model    = AnomalyDetector()
        features = torch.randn(1, 10, 2131)
        lengths  = _lengths_for(features)
        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert anomaly_scores.shape[0] == 1
        assert class_probs.shape[0]    == 1

    def test_many_samples_batch(self):
        model    = AnomalyDetector()
        features = torch.randn(64, 10, 2131)
        lengths  = _lengths_for(features)
        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert anomaly_scores.shape[0] == 64
        assert class_probs.shape[0]    == 64

    def test_identical_frames_across_time(self):
        model    = AnomalyDetector()
        features = torch.randn(2, 1, 2131).expand(2, 10, 2131)
        lengths  = _lengths_for(features)
        anomaly_scores, class_probs = model(features, lengths=lengths)
        assert not torch.isnan(anomaly_scores).any()
        assert not torch.isnan(class_probs).any()

    def test_lengths_none_fallback(self):
        """Explicitly verify that passing lengths=None works identically to
        not passing lengths at all (no-packing code path)."""
        model    = AnomalyDetector()
        features = torch.randn(2, 10, 2131)

        model.eval()
        with torch.no_grad():
            out_none     = model(features, lengths=None)
            out_implicit = model(features)

        assert torch.allclose(out_none[0], out_implicit[0])
        assert torch.allclose(out_none[1], out_implicit[1])


# ============================================================================
# 9. Component Consistency Tests
# ============================================================================

class TestAnomalyDetectorComponentConsistency:

    def test_output_shapes_consistent_with_input(self):
        model = AnomalyDetector(num_classes=14)
        for batch in [1, 4, 8]:
            for seq_len in [5, 10, 20]:
                features = torch.randn(batch, seq_len, 2131)
                lengths  = _lengths_for(features)
                anomaly_scores, class_probs = model(features, lengths=lengths)
                assert anomaly_scores.shape == (batch, seq_len, 1)
                assert class_probs.shape    == (batch, seq_len, 14)

    def test_different_classes_produce_different_outputs(self):
        features = torch.randn(2, 10, 2131)
        lengths  = _lengths_for(features)
        for num_classes in [10, 14, 20]:
            model = AnomalyDetector(num_classes=num_classes)
            _, class_probs = model(features, lengths=lengths)
            assert class_probs.shape[-1] == num_classes


# ============================================================================
# 10. Reproducibility Tests
# ============================================================================

class TestAnomalyDetectorReproducibility:

    def test_deterministic_with_seed(self):
        features = torch.randn(2, 10, 2131)
        lengths  = _lengths_for(features)

        torch.manual_seed(42)
        model_1 = AnomalyDetector()
        model_1.eval()
        scores_1, probs_1 = model_1(features, lengths=lengths)

        torch.manual_seed(42)
        model_2 = AnomalyDetector()
        model_2.eval()
        scores_2, probs_2 = model_2(features, lengths=lengths)

        assert torch.allclose(scores_1, scores_2)
        assert torch.allclose(probs_1,  probs_2)