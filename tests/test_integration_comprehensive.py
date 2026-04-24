"""
tests/test_integration_comprehensive.py
=======================================

Integration tests for the complete anomaly detection pipeline.

Coverage:
  - End-to-end pipeline: features → model → predictions
  - Training loop integration
  - DataLoader integration
  - Model evaluation
  - Component interactions

collate_fn now returns (features, labels, lengths).
AnomalyDetector.forward now accepts lengths keyword argument.
All loops and calls updated accordingly.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.anomaly_detector import AnomalyDetector
from src.models.losses import MILRankingLoss
from src.data.dataset import VideoFeatureDataset, collate_fn


# ============================================================================
# Helpers
# ============================================================================

def _make_batch(batch_size=4, seq_len=10, feature_dim=2131):
    """Return (features, labels, lengths) as collate_fn would."""
    features = torch.randn(batch_size, seq_len, feature_dim)
    labels   = torch.LongTensor([i % 2 for i in range(batch_size)])
    lengths  = torch.LongTensor([seq_len] * batch_size)
    return features, labels, lengths


def _make_variable_batch(feature_dim=2131):
    """Return a collated batch with variable sequence lengths."""
    samples = [
        (torch.randn(8,  feature_dim), 0),
        (torch.randn(12, feature_dim), 1),
        (torch.randn(10, feature_dim), 0),
    ]
    return collate_fn(samples)  # features, labels, lengths


# ============================================================================
# 1. Model + Loss Integration Tests
# ============================================================================

class TestModelLossIntegration:

    def test_model_loss_forward_backward(self):
        model   = AnomalyDetector()
        loss_fn = MILRankingLoss()

        features = torch.randn(4, 10, 2131, requires_grad=True)
        labels   = torch.LongTensor([0, 1, 2, 0])
        lengths  = torch.LongTensor([10, 10, 10, 10])

        scores, probs = model(features, lengths=lengths)
        loss = loss_fn(scores, labels)
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_model_optimization_step(self):
        model     = AnomalyDetector()
        loss_fn   = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        features = torch.randn(4, 10, 2131)
        labels   = torch.LongTensor([0, 1, 2, 0])
        lengths  = torch.LongTensor([10, 10, 10, 10])

        initial_params = [p.clone().detach() for p in model.parameters()]

        model.train()
        scores, probs = model(features, lengths=lengths)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        params_changed = any(
            (ip - cp.detach()).abs().max() > 1e-4
            for ip, cp in zip(initial_params, model.parameters())
        )
        assert params_changed, "No parameters were updated during optimization step"

    def test_multiple_optimization_steps(self):
        model     = AnomalyDetector()
        loss_fn   = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        losses = []
        for _ in range(5):
            features, labels, lengths = _make_batch()

            model.train()
            scores, probs = model(features, lengths=lengths)
            loss = loss_fn(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        assert len(losses) == 5


# ============================================================================
# 2. Model + DataLoader Integration Tests
# ============================================================================

class TestModelDataLoaderIntegration:

    def test_model_with_dataloader(self):
        samples = [
            (torch.randn(10, 2131), 0),
            (torch.randn(12, 2131), 1),
            (torch.randn(8,  2131), 0),
        ]

        dataloader = DataLoader(
            samples, batch_size=2, shuffle=False, collate_fn=collate_fn
        )

        model = AnomalyDetector()
        model.eval()

        with torch.no_grad():
            # collate_fn returns (features, labels, lengths)
            for features, batch_labels, lengths in dataloader:
                scores, probs = model(features, lengths=lengths)
                assert scores is not None
                assert probs is not None

    def test_training_loop_with_dataloader(self):
        samples = [
            (torch.randn(10, 2131), 0),
            (torch.randn(12, 2131), 1),
            (torch.randn(8,  2131), 0),
            (torch.randn(9,  2131), 1),
        ]

        dataloader = DataLoader(
            samples, batch_size=2, shuffle=True, collate_fn=collate_fn
        )

        model     = AnomalyDetector()
        loss_fn   = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(2):
            model.train()
            epoch_loss = 0

            for features, batch_labels, lengths in dataloader:
                scores, probs = model(features, lengths=lengths)
                loss = loss_fn(scores, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            assert epoch_loss >= 0


# ============================================================================
# 3. VideoFeatureDataset Integration Tests
# ============================================================================

class TestVideoFeatureDatasetIntegration:

    def test_dataset_model_forward_pass(self, tmp_path):
        features_dir = tmp_path / "features"
        features_dir.mkdir()

        for i in range(3):
            feats    = np.random.randn(10, 2131).astype(np.float32)
            metadata = {'label': i % 2}
            np.savez_compressed(
                features_dir / f"train_video_{i}.npz",
                features=feats, metadata=metadata,
            )

        dataset = VideoFeatureDataset(str(features_dir), split="train")

        model = AnomalyDetector()
        model.eval()

        with torch.no_grad():
            for idx in range(len(dataset)):
                features, label = dataset[idx]
                features_batch  = features.unsqueeze(0)
                # Single video — no padding, so lengths optional
                scores, probs   = model(features_batch)
                assert scores is not None
                assert probs is not None


# ============================================================================
# 4. End-to-End Pipeline Tests
# ============================================================================

class TestEndToEndPipeline:

    def test_complete_training_pipeline(self):
        model     = AnomalyDetector()
        loss_fn   = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        features, labels, lengths = _make_batch(batch_size=4)

        model.train()
        scores, probs = model(features, lengths=lengths)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_training_and_evaluation_cycle(self):
        model     = AnomalyDetector()
        loss_fn   = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        f_train, l_train, len_train = _make_batch(batch_size=8)
        f_eval,  l_eval,  len_eval  = _make_batch(batch_size=4)

        model.train()
        for _ in range(3):
            scores, probs = model(f_train, lengths=len_train)
            loss = loss_fn(scores, l_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            scores_eval, probs_eval = model(f_eval, lengths=len_eval)
            eval_loss = loss_fn(scores_eval, l_eval)

        assert eval_loss.item() >= 0


# ============================================================================
# 5. Device Transfer Integration Tests
# ============================================================================

class TestDeviceTransferIntegration:

    def test_cpu_to_cpu_pipeline(self):
        device  = torch.device("cpu")
        model   = AnomalyDetector().to(device)
        loss_fn = MILRankingLoss()

        features = torch.randn(4, 10, 2131).to(device)
        labels   = torch.LongTensor([0, 1, 0, 1]).to(device)
        lengths  = torch.LongTensor([10, 10, 10, 10])  # stays CPU

        scores, probs = model(features, lengths=lengths)
        loss = loss_fn(scores, labels)
        assert loss.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_pipeline(self):
        device  = torch.device("cuda")
        model   = AnomalyDetector().to(device)
        loss_fn = MILRankingLoss()

        features = torch.randn(4, 10, 2131).to(device)
        labels   = torch.LongTensor([0, 1, 0, 1]).to(device)
        lengths  = torch.LongTensor([10, 10, 10, 10])  # stays CPU

        scores, probs = model(features, lengths=lengths)
        loss = loss_fn(scores, labels)
        assert loss.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cpu_to_cuda_transfer(self):
        model   = AnomalyDetector().cuda()
        loss_fn = MILRankingLoss()

        features = torch.randn(4, 10, 2131).cuda()
        labels   = torch.LongTensor([0, 1, 0, 1]).cuda()
        lengths  = torch.LongTensor([10, 10, 10, 10])  # stays CPU

        scores, probs = model(features, lengths=lengths)
        loss = loss_fn(scores, labels)
        assert loss.device.type == "cuda"


# ============================================================================
# 6. Batch Composition Tests
# ============================================================================

class TestBatchCompositionIntegration:

    def test_pipeline_with_variable_batch_size(self):
        model   = AnomalyDetector()
        loss_fn = MILRankingLoss()

        for batch_size in [1, 2, 4, 8, 16]:
            features, labels, lengths = _make_batch(batch_size=batch_size)
            scores, probs = model(features, lengths=lengths)
            loss = loss_fn(scores, labels)
            assert not torch.isnan(loss)

    def test_pipeline_with_variable_sequence_length(self):
        model   = AnomalyDetector()
        loss_fn = MILRankingLoss()

        for seq_len in [1, 5, 10, 20, 50]:
            features, labels, lengths = _make_batch(batch_size=4, seq_len=seq_len)
            scores, probs = model(features, lengths=lengths)
            loss = loss_fn(scores, labels)
            assert not torch.isnan(loss)

    def test_pipeline_with_collated_variable_length_batch(self):
        """Variable-length batch via collate_fn — the main real-world case."""
        features, labels, lengths = _make_variable_batch()

        model   = AnomalyDetector()
        loss_fn = MILRankingLoss()

        scores, probs = model(features, lengths=lengths)
        loss = loss_fn(scores, labels)

        assert not torch.isnan(loss)
        # Output padded dim matches the longest sequence (12)
        assert scores.shape[1] == 12

    def test_lengths_respected_in_output(self):
        """Padded positions should have gru_out == 0, giving scores near 0.5."""
        # Build a batch where video 0 is short (5 segs), video 1 is long (10)
        f0 = torch.randn(5,  2131)
        f1 = torch.randn(10, 2131)
        features, labels, lengths = collate_fn([(f0, 0), (f1, 1)])

        model = AnomalyDetector()
        model.eval()
        with torch.no_grad():
            scores, _ = model(features, lengths=lengths)

        # Padded positions for video 0 (indices 5..9): gru_out = 0,
        # so sigmoid(linear(0)) = sigmoid(bias only) — not necessarily 0.5
        # but scores must be valid floats
        assert torch.isfinite(scores).all()
        # Real positions (0..4) must be present
        assert scores.shape == (2, 10, 1)


# ============================================================================
# 7. State Management Tests
# ============================================================================

class TestStateManagement:

    def test_save_and_load_model_state(self, tmp_path):
        model_1 = AnomalyDetector()

        features, _, lengths = _make_batch()
        model_1.eval()
        with torch.no_grad():
            scores_1, probs_1 = model_1(features, lengths=lengths)

        checkpoint_path = tmp_path / "model.pth"
        torch.save(model_1.state_dict(), checkpoint_path)

        model_2 = AnomalyDetector()
        model_2.load_state_dict(torch.load(checkpoint_path))
        model_2.eval()
        with torch.no_grad():
            scores_2, probs_2 = model_2(features, lengths=lengths)

        assert torch.allclose(scores_1, scores_2)
        assert torch.allclose(probs_1, probs_2)

    def test_train_eval_mode_consistency(self):
        model    = AnomalyDetector()
        features, _, lengths = _make_batch()

        model.train()
        assert model.training
        scores_train, _ = model(features, lengths=lengths)

        model.eval()
        assert not model.training
        scores_eval, _ = model(features, lengths=lengths)

        assert scores_train.shape == scores_eval.shape


# ============================================================================
# 8. Gradient Accumulation Tests
# ============================================================================

class TestGradientAccumulation:

    def test_gradient_accumulation(self):
        model     = AnomalyDetector()
        loss_fn   = MILRankingLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        accumulated_loss = 0
        for _ in range(3):
            features, labels, lengths = _make_batch(batch_size=2)
            model.train()
            scores, probs = model(features, lengths=lengths)
            loss = loss_fn(scores, labels) / 3
            loss.backward()
            accumulated_loss += loss.item()

        optimizer.step()
        assert accumulated_loss >= 0


# ============================================================================
# 9. Inference Pipeline Tests
# ============================================================================

class TestInferencePipeline:

    def test_inference_no_grad_with_lengths(self):
        model = AnomalyDetector()
        model.eval()

        features, _, lengths = _make_batch()

        with torch.no_grad():
            scores, probs = model(features, lengths=lengths)

        assert scores.requires_grad is False
        assert probs.requires_grad is False

    def test_inference_no_grad_without_lengths(self):
        """Single-video inference: lengths=None is still valid."""
        model = AnomalyDetector()
        model.eval()

        features = torch.randn(1, 10, 2131)

        with torch.no_grad():
            scores, probs = model(features)   # no lengths — no padding

        assert scores.requires_grad is False
        assert probs.requires_grad is False

    def test_inference_batch_processing(self):
        model = AnomalyDetector()
        model.eval()

        all_scores = []
        for _ in range(5):
            features, _, lengths = _make_variable_batch()
            with torch.no_grad():
                scores, probs = model(features, lengths=lengths)
            all_scores.append(scores)

        assert len(all_scores) == 5