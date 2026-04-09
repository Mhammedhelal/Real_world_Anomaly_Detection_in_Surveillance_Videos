"""
tests/test_integration_comprehensive.py
=======================================

Integration tests for the complete anomaly detection pipeline.

Coverage:
  - End-to-end pipeline: features → model → predictions
  - Feature extraction pipeline
  - Training loop integration
  - DataLoader integration
  - Model evaluation
  - Component interactions
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
# 1. Model + Loss Integration Tests
# ============================================================================

class TestModelLossIntegration:
    """Test model and loss function together."""

    def test_model_loss_forward_backward(self):
        """Test complete forward/backward pass with model and loss."""
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()

        features = torch.randn(4, 10, 2131, requires_grad=True)
        labels = torch.LongTensor([0, 1, 2, 0])

        # Forward pass
        scores, probs = model(features)
        loss = loss_fn(scores, labels)

        # Backward pass
        loss.backward()

        # Verify gradients
        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_model_optimization_step(self):
        """Test model parameter optimization."""
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate

        features = torch.randn(4, 10, 2131)
        labels = torch.LongTensor([0, 1, 2, 0])

        # Get initial parameters
        initial_params = [p.clone().detach() for p in model.parameters()]

        # Training step
        model.train()
        scores, probs = model(features)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify at least one parameter changed (some may remain the same)
        params_changed = False
        for initial_p, current_p in zip(initial_params, model.parameters()):
            # Check if any parameter changed by more than 1e-4
            if (initial_p - current_p.detach()).abs().max() > 1e-4:
                params_changed = True
                break
        assert params_changed, "No parameters were updated during optimization step"

    def test_multiple_optimization_steps(self):
        """Test multiple optimization steps."""
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        losses = []

        for step in range(5):
            features = torch.randn(4, 10, 2131)
            labels = torch.LongTensor([0, 1, 2, 0])

            model.train()
            scores, probs = model(features)
            loss = loss_fn(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Should have 5 loss values
        assert len(losses) == 5


# ============================================================================
# 2. Model + DataLoader Integration Tests
# ============================================================================

class TestModelDataLoaderIntegration:
    """Test model with DataLoader."""

    def test_model_with_dataloader(self, tmp_path):
        """Test model with PyTorch DataLoader."""
        # Create synthetic features
        features_list = [
            torch.randn(10, 2131),
            torch.randn(12, 2131),
            torch.randn(8, 2131),
        ]
        labels = [0, 1, 0]

        batch = list(zip(features_list, labels))

        # Create DataLoader
        dataloader = DataLoader(
            batch,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Run model on batches
        model = AnomalyDetector()
        model.eval()

        with torch.no_grad():
            for features, batch_labels in dataloader:
                scores, probs = model(features)

                assert scores is not None
                assert probs is not None

    def test_training_loop_with_dataloader(self, tmp_path):
        """Test complete training loop with DataLoader."""
        # Create dataset
        features_list = [
            torch.randn(10, 2131),
            torch.randn(12, 2131),
            torch.randn(8, 2131),
            torch.randn(9, 2131),
        ]
        labels = [0, 1, 0, 1]

        batch = list(zip(features_list, labels))

        dataloader = DataLoader(
            batch,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Create model and optimizer
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 2
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0

            for features, batch_labels in dataloader:
                scores, probs = model(features)
                loss = loss_fn(scores, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Epoch completed successfully
            assert epoch_loss >= 0


# ============================================================================
# 3. VideoFeatureDataset Integration Tests
# ============================================================================

class TestVideoFeatureDatasetIntegration:
    """Test VideoFeatureDataset with model."""

    def test_dataset_model_forward_pass(self, tmp_path):
        """Test model forward pass with dataset samples."""
        # Create synthetic dataset files
        features_dir = tmp_path / "features"
        features_dir.mkdir()

        for i in range(3):
            features = np.random.randn(10, 2131).astype(np.float32)
            metadata = {'label': i % 2}
            filepath = features_dir / f"train_video_{i}.npz"
            np.savez_compressed(filepath, features=features, metadata=metadata)

        # Load dataset
        dataset = VideoFeatureDataset(str(features_dir), split="train")

        # Run model on each sample
        model = AnomalyDetector()
        model.eval()

        with torch.no_grad():
            for idx in range(len(dataset)):
                features, label = dataset[idx]

                # Add batch dimension
                features_batch = features.unsqueeze(0)

                scores, probs = model(features_batch)

                assert scores is not None
                assert probs is not None


# ============================================================================
# 4. End-to-End Pipeline Tests
# ============================================================================

class TestEndToEndPipeline:
    """Test complete end-to-end pipeline."""

    def test_complete_training_pipeline(self):
        """Test complete training pipeline: data → model → loss → optim."""
        # Setup
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create synthetic batch
        batch_size = 4
        num_segments = 10
        feature_dim = 2131

        features = torch.randn(batch_size, num_segments, feature_dim)
        labels = torch.LongTensor([0, 1, 0, 1])

        # Training step
        model.train()

        # Forward
        scores, probs = model(features)

        # Loss computation
        loss = loss_fn(scores, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify success
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_training_and_evaluation_cycle(self):
        """Test training and evaluation cycle."""
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        features_train = torch.randn(8, 10, 2131)
        labels_train = torch.LongTensor([0, 0, 1, 1, 0, 1, 0, 1])

        features_eval = torch.randn(4, 10, 2131)
        labels_eval = torch.LongTensor([0, 1, 0, 1])

        # Training
        model.train()
        for _ in range(3):
            scores, probs = model(features_train)
            loss = loss_fn(scores, labels_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            scores_eval, probs_eval = model(features_eval)
            eval_loss = loss_fn(scores_eval, labels_eval)

        assert eval_loss.item() >= 0


# ============================================================================
# 5. Device Transfer Integration Tests
# ============================================================================

class TestDeviceTransferIntegration:
    """Test device transfers throughout pipeline."""

    def test_cpu_to_cpu_pipeline(self):
        """Test pipeline on CPU."""
        device = torch.device("cpu")

        model = AnomalyDetector().to(device)
        loss_fn = MILRankingLoss()

        features = torch.randn(4, 10, 2131).to(device)
        labels = torch.LongTensor([0, 1, 0, 1]).to(device)

        scores, probs = model(features)
        loss = loss_fn(scores, labels)

        assert loss.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cuda_pipeline(self):
        """Test pipeline on CUDA."""
        device = torch.device("cuda")

        model = AnomalyDetector().to(device)
        loss_fn = MILRankingLoss()

        features = torch.randn(4, 10, 2131).to(device)
        labels = torch.LongTensor([0, 1, 0, 1]).to(device)

        scores, probs = model(features)
        loss = loss_fn(scores, labels)

        assert loss.device.type == "cuda"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_cpu_to_cuda_transfer(self):
        """Test transferring model and data to CUDA."""
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()

        # Create on CPU
        features_cpu = torch.randn(4, 10, 2131)
        labels_cpu = torch.LongTensor([0, 1, 0, 1])

        # Move to CUDA
        model = model.cuda()
        features_cuda = features_cpu.cuda()
        labels_cuda = labels_cpu.cuda()

        # Forward pass
        scores, probs = model(features_cuda)
        loss = loss_fn(scores, labels_cuda)

        assert loss.device.type == "cuda"


# ============================================================================
# 6. Batch Composition Tests
# ============================================================================

class TestBatchCompositionIntegration:
    """Test different batch compositions."""

    def test_pipeline_with_variable_batch_size(self):
        """Test pipeline with different batch sizes."""
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()

        for batch_size in [1, 2, 4, 8, 16]:
            features = torch.randn(batch_size, 10, 2131)
            labels = torch.LongTensor([i % 2 for i in range(batch_size)])

            scores, probs = model(features)
            loss = loss_fn(scores, labels)

            assert not torch.isnan(loss)

    def test_pipeline_with_variable_sequence_length(self):
        """Test pipeline with different sequence lengths."""
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()

        for seq_len in [1, 5, 10, 20, 50]:
            features = torch.randn(4, seq_len, 2131)
            labels = torch.LongTensor([0, 1, 0, 1])

            scores, probs = model(features)
            loss = loss_fn(scores, labels)

            assert not torch.isnan(loss)

    def test_pipeline_with_collated_batch(self):
        """Test pipeline with collated variable-length batch."""
        # Create variable-length samples
        samples = [
            (torch.randn(8, 2131), 0),
            (torch.randn(12, 2131), 1),
            (torch.randn(10, 2131), 0),
        ]

        # Collate
        features, labels = collate_fn(samples)

        # Forward pass
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()

        scores, probs = model(features)
        loss = loss_fn(scores, labels)

        assert not torch.isnan(loss)


# ============================================================================
# 7. Checkpoint and State Management Tests
# ============================================================================

class TestStateManagement:
    """Test model state management."""

    def test_save_and_load_model_state(self, tmp_path):
        """Test saving and loading model state."""
        model_1 = AnomalyDetector()

        # Get initial predictions
        features = torch.randn(4, 10, 2131)
        model_1.eval()
        with torch.no_grad():
            scores_1, probs_1 = model_1(features)

        # Save state
        checkpoint_path = tmp_path / "model.pth"
        torch.save(model_1.state_dict(), checkpoint_path)

        # Create new model and load state
        model_2 = AnomalyDetector()
        model_2.load_state_dict(torch.load(checkpoint_path))

        # Get predictions from loaded model
        model_2.eval()
        with torch.no_grad():
            scores_2, probs_2 = model_2(features)

        # Should be identical
        assert torch.allclose(scores_1, scores_2)
        assert torch.allclose(probs_1, probs_2)

    def test_train_eval_mode_consistency(self):
        """Test consistency of train/eval mode switches."""
        model = AnomalyDetector()
        features = torch.randn(4, 10, 2131)

        # Switch modes
        model.train()
        assert model.training

        model.eval()
        assert not model.training

        model.train()
        assert model.training

        # Forward pass should work in both modes
        scores_train, _ = model(features)
        model.eval()
        scores_eval, _ = model(features)

        assert scores_train.shape == scores_eval.shape


# ============================================================================
# 8. Gradient Accumulation Tests
# ============================================================================

class TestGradientAccumulation:
    """Test gradient accumulation over batches."""

    def test_gradient_accumulation(self):
        """Test accumulating gradients over multiple batches."""
        model = AnomalyDetector()
        loss_fn = MILRankingLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        accumulated_loss = 0

        for batch_idx in range(3):
            features = torch.randn(2, 10, 2131)
            labels = torch.LongTensor([0, 1])

            model.train()
            scores, probs = model(features)
            loss = loss_fn(scores, labels) / 3  # Scale by number of batches

            loss.backward()  # Accumulate gradients

            accumulated_loss += loss.item()

        # Step once after accumulation
        optimizer.step()

        # Verify accumulation worked
        assert accumulated_loss >= 0


# ============================================================================
# 9. Inference Pipeline Tests
# ============================================================================

class TestInferencePipeline:
    """Test inference-only pipeline."""

    def test_inference_no_grad(self):
        """Test efficient inference with no_grad."""
        model = AnomalyDetector()
        model.eval()

        features = torch.randn(4, 10, 2131)

        with torch.no_grad():
            scores, probs = model(features)

        # Verify no gradients computed
        assert scores.requires_grad is False
        assert probs.requires_grad is False

    def test_inference_batch_processing(self):
        """Test inference on multiple batches."""
        model = AnomalyDetector()
        model.eval()

        all_scores = []
        all_probs = []

        num_batches = 5
        for batch_idx in range(num_batches):
            features = torch.randn(4, 10, 2131)

            with torch.no_grad():
                scores, probs = model(features)

            all_scores.append(scores)
            all_probs.append(probs)

        # Should have processed 5 batches
        assert len(all_scores) == 5
        assert len(all_probs) == 5
