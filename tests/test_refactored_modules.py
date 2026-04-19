"""
tests/test_refactored_modules.py
=================================
Tests specifically covering the three refactored/new modules:

  - src/utils/checkpointing.py   (split into 3 clean functions)
  - src/utils/metrics.py         (full rewrite + MetricsTracker)
  - src/utils/logging.py         (new module)

These tests complement the existing comprehensive test suite and verify
the new API contracts introduced by the refactor.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# ============================================================
# 1. Checkpointing Tests
# ============================================================

class TestSaveCheckpoint:
    """Tests for save_checkpoint."""

    def test_save_creates_file(self, tmp_path):
        from src.utils.checkpointing import save_checkpoint
        path = tmp_path / "test.pt"
        save_checkpoint({"epoch": 1, "loss": 0.5}, path)
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        from src.utils.checkpointing import save_checkpoint
        path = tmp_path / "nested" / "deep" / "model.pt"
        save_checkpoint({"epoch": 1}, path)
        assert path.exists()

    def test_save_returns_path(self, tmp_path):
        from src.utils.checkpointing import save_checkpoint
        path = tmp_path / "ckpt.pt"
        result = save_checkpoint({"a": 1}, path)
        assert result == path

    def test_save_arbitrary_dict(self, tmp_path):
        from src.utils.checkpointing import save_checkpoint
        state = {"epoch": 5, "loss": 0.123, "config": {"lr": 0.001}}
        path = tmp_path / "state.pt"
        save_checkpoint(state, path)
        loaded = torch.load(path, map_location="cpu")
        assert loaded["epoch"] == 5
        assert loaded["config"]["lr"] == 0.001

    def test_save_with_tensor(self, tmp_path):
        from src.utils.checkpointing import save_checkpoint
        state = {"weights": torch.randn(10, 10), "bias": torch.zeros(10)}
        path = tmp_path / "tensors.pt"
        save_checkpoint(state, path)
        loaded = torch.load(path, map_location="cpu")
        assert loaded["weights"].shape == (10, 10)


class TestLoadCheckpoint:
    """Tests for load_checkpoint (raw dict — no model construction)."""

    def test_load_returns_dict(self, tmp_path):
        from src.utils.checkpointing import save_checkpoint, load_checkpoint
        path = tmp_path / "ckpt.pt"
        save_checkpoint({"epoch": 3, "loss": 0.25}, path)
        ckpt = load_checkpoint(path)
        assert isinstance(ckpt, dict)
        assert ckpt["epoch"] == 3

    def test_load_raises_if_missing(self, tmp_path):
        from src.utils.checkpointing import load_checkpoint
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_load_roundtrip(self, tmp_path):
        from src.utils.checkpointing import save_checkpoint, load_checkpoint
        model = torch.nn.Linear(4, 2)
        state = {"model_state_dict": model.state_dict(), "epoch": 7}
        path = tmp_path / "round.pt"
        save_checkpoint(state, path)
        ckpt = load_checkpoint(path, device="cpu")
        assert "model_state_dict" in ckpt
        assert ckpt["epoch"] == 7

    def test_load_does_not_build_model(self, tmp_path):
        """load_checkpoint must NOT instantiate any model."""
        from src.utils.checkpointing import save_checkpoint, load_checkpoint
        path = tmp_path / "raw.pt"
        save_checkpoint({"data": [1, 2, 3]}, path)
        ckpt = load_checkpoint(path)
        # Must be a plain dict, not a nn.Module
        assert not isinstance(ckpt, torch.nn.Module)

    def test_load_checkpoint_cpu_map(self, tmp_path):
        """load_checkpoint should map to CPU by default."""
        from src.utils.checkpointing import save_checkpoint, load_checkpoint
        path = tmp_path / "cpu.pt"
        save_checkpoint({"t": torch.randn(3)}, path)
        ckpt = load_checkpoint(path, device="cpu")
        assert ckpt["t"].device.type == "cpu"


class TestLoadModelFromCheckpoint:
    """Tests for load_model_from_checkpoint convenience function."""

    def _make_ckpt(self, tmp_path) -> Path:
        from src.models.anomaly_detector import AnomalyDetector
        from src.utils.checkpointing import save_checkpoint
        model = AnomalyDetector(input_size=2131, hidden_size=256, num_classes=14)
        path = tmp_path / "model.pt"
        save_checkpoint({
            "epoch": 2,
            "loss": 0.4,
            "model_state_dict": model.state_dict(),
            "config": {"input_size": 2131, "hidden_size": 256, "num_classes": 14},
        }, path)
        return path

    def test_returns_anomaly_detector(self, tmp_path):
        from src.utils.checkpointing import load_model_from_checkpoint
        from src.models.anomaly_detector import AnomalyDetector
        path = self._make_ckpt(tmp_path)
        model = load_model_from_checkpoint(path, device="cpu")
        assert isinstance(model, AnomalyDetector)

    def test_model_in_eval_mode(self, tmp_path):
        from src.utils.checkpointing import load_model_from_checkpoint
        path = self._make_ckpt(tmp_path)
        model = load_model_from_checkpoint(path, device="cpu")
        assert not model.training

    def test_model_produces_correct_output_shape(self, tmp_path):
        from src.utils.checkpointing import load_model_from_checkpoint
        path = self._make_ckpt(tmp_path)
        model = load_model_from_checkpoint(path, device="cpu")
        x = torch.randn(2, 10, 2131)
        with torch.no_grad():
            scores, probs = model(x)
        assert scores.shape == (2, 10, 1)
        assert probs.shape == (2, 10, 14)

    def test_weights_match_original(self, tmp_path):
        from src.models.anomaly_detector import AnomalyDetector
        from src.utils.checkpointing import save_checkpoint, load_model_from_checkpoint

        original = AnomalyDetector()
        path = tmp_path / "orig.pt"
        save_checkpoint({
            "epoch": 1, "loss": 0.1,
            "model_state_dict": original.state_dict(),
            "config": {"input_size": 2131, "hidden_size": 256, "num_classes": 14},
        }, path)

        loaded = load_model_from_checkpoint(path, device="cpu")
        for (k1, v1), (k2, v2) in zip(
            original.state_dict().items(), loaded.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"Mismatch in {k1}"


# ============================================================
# 2. Metrics Tests
# ============================================================

class TestComputeAUC:
    def test_perfect_classifier(self):
        from src.utils.metrics import compute_auc
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_auc(y_true, y_score) == pytest.approx(1.0, abs=1e-6)

    def test_random_classifier_near_half(self):
        from src.utils.metrics import compute_auc
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 1000)
        y_score = rng.random(1000)
        auc = compute_auc(y_true, y_score)
        assert 0.4 < auc < 0.6

    def test_all_same_label_returns_nan(self):
        from src.utils.metrics import compute_auc
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.5, 0.9])
        result = compute_auc(y_true, y_score)
        assert np.isnan(result)

    def test_shape_mismatch_raises(self):
        from src.utils.metrics import compute_auc
        with pytest.raises(ValueError):
            compute_auc(np.array([0, 1]), np.array([0.5]))

    def test_output_is_float(self):
        from src.utils.metrics import compute_auc
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7])
        assert isinstance(compute_auc(y_true, y_score), float)


class TestComputeAccuracy:
    def test_perfect_accuracy(self):
        from src.utils.metrics import compute_accuracy
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        assert compute_accuracy(y_true, y_pred) == 1.0

    def test_zero_accuracy(self):
        from src.utils.metrics import compute_accuracy
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_partial_accuracy(self):
        from src.utils.metrics import compute_accuracy
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.75)

    def test_output_is_float(self):
        from src.utils.metrics import compute_accuracy
        assert isinstance(compute_accuracy(np.array([0]), np.array([0])), float)


class TestComputePerClassAccuracy:
    def test_all_correct(self):
        from src.utils.metrics import compute_per_class_accuracy
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        result = compute_per_class_accuracy(y_true, y_pred, 3)
        assert all(v == 1.0 for v in result.values())

    def test_missing_class_excluded(self):
        from src.utils.metrics import compute_per_class_accuracy
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        result = compute_per_class_accuracy(y_true, y_pred, 3)  # class 2 absent
        assert "class_2" not in result

    def test_custom_class_names(self):
        from src.utils.metrics import compute_per_class_accuracy
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = compute_per_class_accuracy(y_true, y_pred, 2, ["Normal", "Assault"])
        assert "Normal" in result and "Assault" in result

    def test_returns_dict(self):
        from src.utils.metrics import compute_per_class_accuracy
        result = compute_per_class_accuracy(np.array([0, 1]), np.array([0, 0]), 2)
        assert isinstance(result, dict)


class TestComputeConfusionMatrix:
    def test_shape(self):
        from src.utils.metrics import compute_confusion_matrix
        cm = compute_confusion_matrix(np.array([0, 1, 2]), np.array([0, 1, 2]), 3)
        assert cm.shape == (3, 3)

    def test_perfect_predictions(self):
        from src.utils.metrics import compute_confusion_matrix
        y = np.array([0, 1, 2, 0, 1, 2])
        cm = compute_confusion_matrix(y, y, 3)
        assert (np.diag(cm) == np.array([2, 2, 2])).all()

    def test_all_misclassified(self):
        from src.utils.metrics import compute_confusion_matrix
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        cm = compute_confusion_matrix(y_true, y_pred, 2)
        assert cm[0, 0] == 0 and cm[1, 1] == 0

    def test_out_of_range_ignored(self):
        from src.utils.metrics import compute_confusion_matrix
        y_true = np.array([0, 1, 99])
        y_pred = np.array([0, 1, 99])
        cm = compute_confusion_matrix(y_true, y_pred, 3)
        assert cm.shape == (3, 3)
        assert cm.sum() == 2  # 99 is out of range and ignored


class TestMetricsTracker:
    def test_single_metric_update(self):
        from src.utils.metrics import MetricsTracker
        t = MetricsTracker()
        t.update("loss", 1.0)
        t.update("loss", 0.5)
        assert t.average("loss") == pytest.approx(0.75)

    def test_multiple_metrics(self):
        from src.utils.metrics import MetricsTracker
        t = MetricsTracker()
        t.update("loss", 2.0)
        t.update("acc", 0.9)
        summary = t.summary()
        assert "loss" in summary and "acc" in summary

    def test_reset_clears_all(self):
        from src.utils.metrics import MetricsTracker
        t = MetricsTracker()
        t.update("loss", 1.0)
        t.reset()
        assert np.isnan(t.average("loss"))

    def test_unknown_metric_returns_nan(self):
        from src.utils.metrics import MetricsTracker
        t = MetricsTracker()
        assert np.isnan(t.average("nonexistent"))

    def test_n_weighting(self):
        from src.utils.metrics import MetricsTracker
        t = MetricsTracker()
        t.update("loss", 1.0, n=3)  # represents 3 samples of value 1.0
        t.update("loss", 0.0, n=1)  # represents 1 sample of value 0.0
        # internal storage is 4 values: [1, 1, 1, 0]
        assert t.average("loss") == pytest.approx(0.75)

    def test_window_limits_history(self):
        from src.utils.metrics import MetricsTracker
        t = MetricsTracker(window=3)
        for v in [10.0, 10.0, 10.0, 0.0, 0.0, 0.0]:
            t.update("x", v)
        # Only last 3 values (0, 0, 0) retained
        assert t.average("x") == pytest.approx(0.0)

    def test_summary_returns_dict_of_floats(self):
        from src.utils.metrics import MetricsTracker
        t = MetricsTracker()
        t.update("a", 0.5)
        t.update("b", 1.5)
        s = t.summary()
        assert all(isinstance(v, float) for v in s.values())


# ============================================================
# 3. Logging Tests
# ============================================================

class TestSetupLogging:
    def test_returns_log_path(self, tmp_path):
        from src.utils.logging import setup_logging
        path = setup_logging(log_dir=tmp_path, run_name="test_run")
        assert path is not None
        assert path.exists()
        assert path.suffix == ".log"

    def test_returns_none_when_no_file(self, tmp_path):
        from src.utils.logging import setup_logging
        path = setup_logging(log_dir=tmp_path, log_to_file=False)
        assert path is None

    def test_log_file_contains_output(self, tmp_path):
        from src.utils.logging import setup_logging, get_logger
        setup_logging(log_dir=tmp_path, run_name="content_test")
        logger = get_logger("test_content")
        logger.info("hello from test")
        log_file = tmp_path / "content_test.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "hello from test" in content

    def test_idempotent_no_duplicate_handlers(self, tmp_path):
        from src.utils.logging import setup_logging
        setup_logging(log_dir=tmp_path, run_name="idem1")
        setup_logging(log_dir=tmp_path, run_name="idem2")
        root = logging.getLogger()
        # Handler count should not grow unboundedly
        assert len(root.handlers) <= 4  # console + file per call, max 2 each


class TestGetLogger:
    def test_returns_logger(self):
        from src.utils.logging import get_logger
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"

    def test_same_name_returns_same_instance(self):
        from src.utils.logging import get_logger
        l1 = get_logger("shared.module")
        l2 = get_logger("shared.module")
        assert l1 is l2


class TestTrainingLogger:
    def test_creates_jsonl_file(self, tmp_path):
        from src.utils.logging import TrainingLogger
        tl = TrainingLogger(log_dir=tmp_path, run_name="run_01")
        tl.on_epoch_start(1, 10)
        tl.on_epoch_end(1, 10, {"loss": 0.5, "rank": 0.3})
        tl.close()
        jsonl = tmp_path / "run_01_metrics.jsonl"
        assert jsonl.exists()

    def test_jsonl_is_valid_json(self, tmp_path):
        from src.utils.logging import TrainingLogger
        tl = TrainingLogger(log_dir=tmp_path, run_name="run_02")
        tl.on_epoch_end(1, 5, {"loss": 0.4})
        tl.close()
        jsonl = tmp_path / "run_02_metrics.jsonl"
        lines = jsonl.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["epoch"] == 1
        assert record["loss"] == pytest.approx(0.4)

    def test_is_best_flag_recorded(self, tmp_path):
        from src.utils.logging import TrainingLogger
        tl = TrainingLogger(log_dir=tmp_path, run_name="run_03")
        tl.on_epoch_end(1, 5, {"loss": 0.9}, is_best=False)
        tl.on_epoch_end(2, 5, {"loss": 0.3}, is_best=True)
        tl.close()
        lines = (tmp_path / "run_03_metrics.jsonl").read_text().strip().splitlines()
        records = [json.loads(l) for l in lines]
        assert records[0]["is_best"] is False
        assert records[1]["is_best"] is True

    def test_multiple_epochs_produce_multiple_lines(self, tmp_path):
        from src.utils.logging import TrainingLogger
        tl = TrainingLogger(log_dir=tmp_path, run_name="run_04")
        for ep in range(1, 6):
            tl.on_epoch_end(ep, 5, {"loss": 1.0 / ep})
        tl.close()
        lines = (tmp_path / "run_04_metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 5

    def test_log_eval_appends_to_jsonl(self, tmp_path):
        from src.utils.logging import TrainingLogger
        tl = TrainingLogger(log_dir=tmp_path, run_name="run_05")
        tl.on_epoch_end(1, 1, {"loss": 0.5})
        tl.log_eval({"auc": 0.85, "accuracy": 0.72}, split="test")
        tl.close()
        lines = (tmp_path / "run_05_metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        eval_record = json.loads(lines[1])
        assert eval_record["type"] == "eval"
        assert eval_record["auc"] == pytest.approx(0.85)

    def test_metrics_path_property(self, tmp_path):
        from src.utils.logging import TrainingLogger
        tl = TrainingLogger(log_dir=tmp_path, run_name="run_06")
        assert tl.metrics_path == tmp_path / "run_06_metrics.jsonl"
        tl.close()
