"""
src/engine/trainer.py
---------------------
Training engine for the UCF-Crime Anomaly Detector.

Fixes applied
-------------
- num_workers read from config (was hard-coded to 0).
- Mixed-precision training via torch.cuda.amp (GradScaler + autocast).
- Checkpoint logic delegates to save_checkpoint() / load_checkpoint().
- All print() replaced with structured logging.
- Metrics accumulated via MetricsTracker.
- collate_fn now returns (features, labels, lengths); lengths passed to
  AnomalyDetector.forward so the BiGRU packs sequences correctly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from src.config import Config
from src.data.dataset import VideoFeatureDataset, collate_fn
from src.models.anomaly_detector import AnomalyDetector
from src.models.losses import MILRankingLoss
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.logging import TrainingLogger, get_logger
from src.utils.metrics import MetricsTracker

logger = get_logger(__name__)


class Trainer:
    """
    Training engine for anomaly detection.

    Parameters
    ----------
    model       : AnomalyDetector
    train_loader: DataLoader
    config      : Config | None
    device      : str | None
    save_dir    : str | Path | None
    run_name    : str | None
    """

    def __init__(
        self,
        model: AnomalyDetector,
        train_loader: DataLoader,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        save_dir: Optional[str | Path] = None,
        run_name: Optional[str] = None,
    ) -> None:
        if config is None:
            config_path = (
                Path(__file__).resolve().parent.parent.parent
                / "configs" / "default.yaml"
            )
            config = Config.from_yaml(config_path)
        self.config = config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        for param in self.model.parameters(): param.requires_grad = True
        self.train_loader = train_loader

        self.num_epochs: int    = config.training.num_epochs
        self.log_interval: int  = getattr(config.logging, "log_interval", 10)
        self.save_interval: int = getattr(config.logging, "save_interval", 50)

        # Mixed-precision: only valid on CUDA
        self._use_amp: bool = (
            getattr(config.hardware, "mixed_precision", False)
            and self.device.type == "cuda"
        )
        self._scaler = GradScaler(enabled=self._use_amp)
        logger.info(
            "Mixed-precision (AMP): %s", "enabled" if self._use_amp else "disabled"
        )

        if save_dir is None:
            save_dir = config.logging.checkpoint_dir
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        log_dir = getattr(config.logging, "log_dir", "outputs/logs")
        self._training_logger = TrainingLogger(
            log_dir=log_dir,
            run_name=run_name,
            print_every=self.log_interval,
        )
        self._tracker = MetricsTracker()

        self._setup_optimizer()
        self._setup_loss_functions()

        self.current_epoch: int  = 0
        self.loss_history: List[float] = []
        self.best_loss: float = float("inf")

        logger.info(
            "Trainer ready | device=%s | epochs=%d | amp=%s | save_dir=%s",
            self.device, self.num_epochs, self._use_amp, self.save_dir,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> None:
        opt = self.config.optimizer
        t   = opt.type.lower()
        if t == "adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=opt.learning_rate, rho=opt.rho, eps=opt.eps,
            )
        elif t == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=opt.learning_rate,
            )
        elif t == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt.learning_rate,
                momentum=getattr(opt, "momentum", 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer: {t}")
        logger.info("Optimizer: %s  lr=%s", t, opt.learning_rate)

    def _setup_loss_functions(self) -> None:
        loss_cfg = self.config.loss
        self.criterion_mil   = MILRankingLoss(
            lambda1=loss_cfg.lambda_smoothness,
            lambda2=loss_cfg.lambda_sparsity,
        )
        self.criterion_class = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. Returns aggregated metrics dict."""
        self.model.train()
        self._tracker.reset()
        num_classes = self.config.model.num_classes

        # collate_fn yields (features, labels, lengths)
        for batch_idx, (features, labels, lengths) in enumerate(self.train_loader):
            features = features.to(self.device, non_blocking=True)
            labels   = labels.to(self.device,   non_blocking=True)
            # lengths stays on CPU — pack_padded_sequence requires it there

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=self._use_amp):
                # Pass lengths so the GRU packs and ignores padding zeros
                anomaly_scores, class_logits = self.model(features, lengths=lengths)

                loss_ranking = self.criterion_mil(anomaly_scores, labels)

                num_segments     = features.size(1)
                labels_expanded  = labels.repeat_interleave(num_segments)
                loss_class       = self.criterion_class(
                    class_logits.view(-1, num_classes), labels_expanded
                )
                total_loss = loss_ranking + loss_class

            # Backward — GradScaler is a no-op when AMP is disabled
            self._scaler.scale(total_loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()

            bs = features.size(0)
            self._tracker.update("loss",         total_loss.item(),    bs)
            self._tracker.update("ranking_loss", loss_ranking.item(),  bs)
            self._tracker.update("class_loss",   loss_class.item(),    bs)

            self._training_logger.on_batch_end(
                batch_idx, len(self.train_loader),
                {"loss": total_loss.item(), "rank": loss_ranking.item()},
            )

        return self._tracker.summary()

    def train(self, start_epoch: int = 0) -> AnomalyDetector:
        """Run the full training loop."""
        logger.info(
            "Training started: epochs=%d  batches/epoch=%d  amp=%s",
            self.num_epochs, len(self.train_loader), self._use_amp,
        )
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch + 1
            self._training_logger.on_epoch_start(self.current_epoch, self.num_epochs)

            metrics    = self.train_epoch()
            epoch_loss = metrics["loss"]
            self.loss_history.append(epoch_loss)

            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss
                ckpt = self.save_checkpoint(is_best=True)
                self._training_logger.log_checkpoint(ckpt, self.current_epoch)

            self._training_logger.on_epoch_end(
                self.current_epoch, self.num_epochs, metrics, is_best=is_best
            )

            if (
                self.current_epoch % self.save_interval == 0
                or self.current_epoch == self.num_epochs
            ):
                ckpt = self.save_checkpoint(is_best=False)
                self._training_logger.log_checkpoint(ckpt, self.current_epoch)

        self._training_logger.close()
        logger.info("Training finished. Best loss: %.4f", self.best_loss)
        return self.model

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_best: bool = False) -> Path:
        fname = (
            "best_model.pt"
            if is_best
            else f"anomaly_detector_epoch{self.current_epoch:04d}.pt"
        )
        path = self.save_dir / fname
        save_checkpoint(
            {
                "epoch":                self.current_epoch,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict":    self._scaler.state_dict(),
                "loss":                 self.loss_history[-1] if self.loss_history else 0.0,
                "loss_history":         self.loss_history,
                "best_loss":            self.best_loss,
                "input_size":  self.model.input_size,
                "hidden_size": self.model.hidden_size,
                "num_classes": self.model.num_classes,

            },
            path,
        )
        return path

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        ckpt = load_checkpoint(checkpoint_path, device=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        for param in self.model.parameters(): param.requires_grad = True
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            self._scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.loss_history  = ckpt.get("loss_history", [])
        self.best_loss     = ckpt.get("best_loss", float("inf"))
        logger.info("Resumed from epoch %d", self.current_epoch)
        return self.current_epoch


# ---------------------------------------------------------------------------
# Convenience function (backward-compatible)
# ---------------------------------------------------------------------------

def train(
    features_dir: str,
    save_dir: str = "./checkpoints",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    config: Optional[Config] = None,
    device: Optional[str] = None,
    resume_from: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Tuple[AnomalyDetector, List[float]]:
    """Build dataset, model, trainer and run training."""
    if config is None:
        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs" / "default.yaml"
        )
        config = Config.from_yaml(config_path)

    if epochs     is not None: config.training.num_epochs      = epochs
    if batch_size is not None: config.training.batch_size      = batch_size
    if lr         is not None: config.optimizer.learning_rate  = lr

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading training data from: %s", features_dir)
    dataset = VideoFeatureDataset(features_dir, split="train")

    if len(dataset) == 0:
        raise RuntimeError(
            f"No 'train_*.npz' files found in {features_dir}. "
            "Run extract_features.py first."
        )
    logger.info("Loaded %d training samples (%d skipped)", len(dataset), dataset.n_skipped)

    # num_workers from config (was hard-coded to 0)
    num_workers = getattr(config.training, "num_workers", 4)
    # pin_memory only useful on CUDA
    pin_memory  = getattr(config.training, "pin_memory", True) and (device == "cuda")

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    logger.info(
        "DataLoader: batch_size=%d  num_workers=%d  pin_memory=%s",
        config.training.batch_size, num_workers, pin_memory,
    )

    model = AnomalyDetector(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
    )

    trainer = Trainer(
        model=model, train_loader=loader, config=config,
        device=device, save_dir=save_dir, run_name=run_name,
    )

    start_epoch = 0
    if resume_from:
        start_epoch = trainer.load_checkpoint(resume_from)

    trained_model = trainer.train(start_epoch=start_epoch)
    return trained_model, trainer.loss_history


# backward-compat alias
train_model = train