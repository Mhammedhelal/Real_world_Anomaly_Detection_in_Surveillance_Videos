"""
src/engine/trainer.py
---------------------
Training engine for the UCF-Crime Anomaly Detector.

Changes from original
---------------------
- All print() calls replaced with structured logging via TrainingLogger.
- Epoch metrics collected via MetricsTracker (no manual summing).
- Checkpoint logic delegates to save_checkpoint(); no torch.save() here.
- train() convenience function signature unchanged — backward-compatible.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import Config
from src.data.dataset import VideoFeatureDataset, collate_fn
from src.models.anomaly_detector import AnomalyDetector
from src.models.losses import MILRankingLoss
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.logging import TrainingLogger, get_logger
from src.utils.metrics import MetricsTracker

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training engine for anomaly detection.

    Parameters
    ----------
    model : AnomalyDetector
    train_loader : DataLoader
    config : Config | None
    device : str | None
    save_dir : str | Path | None
    run_name : str | None
        Identifier written into checkpoint filenames and the metrics log.
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
        # Config
        if config is None:
            config_path = (
                Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
            )
            config = Config.from_yaml(config_path)
        self.config = config

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader

        # Hyperparameters from config
        self.num_epochs: int = config.training.num_epochs
        self.log_interval: int = getattr(config.logging, "log_interval", 10)
        self.save_interval: int = getattr(config.logging, "save_interval", 50)

        # Directories
        if save_dir is None:
            save_dir = config.logging.checkpoint_dir
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        log_dir = getattr(config.logging, "log_dir", "outputs/logs")

        # Structured logger
        self._training_logger = TrainingLogger(
            log_dir=log_dir,
            run_name=run_name,
            print_every=self.log_interval,
        )

        # Metrics tracker (resets each epoch)
        self._tracker = MetricsTracker()

        # Optimiser and losses
        self._setup_optimizer()
        self._setup_loss_functions()

        # State
        self.current_epoch: int = 0
        self.loss_history: List[float] = []
        self.best_loss: float = float("inf")

        logger.info(
            "Trainer ready | device=%s | epochs=%d | save_dir=%s",
            self.device,
            self.num_epochs,
            self.save_dir,
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> None:
        opt = self.config.optimizer
        opt_type = opt.type.lower()

        if opt_type == "adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=opt.learning_rate,
                rho=opt.rho,
                eps=opt.eps,
            )
        elif opt_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt.learning_rate,
            )
        elif opt_type == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt.learning_rate,
                momentum=getattr(opt, "momentum", 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        logger.info("Optimizer: %s  lr=%s", opt_type, opt.learning_rate)

    def _setup_loss_functions(self) -> None:
        loss_cfg = self.config.loss
        self.criterion_mil = MILRankingLoss(
            lambda1=loss_cfg.lambda_smoothness,
            lambda2=loss_cfg.lambda_sparsity,
        )
        self.criterion_class = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns
        -------
        dict[str, float]
            Aggregated metrics: ``{"loss", "ranking_loss", "class_loss"}``.
        """
        self.model.train()
        self._tracker.reset()

        num_classes = self.config.model.num_classes

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward
            anomaly_scores, class_logits = self.model(features)

            # MIL ranking loss
            loss_ranking = self.criterion_mil(anomaly_scores, labels)

            # Classification loss (labels broadcast across segments)
            num_segments = features.size(1)
            labels_expanded = labels.repeat_interleave(num_segments)
            loss_class = self.criterion_class(
                class_logits.view(-1, num_classes),
                labels_expanded,
            )

            total_loss = loss_ranking + loss_class

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            bs = features.size(0)
            self._tracker.update("loss", total_loss.item(), bs)
            self._tracker.update("ranking_loss", loss_ranking.item(), bs)
            self._tracker.update("class_loss", loss_class.item(), bs)

            self._training_logger.on_batch_end(
                batch_idx,
                len(self.train_loader),
                {
                    "loss": total_loss.item(),
                    "rank": loss_ranking.item(),
                    "cls": loss_class.item(),
                },
            )

        return self._tracker.summary()

    def train(self, start_epoch: int = 0) -> AnomalyDetector:
        """
        Run the full training loop from *start_epoch* to *num_epochs*.

        Returns
        -------
        AnomalyDetector
            The trained model (still on self.device).
        """
        logger.info(
            "Training started: epochs=%d  batches/epoch=%d",
            self.num_epochs,
            len(self.train_loader),
        )

        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch + 1
            self._training_logger.on_epoch_start(self.current_epoch, self.num_epochs)

            epoch_metrics = self.train_epoch()
            epoch_loss = epoch_metrics["loss"]
            self.loss_history.append(epoch_loss)

            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss
                ckpt_path = self.save_checkpoint(is_best=True)
                self._training_logger.log_checkpoint(ckpt_path, self.current_epoch)

            self._training_logger.on_epoch_end(
                self.current_epoch,
                self.num_epochs,
                epoch_metrics,
                is_best=is_best,
            )

            # Periodic checkpoint
            if (
                self.current_epoch % self.save_interval == 0
                or self.current_epoch == self.num_epochs
            ):
                ckpt_path = self.save_checkpoint(is_best=False)
                self._training_logger.log_checkpoint(ckpt_path, self.current_epoch)

        self._training_logger.close()
        logger.info("Training finished. Best loss: %.4f", self.best_loss)
        return self.model

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_best: bool = False) -> Path:
        """Save current state; returns path written."""
        filename = (
            "best_model.pt"
            if is_best
            else f"anomaly_detector_epoch{self.current_epoch:04d}.pt"
        )
        filepath = self.save_dir / filename

        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss_history[-1] if self.loss_history else 0.0,
            "loss_history": self.loss_history,
            "best_loss": self.best_loss,
            "config": {
                "input_size": self.config.model.input_size,
                "hidden_size": self.config.model.hidden_size,
                "num_classes": self.config.model.num_classes,
            },
        }

        save_checkpoint(state, filepath)
        return filepath

    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """
        Load checkpoint to resume training.

        Returns
        -------
        int
            Epoch number to resume from (i.e. pass as *start_epoch*).
        """
        ckpt = load_checkpoint(checkpoint_path, device=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.loss_history = ckpt.get("loss_history", [])
        self.best_loss = ckpt.get("best_loss", float("inf"))

        logger.info("Resumed from epoch %d", self.current_epoch)
        return self.current_epoch


# ---------------------------------------------------------------------------
# Convenience function (backward-compatible API)
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
    """
    Convenience wrapper: build dataset, model, trainer and run training.

    Parameters
    ----------
    features_dir : str
        Directory containing ``train_*.npz`` feature files.
    save_dir : str
        Directory for checkpoints.
    epochs, batch_size, lr : optional overrides
    config : Config | None
    device : str | None
    resume_from : str | None
        Path to ``.pt`` checkpoint to resume from.
    run_name : str | None
        Identifier written into log filenames.

    Returns
    -------
    (AnomalyDetector, List[float])
        Trained model and per-epoch loss history.
    """
    # Load / build config
    if config is None:
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
        )
        config = Config.from_yaml(config_path)

    if epochs is not None:
        config.training.num_epochs = epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if lr is not None:
        config.optimizer.learning_rate = lr

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    logger.info("Loading training data from: %s", features_dir)
    dataset = VideoFeatureDataset(features_dir, split="train")

    if len(dataset) == 0:
        raise RuntimeError(
            f"No 'train_*.npz' files found in {features_dir}. "
            "Run extract_features.py first."
        )

    logger.info("Loaded %d training samples", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Model
    model = AnomalyDetector(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        config=config,
        device=device,
        save_dir=save_dir,
        run_name=run_name,
    )

    start_epoch = 0
    if resume_from:
        start_epoch = trainer.load_checkpoint(resume_from)

    trained_model = trainer.train(start_epoch=start_epoch)
    return trained_model, trainer.loss_history
