"""
src/engine/trainer.py
---------------------
Training engine for anomaly detection model.

From: AnomalyDetector_helal_Feb_23.ipynb

Refactored to:
  - Support configuration-driven training
  - Provide checkpoint management
  - Enable training resumption
  - Support multiple optimizers
  - Integrate with evaluation
"""

import os
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.anomaly_detector import AnomalyDetector
from src.models.losses import MILRankingLoss
from src.data.dataset import VideoFeatureDataset, collate_fn
from src.utils.checkpointing import save_checkpoint
from src.config import Config


class Trainer:
    """
    Training engine for anomaly detection model.
    
    Parameters
    ----------
    model : AnomalyDetector
        The model to train
    train_loader : DataLoader
        DataLoader for training data
    config : Config | None
        Configuration object. If None, loads defaults.
    device : str | None
        Device ('cuda', 'cpu', or None for auto-detect)
    save_dir : str | Path | None
        Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: AnomalyDetector,
        train_loader: DataLoader,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        save_dir: Optional[str | Path] = None,
    ):
        # Load config if not provided
        if config is None:
            config_path = Path(__file__).parent.parent.parent / 'configs' / 'default.yaml'
            config = Config.from_yaml(config_path)
        self.config = config
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Model and data
        self.model = model.to(self.device)
        self.train_loader = train_loader
        
        # Training parameters from config
        self.num_epochs = config.training.num_epochs
        self.log_interval = getattr(config.logging, 'log_interval', 10)
        self.save_interval = getattr(config.logging, 'save_interval', 50)
        
        # Checkpoint directory
        if save_dir is None:
            save_dir = config.logging.checkpoint_dir
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer setup
        self._setup_optimizer()
        
        # Loss functions
        self._setup_loss_functions()
        
        # Training state
        self.current_epoch = 0
        self.loss_history: List[float] = []
        self.best_loss = float('inf')
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer from config."""
        opt_config = self.config.optimizer
        opt_type = opt_config.type.lower()
        
        if opt_type == 'adadelta':
            self.optimizer = torch.optim.Adadelta(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                rho=opt_config.rho,
                eps=opt_config.eps,
            )
        elif opt_type == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.learning_rate,
            )
        elif opt_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                momentum=getattr(opt_config, 'momentum', 0.9),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    def _setup_loss_functions(self) -> None:
        """Setup loss functions from config."""
        loss_config = self.config.loss
        
        self.criterion_mil = MILRankingLoss(
            lambda1=loss_config.lambda_smoothness,
            lambda2=loss_config.lambda_sparsity,
        )
        self.criterion_class = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns
        -------
        float
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            anomaly_scores, class_logits = self.model(features)
            
            # Multi-task loss
            loss_ranking = self.criterion_mil(anomaly_scores, labels)
            
            # Classification loss: flatten time dimension
            num_segments = features.size(1)
            num_classes = self.config.model.num_classes
            labels_expanded = labels.repeat_interleave(num_segments)
            loss_classification = self.criterion_class(
                class_logits.view(-1, num_classes),
                labels_expanded,
            )
            
            total_loss = loss_ranking + loss_classification
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.log_interval == 0 and batch_idx > 0:
                avg_loss = epoch_loss / num_batches
                print(
                    f"  Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {avg_loss:.4f}"
                )
        
        return epoch_loss / max(num_batches, 1)
    
    def train(self, start_epoch: int = 0) -> AnomalyDetector:
        """
        Run full training loop.
        
        Parameters
        ----------
        start_epoch : int
            Starting epoch (for resuming training)
        
        Returns
        -------
        AnomalyDetector
            Trained model
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Learning rate: {self.config.optimizer.learning_rate}")
        print(f"Optimizer: {self.config.optimizer.type}")
        print(f"Save directory: {self.save_dir}")
        print("=" * 70 + "\n")
        
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch + 1
            
            # Train epoch
            epoch_loss = self.train_epoch()
            self.loss_history.append(epoch_loss)
            
            # Track best model
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint(is_best=True)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Loss: {epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0 or epoch == self.num_epochs - 1:
                self.save_checkpoint()
        
        print(f"\n✅ Training completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        
        return self.model
    
    def save_checkpoint(self, is_best: bool = False) -> Path:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        is_best : bool
            Whether this is the best model so far
        
        Returns
        -------
        Path
            Path to saved checkpoint
        """
        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"anomaly_detector_epoch{self.current_epoch:04d}.pt"
        
        filepath = self.save_dir / filename
        
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_history[-1] if self.loss_history else 0.0,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'config': {
                'input_size': self.config.model.input_size,
                'hidden_size': self.config.model.hidden_size,
                'num_classes': self.config.model.num_classes,
            },
        }
        
        save_checkpoint(state, str(filepath))
        return filepath
    
    def load_checkpoint(self, checkpoint_path: str | Path) -> int:
        """
        Load checkpoint to resume training.
        
        Parameters
        ----------
        checkpoint_path : str | Path
            Path to checkpoint
        
        Returns
        -------
        int
            Epoch to resume from
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"✅ Resumed from epoch {self.current_epoch}")
        return self.current_epoch


def train(
    features_dir: str,
    save_dir: str = "./checkpoints",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    config: Optional[Config] = None,
    device: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> tuple[AnomalyDetector, List[float]]:
    """
    Convenience function for training.
    
    Parameters
    ----------
    features_dir : str
        Directory containing .npz feature files
    save_dir : str
        Directory to save checkpoints
    epochs : int | None
        Number of epochs (overrides config)
    batch_size : int | None
        Batch size (overrides config)
    lr : float | None
        Learning rate (overrides config)
    config : Config | None
        Configuration object
    device : str | None
        Device to use
    resume_from : str | None
        Path to checkpoint to resume from
    
    Returns
    -------
    tuple[AnomalyDetector, List[float]]
        Trained model and loss history
    """
    # Load config
    if config is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'default.yaml'
        config = Config.from_yaml(config_path)
    
    # Apply overrides
    if epochs is not None:
        config.training.num_epochs = epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if lr is not None:
        config.optimizer.learning_rate = lr
    
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    
    # Load dataset
    print(f"📂 Loading training data from: {features_dir}")
    dataset = VideoFeatureDataset(features_dir, split="train")
    
    if len(dataset) == 0:
        raise RuntimeError(
            f"No 'train_*.npz' files found in {features_dir}. "
            "Run extract_features.py first."
        )
    
    print(f"✅ Loaded {len(dataset)} training samples")
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    
    # Create model
    model = AnomalyDetector(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        num_classes=config.model.num_classes,
    ).to(device_obj)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        config=config,
        device=device,
        save_dir=save_dir,
    )
    
    # Resume if requested
    start_epoch = 0
    if resume_from:
        start_epoch = trainer.load_checkpoint(resume_from)
    
    # Train
    trained_model = trainer.train(start_epoch=start_epoch)
    
    return trained_model, trainer.loss_history