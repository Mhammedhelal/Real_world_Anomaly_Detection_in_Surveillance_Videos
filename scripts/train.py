"""
scripts/train.py
----------------

Training script for the anomaly detection model.

Responsibilities
----------------

1. Load configuration
2. Set random seeds
3. Build datasets
4. Build dataloaders
5. Build model
6. Build optimizer
7. Build trainer
8. Run training loop
"""

import argparse
from pathlib import Path
import random
import sys
from typing import List, Optional

from torch.utils.data import DataLoader
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import Config
from src.data.dataset import VideoDataset, collate_fn_fixed_length, collate_fn_variable_length, create_dataloaders
from src.models import AnomalyDetector
from src.engine.trainer import Trainer


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def set_seed(seed: int):
    """
    Ensure reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def build_model(cfg: Config) -> AnomalyDetector:
    return AnomalyDetector(
        input_size = cfg.model.input_size,      # 2131
        hidden_size = cfg.model.hidden_size,    # 256
        num_classes = cfg.model.num_classes     # 14
    )

def build_loader(
    cfg : Config,
    features_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    collate_type: str = 'variable_length',
    fixed_length: Optional[int] = None
) -> DataLoader:
    """
    Create training dataloader from extracted features.
    
    Args:
        features_dir: Directory containing .npz feature files
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle training data
        collate_type: 'variable_length' or 'fixed_length'
        fixed_length: Length for fixed-length collate (required if collate_type='fixed_length')
    
    Returns:
        train_loader
    """
    import os
    
    all_features = []
    all_labels = []
    
    # Load all features from directory
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")
    
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
    
    for filename in sorted(feature_files):
        file_path = os.path.join(features_dir, filename)
        data = np.load(file_path, allow_pickle=True)
        
        # Load features: Shape [Segments, 2131]
        features = data['features']
        
        # Extract metadata to get label
        metadata = data['metadata'].item()
        label = int(metadata['label'])
        
        all_features.append(torch.FloatTensor(features))
        all_labels.append(label)
    
    if len(all_features) == 0:
        raise ValueError(f"No feature files found in {features_dir}")
    
    print(f"✅ Loaded {len(all_features)} video feature sets from {features_dir}")
    
    # Create dataset
    dataset = VideoDataset(all_features, all_labels)
    
    # Choose collate function
    if collate_type == 'fixed_length':
        if fixed_length is None:
            raise ValueError("fixed_length must be specified for fixed_length collate")
        collate_fn = lambda batch: collate_fn_fixed_length(batch, fixed_length)
    else:
        collate_fn = collate_fn_variable_length
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader

def build_trainer(
    cfg:    Config,
    model:  AnomalyDetector,
    loader: DataLoader,
) -> Trainer:
    return Trainer(
        model = model,
        train_loader = loader,
        device = cfg.hardware.device,
        learning_rate = cfg.optimizer.learning_rate,
        num_epochs = cfg.training.num_epochs,
        rho = cfg.optimizer.rho,
        eps = cfg.optimizer.eps
    )
# -------------------------------------------------
# CLI
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the anomaly detector model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--lr",            type=float, default=None, help="Learning rate (both stages)")
    parser.add_argument("--batch_size",    type=int,   default=None, help="Batch size (both stages)")
    parser.add_argument("--device",        type=str,   default=None, help="cuda | cpu")
    parser.add_argument("--num_epochs",    type=int,   default=None, help="Epochs per stage")
    parser.add_argument("--num_subgroups", type=int,   default=None, help="1 | 2 | 4")
    return parser.parse_args()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    args = parse_args()

    # -- Config ---------------------------------------------
    cfg = Config.from_yaml(args.config)

    overrides: dict = {}
    if args.lr is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["lr"] = args.lr
        overrides.setdefault("training", {}).setdefault("stage2", {})["lr"] = args.lr
    if args.batch_size is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["batch_size"] = args.batch_size
        overrides.setdefault("training", {}).setdefault("stage2", {})["batch_size"] = args.batch_size
    if args.device is not None:
        overrides.setdefault("training", {})["device"] = args.device
    if args.num_epochs is not None:
        overrides.setdefault("training", {}).setdefault("stage1", {})["epochs"] = args.num_epochs
        overrides.setdefault("training", {}).setdefault("stage2", {})["epochs"] = args.num_epochs
    if args.num_subgroups is not None:
        overrides.setdefault("pooling", {})["num_subgroups"] = args.num_subgroups
    if overrides:
        cfg.merge(overrides)

    # -- Reproducibility ---------------------------------------------
    set_seed(cfg.training.seed)

    # -- Persist run config ---------------------------------------------
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(Path(cfg.paths.output_dir) / "run_config.yaml")
    print(cfg)

    # -- Data ---------------------------------------------
    train_loader = build_loader(
        cfg,
        features_dir = cfg.dataset.input_video_dir,
        batch_size = cfg.training.batch_size,
        num_workers= cfg.training.num_workers,
        pin_memory = cfg.training.pin_memory,
        shuffle = True,
        collate_type = cfg.dataset.collate_type,
        fixed_length = cfg.dataset.collate_fixed_length
    )

    # -- Model ---------------------------------------------
    model = build_model(cfg)

    # -- Train ---------------------------------------------
    build_trainer(cfg, model,train_loader).train()

if __name__ == "__main__":
    main()