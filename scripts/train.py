"""
scripts/train.py
----------------
CLI entrypoint for training the UCF-Crime Anomaly Detector.

Handles:
  - Argument parsing
  - Configuration loading and CLI overrides
  - Orchestration (delegates to src.engine.trainer)

Usage:
    # Use defaults from config
    python scripts/train.py --features-dir data/features/extracted

    # Override config values
    python scripts/train.py \
        --features-dir data/features/extracted \
        --epochs 50 \
        --batch-size 16 \
        --lr 0.001

    # Resume training
    python scripts/train.py \
        --features-dir data/features/extracted \
        --resume checkpoints/best_model.pt

    # Use custom config
    python scripts/train.py \
        --features-dir data/features/extracted \
        --config configs/custom.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.engine.trainer import train


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the UCF-Crime anomaly detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory containing .npz feature files (train_*.npz)",
    )
    
    # Output directory
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    
    # Training parameters (override config)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    
    # Model parameters (override config)
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Feature vector dimension (overrides config)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Bi-GRU hidden size (overrides config)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of crime categories (overrides config)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (auto-detected if not specified)",
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    # Logging
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Print training stats every N batches (overrides config)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (overrides config)",
    )
    
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> Config:
    """
    Load configuration and apply CLI overrides.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    
    Returns
    -------
    Config
        Configuration object with overrides applied
    """
    # Load base config
    if args.config:
        config = Config.from_yaml(args.config)
        print(f"📄 Loaded config from: {args.config}")
    else:
        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = Config.from_yaml(config_path)
        print(f"📄 Loaded default config from: {config_path}")
    
    # Apply CLI overrides
    overrides = {}
    
    # Training parameters
    if args.epochs is not None:
        overrides.setdefault('training', {})['num_epochs'] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault('training', {})['batch_size'] = args.batch_size
    if args.lr is not None:
        overrides.setdefault('optimizer', {})['learning_rate'] = args.lr
    
    # Model parameters
    if args.input_size is not None:
        overrides.setdefault('model', {})['input_size'] = args.input_size
    if args.hidden_size is not None:
        overrides.setdefault('model', {})['hidden_size'] = args.hidden_size
    if args.num_classes is not None:
        overrides.setdefault('model', {})['num_classes'] = args.num_classes
    
    # Logging parameters
    if args.log_interval is not None:
        overrides.setdefault('logging', {})['log_interval'] = args.log_interval
    if args.save_every is not None:
        overrides.setdefault('logging', {})['save_interval'] = args.save_every
    
    # Apply overrides
    if overrides:
        print(f"⚙️  Applying CLI overrides: {overrides}")
        config.merge(overrides)
    
    return config


def main() -> None:
    """Main training entrypoint."""
    # Parse arguments
    args = parse_args()
    
    # Load config with overrides
    config = load_config_with_overrides(args)
    
    # Print configuration summary
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Features dir:    {args.features_dir}")
    print(f"Save dir:        {args.save_dir}")
    print(f"Epochs:          {config.training.num_epochs}")
    print(f"Batch size:      {config.training.batch_size}")
    print(f"Learning rate:   {config.optimizer.learning_rate}")
    print(f"Optimizer:       {config.optimizer.type}")
    print(f"Input size:      {config.model.input_size}")
    print(f"Hidden size:     {config.model.hidden_size}")
    print(f"Num classes:     {config.model.num_classes}")
    print(f"Device:          {args.device or 'auto'}")
    if args.resume:
        print(f"Resume from:     {args.resume}")
    print("=" * 70 + "\n")
    
    try:
        # Train model (delegates to engine)
        trained_model, loss_history = train(
            features_dir=args.features_dir,
            save_dir=args.save_dir,
            config=config,
            device=args.device,
            resume_from=args.resume,
        )
        
        # Plot training loss
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 4))
            plt.plot(loss_history, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = Path(args.save_dir) / "training_loss.png"
            plt.savefig(plot_path, dpi=150)
            print(f"\n📈 Training loss plot saved to: {plot_path}")
            plt.show()
        except ImportError:
            print("\n⚠️  matplotlib not installed. Skipping loss plot.")
        except Exception as e:
            print(f"\n⚠️  Could not generate loss plot: {e}")
        
        print("\n✅ Training completed successfully!")
        print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"Best loss:  {min(loss_history):.4f}")
        print(f"Checkpoints saved to: {args.save_dir}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()