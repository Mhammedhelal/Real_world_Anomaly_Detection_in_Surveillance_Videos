"""
scripts/evaluate.py
-------------------
CLI entrypoint for evaluating the UCF-Crime Anomaly Detector.

Handles:
  - Argument parsing
  - Configuration loading and CLI overrides
  - Orchestration (delegates to src.engine.evaluator)

Usage:
    # Basic evaluation
    python scripts/evaluate.py \
        --features-dir data/features/extracted \
        --checkpoint checkpoints/best_model.pt

    # Evaluate on train set (for debugging)
    python scripts/evaluate.py \
        --features-dir data/features/extracted \
        --checkpoint checkpoints/best_model.pt \
        --split train

    # Save plots and results
    python scripts/evaluate.py \
        --features-dir data/features/extracted \
        --checkpoint checkpoints/best_model.pt \
        --save-dir outputs/evaluation

    # Override batch size
    python scripts/evaluate.py \
        --features-dir data/features/extracted \
        --checkpoint checkpoints/best_model.pt \
        --batch-size 32

    # Use custom config
    python scripts/evaluate.py \
        --features-dir data/features/extracted \
        --checkpoint checkpoints/best_model.pt \
        --config configs/custom.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.engine.evaluator import evaluate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the UCF-Crime anomaly detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory containing .npz feature files",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of crime categories (overrides config)",
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (auto-detected if not specified)",
    )
    
    # Output options
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save evaluation plots and results",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable interactive plot display",
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
    
    if args.batch_size is not None:
        overrides.setdefault('training', {})['batch_size'] = args.batch_size
    if args.num_classes is not None:
        overrides.setdefault('model', {})['num_classes'] = args.num_classes
    
    # Apply overrides
    if overrides:
        print(f"⚙️  Applying CLI overrides: {overrides}")
        config.merge(overrides)
    
    return config


def main() -> None:
    """Main evaluation entrypoint."""
    # Parse arguments
    args = parse_args()
    
    # Load config with overrides
    config = load_config_with_overrides(args)
    
    # Get effective values
    batch_size = args.batch_size or config.training.batch_size
    num_classes = args.num_classes or config.model.num_classes
    
    # Print configuration summary
    print("\n" + "=" * 70)
    print("EVALUATION CONFIGURATION")
    print("=" * 70)
    print(f"Features dir:    {args.features_dir}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Split:           {args.split}")
    print(f"Batch size:      {batch_size}")
    print(f"Num classes:     {num_classes}")
    print(f"Device:          {args.device or 'auto'}")
    if args.save_dir:
        print(f"Save dir:        {args.save_dir}")
    print("=" * 70 + "\n")
    
    try:
        # Run evaluation (delegates to engine)
        results = evaluate(
            features_dir=args.features_dir,
            checkpoint_path=args.checkpoint,
            split=args.split,
            batch_size=batch_size,
            num_classes=num_classes,
            device=args.device,
            plot=not args.no_plot,
            save_dir=args.save_dir,
            config=config,
        )
        
        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"AUC-ROC:         {results['auc']:.4f}")
        print(f"Accuracy:        {results['accuracy']*100:.2f}%")
        print("\nPer-class accuracy:")
        for cls_name, acc in results['per_class_accuracy'].items():
            print(f"  {cls_name:<15} {acc*100:.1f}%")
        print("=" * 70)
        
        if args.save_dir:
            print(f"\n💾 Results saved to: {args.save_dir}")
        
        print("\n✅ Evaluation completed successfully!")
    
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()