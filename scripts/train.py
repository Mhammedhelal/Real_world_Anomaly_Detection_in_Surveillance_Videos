"""
scripts/train.py
----------------
CLI entry point for training the UCF-Crime Anomaly Detector.

Usage
-----
    python scripts/train.py --features-dir data/features/extracted
    python scripts/train.py --features-dir data/features/extracted \\
        --epochs 50 --batch-size 16 --lr 0.001
    python scripts/train.py --features-dir data/features/extracted \\
        --resume checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.engine.trainer import train
from src.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the UCF-Crime anomaly detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features-dir", required=True,
                   help="Directory with train_*.npz feature files")
    p.add_argument("--save-dir", default="./outputs/checkpoints",
                   help="Directory for model checkpoints")
    p.add_argument("--config", default=None,
                   help="Path to YAML config file")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--resume", default=None,
                   help="Checkpoint path to resume training from")
    p.add_argument("--run-name", default=None,
                   help="Identifier written into log filenames")
    p.add_argument("--log-dir", default="outputs/logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Structured logging — must be called before any logger.info()
    log_path = setup_logging(log_dir=args.log_dir, run_name=args.run_name)
    logger = get_logger(__name__)

    # Config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        config = Config.from_yaml(config_path)

    overrides = {}
    if args.epochs:
        overrides.setdefault("training", {})["num_epochs"] = args.epochs
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        overrides.setdefault("optimizer", {})["learning_rate"] = args.lr
    if args.input_size:
        overrides.setdefault("model", {})["input_size"] = args.input_size
    if args.hidden_size:
        overrides.setdefault("model", {})["hidden_size"] = args.hidden_size
    if args.num_classes:
        overrides.setdefault("model", {})["num_classes"] = args.num_classes
    if overrides:
        config.merge(overrides)

    logger.info("features_dir=%s  save_dir=%s", args.features_dir, args.save_dir)
    logger.info("epochs=%d  batch_size=%d  lr=%s",
                config.training.num_epochs,
                config.training.batch_size,
                config.optimizer.learning_rate)

    try:
        trained_model, loss_history = train(
            features_dir=args.features_dir,
            save_dir=args.save_dir,
            config=config,
            device=args.device,
            resume_from=args.resume,
            run_name=args.run_name,
        )

        logger.info("Training complete. Final loss=%.4f  Best=%.4f",
                    loss_history[-1], min(loss_history))

        # Optional: plot training loss
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
            logger.info("Loss plot saved → %s", plot_path)
        except ImportError:
            pass

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
