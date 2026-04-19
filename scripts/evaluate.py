"""
scripts/evaluate.py
-------------------
CLI entry point for evaluating the UCF-Crime Anomaly Detector.

Usage
-----
    python scripts/evaluate.py \\
        --features-dir data/features/extracted \\
        --checkpoint outputs/checkpoints/best_model.pt

    python scripts/evaluate.py \\
        --features-dir data/features/extracted \\
        --checkpoint outputs/checkpoints/best_model.pt \\
        --split train --save-dir outputs/evaluation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.engine.evaluator import evaluate
from src.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the UCF-Crime anomaly detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--log-dir", default="outputs/logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging(log_dir=args.log_dir, run_name="evaluate")
    logger = get_logger(__name__)

    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        config = Config.from_yaml(config_path)

    overrides = {}
    if args.batch_size:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.num_classes:
        overrides.setdefault("model", {})["num_classes"] = args.num_classes
    if overrides:
        config.merge(overrides)

    batch_size = args.batch_size or config.training.batch_size
    num_classes = args.num_classes or config.model.num_classes

    logger.info(
        "Evaluating: features=%s  checkpoint=%s  split=%s",
        args.features_dir, args.checkpoint, args.split,
    )

    try:
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

        logger.info(
            "AUC=%.4f  Accuracy=%.2f%%", results["auc"], results["accuracy"] * 100
        )

        if args.save_dir:
            logger.info("Results saved → %s", args.save_dir)

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
