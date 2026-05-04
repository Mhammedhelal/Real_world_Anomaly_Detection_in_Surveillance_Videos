"""
scripts/generate_visualizations.py
------------------------------------
Generate all evaluation visualizations and save them to outputs/evaluation/.

Reads a saved evaluation results JSON (produced by scripts/evaluate.py
--save-dir) and an optional loss-history JSON, then calls every plot
function in src/utils/visualization.py.

Usage
-----
    # Minimal — reads results from outputs/evaluation/
    python scripts/generate_visualizations.py

    # Explicit paths
    python scripts/generate_visualizations.py \\
        --results-json  outputs/evaluation/evaluation_results.json \\
        --loss-json     outputs/logs/loss_history.json \\
        --save-dir      outputs/evaluation \\
        --checkpoint    outputs/checkpoints/best_model.pt \\
        --features-dir  data/features/extracted

    # Skip dashboard (useful when loss history is unavailable)
    python scripts/generate_visualizations.py --no-dashboard

Outputs
-------
    outputs/evaluation/
        roc_curve.png
        confusion_matrix.png
        per_class_accuracy.png
        training_loss.png        (only if --loss-json is given)
        dashboard.png            (only if --loss-json is given)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make src.* importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logging, get_logger
from src.utils.visualization import (
    ANOMALY_CLASSES,
    generate_all_evaluation_plots,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_roc_curve,
    plot_training_loss,
    plot_evaluation_dashboard,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all evaluation visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-json",
        default="outputs/evaluation/evaluation_results.json",
        help="Path to evaluation_results.json written by scripts/evaluate.py",
    )
    p.add_argument(
        "--loss-json",
        default=None,
        help=(
            "Path to a JSON file containing a list of per-epoch losses "
            "(e.g. outputs/logs/loss_history.json).  "
            "If omitted the script tries to load "
            "outputs/logs/loss_history.json automatically."
        ),
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to a .pt checkpoint.  When given, the script re-runs "
            "evaluation on --features-dir and generates plots from fresh results "
            "(overrides --results-json)."
        ),
    )
    p.add_argument(
        "--features-dir",
        default=None,
        help="Feature directory required when --checkpoint is supplied.",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate (used with --checkpoint).",
    )
    p.add_argument(
        "--save-dir",
        default="outputs/evaluation",
        help="Directory where PNG files are written.",
    )
    p.add_argument(
        "--log-dir",
        default="outputs/logs",
        help="Directory for log files.",
    )
    p.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Skip the combined dashboard plot.",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Device for re-evaluation (only used with --checkpoint).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results_from_json(path: Path) -> dict:
    """Load evaluation results dict from the JSON written by Evaluator.save_results()."""
    with open(path) as f:
        data = json.load(f)

    # The JSON has scalar metrics + lists; rebuild numpy arrays where needed
    results = {
        "auc_roc":  float(data.get("auc_roc", data.get("auc", 0.0))),
        "auc_pr":   float(data.get("auc_pr", 0.0)),
        "accuracy": float(data.get("accuracy", 0.0)),
        "per_class_accuracy": data.get("per_class_accuracy", {}),
        "confusion_matrix": np.array(data["confusion_matrix"]),
    }

    # anomaly_scores / binary_labels may be absent in a summary JSON;
    # reconstruct dummy arrays so ROC plotting still works gracefully
    if "anomaly_scores" in data:
        results["anomaly_scores"] = np.array(data["anomaly_scores"])
        results["binary_labels"]  = np.array(data["binary_labels"])
    else:
        # Build a minimal synthetic ROC from auc_roc for display purposes
        results["anomaly_scores"] = np.array([])
        results["binary_labels"]  = np.array([])

    return results


def _load_losses(loss_json: Path | None) -> list | None:
    """Try loading loss history; return None if not available."""
    candidates = [
        loss_json,
        Path("outputs/logs/loss_history.json"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            with open(p) as f:
                data = json.load(f)
            # Accept a plain list or {"losses": [...]}
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "losses" in data:
                return data["losses"]
    return None


def _run_fresh_evaluation(checkpoint: str, features_dir: str,
                           split: str, device: str | None) -> dict:
    """Run Evaluator and return results dict (includes anomaly_scores etc.)."""
    from src.engine.evaluator import Evaluator
    ev = Evaluator(
        features_dir=features_dir,
        checkpoint_path=checkpoint,
        device=device,
    )
    return ev.evaluate(split=split)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    setup_logging(log_dir=args.log_dir, run_name="generate_visualizations")
    logger = get_logger(__name__)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Obtain evaluation results ─────────────────────────────────────────
    if args.checkpoint:
        if not args.features_dir:
            logger.error("--features-dir is required when --checkpoint is supplied.")
            sys.exit(1)
        logger.info("Re-running evaluation from checkpoint: %s", args.checkpoint)
        results = _run_fresh_evaluation(
            args.checkpoint, args.features_dir, args.split, args.device
        )
    else:
        results_path = Path(args.results_json)
        if not results_path.exists():
            logger.error(
                "Results JSON not found: %s\n"
                "Run scripts/evaluate.py --save-dir %s first, or supply "
                "--checkpoint + --features-dir to re-evaluate.",
                results_path, args.save_dir,
            )
            sys.exit(1)
        logger.info("Loading evaluation results from: %s", results_path)
        results = _load_results_from_json(results_path)

    # ── 2. Load loss history (optional) ──────────────────────────────────────
    losses = _load_losses(
        Path(args.loss_json) if args.loss_json else None
    )
    if losses:
        logger.info("Loaded %d epoch losses.", len(losses))
    else:
        logger.info(
            "No loss history found — training_loss.png and dashboard.png "
            "will be skipped.  Supply --loss-json to include them."
        )

    auc_roc   = results["auc_roc"]
    auc_pr    = results.get("auc_pr", 0.0)
    accuracy  = results.get("accuracy")
    cm        = results["confusion_matrix"]
    per_class = results["per_class_accuracy"]

    logger.info(
        "Results summary: AUC-ROC=%.4f  AUC-PR=%.4f  Accuracy=%.2f%%",
        auc_roc, auc_pr, (accuracy or 0) * 100,
    )

    # ── 3. ROC Curve ──────────────────────────────────────────────────────────
    y_score = results.get("anomaly_scores", np.array([]))
    y_true  = results.get("binary_labels",  np.array([]))

    if len(y_score) > 0:
        logger.info("Generating ROC curve …")
        plot_roc_curve(
            y_true=y_true,
            y_score=y_score,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            accuracy=accuracy,
            save_path=save_dir / "roc_curve.png",
        )
    else:
        logger.warning(
            "anomaly_scores not in results — ROC curve skipped. "
            "Use --checkpoint + --features-dir for full re-evaluation."
        )

    # ── 4. Confusion Matrix ───────────────────────────────────────────────────
    logger.info("Generating confusion matrix …")
    plot_confusion_matrix(
        confusion_matrix=cm,
        class_names=ANOMALY_CLASSES[:cm.shape[0]],
        normalize=True,
        save_path=save_dir / "confusion_matrix.png",
    )

    # ── 5. Per-Class Accuracy ─────────────────────────────────────────────────
    if per_class:
        logger.info("Generating per-class accuracy chart …")
        plot_per_class_accuracy(
            per_class_accuracy=per_class,
            save_path=save_dir / "per_class_accuracy.png",
        )
    else:
        logger.warning("per_class_accuracy missing from results — chart skipped.")

    # ── 6. Training Loss ──────────────────────────────────────────────────────
    if losses:
        logger.info("Generating training loss curve …")
        plot_training_loss(
            losses=losses,
            save_path=save_dir / "training_loss.png",
        )

    # ── 7. Dashboard ─────────────────────────────────────────────────────────
    if losses and not args.no_dashboard and len(y_score) > 0 and per_class:
        logger.info("Generating combined dashboard …")
        plot_evaluation_dashboard(
            losses=losses,
            y_true=y_true,
            y_score=y_score,
            confusion_matrix=cm,
            per_class_accuracy=per_class,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            accuracy=accuracy,
            class_names=ANOMALY_CLASSES[:cm.shape[0]],
            save_path=save_dir / "dashboard.png",
        )
    elif args.no_dashboard:
        logger.info("Dashboard skipped (--no-dashboard).")
    else:
        logger.info(
            "Dashboard skipped (need loss history + anomaly_scores + per_class)."
        )

    # ── 8. Summary ────────────────────────────────────────────────────────────
    generated = sorted(save_dir.glob("*.png"))
    logger.info(
        "Done — %d plot(s) saved to %s:\n  %s",
        len(generated),
        save_dir,
        "\n  ".join(p.name for p in generated),
    )
    print(f"\n✅  {len(generated)} plot(s) saved to {save_dir}/")
    for p in generated:
        print(f"   • {p.name}")


if __name__ == "__main__":
    main()