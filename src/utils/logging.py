"""
src/utils/logging.py
--------------------
Structured logging for the anomaly detection project.

Why a dedicated module?
-----------------------
The original codebase mixed ``print()`` calls throughout training, evaluation,
and feature extraction.  This module provides:

  1. A single ``setup_logging()`` call that configures the root logger once.
  2. A ``TrainingLogger`` that writes epoch / batch summaries to both the
     terminal and a JSON-lines log file — making results reproducible and
     parseable.
  3. Helper functions used by the engine modules so they don't need to
     ``import logging`` and configure handlers themselves.

Usage
-----
In your main script (train.py, evaluate.py, etc.):

    from src.utils.logging import setup_logging, get_logger

    setup_logging(log_dir="outputs/logs", run_name="run_001")
    logger = get_logger(__name__)
    logger.info("Training started")

In library modules (trainer.py, evaluator.py, etc.):

    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Epoch %d  loss=%.4f", epoch, loss)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Module-level logger (used within this file only)
# ---------------------------------------------------------------------------
_module_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public: configure root logger
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str | Path = "outputs/logs",
    run_name: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> Path | None:
    """
    Configure the root logger with a console handler and an optional file
    handler.  Call this **once** at the start of your main script.

    Parameters
    ----------
    log_dir : str | Path
        Directory where log files are written.
    run_name : str | None
        Human-readable run identifier (e.g. ``"run_001"``).  If None, a
        timestamp is used.
    level : int
        Logging level (default: ``logging.INFO``).
    log_to_file : bool
        Whether to write logs to a ``.log`` file in *log_dir*.

    Returns
    -------
    Path | None
        Path to the log file, or None if ``log_to_file=False``.
    """
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Formatter
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers when called multiple times (e.g. in tests)
    if root.handlers:
        root.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    log_path = None
    if log_to_file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_name}.log"

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        _module_logger.info("Log file: %s", log_path)

    return log_path


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for *name* (typically ``__name__`` of the calling module).

    This is just a thin wrapper around ``logging.getLogger`` so callers
    do not need to import the standard ``logging`` module directly.
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# TrainingLogger — epoch / batch summary writer
# ---------------------------------------------------------------------------

class TrainingLogger:
    """
    Logs per-epoch training stats to the console AND a JSON-lines file.

    The JSON-lines format (one JSON object per line) is machine-readable
    and can be loaded later for plotting or analysis::

        import json
        with open("outputs/logs/run_001_metrics.jsonl") as f:
            records = [json.loads(line) for line in f]

    Parameters
    ----------
    log_dir : str | Path
        Directory where ``<run_name>_metrics.jsonl`` is written.
    run_name : str | None
        Identifier for this training run.
    print_every : int
        Print batch-level stats every N batches (0 = never).
    """

    def __init__(
        self,
        log_dir: str | Path = "outputs/logs",
        run_name: Optional[str] = None,
        print_every: int = 10,
    ) -> None:
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_name = run_name
        self.print_every = print_every
        self._logger = get_logger(f"training.{run_name}")

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = log_dir / f"{run_name}_metrics.jsonl"
        self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")

        self._epoch_start: float = 0.0
        self._batch_start: float = 0.0

        self._logger.info(
            "TrainingLogger initialised — metrics → %s", self._jsonl_path
        )

    # ------------------------------------------------------------------
    # Epoch-level hooks
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch: int, total_epochs: int) -> None:
        """Call at the beginning of each epoch."""
        self._epoch_start = time.time()
        self._logger.info(
            "Epoch [%d/%d] started", epoch, total_epochs
        )

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """
        Call at the end of each epoch with aggregated metrics.

        Parameters
        ----------
        epoch : int
        total_epochs : int
        metrics : dict[str, float]
            E.g. ``{"loss": 0.312, "ranking_loss": 0.205, ...}``.
        is_best : bool
            Whether this epoch achieved the lowest loss so far.
        """
        elapsed = time.time() - self._epoch_start
        tag = "  ★ best" if is_best else ""

        # Console summary
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self._logger.info(
            "Epoch [%d/%d] %s  |  %s  |  %.1fs%s",
            epoch,
            total_epochs,
            metric_str,
            self._eta(epoch, total_epochs, elapsed),
            elapsed,
            tag,
        )

        # JSON-lines record
        record: Dict[str, Any] = {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "elapsed_s": round(elapsed, 2),
            "is_best": is_best,
            **metrics,
        }
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

    # ------------------------------------------------------------------
    # Batch-level hooks
    # ------------------------------------------------------------------

    def on_batch_start(self) -> None:
        """Optional: call at batch start for timing."""
        self._batch_start = time.time()

    def on_batch_end(
        self,
        batch_idx: int,
        total_batches: int,
        metrics: Dict[str, float],
    ) -> None:
        """
        Call after each training batch.  Prints every *print_every* batches.
        """
        if self.print_every <= 0:
            return
        if batch_idx % self.print_every != 0:
            return

        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self._logger.debug(
            "  Batch [%d/%d] %s", batch_idx, total_batches, metric_str
        )

    # ------------------------------------------------------------------
    # Extra events
    # ------------------------------------------------------------------

    def log_checkpoint(self, path: str | Path, epoch: int) -> None:
        """Log that a checkpoint was saved."""
        self._logger.info("Checkpoint saved → %s  (epoch %d)", path, epoch)

    def log_eval(self, metrics: Dict[str, float], split: str = "test") -> None:
        """Log evaluation metrics."""
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self._logger.info("Eval [%s] %s", split, metric_str)

        record: Dict[str, Any] = {"type": "eval", "split": split, **metrics}
        self._jsonl_file.write(json.dumps(record) + "\n")
        self._jsonl_file.flush()

    def close(self) -> None:
        """Flush and close the JSON-lines file."""
        self._jsonl_file.flush()
        self._jsonl_file.close()
        self._logger.info("TrainingLogger closed.")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eta(epoch: int, total: int, elapsed: float) -> str:
        remaining = total - epoch
        if epoch == 0:
            return "ETA: --"
        secs = elapsed * remaining
        m, s = divmod(int(secs), 60)
        h, m = divmod(m, 60)
        return f"ETA: {h:02d}:{m:02d}:{s:02d}"

    @property
    def metrics_path(self) -> Path:
        """Path to the JSON-lines metrics file."""
        return self._jsonl_path
