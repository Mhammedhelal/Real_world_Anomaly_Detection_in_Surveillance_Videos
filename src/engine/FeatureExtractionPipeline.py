"""
src/engine/FeatureExtractionPipeline.py
----------------------------------------
Feature extraction pipeline — source-agnostic.

Data flow
---------
  AbstractFrameSource  →  VideoPreprocessor  →  FusionExtractor  →  Sink
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from src.data.sources.base import AbstractFrameSource
from src.data.sources.disk_source import DiskVideoSource
from src.models.video_preprocessor import VideoPreprocessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureExtractionPipeline:
    """
    Orchestrates AbstractFrameSource → features.

    Parameters
    ----------
    source : AbstractFrameSource
    preprocessor : VideoPreprocessor
    feature_extractor : object  (must implement extract_features(segments) → np.ndarray)
    features_dir : str | Path
    metadata_dir : str | Path
    device : str
    """

    def __init__(
        self,
        source: AbstractFrameSource,
        preprocessor: VideoPreprocessor,
        feature_extractor,
        features_dir: str | Path = 'data/features/extracted',
        metadata_dir: str | Path = 'data/features/metadata',
        device: str = 'cuda',
    ) -> None:
        self.source = source
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.features_dir = Path(features_dir)
        self.metadata_dir = Path(metadata_dir)
        self.device = device

        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self._progress_file = self.metadata_dir / 'extraction_progress.json'
        self._progress = self._load_progress()

        self._stats: Dict[str, Any] = {
            'total_videos': 0, 'successful': 0, 'failed': 0,
            'total_features': 0, 'total_size_mb': 0.0,
        }

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def _load_progress(self) -> dict:
        if self._progress_file.exists():
            with open(self._progress_file) as f:
                data = json.load(f)
            # Migrate legacy progress files that predate last_checkpoint
            data.setdefault('last_checkpoint', None)
            return data
        return {
            'processed': [],
            'failed': [],
            'last_checkpoint': None,   # filename of the last video we *started*
            'start_time': datetime.now().isoformat(),
        }

    def _save_progress(self) -> None:
        with open(self._progress_file, 'w') as f:
            json.dump(self._progress, f, indent=2)

    # ------------------------------------------------------------------
    # Core extraction — source-agnostic
    # ------------------------------------------------------------------

    def extract_from_source(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """Yield (features, metadata) pairs from the injected source."""
        accumulated_frames: List[np.ndarray] = []

        for frame_batch in self.source.stream():
            accumulated_frames.extend(frame_batch)

            if len(accumulated_frames) >= self.preprocessor.segment_length:
                segments = self.preprocessor.to_segments(accumulated_frames)
                if segments:
                    features = self.feature_extractor.extract_features(segments)
                    yield features, self.source.metadata()
                accumulated_frames = []

        if accumulated_frames:
            segments = self.preprocessor.to_segments(accumulated_frames)
            if segments:
                features = self.feature_extractor.extract_features(segments)
                yield features, self.source.metadata()

    # ------------------------------------------------------------------
    # Training mode — per-video extraction + save to disk
    # ------------------------------------------------------------------

    def extract_all_features(
        self,
        resume: bool = True,
        force_reprocess: bool = False,
        max_videos: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Extract features for every video provided by the DiskVideoSource.

        Resume behaviour
        ----------------
        When ``resume=True`` (default) and ``force_reprocess=False``:

        * Any video whose filename appears in ``progress['processed']`` is
          skipped — it completed successfully in a prior run.
        * Additionally, if the pipeline was interrupted mid-video, that video's
          filename is stored in ``progress['last_checkpoint']``.  On the next
          run the pipeline rewinds to that video and re-processes it from
          scratch, because we cannot know whether its .npz was written cleanly.
        * Videos that appear in ``progress['failed']`` are *retried* on every
          resume so that transient errors (e.g. a locked GPU) are self-healing.

        Pass ``force_reprocess=True`` (``--force`` on the CLI) to ignore all
        prior progress and re-extract everything.
        """
        if not isinstance(self.source, DiskVideoSource):
            raise TypeError(
                "extract_all_features() requires DiskVideoSource. "
                "For live inference use extract_from_source()."
            )

        video_paths = self.source._video_paths
        if max_videos:
            video_paths = video_paths[:max_videos]

        logger.info("Feature extraction: %d videos → %s", len(video_paths), self.features_dir)

        # Build a fast lookup of already-completed filenames
        completed: set[str] = {
            item['filename'] for item in self._progress['processed']
        }

        # The last video we *started* before a crash — must be re-processed
        # even if it somehow ended up in `completed` with stale data.
        interrupted_filename: Optional[str] = self._progress.get('last_checkpoint')
        if resume and interrupted_filename and not force_reprocess:
            logger.info(
                "Resuming from last checkpoint: %s  (will re-process to ensure integrity)",
                interrupted_filename,
            )
            # Remove it from completed so it gets re-processed below
            completed.discard(interrupted_filename)

        for video_path in video_paths:
            filename = video_path.name

            if not force_reprocess and resume and filename in completed:
                logger.info("Skipping (already processed): %s", filename)
                self._stats['successful'] += 1
                continue

            # --- Mark that we are *starting* this video before any work ---
            # If the process is killed mid-extraction, the next resume will
            # see this filename in last_checkpoint and re-process it.
            self._progress['last_checkpoint'] = filename
            self._save_progress()

            success = self._process_single_video(video_path)
            self._stats['total_videos'] += 1

            if success:
                self._stats['successful'] += 1
                # Clear the checkpoint only after a confirmed successful save
                self._progress['last_checkpoint'] = None
                self._save_progress()
            else:
                self._stats['failed'] += 1
                # Leave last_checkpoint set so the next resume retries this video

        self._print_summary()
        return self._stats['successful'], self._stats['failed']

    def _process_single_video(self, video_path: Path) -> bool:
        filename = video_path.name
        video_meta = self.source.video_metadata_for(video_path)
        logger.info("Processing: %s  label=%d (%s)",
                    filename, video_meta['label'], video_meta['class'])

        try:
            all_frames: List[np.ndarray] = []
            for frame_batch in self.source._stream_single_video(video_path):
                all_frames.extend(frame_batch)

            if not all_frames:
                logger.warning("No frames read from %s", filename)
                self._log_failure(filename, "no frames read")
                return False

            segments = self.preprocessor.to_segments(all_frames)
            if not segments:
                logger.warning("No segments created for %s", filename)
                self._log_failure(filename, "no segments created")
                return False

            t0 = time.time()
            features = self.feature_extractor.extract_features(segments)
            elapsed = time.time() - t0

            if features.shape[0] == 0:
                logger.warning("Empty feature array for %s", filename)
                self._log_failure(filename, "empty feature array")
                return False

            logger.info(
                "%s: %d segments × %d features  (%.1fs)",
                filename, features.shape[0], features.shape[1], elapsed,
            )
            self._save_npz(features, video_meta)

            self._progress['processed'].append({
                'filename': filename,
                'split': video_meta['split'],
                'features_shape': list(features.shape),
                'extraction_time': elapsed,
                'timestamp': datetime.now().isoformat(),
            })
            self._save_progress()

            del all_frames, segments, features
            torch.cuda.empty_cache()
            return True

        except Exception as exc:
            import traceback
            logger.error("Error processing %s: %s", filename, exc)
            traceback.print_exc()
            self._log_failure(filename, str(exc))
            return False

    # ------------------------------------------------------------------
    # .npz saving
    # ------------------------------------------------------------------

    def _save_npz(self, features: np.ndarray, video_meta: dict) -> Path:
        split = video_meta.get('split', 'train')
        stem = Path(video_meta['filename']).stem
        out_path = self.features_dir / f"{split}_{stem}.npz"

        metadata = {
            **video_meta,
            'feature_dim': features.shape[1],
            'num_segments': features.shape[0],
            'extraction_time': datetime.now().isoformat(),
            'dataset_type': 'normal_only' if video_meta['label'] == 0 else 'anomalous',
        }

        np.savez_compressed(out_path, features=features.astype(np.float32), metadata=metadata)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        self._stats['total_features'] += 1
        self._stats['total_size_mb'] += size_mb
        logger.info("Saved: %s  (%.2f MB)", out_path.name, size_mb)
        return out_path

    def _log_failure(self, filename: str, reason: str) -> None:
        self._progress['failed'].append({
            'filename': filename,
            'error': reason,
            'timestamp': datetime.now().isoformat(),
        })
        self._save_progress()

    def _print_summary(self) -> None:
        s = self._stats
        logger.info(
            "Extraction complete: %d total  %d ok  %d failed  %.2f MB",
            s['total_videos'], s['successful'], s['failed'], s['total_size_mb'],
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def check_status(self) -> None:
        npz_files = list(self.features_dir.glob('*.npz'))
        train_files = [f for f in npz_files if f.name.startswith('train_')]
        test_files = [f for f in npz_files if f.name.startswith('test_')]
        total_mb = sum(f.stat().st_size for f in npz_files) / (1024 * 1024)

        logger.info(
            "Status [%s]: %d files  (%d train, %d test)  %.2f MB  "
            "processed=%d  failed=%d  last_checkpoint=%s",
            self.features_dir,
            len(npz_files), len(train_files), len(test_files), total_mb,
            len(self._progress['processed']), len(self._progress['failed']),
            self._progress.get('last_checkpoint') or 'none',
        )


# ---------------------------------------------------------------------------
# Backward-compat aliases (scripts imported these from the old engine __init__)
# ---------------------------------------------------------------------------

def process_in_batches(*args, **kwargs):
    raise NotImplementedError("Use FeatureExtractionPipeline.extract_all_features() instead.")


def check_status(pipeline: FeatureExtractionPipeline) -> None:
    pipeline.check_status()