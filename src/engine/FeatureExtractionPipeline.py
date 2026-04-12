"""
src/engine/FeatureExtractionPipeline.py
----------------------------------------
Feature extraction pipeline — fully decoupled from data source.

Data flow
---------
  AbstractFrameSource                (disk OR camera — pipeline doesn't care)
      ↓  List[np.ndarray]  raw RGB frames
  VideoPreprocessor.to_segments()
      ↓  List[Tensor]  normalised, fixed-length clips
  FusionExtractor.extract()
      ↓  np.ndarray  [num_segments, feature_dim]
  Sink (training mode)  →  save .npz to disk
  Sink (inference mode) →  return features / pass to model

Design
------
The pipeline never imports cv2, never opens a file, never constructs a
VideoCapture.  All of that is hidden behind AbstractFrameSource.

The two concrete flows are:

  Training (offline, batch):
    DiskVideoSource → FeatureExtractionPipeline → DiskSink (.npz)

  Inference (real-time):
    CameraStreamSource → FeatureExtractionPipeline → ModelInferenceSink

The pipeline code is identical for both.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from src.data.sources.base import AbstractFrameSource
from src.data.sources.disk_source import DiskVideoSource
from src.models.video_preprocessor import VideoPreprocessor
from src.config import Config


# ---------------------------------------------------------------------------
# FeatureExtractionPipeline
# ---------------------------------------------------------------------------

class FeatureExtractionPipeline:
    """
    Orchestrates the flow from AbstractFrameSource → features.

    The pipeline does NOT know whether the source is a disk directory or
    a live camera.  It consumes the AbstractFrameSource.stream() iterator
    and calls the preprocessor and extractor on each batch.

    Parameters
    ----------
    source : AbstractFrameSource
        Any frame source (DiskVideoSource, CameraStreamSource, …).
    preprocessor : VideoPreprocessor
        Converts raw RGB frames to normalised segment tensors.
    feature_extractor : object
        Must implement extract_features(segments) → np.ndarray.
        Typically a TwoStreamFeatureExtractor / FusionExtractor.
    features_dir : str | Path
        Where to save .npz files (training mode only).
    metadata_dir : str | Path
        Where to save extraction progress JSON.
    device : str
        'cuda' or 'cpu'.
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

        self._stats: Dict[str, int | float] = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'total_features': 0,
            'total_size_mb': 0.0,
        }

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def _load_progress(self) -> dict:
        if self._progress_file.exists():
            with open(self._progress_file) as f:
                return json.load(f)
        return {
            'processed': [],
            'failed': [],
            'start_time': datetime.now().isoformat(),
        }

    def _save_progress(self) -> None:
        with open(self._progress_file, 'w') as f:
            json.dump(self._progress, f, indent=2)

    # ------------------------------------------------------------------
    # Core extraction — source-agnostic
    # ------------------------------------------------------------------

    def extract_from_source(self) -> Iterator[Tuple[np.ndarray, dict]]:
        """
        Stream features from whatever source was injected.

        Yields
        ------
        (features, metadata) pairs where:
            features : np.ndarray  [num_segments, feature_dim]
            metadata : dict        from source.metadata()

        The caller decides what to do with each pair (save to disk,
        feed to model, buffer in FeatureBuffer, …).

        This method works identically for DiskVideoSource and
        CameraStreamSource — the pipeline never branches on source type.
        """
        accumulated_frames: List[np.ndarray] = []

        for frame_batch in self.source.stream():
            # Accumulate raw frames
            accumulated_frames.extend(frame_batch)

            # Once we have enough for at least one segment, process
            if len(accumulated_frames) >= self.preprocessor.segment_length:
                segments = self.preprocessor.to_segments(accumulated_frames)
                if segments:
                    features = self.feature_extractor.extract_features(segments)
                    yield features, self.source.metadata()
                accumulated_frames = []

        # Process any leftover frames (tail of last video / stream flush)
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
        Extract features from all videos in a DiskVideoSource and save
        them as .npz files.

        Designed for offline/training use.  Uses DiskVideoSource.iter_videos()
        so each video's features + metadata stay correctly paired.

        Parameters
        ----------
        resume : bool
            Skip videos already in the progress log.
        force_reprocess : bool
            Reprocess even if already logged.
        max_videos : int | None
            Process at most this many videos (for testing).

        Returns
        -------
        (successful, failed) counts.
        """
        if not isinstance(self.source, DiskVideoSource):
            raise TypeError(
                "extract_all_features() requires a DiskVideoSource. "
                "For live inference use extract_from_source() instead."
            )

        video_paths = self.source._video_paths
        if max_videos:
            video_paths = video_paths[:max_videos]

        print("\n" + "=" * 70)
        print(f"FEATURE EXTRACTION — {len(video_paths)} videos")
        print(f"Source   : {self.source}")
        print(f"Output   : {self.features_dir}")
        print("=" * 70)

        for video_path in video_paths:
            filename = video_path.name

            already_done = any(
                item['filename'] == filename
                for item in self._progress['processed']
            )
            if resume and already_done and not force_reprocess:
                print(f"⏭️  Skipping (already processed): {filename}")
                self._stats['successful'] += 1
                continue

            success = self._process_single_video(video_path)
            self._stats['total_videos'] += 1
            if success:
                self._stats['successful'] += 1
            else:
                self._stats['failed'] += 1

        self._print_summary()
        return self._stats['successful'], self._stats['failed']

    def _process_single_video(self, video_path: Path) -> bool:
        """
        Extract features for one video file and save to .npz.

        The source streams frames; this method processes and saves them.
        """
        filename = video_path.name
        video_meta = self.source.video_metadata_for(video_path)

        print(f"\n{'='*60}")
        print(f"Processing : {filename}")
        print(f"Label      : {video_meta['label']} ({video_meta['class']})")
        print(f"{'='*60}")

        try:
            # Collect all frames for this video from the source
            all_frames: List[np.ndarray] = []
            for frame_batch in self.source._stream_single_video(video_path):
                all_frames.extend(frame_batch)

            if not all_frames:
                print(f"❌ No frames read from {filename}")
                self._log_failure(filename, "no frames read")
                return False

            # Preprocess: frames → segments
            segments = self.preprocessor.to_segments(all_frames)
            if not segments:
                print(f"❌ No segments created for {filename}")
                self._log_failure(filename, "no segments created")
                return False

            # Extract features
            print(f"🔍 Extracting features for {len(segments)} segments …")
            t0 = time.time()
            features = self.feature_extractor.extract_features(segments)
            elapsed = time.time() - t0

            if features.shape[0] == 0:
                print(f"❌ Empty feature array for {filename}")
                self._log_failure(filename, "empty feature array")
                return False

            print(f"✅ {features.shape[0]} segments × {features.shape[1]} features  ({elapsed:.1f}s)")

            # Save .npz
            self._save_npz(features, video_meta)

            # Log progress
            self._progress['processed'].append({
                'filename':        filename,
                'split':           video_meta['split'],
                'features_shape':  list(features.shape),
                'extraction_time': elapsed,
                'timestamp':       datetime.now().isoformat(),
            })
            self._save_progress()

            # Free GPU memory
            del all_frames, segments, features
            torch.cuda.empty_cache()

            return True

        except Exception as exc:
            import traceback
            print(f"❌ Error processing {filename}: {exc}")
            traceback.print_exc()
            self._log_failure(filename, str(exc))
            return False

    # ------------------------------------------------------------------
    # .npz saving (training sink — only called from training path)
    # ------------------------------------------------------------------

    def _save_npz(self, features: np.ndarray, video_meta: dict) -> Path:
        """Save feature array + metadata to a compressed .npz file."""
        split    = video_meta.get('split', 'train')
        stem     = Path(video_meta['filename']).stem
        out_path = self.features_dir / f"{split}_{stem}.npz"

        metadata = {
            **video_meta,
            'feature_dim':   features.shape[1],
            'num_segments':  features.shape[0],
            'extraction_time': datetime.now().isoformat(),
            'dataset_type':  'normal_only' if video_meta['label'] == 0 else 'anomalous',
        }

        np.savez_compressed(
            out_path,
            features=features.astype(np.float32),
            metadata=metadata,
        )

        size_mb = out_path.stat().st_size / (1024 * 1024)
        self._stats['total_features'] += 1
        self._stats['total_size_mb'] += size_mb

        print(f"💾 Saved: {out_path.name}  ({size_mb:.2f} MB)")
        return out_path

    def _log_failure(self, filename: str, reason: str) -> None:
        self._progress['failed'].append({
            'filename':  filename,
            'error':     reason,
            'timestamp': datetime.now().isoformat(),
        })
        self._save_progress()

    def _print_summary(self) -> None:
        s = self._stats
        print(f"\n{'='*70}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total videos      : {s['total_videos']}")
        print(f"Successful        : {s['successful']}")
        print(f"Failed            : {s['failed']}")
        print(f"Feature files     : {s['total_features']}")
        print(f"Storage used      : {s['total_size_mb']:.2f} MB")

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def check_status(self) -> None:
        """Print a summary of what has been extracted so far."""
        npz_files = list(self.features_dir.glob('*.npz'))
        train_files = [f for f in npz_files if f.name.startswith('train_')]
        test_files  = [f for f in npz_files if f.name.startswith('test_')]
        total_mb = sum(f.stat().st_size for f in npz_files) / (1024 * 1024)

        print(f"\n{'='*70}")
        print(f"STATUS — {self.features_dir}")
        print(f"{'='*70}")
        print(f"Feature files : {len(npz_files)}")
        print(f"  • Train     : {len(train_files)}")
        print(f"  • Test      : {len(test_files)}")
        print(f"Storage       : {total_mb:.2f} MB")
        print(f"Processed log : {len(self._progress['processed'])} entries")
        print(f"Failed log    : {len(self._progress['failed'])} entries")