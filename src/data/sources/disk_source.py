"""
src/sources/disk_source.py
--------------------------
Disk-based frame source.

ALL disk I/O that was previously spread across FeatureExtractionPipeline
and VideoPreprocessor is now centralised here.  The pipeline and
preprocessor receive only raw RGB numpy frames — they never touch a file
path or a cv2.VideoCapture.

Responsibilities
----------------
- Walk a video directory and index files
- Open each video with cv2.VideoCapture
- Downsample frames to a target FPS
- Convert BGR → RGB (OpenCV quirk, handled once, here)
- Yield fixed-size batches of frames to the pipeline
- Expose per-video metadata (label, filename, split) for the sink
- Clean up capture handles even on errors

NOT responsible for
-------------------
- Resizing or normalising frames  →  VideoPreprocessor
- Segmenting frames into clips    →  VideoPreprocessor
- Saving .npz files               →  DiskSink / FeatureExtractionPipeline
- Feature extraction              →  FusionExtractor
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import cv2
import numpy as np

from src.data.sources.base import AbstractFrameSource


# ---------------------------------------------------------------------------
# Default label inference
# ---------------------------------------------------------------------------

def _label_from_path(path: Path) -> int:
    """
    Derive a numeric label from the video's parent folder name.

    Convention (matches existing project layout):
        data/videos/normal/    → 0
        data/videos/anomalous/ → 1   (or any non-'normal' folder)

    Override by passing a custom ``label_fn`` to DiskVideoSource.
    """
    folder = path.parent.name.lower()
    if folder in ("normal", "0"):
        return 0
    # Try to parse an integer folder name ('1', '2', …)
    try:
        return int(folder)
    except ValueError:
        return 1  # treat anything non-normal as anomalous


# ---------------------------------------------------------------------------
# DiskVideoSource
# ---------------------------------------------------------------------------

class DiskVideoSource(AbstractFrameSource):
    """
    Streams raw RGB frames from video files stored on disk.

    Parameters
    ----------
    video_dir : str | Path
        Root directory to search for video files (recursive).
    extensions : tuple[str]
        File extensions to include.
    batch_size : int
        Number of frames per yielded batch.  Should equal the segment
        length expected by VideoPreprocessor so one batch == one segment.
    target_fps : int
        Desired output frame rate.  Native FPS is downsampled by skipping
        frames so that the effective rate matches target_fps.
    max_frames : int
        Maximum frames to read from a single video (safety cap).
    label_fn : Callable[[Path], int] | None
        Function that maps a video Path to a numeric label.
        Defaults to ``_label_from_path`` (folder-name convention).
    split : str
        'train' or 'test' — stored in metadata, used by the sink when
        naming .npz files.
    """

    VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv')

    def __init__(
        self,
        video_dir: str | Path,
        extensions: tuple = VIDEO_EXTENSIONS,
        batch_size: int = 16,
        target_fps: int = 8,
        max_frames: int = 3000,
        label_fn: Optional[Callable[[Path], int]] = None,
        split: str = 'train',
    ) -> None:
        self.video_dir = Path(video_dir)
        self.extensions = extensions
        self.batch_size = batch_size
        self.target_fps = target_fps
        self.max_frames = max_frames
        self.label_fn = label_fn or _label_from_path
        self.split = split

        self._video_paths: List[Path] = self._index_videos()
        self._current_video_path: Optional[Path] = None  # set during stream()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_videos(self) -> List[Path]:
        """Walk video_dir once at construction time. Fail fast if missing."""
        if not self.video_dir.exists():
            raise FileNotFoundError(
                f"DiskVideoSource: video directory not found: {self.video_dir}"
            )

        paths = [
            Path(root) / fname
            for root, _, files in os.walk(self.video_dir)
            for fname in files
            if fname.lower().endswith(self.extensions)
        ]
        paths.sort()

        if not paths:
            raise FileNotFoundError(
                f"DiskVideoSource: no video files found under {self.video_dir} "
                f"with extensions {self.extensions}"
            )

        print(f"✅ DiskVideoSource: indexed {len(paths)} videos in {self.video_dir}")
        return paths

    # ------------------------------------------------------------------
    # AbstractFrameSource interface
    # ------------------------------------------------------------------

    def stream(self) -> Iterator[List[np.ndarray]]:
        """
        Iterate over all indexed videos and yield frame batches.

        For each video:
          1. Open with cv2.VideoCapture
          2. Compute frame-skip interval to hit target_fps
          3. Read frames, skipping as needed, up to max_frames
          4. Convert BGR → RGB
          5. Accumulate into batches of batch_size and yield
          6. Yield any leftover frames as a final (shorter) batch
          7. Release the capture handle

        The caller (FeatureExtractionPipeline) iterates this generator;
        it never knows it is reading from disk.
        """
        for video_path in self._video_paths:
            self._current_video_path = video_path
            yield from self._stream_single_video(video_path)
        self._current_video_path = None

    def _stream_single_video(self, path: Path) -> Iterator[List[np.ndarray]]:
        """Open one video file and yield frame batches from it."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"⚠️  DiskVideoSource: cannot open {path.name}, skipping")
            return

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # How many native frames to skip to approximate target_fps
        interval = max(1, int(round(native_fps / self.target_fps)))
        effective_fps = native_fps / interval

        print(
            f"🎬 {path.name}  "
            f"native={native_fps:.1f}fps  effective={effective_fps:.1f}fps  "
            f"frames={total_frames}  size={width}x{height}"
        )

        batch: List[np.ndarray] = []
        frame_idx = 0
        sampled = 0

        try:
            while sampled < self.max_frames:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval == 0:
                    # BGR → RGB (done once, here, never in the pipeline)
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    batch.append(rgb_frame)
                    sampled += 1

                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

                frame_idx += 1

        finally:
            cap.release()

        # Yield any remaining frames (shorter batch — preprocessor pads)
        if batch:
            yield batch

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return f"DiskVideoSource({self.video_dir.name})"

    def metadata(self) -> dict:
        """
        Return metadata for the *currently streaming* video.

        Called by the pipeline after each video completes so the sink
        can write correct labels and filenames into the .npz file.
        """
        if self._current_video_path is None:
            return {}
        path = self._current_video_path
        return {
            'filename':   path.name,
            'video_path': str(path),
            'label':      self.label_fn(path),
            'class':      'Normal' if self.label_fn(path) == 0 else 'Anomalous',
            'split':      self.split,
            'source':     'disk',
        }

    def video_metadata_for(self, path: Path) -> dict:
        """
        Return full metadata for an explicit path (used by the pipeline
        when iterating per-video to build the sink payload).
        """
        label = self.label_fn(path)
        return {
            'filename':   path.name,
            'video_path': str(path),
            'full_path':  str(path),
            'label':      label,
            'class':      'Normal' if label == 0 else 'Anomalous',
            'split':      self.split,
            'source':     'disk',
            'directory':  path.parent.name,
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._video_paths)

    def __repr__(self) -> str:
        return (
            f"DiskVideoSource(video_dir={self.video_dir!r}, "
            f"videos={len(self._video_paths)}, "
            f"target_fps={self.target_fps}, "
            f"batch_size={self.batch_size})"
        )

    def iter_videos(self) -> Iterator[tuple[Path, Iterator[List[np.ndarray]]]]:
        """
        Yield (video_path, frame_batch_iterator) pairs.

        Used by FeatureExtractionPipeline so it can associate per-video
        metadata with the feature output without breaking the abstraction.
        The pipeline still only calls _stream_single_video — it never
        opens the file itself.
        """
        for path in self._video_paths:
            self._current_video_path = path
            yield path, self._stream_single_video(path)
        self._current_video_path = None