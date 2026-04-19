"""
src/data/sources/disk_source.py
--------------------------------
Disk-based frame source — all disk I/O centralised here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import cv2
import numpy as np

from src.data.sources.base import AbstractFrameSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _label_from_path(path: Path) -> int:
    folder = path.parent.name.lower()
    if folder in ("normal", "0"):
        return 0
    try:
        return int(folder)
    except ValueError:
        return 1


class DiskVideoSource(AbstractFrameSource):
    """
    Streams raw RGB frames from video files stored on disk.

    Parameters
    ----------
    video_dir : str | Path
    extensions : tuple[str]
    batch_size : int
    target_fps : int
    max_frames : int
    label_fn : Callable[[Path], int] | None
    split : str
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
        self._current_video_path: Optional[Path] = None

    def _index_videos(self) -> List[Path]:
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
                f"DiskVideoSource: no video files found under {self.video_dir}"
            )
        logger.info("DiskVideoSource: indexed %d videos in %s", len(paths), self.video_dir)
        return paths

    def stream(self) -> Iterator[List[np.ndarray]]:
        for video_path in self._video_paths:
            self._current_video_path = video_path
            yield from self._stream_single_video(video_path)
        self._current_video_path = None

    def _stream_single_video(self, path: Path) -> Iterator[List[np.ndarray]]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.warning("Cannot open %s — skipping", path.name)
            return

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        interval = max(1, int(round(native_fps / self.target_fps)))
        effective_fps = native_fps / interval

        logger.info(
            "%s  native=%.1ffps  effective=%.1ffps  frames=%d  size=%dx%d",
            path.name, native_fps, effective_fps, total_frames, width, height,
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
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    batch.append(rgb_frame)
                    sampled += 1
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                frame_idx += 1
        finally:
            cap.release()

        if batch:
            yield batch

    @property
    def source_id(self) -> str:
        return f"DiskVideoSource({self.video_dir.name})"

    def metadata(self) -> dict:
        if self._current_video_path is None:
            return {}
        path = self._current_video_path
        label = self.label_fn(path)
        return {
            'filename': path.name,
            'video_path': str(path),
            'label': label,
            'class': 'Normal' if label == 0 else 'Anomalous',
            'split': self.split,
            'source': 'disk',
        }

    def video_metadata_for(self, path: Path) -> dict:
        label = self.label_fn(path)
        return {
            'filename': path.name,
            'video_path': str(path),
            'full_path': str(path),
            'label': label,
            'class': 'Normal' if label == 0 else 'Anomalous',
            'split': self.split,
            'source': 'disk',
            'directory': path.parent.name,
        }

    def __len__(self) -> int:
        return len(self._video_paths)

    def __repr__(self) -> str:
        return (
            f"DiskVideoSource(video_dir={self.video_dir!r}, "
            f"videos={len(self._video_paths)}, "
            f"target_fps={self.target_fps})"
        )

    def iter_videos(self) -> Iterator[tuple]:
        for path in self._video_paths:
            self._current_video_path = path
            yield path, self._stream_single_video(path)
        self._current_video_path = None
