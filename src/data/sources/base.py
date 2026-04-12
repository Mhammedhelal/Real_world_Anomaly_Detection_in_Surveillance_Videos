"""
src/sources/base.py
-------------------
Abstract base class for all frame sources.

The pipeline and preprocessor consume ONLY this interface — they have
zero knowledge of whether frames come from disk, an RTSP stream, a
webcam, or a test fixture.

Contract
--------
stream() yields batches of raw RGB frames as numpy arrays.
Each item is List[np.ndarray], shape (H, W, 3), dtype uint8.
The batch size is source-defined (e.g. one segment's worth of frames).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np


class AbstractFrameSource(ABC):
    """
    Unified interface for all frame sources.

    Implementors
    ------------
    DiskVideoSource     — iterates video files from a directory
    CameraStreamSource  — reads a live RTSP / webcam feed

    Usage
    -----
    Any source can be dropped into FeatureExtractionPipeline or
    InferencePipeline without changing a single line of pipeline code:

        pipeline = FeatureExtractionPipeline(
            source=DiskVideoSource(video_dir="data/videos/normal"),
            ...
        )

        pipeline = FeatureExtractionPipeline(
            source=CameraStreamSource(rtsp_url="rtsp://..."),
            ...
        )
    """

    @abstractmethod
    def stream(self) -> Iterator[List[np.ndarray]]:
        """
        Yield batches of raw RGB frames.

        Yields
        ------
        List[np.ndarray]
            Each list is one batch of frames.
            Each frame: shape (H, W, 3), dtype uint8, RGB colour order.

        Notes
        -----
        - The pipeline never calls open() / close() separately; this
          method is responsible for all resource acquisition and release.
        - Raising StopIteration (or returning) signals end-of-stream.
          For live sources this may never happen unless the feed drops.
        """
        ...

    @property
    def source_id(self) -> str:
        """Human-readable identifier for logging / metadata."""
        return self.__class__.__name__

    def metadata(self) -> dict:
        """
        Optional per-source metadata passed through to the sink.

        Override to supply label, filename, split, etc.
        Default returns an empty dict so callers can always do
        ``.update(source.metadata())`` safely.
        """
        return {}