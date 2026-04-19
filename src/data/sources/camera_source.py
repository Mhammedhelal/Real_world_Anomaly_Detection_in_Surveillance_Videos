"""
src/data/sources/camera_source.py
----------------------------------
Live camera / RTSP stream source.
"""

from __future__ import annotations

import time
from typing import Iterator, List, Optional, Union

import cv2
import numpy as np

from src.data.sources.base import AbstractFrameSource
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CameraStreamSource(AbstractFrameSource):
    """
    Streams raw RGB frames from a live camera or RTSP feed.

    Parameters
    ----------
    source : str | int
    batch_size : int
    target_fps : int
    max_batches : int | None
    reconnect : bool
    reconnect_delay : float
    max_reconnects : int
    camera_id : str
    """

    def __init__(
        self,
        source: Union[str, int] = 0,
        batch_size: int = 16,
        target_fps: int = 8,
        max_batches: Optional[int] = None,
        reconnect: bool = True,
        reconnect_delay: float = 2.0,
        max_reconnects: int = 5,
        camera_id: str = 'camera-0',
    ) -> None:
        self.source = source
        self.batch_size = batch_size
        self.target_fps = target_fps
        self.max_batches = max_batches
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self.camera_id = camera_id
        self._stop_requested = False

    def stream(self) -> Iterator[List[np.ndarray]]:
        self._stop_requested = False
        reconnect_attempts = 0
        batches_yielded = 0

        while not self._stop_requested:
            cap = self._open_capture()
            if cap is None:
                if not self.reconnect or reconnect_attempts >= self.max_reconnects:
                    logger.error("CameraStreamSource: giving up on %s", self.source)
                    return
                reconnect_attempts += 1
                logger.warning(
                    "CameraStreamSource: reconnect %d/%d in %.1fs",
                    reconnect_attempts, self.max_reconnects, self.reconnect_delay,
                )
                time.sleep(self.reconnect_delay)
                continue

            reconnect_attempts = 0
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = max(1, int(round(native_fps / self.target_fps)))

            logger.info(
                "CameraStreamSource connected: %s  native=%.1ffps  effective=%.1ffps",
                self.source, native_fps, native_fps / interval,
            )

            batch: List[np.ndarray] = []
            frame_idx = 0

            try:
                while not self._stop_requested:
                    ret, bgr_frame = cap.read()
                    if not ret:
                        logger.warning("CameraStreamSource: stream read failed (%s)", self.source)
                        break

                    if frame_idx % interval == 0:
                        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        batch.append(rgb_frame)

                        if len(batch) == self.batch_size:
                            yield batch
                            batch = []
                            batches_yielded += 1

                            if self.max_batches and batches_yielded >= self.max_batches:
                                self._stop_requested = True
                                break

                    frame_idx += 1
            finally:
                cap.release()

            if self._stop_requested:
                break

            if self.reconnect and reconnect_attempts < self.max_reconnects:
                reconnect_attempts += 1
                logger.warning(
                    "CameraStreamSource: stream lost, reconnecting %d/%d",
                    reconnect_attempts, self.max_reconnects,
                )
                time.sleep(self.reconnect_delay)
            else:
                logger.info("CameraStreamSource: stream ended.")
                break

    def stop(self) -> None:
        self._stop_requested = True

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        try:
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                return cap
            cap.release()
            logger.warning("CameraStreamSource: could not open %s", self.source)
            return None
        except Exception as exc:
            logger.warning("CameraStreamSource: exception opening %s: %s", self.source, exc)
            return None

    @property
    def source_id(self) -> str:
        return f"CameraStreamSource({self.camera_id})"

    def metadata(self) -> dict:
        return {
            'source': 'camera',
            'camera_id': self.camera_id,
            'stream': str(self.source),
            'label': -1,
            'class': 'Unknown',
            'split': 'inference',
        }

    def __repr__(self) -> str:
        return (
            f"CameraStreamSource(source={self.source!r}, "
            f"batch_size={self.batch_size}, target_fps={self.target_fps})"
        )
