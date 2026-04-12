"""
src/sources/camera_source.py
-----------------------------
Live camera / RTSP stream source.

Implements AbstractFrameSource for real-time inference.  The pipeline
receives the same List[np.ndarray] batches as it does from DiskVideoSource
— it cannot tell the difference.

Responsibilities
----------------
- Connect to a camera device or RTSP URL via cv2.VideoCapture
- Accumulate frames into sliding-window batches (size == segment_length)
- Convert BGR → RGB
- Yield continuously until the stream drops or stop() is called
- Reconnect automatically on transient stream failures (optional)

NOT responsible for
-------------------
- Resizing / normalising frames   →  VideoPreprocessor
- Segmenting clips                →  VideoPreprocessor
- Feature extraction              →  FusionExtractor
- Saving results                  →  Sink
"""

from __future__ import annotations

import time
from typing import Iterator, List, Optional, Union

import cv2
import numpy as np

from src.data.sources.base import AbstractFrameSource


class CameraStreamSource(AbstractFrameSource):
    """
    Streams raw RGB frames from a live camera or RTSP feed.

    Parameters
    ----------
    source : str | int
        RTSP URL string  (e.g. 'rtsp://192.168.1.10/live')
        or integer device index  (e.g. 0 for the default webcam).
    batch_size : int
        Frames per yielded batch.  Should match the segment_length
        expected by VideoPreprocessor (default 16).
    target_fps : int
        Desired capture rate.  Frames are dropped to approximate this
        rate when the camera native FPS is higher.
    max_batches : int | None
        Stop after this many batches.  None = stream indefinitely.
        Useful for testing or fixed-duration inference windows.
    reconnect : bool
        If True, attempt to reconnect when the stream drops instead of
        raising StopIteration immediately.
    reconnect_delay : float
        Seconds to wait between reconnection attempts.
    max_reconnects : int
        Maximum number of reconnection attempts before giving up.
    camera_id : str
        Human-readable name used in logs and metadata.
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

    # ------------------------------------------------------------------
    # AbstractFrameSource interface
    # ------------------------------------------------------------------

    def stream(self) -> Iterator[List[np.ndarray]]:
        """
        Connect to the camera and yield frame batches indefinitely.

        Each batch is List[np.ndarray] of length batch_size, where each
        frame is (H, W, 3) uint8 RGB — identical contract to DiskVideoSource.

        The generator yields until:
          - stop() is called
          - max_batches is reached
          - the stream drops and reconnection is disabled / exhausted
        """
        self._stop_requested = False
        reconnect_attempts = 0
        batches_yielded = 0

        while not self._stop_requested:
            cap = self._open_capture()
            if cap is None:
                # Could not connect at all
                if not self.reconnect or reconnect_attempts >= self.max_reconnects:
                    print(f"❌ CameraStreamSource: giving up on {self.source}")
                    return
                reconnect_attempts += 1
                print(
                    f"⏳ CameraStreamSource: reconnect attempt "
                    f"{reconnect_attempts}/{self.max_reconnects} "
                    f"in {self.reconnect_delay}s …"
                )
                time.sleep(self.reconnect_delay)
                continue

            # Successful connection — reset counter
            reconnect_attempts = 0
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            interval = max(1, int(round(native_fps / self.target_fps)))

            print(
                f"📷 CameraStreamSource connected: {self.source}  "
                f"native={native_fps:.1f}fps  "
                f"effective={native_fps/interval:.1f}fps"
            )

            batch: List[np.ndarray] = []
            frame_idx = 0

            try:
                while not self._stop_requested:
                    ret, bgr_frame = cap.read()

                    if not ret:
                        print(f"⚠️  CameraStreamSource: stream read failed ({self.source})")
                        break  # exit inner loop → attempt reconnect

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

            # Stream dropped — try reconnecting
            if self.reconnect and reconnect_attempts < self.max_reconnects:
                reconnect_attempts += 1
                print(
                    f"🔄 CameraStreamSource: stream lost, reconnecting "
                    f"({reconnect_attempts}/{self.max_reconnects}) …"
                )
                time.sleep(self.reconnect_delay)
            else:
                print("❌ CameraStreamSource: stream ended, no reconnect.")
                break

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the stream() generator to stop after the current batch."""
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Open the capture handle and verify it works.

        Returns None instead of raising so the stream() loop can handle
        retries gracefully.
        """
        try:
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                return cap
            cap.release()
            print(f"⚠️  CameraStreamSource: could not open {self.source}")
            return None
        except Exception as exc:
            print(f"⚠️  CameraStreamSource: exception opening {self.source}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def source_id(self) -> str:
        return f"CameraStreamSource({self.camera_id})"

    def metadata(self) -> dict:
        """
        Live sources don't have pre-known labels or filenames.
        Downstream sinks should use wall-clock timestamps instead.
        """
        return {
            'source':    'camera',
            'camera_id': self.camera_id,
            'stream':    str(self.source),
            'label':     -1,   # unknown at capture time
            'class':     'Unknown',
            'split':     'inference',
        }

    def __repr__(self) -> str:
        return (
            f"CameraStreamSource(source={self.source!r}, "
            f"batch_size={self.batch_size}, "
            f"target_fps={self.target_fps})"
        )