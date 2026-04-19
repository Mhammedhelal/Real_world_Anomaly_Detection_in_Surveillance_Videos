"""
src/utils/video.py
------------------
General-purpose video helpers — no dataset/label logic.
"""

import os
import cv2

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_video_info(video_path: str) -> dict | None:
    """Return basic metadata for a video using OpenCV."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration,
            'size_mb': os.path.getsize(video_path) / (1024 * 1024),
        }
    except Exception as exc:
        logger.debug("get_video_info failed for %s: %s", video_path, exc)
        return None
