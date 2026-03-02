"""
Utility functions related to video files.

This module contains only general-purpose video helpers and does not
depend on dataset or label logic.
"""

import os
import cv2


def get_video_info(video_path):
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
            'size_mb': os.path.getsize(video_path) / (1024 * 1024)
        }
    except Exception:
        return None
