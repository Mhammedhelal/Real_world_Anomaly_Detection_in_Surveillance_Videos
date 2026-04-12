"""
src/sources/__init__.py
-----------------------
Frame source implementations.
"""

from src.data.sources.base import AbstractFrameSource
from src.data.sources.disk_source import DiskVideoSource
from src.data.sources.camera_source import CameraStreamSource

__all__ = [
    'AbstractFrameSource',
    'DiskVideoSource',
    'CameraStreamSource',
]