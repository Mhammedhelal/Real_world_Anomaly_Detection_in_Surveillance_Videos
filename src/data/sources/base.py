"""
src/data/sources/base.py
------------------------
Abstract base class for all frame sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List

import numpy as np


class AbstractFrameSource(ABC):
    """
    Unified interface for all frame sources (disk, camera, test fixture).

    stream() yields List[np.ndarray] batches of raw RGB frames (H, W, 3) uint8.
    """

    @abstractmethod
    def stream(self) -> Iterator[List[np.ndarray]]:
        """Yield batches of raw RGB frames."""
        ...

    @property
    def source_id(self) -> str:
        return self.__class__.__name__

    def metadata(self) -> dict:
        """Optional per-source metadata (label, filename, split, …)."""
        return {}
