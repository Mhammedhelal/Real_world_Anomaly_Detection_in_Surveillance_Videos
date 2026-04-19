"""
src/models/video_preprocessor.py
---------------------------------
Pure frame-transformation preprocessor — no disk I/O.

Receives raw RGB numpy frames, resizes/normalises/converts to tensors,
groups into fixed-length segments.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_config() -> Config:
    config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'default.yaml'
    return Config.from_yaml(config_path)


def _parse_frame_size(frame_size_raw) -> tuple:
    if isinstance(frame_size_raw, str):
        try:
            return ast.literal_eval(frame_size_raw)
        except (ValueError, SyntaxError):
            return (224, 224)
    elif isinstance(frame_size_raw, (tuple, list)):
        return tuple(frame_size_raw)
    return (224, 224)


def _build_default_transform(
    frame_size: tuple = (224, 224),
    mean: list | None = None,
    std: list | None = None,
) -> T.Compose:
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    return T.Compose([
        T.ToPILImage(),
        T.Resize(frame_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


class VideoPreprocessor:
    """
    Stateless frame transformer.

    Parameters
    ----------
    frame_size : tuple | None
    segment_length : int | None
    transform : torchvision.transforms.Compose | None
    config : Config | None
    """

    def __init__(
        self,
        frame_size: tuple | None = None,
        segment_length: int | None = None,
        transform: Optional[T.Compose] = None,
        config: Optional[Config] = None,
    ) -> None:
        if config is None:
            config = _load_config()
        self.config = config

        if frame_size is None:
            frame_size_raw = getattr(self.config.dataset, 'frame_size', (224, 224))
            frame_size = _parse_frame_size(frame_size_raw)
        self.frame_size = frame_size

        if segment_length is None:
            segment_length = getattr(self.config.dataset, 'segment_length', 16)
        self.segment_length = segment_length

        mean = getattr(self.config.dataset, 'mean', [0.485, 0.456, 0.406])
        std = getattr(self.config.dataset, 'std', [0.229, 0.224, 0.225])

        self.transform = transform or _build_default_transform(self.frame_size, mean=mean, std=std)

        logger.debug(
            "VideoPreprocessor: frame_size=%s  segment_length=%d",
            self.frame_size, self.segment_length,
        )

    def process_batch(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """Convert raw RGB numpy frames → normalised tensors."""
        return [self.transform(frame) for frame in frames]

    def to_segments(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """Raw frames → normalised tensors → fixed-length segments."""
        tensors = self.process_batch(frames)
        return self.create_segments(tensors)

    def create_segments(self, frame_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Group frame tensors into fixed-length segments (pads short tails)."""
        if not frame_tensors:
            return []

        if len(frame_tensors) < self.segment_length:
            pad = self.segment_length - len(frame_tensors)
            frame_tensors = frame_tensors + [frame_tensors[-1]] * pad

        segments = []
        for start in range(0, len(frame_tensors), self.segment_length):
            chunk = frame_tensors[start: start + self.segment_length]
            if len(chunk) < self.segment_length:
                pad = self.segment_length - len(chunk)
                chunk = chunk + [chunk[-1]] * pad
            segments.append(torch.stack(chunk))
        return segments

    def save_sample_frames(
        self,
        frames: List[np.ndarray],
        output_dir: str | Path,
        name_prefix: str = 'frame',
        max_save: int = 10,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        indices = np.linspace(0, len(frames) - 1, min(max_save, len(frames)), dtype=int)
        for i, idx in enumerate(indices):
            Image.fromarray(frames[idx]).save(output_dir / f"{name_prefix}_frame_{i:03d}.jpg")
        logger.info("Saved %d sample frames → %s", len(indices), output_dir)

    def __repr__(self) -> str:
        return (
            f"VideoPreprocessor(frame_size={self.frame_size}, "
            f"segment_length={self.segment_length})"
        )


# Backward-compat alias
FramePreprocessor = VideoPreprocessor
