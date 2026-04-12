"""
src/models/video_preprocessor.py
---------------------------------
Pure frame-transformation preprocessor.

ALL disk I/O has been removed.  This class:
  - Receives raw RGB numpy frames  (from ANY AbstractFrameSource)
  - Resizes, normalises, and converts to tensors
  - Groups frames into fixed-length segments

It never opens files, never reads paths, never calls cv2.VideoCapture.
That logic lives entirely in DiskVideoSource / CameraStreamSource.

Before (mixed concerns)
-----------------------
    preprocessor.read_video(video_path, ...)    ← disk I/O, BGR→RGB
    preprocessor.create_segments(frames, ...)   ← pure transform

After (decoupled)
-----------------
    # disk/camera source handles I/O and BGR→RGB conversion
    for frame_batch in source.stream():
        segments = preprocessor.to_segments(frame_batch)
        features = extractor.extract(segments)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

import torchvision.transforms as T

from src.config import Config


# ---------------------------------------------------------------------------
# Load configuration from YAML
# ---------------------------------------------------------------------------

def _load_config() -> Config:
    """Load config from default.yaml relative to project root."""
    config_path = Path(__file__).resolve().parent.parent.parent / 'configs' / 'default.yaml'
    return Config.from_yaml(config_path)


def _parse_frame_size(frame_size_raw) -> tuple:
    """Parse frame_size from config, handling both tuple and string formats."""
    if isinstance(frame_size_raw, str):
        # Parse string like "(224, 224)" to tuple
        try:
            return ast.literal_eval(frame_size_raw)
        except (ValueError, SyntaxError):
            return (224, 224)
    elif isinstance(frame_size_raw, (tuple, list)):
        return tuple(frame_size_raw)
    else:
        return (224, 224)


# ---------------------------------------------------------------------------
# Default transform (ImageNet normalisation)
# ---------------------------------------------------------------------------

def _build_default_transform(
    frame_size: tuple = (224, 224),
    mean: list | None = None,
    std: list | None = None,
) -> T.Compose:
    """Build transform pipeline with configurable normalization."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    return T.Compose([
        T.ToPILImage(),
        T.Resize(frame_size),
        T.ToTensor(),
        T.Normalize(
            mean=mean,
            std=std,
        ),
    ])


# ---------------------------------------------------------------------------
# VideoPreprocessor
# ---------------------------------------------------------------------------

class VideoPreprocessor:
    """
    Stateless frame transformer consumed by the feature extraction pipeline.

    Loads configuration from configs/default.yaml.

    Parameters
    ----------
    frame_size : tuple[int, int] | None
        (height, width) to resize each frame to.
        If None, loads from config.dataset.frame_size.
    segment_length : int | None
        Number of frames per segment / clip fed to the feature extractor.
        If None, loads from config.dataset.segment_length.
    transform : torchvision.transforms.Compose | None
        Custom transform pipeline. Defaults to resize + ImageNet normalise.
    config : Config | None
        Config object. If None, loads from default.yaml.
    """

    def __init__(
        self,
        frame_size: tuple | None = None,
        segment_length: int | None = None,
        transform: Optional[T.Compose] = None,
        config: Optional[Config] = None,
    ) -> None:
        # Load config if not provided
        if config is None:
            config = _load_config()
        self.config = config

        # Load frame_size from config if not provided
        if frame_size is None:
            frame_size_raw = getattr(self.config.dataset, 'frame_size', (224, 224))
            frame_size = _parse_frame_size(frame_size_raw)
        self.frame_size = frame_size

        # Load segment_length from config if not provided
        if segment_length is None:
            segment_length = getattr(self.config.dataset, 'segment_length', 16)
        self.segment_length = segment_length

        # Load normalization parameters from config
        mean = getattr(self.config.dataset, 'mean', [0.485, 0.456, 0.406])
        std = getattr(self.config.dataset, 'std', [0.229, 0.224, 0.225])

        # Use provided transform or build default from config
        if transform is None:
            transform = _build_default_transform(self.frame_size, mean=mean, std=std)
        self.transform = transform

    # ------------------------------------------------------------------
    # Primary API — called by the pipeline
    # ------------------------------------------------------------------

    def process_batch(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Convert a batch of raw RGB numpy frames to normalised tensors.

        Parameters
        ----------
        frames : List[np.ndarray]
            Raw RGB frames, shape (H, W, 3), dtype uint8.
            Comes directly from AbstractFrameSource.stream().

        Returns
        -------
        List[torch.Tensor]
            Each tensor: (C, H, W), float32, ImageNet-normalised.
        """
        return [self.transform(frame) for frame in frames]

    def to_segments(self, frames: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Convert raw frames → normalised tensors → fixed-length segments.

        Combines process_batch + create_segments in one call.
        This is the method the pipeline should prefer.

        Parameters
        ----------
        frames : List[np.ndarray]
            Raw RGB frames from the source.

        Returns
        -------
        List[torch.Tensor]
            Each tensor: (segment_length, C, H, W).
        """
        tensors = self.process_batch(frames)
        return self.create_segments(tensors)

    def create_segments(
        self,
        frame_tensors: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Group a list of frame tensors into fixed-length segments.

        Short tail segments are padded by repeating the last frame.

        Parameters
        ----------
        frame_tensors : List[torch.Tensor]
            Normalised frame tensors, each (C, H, W).

        Returns
        -------
        List[torch.Tensor]
            Each tensor: (segment_length, C, H, W).
        """
        if not frame_tensors:
            return []

        # Pad if fewer frames than one full segment
        if len(frame_tensors) < self.segment_length:
            pad = self.segment_length - len(frame_tensors)
            frame_tensors = frame_tensors + [frame_tensors[-1]] * pad

        segments = []
        for start in range(0, len(frame_tensors), self.segment_length):
            chunk = frame_tensors[start : start + self.segment_length]

            # Pad the last (possibly short) chunk
            if len(chunk) < self.segment_length:
                pad = self.segment_length - len(chunk)
                chunk = chunk + [chunk[-1]] * pad

            segments.append(torch.stack(chunk))   # (S, C, H, W)

        return segments

    # ------------------------------------------------------------------
    # Visualisation helper (unchanged, still useful for debugging)
    # ------------------------------------------------------------------

    def save_sample_frames(
        self,
        frames: List[np.ndarray],
        output_dir: str | Path,
        name_prefix: str = 'frame',
        max_save: int = 10,
    ) -> None:
        """
        Save a sample of raw RGB frames to disk for debugging.

        This is the only method that writes to disk, and it is a
        debug/visualisation helper — not part of the extraction path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        indices = np.linspace(0, len(frames) - 1, min(max_save, len(frames)), dtype=int)
        for i, idx in enumerate(indices):
            Image.fromarray(frames[idx]).save(
                output_dir / f"{name_prefix}_frame_{i:03d}.jpg"
            )
        print(f"💾 Saved {len(indices)} frames to {output_dir}")

    def __repr__(self) -> str:
        return (
            f"VideoPreprocessor("
            f"frame_size={self.frame_size}, "
            f"segment_length={self.segment_length})"
        )


# ---------------------------------------------------------------------------
# Backward-compat alias so existing imports don't break immediately
# ---------------------------------------------------------------------------
FramePreprocessor = VideoPreprocessor