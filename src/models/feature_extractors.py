"""
src/models/feature_extractors.py
---------------------------------
Feature Extractors for Video Analysis.

I3D, R3D, Lightweight 3D CNN, YOLOv8 object features, two-stream fusion.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import os

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseFeatureExtractor(nn.Module, ABC):
    """Base class for video feature extractors."""

    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.model = None
        self.feature_dim = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract_features(self, segments: List[torch.Tensor]) -> np.ndarray:
        self.eval()
        features = []
        with torch.no_grad():
            for segment in segments:
                segment_batch = segment.unsqueeze(0).to(self.device)
                segment_batch = segment_batch.permute(0, 2, 1, 3, 4)
                feat = self.forward(segment_batch)
                features.append(feat.cpu().numpy().flatten())
        return np.array(features)


class I3DFeatureExtractor(BaseFeatureExtractor):
    """I3D-ResNet50 feature extractor. Output dim: 2048."""

    def __init__(self, device: str = "cuda", pretrained: bool = True, freeze: bool = True):
        super().__init__(device)
        try:
            from pytorchvideo.models.hub import i3d_r50
            self.model = i3d_r50(pretrained=pretrained)
            self.model.blocks[-1] = nn.Identity()
            self.feature_dim = 2048
            if freeze:
                for p in self.model.parameters():
                    p.requires_grad = False
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("Loaded I3D-ResNet50 (feature_dim=%d)", self.feature_dim)
        except ImportError as e:
            raise ImportError(f"pytorchvideo not installed: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model(x)
        return feat.view(feat.size(0), -1)


class ResidualBlock3D(nn.Module):
    """3D Residual Block."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class R3DFeatureExtractor(BaseFeatureExtractor):
    """R3D-18 feature extractor. Output dim: 512."""

    def __init__(self, device: str = 'cuda', pretrained: bool = True):
        super().__init__(device)
        self.feature_dim = 512
        try:
            import torchvision.models.video as video_models
            self.model = video_models.r3d_18(pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model = nn.Sequential(self.model, nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten())
            with torch.no_grad():
                dummy = torch.randn(1, 3, 16, 112, 112).to(device)
                self.feature_dim = self.model(dummy).shape[1]
            self.model = self.model.to(device)
            self.model.eval()
            logger.info("Loaded R3D-18 (feature_dim=%d)", self.feature_dim)
        except ImportError as e:
            logger.warning("Could not load R3D-18: %s — building simple fallback", e)
            self.model = self._build_simple_r3d().to(device)
            self.model.eval()

    def _build_simple_r3d(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm3d(out_ch)
            )
        layers = [ResidualBlock3D(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LightweightFeatureExtractor(BaseFeatureExtractor):
    """Lightweight 3D CNN. Output dim: 512."""

    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.feature_dim = 512
        self.model = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(True), nn.MaxPool3d(2),
            nn.Conv3d(256, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
        ).to(device)
        self.model.eval()
        logger.info("Loaded Lightweight 3D CNN")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class YOLOObjectFeatureExtractor:
    """YOLOv8 object-centric features. Output dim: 83 (80 COCO + 3 bbox stats)."""

    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cuda"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.model.model.eval()
            self.device = device
            self.num_classes = 80
            self.feature_dim = 83
            logger.info("Loaded YOLOv8 (feature_dim=%d)", self.feature_dim)
        except ImportError:
            raise ImportError("ultralytics not installed: pip install ultralytics")

    def extract_segment_features(self, frames_np: List[np.ndarray]) -> np.ndarray:
        results = self.model(frames_np, verbose=False)
        obj_counts = np.zeros(self.num_classes, dtype=np.float32)
        bbox_stats = []
        for res in results:
            for box in res.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0]
                if cls < self.num_classes:
                    obj_counts[cls] += 1
                w = (x2 - x1).item() if hasattr(x2 - x1, 'item') else (x2 - x1)
                h = (y2 - y1).item() if hasattr(y2 - y1, 'item') else (y2 - y1)
                bbox_stats.append([w, h, conf])
        obj_counts /= max(len(frames_np), 1)
        if bbox_stats:
            bbox_arr = np.array(bbox_stats)
            bbox_features = np.concatenate([bbox_arr.mean(0), bbox_arr.std(0)])
        else:
            bbox_features = np.zeros(6, dtype=np.float32)
        return np.concatenate([obj_counts, bbox_features])


class YOLOFeatureAdapter:
    """Adapter for YOLO in multi-stream pipelines."""

    def __init__(self, yolo_extractor: YOLOObjectFeatureExtractor, device: str = "cuda"):
        self.yolo = yolo_extractor
        self.device = device
        self.feature_dim = yolo_extractor.feature_dim

    def extract_features(self, segments: List[torch.Tensor]) -> np.ndarray:
        features = []
        for seg in segments:
            frames_np = [
                (f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                for f in seg
            ]
            features.append(self.yolo.extract_segment_features(frames_np))
        return np.stack(features)


class FeatureExtractorFactory:
    """Factory for creating feature extractors."""

    @staticmethod
    def create_extractor(model_type: str = 'i3d', device: str = 'cuda') -> BaseFeatureExtractor:
        model_type = model_type.lower()
        if model_type == 'i3d':
            return I3DFeatureExtractor(device=device)
        elif model_type == 'r3d':
            return R3DFeatureExtractor(device=device)
        elif model_type == 'lightweight':
            return LightweightFeatureExtractor(device=device)
        raise ValueError(f"Unknown model type: {model_type}")


class TwoStreamFeatureExtractor:
    """Fuses motion (3D CNN) + object (YOLO) features. Default: 2048+83=2131."""

    def __init__(self, motion_extractor: BaseFeatureExtractor, object_extractor: YOLOFeatureAdapter):
        self.motion_extractor = motion_extractor
        self.object_extractor = object_extractor
        self.feature_dim = motion_extractor.feature_dim + object_extractor.feature_dim
        logger.info("TwoStreamFeatureExtractor: feature_dim=%d", self.feature_dim)

    def extract_features(self, segments: List[torch.Tensor]) -> np.ndarray:
        motion_feats = self.motion_extractor.extract_features(segments)
        object_feats = self.object_extractor.extract_features(segments)
        assert motion_feats.shape[0] == object_feats.shape[0]
        return np.concatenate([motion_feats, object_feats], axis=1)
