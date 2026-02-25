"""
Feature Extractors for Video Analysis.

Implements different 3D CNN architectures and YOLO-based object detection
for extracting motion and object features from video segments.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
import os


class BaseFeatureExtractor(nn.Module, ABC):
    """Base class for video feature extractors."""
    
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.model = None
        self.feature_dim = None
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor."""
        raise NotImplementedError
    
    def extract_features(self, segments: List[torch.Tensor]) -> np.ndarray:
        """
        Extract features from video segments.
        
        Args:
            segments: List of segments [segment_length, C, H, W]
        
        Returns:
            features: numpy array [num_segments, feature_dim]
        """
        self.eval()
        features = []
        
        with torch.no_grad():
            for segment in segments:
                # Add batch dimension: [1, segment_length, C, H, W]
                segment_batch = segment.unsqueeze(0).to(self.device)
                
                # Permute from [B, T, C, H, W] to [B, C, T, H, W]
                segment_batch = segment_batch.permute(0, 2, 1, 3, 4)
                
                # Extract features
                feat = self.forward(segment_batch)
                features.append(feat.cpu().numpy().flatten())
        
        return np.array(features)


# ============================================================================
# I3D Feature Extractor (from PyTorchVideo)
# ============================================================================

class I3DFeatureExtractor(BaseFeatureExtractor):
    """
    I3D (Inflated 3D) Feature Extractor using pretrained I3D-ResNet50.
    
    Strong temporal and spatial feature learning.
    Output dimension: 2048
    """
    
    def __init__(
        self,
        device: str = "cuda",
        pretrained: bool = True,
        freeze: bool = True
    ):
        super().__init__(device)
        
        try:
            from pytorchvideo.models.hub import i3d_r50
            
            # Load pretrained I3D
            self.model = i3d_r50(pretrained=pretrained)
            
            # Remove classification head to get features
            self.model.blocks[-1] = nn.Identity()
            
            # Feature dimension after global pooling
            self.feature_dim = 2048
            
            if freeze:
                for p in self.model.parameters():
                    p.requires_grad = False
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Loaded I3D-ResNet50 feature extractor (feature_dim={self.feature_dim})")
        
        except ImportError as e:
            raise ImportError(f"pytorchvideo not installed: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, C, T, H, W]
        
        Returns:
            features: Tensor [B, 2048]
        """
        # Forward through backbone
        feat = self.model(x)  # [B, 2048, 1, 1, 1]
        
        # Flatten
        feat = feat.view(feat.size(0), -1)
        
        return feat


# ============================================================================
# R3D Feature Extractor
# ============================================================================

class R3DFeatureExtractor(BaseFeatureExtractor):
    """
    R3D-18 Feature Extractor.
    
    Lighter weight than I3D but still effective for motion capture.
    Output dimension: 512
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        pretrained: bool = True
    ):
        super().__init__(device)
        self.feature_dim = 512
        
        try:
            import torchvision.models.video as video_models
            self.model = video_models.r3d_18(pretrained=pretrained)
            
            # Remove classification head
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
            # Add adaptive pooling for consistency
            self.model = nn.Sequential(
                self.model,
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten()
            )
            
            # Verify feature dimension
            with torch.no_grad():
                dummy = torch.randn(1, 3, 16, 112, 112).to(device)
                output = self.model(dummy)
                self.feature_dim = output.shape[1]
            
            self.model = self.model.to(device)
            self.model.eval()
            
            print(f"✅ Loaded R3D-18 feature extractor (feature_dim={self.feature_dim})")
        
        except ImportError as e:
            print(f"⚠️ Could not load R3D-18: {e}")
            self.model = self._build_simple_r3d()
            self.model = self.model.to(device)
            self.model.eval()
    
    def _build_simple_r3d(self) -> nn.Module:
        """Build simplified R3D model."""
        model = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        return model
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a layer with residual blocks."""
        layers = []
        
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        
        layers.append(ResidualBlock3D(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock3D(nn.Module):
    """3D Residual Block for R3D."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


# ============================================================================
# Lightweight Feature Extractor
# ============================================================================

class LightweightFeatureExtractor(BaseFeatureExtractor):
    """
    Lightweight 3D CNN for resource-constrained environments.
    
    Smaller model suitable for Colab or limited GPU memory.
    Output dimension: 512
    """
    
    def __init__(self, device: str = 'cuda'):
        super().__init__(device)
        self.feature_dim = 512
        
        self.model = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        
        self.model = self.model.to(device)
        self.model.eval()
        print("✅ Loaded lightweight 3D CNN feature extractor")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# YOLO Object Feature Extractor
# ============================================================================

class YOLOObjectFeatureExtractor:
    """
    Object-centric feature extractor using YOLOv8.
    
    Extracts object detection features (class counts + bounding box statistics).
    Output dimension: 83 (80 COCO classes + 3 bbox stats)
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cuda"):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.model.model.eval()
            self.device = device
            
            # COCO classes + bbox stats
            self.num_classes = 80
            self.feature_dim = 80 + 3  # counts + (mean w, h, conf)
            
            print(f"✅ Loaded YOLOv8 object detector (feature_dim={self.feature_dim})")
        
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
    
    def extract_segment_features(self, frames_np: List[np.ndarray]) -> np.ndarray:
        """
        Extract YOLO features from video segment.
        
        Args:
            frames_np: List of RGB uint8 images [H, W, 3]
        
        Returns:
            Feature vector [feature_dim]
        """
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
                
                bbox_stats.append([
                    (x2 - x1).item() if hasattr(x2 - x1, 'item') else (x2 - x1),
                    (y2 - y1).item() if hasattr(y2 - y1, 'item') else (y2 - y1),
                    conf
                ])
        
        # Normalize counts by number of frames
        obj_counts /= max(len(frames_np), 1)
        
        if bbox_stats:
            bbox_stats = np.array(bbox_stats)
            mean = bbox_stats.mean(axis=0)
            std = bbox_stats.std(axis=0)
            bbox_features = np.concatenate([mean, std])
        else:
            bbox_features = np.zeros(6, dtype=np.float32)
        
        return np.concatenate([obj_counts, bbox_features])


class YOLOFeatureAdapter:
    """Adapter for using YOLO in multi-stream pipelines."""
    
    def __init__(self, yolo_extractor: YOLOObjectFeatureExtractor, device: str = "cuda"):
        self.yolo = yolo_extractor
        self.device = device
        self.feature_dim = yolo_extractor.feature_dim
    
    def extract_features(self, segments: List[torch.Tensor]) -> np.ndarray:
        """
        Extract YOLO features from video segments.
        
        Args:
            segments: List of tensors [T, C, H, W]
        
        Returns:
            Feature array [num_segments, feature_dim]
        """
        features = []
        
        for seg in segments:
            frames_np = [
                (f.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                for f in seg
            ]
            feat = self.yolo.extract_segment_features(frames_np)
            features.append(feat)
        
        return np.stack(features)


# ============================================================================
# Feature Extractor Factory
# ============================================================================

class FeatureExtractorFactory:
    """Factory for creating feature extractors."""
    
    @staticmethod
    def create_extractor(model_type: str = 'i3d', device: str = 'cuda') -> BaseFeatureExtractor:
        """
        Create feature extractor.
        
        Args:
            model_type: 'i3d', 'r3d', 'lightweight'
            device: 'cuda' or 'cpu'
        
        Returns:
            Feature extractor instance
        """
        model_type = model_type.lower()
        
        if model_type == 'i3d':
            return I3DFeatureExtractor(device=device)
        elif model_type == 'r3d':
            return R3DFeatureExtractor(device=device)
        elif model_type == 'lightweight':
            return LightweightFeatureExtractor(device=device)
        else:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Choose from: 'i3d', 'r3d', 'lightweight'")


# ============================================================================
# Two-Stream Feature Extractor (Fusion)
# ============================================================================

class TwoStreamFeatureExtractor:
    """
    Fuses motion (3D CNN) and object (YOLO) features.
    
    Final feature dimension: motion_dim + object_dim
    Default: 2048 (I3D) + 83 (YOLO) = 2131
    """
    
    def __init__(
        self,
        motion_extractor: BaseFeatureExtractor,
        object_extractor: YOLOFeatureAdapter
    ):
        self.motion_extractor = motion_extractor
        self.object_extractor = object_extractor
        
        self.feature_dim = (
            motion_extractor.feature_dim +
            object_extractor.feature_dim
        )
    
    def extract_features(self, segments: List[torch.Tensor]) -> np.ndarray:
        """
        Extract fused features from video segments.
        
        Args:
            segments: List of segments [segment_length, C, H, W]
        
        Returns:
            Fused features [num_segments, motion_dim + object_dim]
        """
        motion_feats = self.motion_extractor.extract_features(segments)
        object_feats = self.object_extractor.extract_features(segments)
        
        assert motion_feats.shape[0] == object_feats.shape[0], \
            f"Feature count mismatch: {motion_feats.shape[0]} vs {object_feats.shape[0]}"
        
        return np.concatenate([motion_feats, object_feats], axis=1)
