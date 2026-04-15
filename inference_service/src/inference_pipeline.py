"""
inference_service/src/inference_pipeline.py
--------------------------
Unified inference pipeline for production deployment.

Combines:
  - VideoPreprocessor (frame preprocessing)
  - FeatureExtractionPipeline (I3D + YOLO feature extraction)
  - AnomalyDetector (BiGRU temporal model)

Provides single entry point for Node.js backend.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import torch

from src.models.anomaly_detector import AnomalyDetector
from src.models.video_preprocessor import VideoPreprocessor
from src.models.feature_extractors import (
    I3DFeatureExtractor,
    YOLOObjectFeatureExtractor,
    YOLOFeatureAdapter,
    TwoStreamFeatureExtractor
)
from src.data.labels import UCF_CRIME_CATEGORIES, get_class_name
from src.utils.checkpointing import load_model_from_checkpoint
from inference_service.src.spatial_localizer import SpatialLocalizer, SpatialLocalization


class InferencePipeline:
    """
    Production inference pipeline for real-time anomaly detection.
    
    **Entry Point for Node.js Backend**
    
    Workflow:
    1. Receives raw RGB frames (numpy arrays)
    2. Preprocesses frames (resize, normalize, segment)
    3. Extracts features (I3D motion + YOLO objects)
    4. Runs anomaly detection (BiGRU model)
    5. Returns anomaly score + classification
    
    Parameters
    ----------
    checkpoint_path : str | Path
        Path to trained AnomalyDetector checkpoint (.pt file)
    config : dict | None
        Configuration dict. If None, uses defaults.
    device : str | None
        Device ('cuda', 'cpu', or None for auto-detect)
    threshold : float
        Anomaly threshold [0.0-1.0]. Scores > threshold → anomaly
    features_dir : str | Path | None
        Optional directory to save extracted features
    metadata_dir : str | Path | None
        Optional directory to save inference metadata
    
    Example
    -------
    >>> pipeline = InferencePipeline(
    ...     checkpoint_path="models/best_model.pt",
    ...     threshold=0.5
    ... )
    >>> 
    >>> # Get frames from camera
    >>> frames = [frame1, frame2, ...]  # List of RGB numpy arrays
    >>> 
    >>> # Run inference
    >>> result = pipeline.predict(frames)
    >>> 
    >>> # Check result
    >>> if result['is_anomaly']:
    ...     print(f"⚠️ Anomaly detected: {result['predicted_class']}")
    ...     print(f"   Score: {result['anomaly_score']:.3f}")
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
        features_dir: Optional[str | Path] = None,
        metadata_dir: Optional[str | Path] = None,
        enable_localization: bool = True,
        localization_strategy: str = 'object',
    ):
        # Configuration
        self.config = config or self._default_config()
        self.threshold = threshold
        
        # Spatial localization
        self.enable_localization = enable_localization
        self.spatial_localizer = None
        if enable_localization:
            self.spatial_localizer = SpatialLocalizer(
                strategy=localization_strategy,
                min_confidence=0.25,
                nms_threshold=0.45
            )
            print(f"🎯 Spatial localization enabled (strategy: {localization_strategy})")
        
        # Directories
        self.features_dir = Path(features_dir) if features_dir else None
        self.metadata_dir = Path(metadata_dir) if metadata_dir else None
        
        if self.features_dir:
            self.features_dir.mkdir(parents=True, exist_ok=True)
        if self.metadata_dir:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        print(f"🖥️  Inference device: {self.device}")
        
        # Load components
        self._load_preprocessor()
        self._load_feature_extractors()
        self._load_model(checkpoint_path)
        
        print("✅ Inference pipeline ready")
    
    def _default_config(self) -> dict:
        """Default configuration."""
        return {
            'frame_size': (224, 224),
            'segment_length': 16,
            'input_size': 2131,  # I3D (2048) + YOLO (83)
            'hidden_size': 256,
            'num_classes': 14,
        }
    
    def _load_preprocessor(self):
        """Load video preprocessor."""
        print("📦 Loading VideoPreprocessor...")
        
        self.preprocessor = VideoPreprocessor(
            frame_size=self.config['frame_size'],
            segment_length=self.config['segment_length']
        )
        
        print(f"   Frame size: {self.config['frame_size']}")
        print(f"   Segment length: {self.config['segment_length']}")
    
    def _load_feature_extractors(self):
        """Load I3D + YOLO feature extractors."""
        print("📦 Loading feature extractors...")
        
        # I3D for motion features
        self.i3d_extractor = I3DFeatureExtractor(device=str(self.device))
        
        # YOLO for object features (keep reference to raw extractor for detections)
        self.yolo_raw_extractor = YOLOObjectFeatureExtractor(device=str(self.device))
        self.yolo_extractor = YOLOFeatureAdapter(self.yolo_raw_extractor, device=str(self.device))
        
        # Combined extractor
        self.feature_extractor = TwoStreamFeatureExtractor(
            motion_extractor=self.i3d_extractor,
            object_extractor=self.yolo_extractor
        )
        
        print(f"   Feature dim: {self.feature_extractor.feature_dim}")
    
    def _load_model(self, checkpoint_path: Optional[str | Path]):
        """Load trained anomaly detector."""
        if checkpoint_path is None:
            # Use default checkpoint path
            checkpoint_path = Path(__file__).parent.parent / "models" / "best_model.pt"
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}\n"
                f"Please provide a valid checkpoint path or train a model first."
            )
        
        print(f"📦 Loading AnomalyDetector from: {checkpoint_path}")
        
        self.model = load_model_from_checkpoint(
            str(checkpoint_path),
            self.device
        )
        
        self.model.eval()
        print("   Model ready for inference")
    
    # ========================================================================
    # Main Inference Entry Point
    # ========================================================================
    
    def predict(
        self,
        frames: List[np.ndarray],
        save_features: bool = False,
        timestamp: Optional[str] = None,
        return_visualization: bool = False
    ) -> Dict:
        """
        Run inference on a batch of frames.
        
        **Main Entry Point for Node.js Backend**
        
        Parameters
        ----------
        frames : List[np.ndarray]
            List of RGB frames as numpy arrays, shape (H, W, 3) uint8.
            Minimum 1 frame, will be padded/sampled to segment_length.
        save_features : bool
            Whether to save extracted features to disk
        timestamp : str | None
            Optional timestamp for this batch (ISO 8601 format)
        return_visualization : bool
            Whether to return visualization frame with bounding boxes
        
        Returns
        -------
        dict
            Inference result with keys:
            - anomaly_score : float [0.0-1.0]
            - is_anomaly : bool
            - predicted_class : str
            - confidence : float
            - threshold_used : float
            - localization : dict (bounding boxes, if enabled)
            - metadata : dict (optional)
        
        Example
        -------
        >>> result = pipeline.predict(frames)
        >>> print(f"Score: {result['anomaly_score']:.3f}")
        >>> print(f"Anomaly: {result['is_anomaly']}")
        >>> print(f"Class: {result['predicted_class']}")
        >>> if result['localization']:
        >>>     print(f"Boxes: {len(result['localization']['bounding_boxes'])}")
        """
        # Validate input
        if not frames:
            raise ValueError("Empty frames list")
        
        if not all(isinstance(f, np.ndarray) for f in frames):
            raise ValueError("All frames must be numpy arrays")
        
        if not all(f.ndim == 3 and f.shape[2] == 3 for f in frames):
            raise ValueError("All frames must be RGB (H, W, 3)")
        
        # Get frame dimensions
        frame_height, frame_width = frames[0].shape[:2]
        
        # Preprocess frames
        segments = self.preprocessor.to_segments(frames)
        
        if not segments:
            raise ValueError("Failed to create segments from frames")
        
        # Extract features AND capture YOLO detections for localization
        yolo_detections = []
        if self.enable_localization:
            # Run YOLO on original frames to get detections
            yolo_detections = self._extract_yolo_detections(frames)
        
        # Extract combined features for anomaly detection
        features = self.feature_extractor.extract_features(segments)
        
        # Save features if requested
        feature_path = None
        if save_features and self.features_dir:
            feature_path = self._save_features(features, timestamp)
        
        # Run anomaly detection
        with torch.no_grad():
            # Convert to tensor: [1, num_segments, feature_dim]
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            
            # Forward pass
            anomaly_scores, class_probs = self.model(features_tensor)
            
            # Get video-level score (max across segments)
            video_score = anomaly_scores.squeeze().max().item()
            
            # Get predicted class (mean probs across segments)
            mean_probs = class_probs.squeeze().mean(dim=0)
            pred_class_idx = torch.argmax(mean_probs).item()
            confidence = mean_probs[pred_class_idx].item()
        
        # Threshold decision
        is_anomaly = video_score > self.threshold
        
        # Get class name
        predicted_class = get_class_name(pred_class_idx)
        
        # Spatial localization (if enabled and anomaly detected)
        localization_result = None
        visualization_frame = None
        
        if self.enable_localization and self.spatial_localizer and yolo_detections:
            localization = self.spatial_localizer.localize(
                yolo_detections=yolo_detections,
                anomaly_score=video_score,
                frame_shape=(frame_height, frame_width),
                frames=frames if self.spatial_localizer.strategy == 'gradient' else None
            )
            
            localization_result = localization.to_dict()
            
            # Generate visualization if requested
            if return_visualization and frames:
                from src.spatial_localizer import visualize_localization
                # Use middle frame for visualization
                mid_frame_idx = len(frames) // 2
                visualization_frame = visualize_localization(
                    frames[mid_frame_idx],
                    localization
                )
        
        # Build result
        result = {
            'anomaly_score': float(video_score),
            'is_anomaly': bool(is_anomaly),
            'predicted_class': str(predicted_class),
            'confidence': float(confidence),
            'threshold_used': float(self.threshold),
        }
        
        # Add localization if available
        if localization_result:
            result['localization'] = localization_result
        
        # Optional metadata
        metadata = {}
        if save_features or feature_path:
            metadata['feature_path'] = str(feature_path) if feature_path else None
            metadata['num_segments'] = len(segments)
            metadata['feature_shape'] = list(features.shape)
        
        metadata['timestamp'] = timestamp or datetime.utcnow().isoformat() + "Z"
        metadata['frame_dimensions'] = {
            'width': frame_width,
            'height': frame_height
        }
        
        if visualization_frame is not None:
            # Convert to base64 for transmission
            import cv2
            import base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(visualization_frame, cv2.COLOR_RGB2BGR))
            metadata['visualization_base64'] = base64.b64encode(buffer).decode('utf-8')
        
        result['metadata'] = metadata
        
        return result
    
    def _extract_yolo_detections(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Extract YOLO detections from frames.
        
        Parameters
        ----------
        frames : List[np.ndarray]
            RGB frames
        
        Returns
        -------
        List[Dict]
            Detection results for each frame
        """
        detections = []
        
        for frame in frames:
            # Run YOLO
            results = self.yolo_raw_extractor.model(frame, verbose=False)
            
            frame_detections = {'boxes': []}
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class name
                    class_name = result.names[cls] if cls < len(result.names) else str(cls)
                    
                    frame_detections['boxes'].append({
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': str(class_name)
                    })
            
            detections.append(frame_detections)
        
        return detections
    
    def _save_features(
        self,
        features: np.ndarray,
        timestamp: Optional[str] = None
    ) -> Path:
        """Save extracted features to disk."""
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat().replace(':', '-')
        
        filename = f"inference_{timestamp}.npz"
        filepath = self.features_dir / filename
        
        np.savez_compressed(
            filepath,
            features=features.astype(np.float32),
            timestamp=timestamp,
            threshold=self.threshold
        )
        
        return filepath
    
    def cleanup(self):
        """Cleanup resources."""
        # Free GPU memory
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'i3d_extractor'):
            del self.i3d_extractor
        if hasattr(self, 'yolo_extractor'):
            del self.yolo_extractor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Pipeline cleanup complete")


# ============================================================================
# Batch Inference (for offline processing)
# ============================================================================

def batch_predict(
    pipeline: InferencePipeline,
    frames_list: List[List[np.ndarray]],
    batch_size: int = 4
) -> List[Dict]:
    """
    Run inference on multiple batches of frames.
    
    Useful for processing multiple camera streams or video clips.
    
    Parameters
    ----------
    pipeline : InferencePipeline
        Initialized inference pipeline
    frames_list : List[List[np.ndarray]]
        List of frame batches (each batch is a list of frames)
    batch_size : int
        Number of batches to process at once
    
    Returns
    -------
    List[Dict]
        List of prediction results
    """
    results = []
    
    for i in range(0, len(frames_list), batch_size):
        batch = frames_list[i:i + batch_size]
        
        for frames in batch:
            result = pipeline.predict(frames)
            results.append(result)
    
    return results