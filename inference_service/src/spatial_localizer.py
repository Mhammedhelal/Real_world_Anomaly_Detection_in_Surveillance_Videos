"""
inference_service/src/spatial_localizer.py
------------------------
Spatial localization of anomalies using YOLO detections and anomaly scores.

Strategy:
1. YOLO provides object bounding boxes with class probabilities
2. Anomaly detector provides temporal anomaly score
3. Combine to identify which spatial regions are anomalous

Approaches:
- Object-based: Attribute anomaly to detected objects
- Region-based: Divide frame into grid, score each region
- Attention-based: Use gradient-based saliency maps
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import cv2
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Bounding box with confidence and class."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    anomaly_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'x1': float(self.x1),
            'y1': float(self.y1),
            'x2': float(self.x2),
            'y2': float(self.y2),
            'width': float(self.x2 - self.x1),
            'height': float(self.y2 - self.y1),
            'confidence': float(self.confidence),
            'class_id': int(self.class_id),
            'class_name': str(self.class_name),
            'anomaly_score': float(self.anomaly_score)
        }
    
    def area(self) -> float:
        """Calculate box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another box."""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class SpatialLocalization:
    """Complete spatial localization result."""
    bounding_boxes: List[BoundingBox]
    anomaly_heatmap: Optional[np.ndarray] = None
    frame_width: int = 0
    frame_height: int = 0
    primary_anomaly_box: Optional[BoundingBox] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'bounding_boxes': [box.to_dict() for box in self.bounding_boxes],
            'num_detections': len(self.bounding_boxes),
            'frame_dimensions': {
                'width': self.frame_width,
                'height': self.frame_height
            },
            'primary_anomaly': self.primary_anomaly_box.to_dict() if self.primary_anomaly_box else None
        }


class SpatialLocalizer:
    """
    Localizes anomalies in video frames.
    
    Three strategies supported:
    1. Object-based: Attribute anomaly to YOLO-detected objects
    2. Region-based: Divide frame into grid and score regions
    3. Gradient-based: Use attention maps from model
    
    Parameters
    ----------
    strategy : str
        'object', 'region', or 'gradient'
    min_confidence : float
        Minimum YOLO detection confidence
    nms_threshold : float
        Non-maximum suppression threshold
    grid_size : tuple
        Grid dimensions for region-based localization
    """
    
    def __init__(
        self,
        strategy: str = 'object',
        min_confidence: float = 0.25,
        nms_threshold: float = 0.45,
        grid_size: Tuple[int, int] = (8, 8)
    ):
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.grid_size = grid_size
    
    def localize(
        self,
        yolo_detections: List[Dict],
        anomaly_score: float,
        frame_shape: Tuple[int, int],
        frames: Optional[List[np.ndarray]] = None
    ) -> SpatialLocalization:
        """
        Localize anomaly in frame.
        
        Parameters
        ----------
        yolo_detections : List[Dict]
            YOLO detection results for all frames in segment
        anomaly_score : float
            Overall anomaly score for the segment
        frame_shape : Tuple[int, int]
            (height, width) of frames
        frames : List[np.ndarray], optional
            Original frames (needed for gradient-based)
        
        Returns
        -------
        SpatialLocalization
            Bounding boxes with anomaly attribution
        """
        if self.strategy == 'object':
            return self._object_based_localization(
                yolo_detections, anomaly_score, frame_shape
            )
        elif self.strategy == 'region':
            return self._region_based_localization(
                yolo_detections, anomaly_score, frame_shape
            )
        elif self.strategy == 'gradient':
            return self._gradient_based_localization(
                yolo_detections, anomaly_score, frame_shape, frames
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _object_based_localization(
        self,
        yolo_detections: List[Dict],
        anomaly_score: float,
        frame_shape: Tuple[int, int]
    ) -> SpatialLocalization:
        """
        Object-based strategy: Attribute anomaly to detected objects.
        
        Logic:
        1. Aggregate YOLO detections across all frames
        2. Filter by confidence
        3. Apply NMS (non-maximum suppression)
        4. Score each object based on:
           - Object class (some classes more likely anomalous)
           - Temporal consistency
           - Spatial coverage
           - Overall anomaly score
        """
        height, width = frame_shape
        
        # Aggregate detections across frames
        all_boxes = []
        for frame_detections in yolo_detections:
            for det in frame_detections.get('boxes', []):
                if det['confidence'] >= self.min_confidence:
                    all_boxes.append(det)
        
        if not all_boxes:
            # No detections - return empty localization
            return SpatialLocalization(
                bounding_boxes=[],
                frame_width=width,
                frame_height=height
            )
        
        # Convert to BoundingBox objects
        boxes = []
        for det in all_boxes:
            box = BoundingBox(
                x1=det['x1'],
                y1=det['y1'],
                x2=det['x2'],
                y2=det['y2'],
                confidence=det['confidence'],
                class_id=det['class_id'],
                class_name=det['class_name']
            )
            boxes.append(box)
        
        # Apply NMS
        boxes = self._apply_nms(boxes)
        
        # Score each box
        boxes = self._score_boxes(boxes, anomaly_score)
        
        # Find primary anomaly
        primary = max(boxes, key=lambda b: b.anomaly_score) if boxes else None
        
        return SpatialLocalization(
            bounding_boxes=boxes,
            frame_width=width,
            frame_height=height,
            primary_anomaly_box=primary
        )
    
    def _region_based_localization(
        self,
        yolo_detections: List[Dict],
        anomaly_score: float,
        frame_shape: Tuple[int, int]
    ) -> SpatialLocalization:
        """
        Region-based strategy: Divide frame into grid and score regions.
        
        Logic:
        1. Divide frame into grid (e.g., 8x8)
        2. For each cell, count YOLO detections
        3. Weight cells by detection density and object classes
        4. Combine with overall anomaly score
        5. Return high-scoring regions as bounding boxes
        """
        height, width = frame_shape
        grid_h, grid_w = self.grid_size
        
        cell_height = height / grid_h
        cell_width = width / grid_w
        
        # Create grid scores
        grid_scores = np.zeros((grid_h, grid_w))
        
        # Aggregate detections into grid cells
        for frame_detections in yolo_detections:
            for det in frame_detections.get('boxes', []):
                if det['confidence'] < self.min_confidence:
                    continue
                
                # Calculate which grid cells this box overlaps
                center_x = (det['x1'] + det['x2']) / 2
                center_y = (det['y1'] + det['y2']) / 2
                
                cell_x = int(center_x / cell_width)
                cell_y = int(center_y / cell_height)
                
                # Ensure within bounds
                cell_x = max(0, min(grid_w - 1, cell_x))
                cell_y = max(0, min(grid_h - 1, cell_y))
                
                # Add score (weighted by confidence and anomaly score)
                grid_scores[cell_y, cell_x] += det['confidence'] * anomaly_score
        
        # Normalize
        if grid_scores.max() > 0:
            grid_scores /= grid_scores.max()
        
        # Find high-scoring regions (threshold at 0.3 * max score)
        threshold = 0.3 * grid_scores.max() if grid_scores.max() > 0 else 0
        
        boxes = []
        for i in range(grid_h):
            for j in range(grid_w):
                if grid_scores[i, j] >= threshold:
                    box = BoundingBox(
                        x1=j * cell_width,
                        y1=i * cell_height,
                        x2=(j + 1) * cell_width,
                        y2=(i + 1) * cell_height,
                        confidence=grid_scores[i, j],
                        class_id=-1,  # Region, not object
                        class_name='region',
                        anomaly_score=grid_scores[i, j]
                    )
                    boxes.append(box)
        
        # Merge adjacent boxes
        boxes = self._merge_adjacent_boxes(boxes, cell_width, cell_height)
        
        primary = max(boxes, key=lambda b: b.anomaly_score) if boxes else None
        
        return SpatialLocalization(
            bounding_boxes=boxes,
            anomaly_heatmap=grid_scores,
            frame_width=width,
            frame_height=height,
            primary_anomaly_box=primary
        )
    
    def _gradient_based_localization(
        self,
        yolo_detections: List[Dict],
        anomaly_score: float,
        frame_shape: Tuple[int, int],
        frames: Optional[List[np.ndarray]] = None
    ) -> SpatialLocalization:
        """
        Gradient-based strategy: Use attention/saliency maps.
        
        This is a placeholder - full implementation requires:
        1. Model with attention mechanisms
        2. Gradient computation through model
        3. CAM/Grad-CAM style visualization
        
        For now, fall back to object-based.
        """
        # TODO: Implement gradient-based attribution
        # For now, use object-based as fallback
        return self._object_based_localization(
            yolo_detections, anomaly_score, frame_shape
        )
    
    def _apply_nms(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """Apply non-maximum suppression to remove duplicate detections."""
        if not boxes:
            return []
        
        # Sort by confidence
        boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
        
        keep = []
        while boxes:
            # Keep highest confidence box
            current = boxes.pop(0)
            keep.append(current)
            
            # Remove boxes with high IoU
            boxes = [
                box for box in boxes
                if current.iou(box) < self.nms_threshold
            ]
        
        return keep
    
    def _score_boxes(
        self,
        boxes: List[BoundingBox],
        anomaly_score: float
    ) -> List[BoundingBox]:
        """
        Score bounding boxes based on anomaly likelihood.
        
        Factors:
        1. Object class (person, vehicle more likely in normal scenes)
        2. Detection confidence
        3. Overall anomaly score
        4. Box size (unusual sizes may indicate anomaly)
        """
        # Anomalous object classes (from COCO dataset)
        # These are more likely to be involved in anomalies
        anomalous_classes = {
            # Weapons/dangerous objects
            'knife', 'scissors', 'baseball bat', 'tennis racket',
            # Unusual objects in surveillance
            'suitcase', 'backpack', 'handbag',
            # Fire-related
            'fire hydrant',
        }
        
        # Normal classes (commonly seen in normal surveillance)
        normal_classes = {
            'person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck',
            'traffic light', 'stop sign', 'bench', 'chair'
        }
        
        for box in boxes:
            # Base score from anomaly detector
            base_score = anomaly_score
            
            # Class weight
            class_weight = 1.0
            if box.class_name.lower() in anomalous_classes:
                class_weight = 1.5  # Boost anomalous classes
            elif box.class_name.lower() in normal_classes:
                class_weight = 0.7  # Reduce normal classes
            
            # Confidence weight
            conf_weight = box.confidence
            
            # Size weight (very large or very small objects unusual)
            area = box.area()
            size_weight = 1.0
            if area < 0.01:  # Very small (< 1% of frame)
                size_weight = 1.2
            elif area > 0.5:  # Very large (> 50% of frame)
                size_weight = 1.2
            
            # Combined score
            box.anomaly_score = base_score * class_weight * conf_weight * size_weight
            
            # Clip to [0, 1]
            box.anomaly_score = np.clip(box.anomaly_score, 0.0, 1.0)
        
        return boxes
    
    def _merge_adjacent_boxes(
        self,
        boxes: List[BoundingBox],
        cell_width: float,
        cell_height: float
    ) -> List[BoundingBox]:
        """Merge adjacent grid cells into larger regions."""
        if not boxes:
            return []
        
        # Sort by position
        boxes = sorted(boxes, key=lambda b: (b.y1, b.x1))
        
        merged = []
        current = boxes[0]
        
        for box in boxes[1:]:
            # Check if adjacent (within 1.5 cells)
            if (abs(box.x1 - current.x2) < 1.5 * cell_width and
                abs(box.y1 - current.y1) < 0.5 * cell_height):
                # Merge
                current = BoundingBox(
                    x1=min(current.x1, box.x1),
                    y1=min(current.y1, box.y1),
                    x2=max(current.x2, box.x2),
                    y2=max(current.y2, box.y2),
                    confidence=max(current.confidence, box.confidence),
                    class_id=-1,
                    class_name='region',
                    anomaly_score=max(current.anomaly_score, box.anomaly_score)
                )
            else:
                merged.append(current)
                current = box
        
        merged.append(current)
        return merged


def visualize_localization(
    frame: np.ndarray,
    localization: SpatialLocalization,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on frame.
    
    Parameters
    ----------
    frame : np.ndarray
        Input frame (RGB)
    localization : SpatialLocalization
        Localization result
    thickness : int
        Box line thickness
    
    Returns
    -------
    np.ndarray
        Frame with boxes drawn
    """
    vis_frame = frame.copy()
    
    for box in localization.bounding_boxes:
        # Color based on anomaly score (green → yellow → red)
        score = box.anomaly_score
        if score < 0.3:
            color = (0, 255, 0)  # Green (normal)
        elif score < 0.7:
            color = (255, 255, 0)  # Yellow (suspicious)
        else:
            color = (255, 0, 0)  # Red (anomaly)
        
        # Draw box
        cv2.rectangle(
            vis_frame,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            color,
            thickness
        )
        
        # Draw label
        label = f"{box.class_name} {box.anomaly_score:.2f}"
        cv2.putText(
            vis_frame,
            label,
            (int(box.x1), int(box.y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    # Highlight primary anomaly
    if localization.primary_anomaly_box:
        box = localization.primary_anomaly_box
        cv2.rectangle(
            vis_frame,
            (int(box.x1) - 2, int(box.y1) - 2),
            (int(box.x2) + 2, int(box.y2) + 2),
            (255, 0, 255),  # Magenta
            thickness + 2
        )
    
    return vis_frame