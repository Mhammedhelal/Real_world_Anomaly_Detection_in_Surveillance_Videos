"""
inference_service/src/spatial_localizer.py
------------------------------------------
Spatial localisation of anomalies using YOLO detections and anomaly scores.

The BiGRU model tells us *when* an anomaly occurred (which segment, score in
[0,1]) but not *where* in the frame.  This module adds the "where" by
combining the temporal anomaly score with YOLO's spatial bounding boxes.

Strategy
--------
For each batch of frames that the inference pipeline scores as anomalous,
YOLO already ran on every frame to produce the 83-dim feature vector.  We
re-use those raw detections here — no second YOLO pass needed.

Two localisation strategies are supported:

object  — Attribute the anomaly score to specific detected objects.
          Each bounding box receives a composite anomaly score derived
          from:
            • The overall segment anomaly score
            • The object class (people/vehicles score higher for crimes)
            • Detection confidence
            • Unusual object size (very small or very large)
          The highest-scoring box is flagged as the primary anomaly region.

region  — Divide the frame into a grid (default 8×8), accumulate YOLO
          detection density into each cell weighted by anomaly score, then
          merge adjacent hot cells into contiguous regions.  Useful when
          no single object dominates (e.g. crowd panic, arson).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Single detection with anomaly attribution score."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float       # YOLO detection confidence
    class_id: int
    class_name: str
    anomaly_score: float = 0.0  # anomaly attribution for this box

    # ---------- geometry helpers ----------

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection-over-Union with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "x1": float(self.x1), "y1": float(self.y1),
            "x2": float(self.x2), "y2": float(self.y2),
            "width": float(self.width), "height": float(self.height),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "anomaly_score": float(self.anomaly_score),
        }


@dataclass
class LocalisationResult:
    """Spatial localisation output for one batch of frames."""
    strategy: str                          # "object" or "region"
    frame_width: int
    frame_height: int
    segment_anomaly_score: float           # the BiGRU score that triggered this
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    primary_box: Optional[BoundingBox] = None   # highest-scoring box
    heatmap: Optional[np.ndarray] = None        # region strategy only, shape [H,W] float32

    @property
    def num_detections(self) -> int:
        return len(self.bounding_boxes)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "segment_anomaly_score": self.segment_anomaly_score,
            "num_detections": self.num_detections,
            "bounding_boxes": [b.to_dict() for b in self.bounding_boxes],
            "primary_box": self.primary_box.to_dict() if self.primary_box else None,
        }


# ---------------------------------------------------------------------------
# Anomaly class weights
# COCO class names that are contextually relevant to UCF-Crime categories
# ---------------------------------------------------------------------------

# Classes whose presence is a strong signal of the relevant crime type
_HIGH_RISK_CLASSES = {
    "knife", "scissors", "baseball bat", "tennis racket",
    "suitcase", "backpack", "handbag",        # common in robbery / shoplifting
    "fire hydrant",                            # proxy for fire proximity (arson)
    "car", "truck", "bus",                     # road accidents
}

# Classes that appear in almost every normal scene — downweight them
_NORMAL_CLASSES = {
    "person", "bicycle", "motorcycle",
    "traffic light", "stop sign",
    "bench", "chair", "potted plant",
}


def _class_weight(class_name: str) -> float:
    name = class_name.lower()
    if name in _HIGH_RISK_CLASSES:
        return 1.6
    if name in _NORMAL_CLASSES:
        return 0.7
    return 1.0


def _size_weight(box: BoundingBox, frame_area: float) -> float:
    """Unusual object sizes (very small or very large) get a slight boost."""
    if frame_area <= 0:
        return 1.0
    frac = box.area / frame_area
    if frac < 0.005 or frac > 0.4:
        return 1.2
    return 1.0


# ---------------------------------------------------------------------------
# Core localiser
# ---------------------------------------------------------------------------

class SpatialLocalizer:
    """
    Localises anomalies in video frames given YOLO detections and an
    anomaly score from the BiGRU model.

    Parameters
    ----------
    strategy : "object" | "region"
        Localisation strategy (see module docstring).
    min_confidence : float
        Minimum YOLO detection confidence to use.
    nms_iou_threshold : float
        IoU threshold for non-maximum suppression.
    grid_rows, grid_cols : int
        Grid dimensions for the "region" strategy.
    """

    def __init__(
        self,
        strategy: str = "object",
        min_confidence: float = 0.25,
        nms_iou_threshold: float = 0.45,
        grid_rows: int = 8,
        grid_cols: int = 8,
    ) -> None:
        if strategy not in ("object", "region"):
            raise ValueError(f"strategy must be 'object' or 'region', got {strategy!r}")
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def localise(
        self,
        yolo_detections: List[Dict],
        anomaly_score: float,
        frame_shape: Tuple[int, int],   # (height, width)
    ) -> LocalisationResult:
        """
        Compute spatial localisation for one inference batch.

        Parameters
        ----------
        yolo_detections : List[Dict]
            One dict per frame.  Each dict has key ``"boxes"`` — a list of
            dicts with keys ``x1, y1, x2, y2, confidence, class_id,
            class_name``.  This is exactly the format produced by
            ``YOLOObjectFeatureExtractor`` when called with ``verbose=False``.
        anomaly_score : float
            Segment-level anomaly score from the BiGRU model (0–1).
        frame_shape : (height, width)
            Pixel dimensions of the original frames.

        Returns
        -------
        LocalisationResult
        """
        height, width = frame_shape
        result = LocalisationResult(
            strategy=self.strategy,
            frame_width=width,
            frame_height=height,
            segment_anomaly_score=anomaly_score,
        )

        if self.strategy == "object":
            self._object_strategy(yolo_detections, anomaly_score, result)
        else:
            self._region_strategy(yolo_detections, anomaly_score, result)

        return result

    # ------------------------------------------------------------------
    # Object strategy
    # ------------------------------------------------------------------

    def _object_strategy(
        self,
        yolo_detections: List[Dict],
        anomaly_score: float,
        result: LocalisationResult,
    ) -> None:
        # Collect all confident detections across all frames in the batch
        raw_boxes: List[BoundingBox] = []
        for frame_det in yolo_detections:
            for det in frame_det.get("boxes", []):
                if det["confidence"] < self.min_confidence:
                    continue
                raw_boxes.append(BoundingBox(
                    x1=float(det["x1"]), y1=float(det["y1"]),
                    x2=float(det["x2"]), y2=float(det["y2"]),
                    confidence=float(det["confidence"]),
                    class_id=int(det["class_id"]),
                    class_name=str(det["class_name"]),
                ))

        if not raw_boxes:
            return

        # NMS to remove duplicate detections across frames
        boxes = _nms(raw_boxes, self.nms_iou_threshold)

        # Score each box
        frame_area = float(result.frame_width * result.frame_height)
        for box in boxes:
            box.anomaly_score = float(np.clip(
                anomaly_score
                * _class_weight(box.class_name)
                * float(box.confidence)
                * _size_weight(box, frame_area),
                0.0, 1.0,
            ))

        # Sort by anomaly score descending
        boxes.sort(key=lambda b: b.anomaly_score, reverse=True)
        result.bounding_boxes = boxes
        result.primary_box = boxes[0] if boxes else None

    # ------------------------------------------------------------------
    # Region strategy
    # ------------------------------------------------------------------

    def _region_strategy(
        self,
        yolo_detections: List[Dict],
        anomaly_score: float,
        result: LocalisationResult,
    ) -> None:
        h, w = result.frame_height, result.frame_width
        cell_h = h / self.grid_rows
        cell_w = w / self.grid_cols
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        for frame_det in yolo_detections:
            for det in frame_det.get("boxes", []):
                if det["confidence"] < self.min_confidence:
                    continue
                cx = (det["x1"] + det["x2"]) / 2
                cy = (det["y1"] + det["y2"]) / 2
                col = int(np.clip(cx / cell_w, 0, self.grid_cols - 1))
                row = int(np.clip(cy / cell_h, 0, self.grid_rows - 1))
                grid[row, col] += float(det["confidence"]) * anomaly_score

        # Normalise
        if grid.max() > 0:
            grid /= grid.max()

        result.heatmap = grid

        # Threshold at 30 % of peak to find hot cells
        threshold = 0.3 * grid.max() if grid.max() > 0 else 0.0
        raw_boxes: List[BoundingBox] = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if grid[row, col] >= threshold:
                    raw_boxes.append(BoundingBox(
                        x1=col * cell_w, y1=row * cell_h,
                        x2=(col + 1) * cell_w, y2=(row + 1) * cell_h,
                        confidence=float(grid[row, col]),
                        class_id=-1, class_name="region",
                        anomaly_score=float(grid[row, col]),
                    ))

        # Merge adjacent cells into larger regions
        boxes = _merge_adjacent(raw_boxes, cell_w, cell_h)
        result.bounding_boxes = boxes
        result.primary_box = max(boxes, key=lambda b: b.anomaly_score) if boxes else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nms(boxes: List[BoundingBox], iou_threshold: float) -> List[BoundingBox]:
    """Simple greedy NMS — keeps highest-confidence box, suppresses overlaps."""
    boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: List[BoundingBox] = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if best.iou(b) < iou_threshold]
    return kept


def _merge_adjacent(
    boxes: List[BoundingBox],
    cell_w: float,
    cell_h: float,
) -> List[BoundingBox]:
    """Merge horizontally adjacent grid cells into contiguous regions."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b.y1, b.x1))
    merged = [boxes[0]]
    for box in boxes[1:]:
        prev = merged[-1]
        # Adjacent horizontally (within 1.5 cells) and same row
        if abs(box.x1 - prev.x2) < 1.5 * cell_w and abs(box.y1 - prev.y1) < 0.5 * cell_h:
            merged[-1] = BoundingBox(
                x1=min(prev.x1, box.x1), y1=min(prev.y1, box.y1),
                x2=max(prev.x2, box.x2), y2=max(prev.y2, box.y2),
                confidence=max(prev.confidence, box.confidence),
                class_id=-1, class_name="region",
                anomaly_score=max(prev.anomaly_score, box.anomaly_score),
            )
        else:
            merged.append(box)
    return merged


# ---------------------------------------------------------------------------
# Visualisation helper (OpenCV, optional)
# ---------------------------------------------------------------------------

def draw_localisation(
    frame: np.ndarray,
    localisation: LocalisationResult,
    draw_all: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes on a frame (in-place copy).

    Colour coding:
        Green  (score < 0.4)  — low anomaly attribution
        Yellow (score < 0.7)  — moderate
        Red    (score ≥ 0.7)  — high anomaly attribution
    Primary box has a thicker magenta outline.

    Parameters
    ----------
    frame : np.ndarray  H×W×3 uint8 BGR
    localisation : LocalisationResult
    draw_all : bool  — if False, only draw the primary box
    thickness : int

    Returns
    -------
    np.ndarray  annotated copy
    """
    vis = frame.copy()
    boxes = localisation.bounding_boxes if draw_all else []
    if localisation.primary_box and not draw_all:
        boxes = [localisation.primary_box]

    for box in boxes:
        s = box.anomaly_score
        if s < 0.4:
            color = (0, 255, 0)       # green
        elif s < 0.7:
            color = (0, 255, 255)     # yellow
        else:
            color = (0, 0, 255)       # red

        cv2.rectangle(vis,
                      (int(box.x1), int(box.y1)),
                      (int(box.x2), int(box.y2)),
                      color, thickness)
        label = f"{box.class_name} {box.anomaly_score:.2f}"
        cv2.putText(vis, label, (int(box.x1), max(int(box.y1) - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Highlight primary box in magenta
    if localisation.primary_box:
        pb = localisation.primary_box
        cv2.rectangle(vis,
                      (int(pb.x1) - 2, int(pb.y1) - 2),
                      (int(pb.x2) + 2, int(pb.y2) + 2),
                      (255, 0, 255), thickness + 2)

    return vis