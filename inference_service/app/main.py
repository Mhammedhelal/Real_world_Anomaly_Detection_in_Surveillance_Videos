"""
inference_service/app/main.py
------------------------------
FastAPI inference service — entry point for the Node.js back-end team.

Endpoints
---------
GET  /health           → service readiness + model status
POST /predict          → run inference on a batch of base64 frames
GET  /threshold        → return current anomaly threshold
PUT  /threshold        → hot-update threshold (no restart needed)

The Node.js back-end sends frames as base64-encoded JPEG/PNG strings and
receives a structured JSON response with the anomaly score, class, and
spatial bounding boxes.  It never touches PyTorch or the ML pipeline.

Note on spatial localisation
-----------------------------
Bounding boxes are only populated when ``is_anomaly=true``.  The Node.js
layer can use ``localisation.primary_box`` for a single highlighted region,
or ``localisation.bounding_boxes`` for all detected objects with their
individual anomaly attribution scores.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# Add project root to path so src.* imports work inside the container
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from inference_service.src.config_loader import load_inference_config
from inference_service.src.inference_pipeline import (
    RealTimeInferencePipeline,
    InferenceResult,
)
from inference_service.src.spatial_localizer import LocalisationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Anomaly Detection Inference API",
    description="Real-time surveillance anomaly detection — ML back-end",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline (loaded once at startup)
_pipeline: Optional[RealTimeInferencePipeline] = None
_config: dict = {}


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    global _pipeline, _config
    logger.info("Loading inference service…")
    try:
        _config = load_inference_config()
        logger.info("Config: device=%s  threshold=%.3f  strategy=%s",
                    _config["device"],
                    _config["anomaly_threshold"],
                    _config["localizer_strategy"])

        # Build pipeline without starting camera threads.
        # The /predict endpoint runs single-batch inference synchronously;
        # the threaded camera path is used when run_inference.py is called
        # directly for live-stream mode.
        _pipeline = RealTimeInferencePipeline.from_checkpoint(
            checkpoint_path     = _config["checkpoint_path"],
            threshold           = float(_config["anomaly_threshold"]),
            device              = str(_config["device"]),
            localizer_strategy  = str(_config["localizer_strategy"]),
        )
        logger.info("Pipeline ready — model loaded on %s", _pipeline.device)
    except Exception as exc:
        logger.exception("Startup failed: %s", exc)
        raise


@app.on_event("shutdown")
async def shutdown() -> None:
    if _pipeline:
        _pipeline.stop()
    logger.info("Service shut down")


# ---------------------------------------------------------------------------
# Request / response models  (Node.js API contract)
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Sent by the Node.js back-end for each camera segment.

    ``frames`` — list of base64-encoded JPEG or PNG frames (RGB).
                 Length should equal segment_length (default 16).
                 The service pads or truncates silently.

    ``timestamp`` — ISO-8601 string from the client (optional, logged).
    ``save_features`` — if true, save extracted .npz to /app/data/features/.
    """
    frames: List[str] = Field(..., min_length=1, max_length=64,
                               description="Base64-encoded RGB frames")
    timestamp: Optional[str] = None
    save_features: bool = False


class BoundingBoxOut(BaseModel):
    x1: float; y1: float; x2: float; y2: float
    width: float; height: float
    confidence: float
    class_id: int; class_name: str
    anomaly_score: float


class LocalisationOut(BaseModel):
    strategy: str
    num_detections: int
    bounding_boxes: List[BoundingBoxOut]
    primary_box: Optional[BoundingBoxOut]


class PredictResponse(BaseModel):
    # Core decision
    anomaly_score: float = Field(..., ge=0.0, le=1.0,
                                  description="Max segment score [0,1]")
    is_anomaly: bool
    threshold_used: float

    # Classification
    predicted_class: str
    predicted_class_id: int
    class_confidence: float = Field(..., ge=0.0, le=1.0)

    # Temporal
    segment_scores: List[float]
    peak_segment_idx: int

    # Spatial (only populated when is_anomaly=True)
    localisation: Optional[LocalisationOut]

    # Meta
    inference_time_ms: float
    timestamp: str


class ThresholdResponse(BaseModel):
    threshold: float


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    threshold: float
    version: str
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {"service": "Anomaly Detection API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status      = "healthy" if _pipeline else "initialising",
        model_loaded= _pipeline is not None,
        device      = str(_pipeline.device) if _pipeline else "unknown",
        threshold   = float(_pipeline.threshold) if _pipeline else 0.0,
        version     = "1.0.0",
        timestamp   = _utc_now(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Run inference on a batch of frames.

    The Node.js back-end sends base64-encoded frames.
    Returns anomaly score, class, segment scores, and spatial bounding boxes.
    """
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    t_start = time.perf_counter()

    # 1. Decode frames
    try:
        frames = _decode_frames(request.frames)
    except ValueError as exc:
        raise HTTPException(400, f"Frame decode error: {exc}")

    # 2. Run single-batch inference (synchronous — FastAPI handles concurrency)
    result = _run_single_batch(_pipeline, frames)
    if result is None:
        raise HTTPException(500, "Inference failed — check service logs")

    ms_total = (time.perf_counter() - t_start) * 1000.0

    # 3. Serialise localisation
    localisation_out: Optional[LocalisationOut] = None
    if result.localisation:
        loc = result.localisation
        localisation_out = LocalisationOut(
            strategy        = loc.strategy,
            num_detections  = loc.num_detections,
            bounding_boxes  = [_box_to_out(b) for b in loc.bounding_boxes],
            primary_box     = _box_to_out(loc.primary_box) if loc.primary_box else None,
        )

    return PredictResponse(
        anomaly_score     = result.anomaly_score,
        is_anomaly        = result.is_anomaly,
        threshold_used    = result.threshold,
        predicted_class   = result.predicted_class,
        predicted_class_id= result.predicted_class_id,
        class_confidence  = result.class_confidence,
        segment_scores    = result.segment_scores,
        peak_segment_idx  = result.peak_segment_idx,
        localisation      = localisation_out,
        inference_time_ms = ms_total,
        timestamp         = request.timestamp or _utc_now(),
    )


@app.get("/threshold", response_model=ThresholdResponse)
async def get_threshold():
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    return ThresholdResponse(threshold=_pipeline.threshold)


@app.put("/threshold", response_model=ThresholdResponse)
async def set_threshold(body: ThresholdUpdate):
    """
    Hot-update the anomaly threshold without restarting the service.

    Call this after running ``calibrate_threshold_from_features`` on a
    validation split to apply the calibrated value at runtime.
    """
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    _pipeline.update_threshold(body.threshold)
    logger.info("Threshold updated to %.4f", body.threshold)
    return ThresholdResponse(threshold=_pipeline.threshold)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_frames(frames_b64: List[str]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i, b64 in enumerate(frames_b64):
        try:
            data  = base64.b64decode(b64)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            out.append(np.array(image, dtype=np.uint8))
        except Exception as exc:
            raise ValueError(f"frame {i}: {exc}")
    return out


def _run_single_batch(
    pipeline: RealTimeInferencePipeline,
    frames: List[np.ndarray],
) -> Optional[InferenceResult]:
    """
    Run the inference pipeline on a single batch without the camera thread.
    The _InferenceThread._process() logic is called directly (same code path
    as the threaded live-stream mode).
    """
    # We re-use the inference thread's _process method by accessing it
    # directly — the pipeline doesn't need to be started for HTTP requests.
    import time as _time

    t = pipeline._inference_thread
    if t is None:
        # Build a temporary inference thread (not started, just for _process)
        from inference_service.src.inference_pipeline import _InferenceThread
        import queue, threading
        t = _InferenceThread(
            preprocessor      = pipeline.preprocessor,
            feature_extractor = pipeline.feature_extractor,
            yolo_raw          = pipeline.yolo_raw,
            model             = pipeline.model,
            localizer         = pipeline.localizer,
            device            = pipeline.device,
            threshold         = pipeline.threshold,
            frame_queue       = queue.Queue(),
            result_queue      = queue.Queue(),
            stop_event        = threading.Event(),
        )

    return t._process(frames, _time.time())


def _box_to_out(b) -> BoundingBoxOut:
    return BoundingBoxOut(
        x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
        width=b.width, height=b.height,
        confidence=b.confidence,
        class_id=b.class_id, class_name=b.class_name,
        anomaly_score=b.anomaly_score,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"