"""
inference_service/main.py
-------
Production FastAPI service for real-time anomaly detection inference.

Entry point for Node.js backend integration.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
import numpy as np
import torch
import io
from PIL import Image
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_service.src.pipeline import InferencePipeline
from inference_service.src.config_loader import load_inference_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Real-Time Anomaly Detection API",
    description="Production inference service for surveillance video anomaly detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for Node.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference pipeline (loaded once at startup)
pipeline: Optional[InferencePipeline] = None


# ============================================================================
# Request/Response Models (API Contract)
# ============================================================================

class FrameInput(BaseModel):
    """Single frame input."""
    frame_base64: str = Field(
        ...,
        description="Base64-encoded frame (JPEG/PNG). RGB format assumed."
    )


class BatchFramesInput(BaseModel):
    """Batch of frames for inference (1 video segment)."""
    frames: List[str] = Field(
        ...,
        description="List of base64-encoded frames (min 1, max 64). "
                    "System will pad/sample to segment_length=16.",
        min_items=1,
        max_items=64
    )
    timestamp: Optional[str] = Field(
        None,
        description="Optional timestamp for this batch (ISO 8601 format)"
    )
    save_features: bool = Field(
        False,
        description="Whether to save extracted features to disk"
    )
    return_visualization: bool = Field(
        False,
        description="Whether to return annotated frame with bounding boxes (base64)"
    )


class BoundingBoxOutput(BaseModel):
    """Bounding box with anomaly attribution."""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    width: float = Field(..., description="Box width in pixels")
    height: float = Field(..., description="Box height in pixels")
    confidence: float = Field(..., description="Detection confidence [0.0-1.0]")
    class_id: int = Field(..., description="Object class ID")
    class_name: str = Field(..., description="Object class name (e.g., 'person', 'car')")
    anomaly_score: float = Field(..., description="Anomaly attribution score for this object [0.0-1.0]")


class LocalizationOutput(BaseModel):
    """Spatial localization of anomaly."""
    bounding_boxes: List[BoundingBoxOutput] = Field(
        ...,
        description="List of bounding boxes with anomaly scores"
    )
    num_detections: int = Field(
        ...,
        description="Total number of detections"
    )
    frame_dimensions: Dict[str, int] = Field(
        ...,
        description="Frame width and height"
    )
    primary_anomaly: Optional[BoundingBoxOutput] = Field(
        None,
        description="Primary anomaly region (highest scoring box)"
    )


class AnomalyPrediction(BaseModel):
    """Anomaly detection result."""
    anomaly_score: float = Field(
        ...,
        description="Anomaly probability [0.0-1.0]. Higher = more anomalous.",
        ge=0.0,
        le=1.0
    )
    is_anomaly: bool = Field(
        ...,
        description="Binary decision: True if score > threshold"
    )
    predicted_class: str = Field(
        ...,
        description="Predicted anomaly category (e.g., 'Normal', 'Theft', 'Violence')"
    )
    confidence: float = Field(
        ...,
        description="Classification confidence [0.0-1.0]",
        ge=0.0,
        le=1.0
    )
    threshold_used: float = Field(
        ...,
        description="Threshold used for binary decision"
    )
    localization: Optional[LocalizationOutput] = Field(
        None,
        description="Spatial localization with bounding boxes (if enabled)"
    )
    timestamp: str = Field(
        ...,
        description="Inference timestamp (ISO 8601)"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    metadata: Optional[dict] = Field(
        None,
        description="Optional metadata (feature paths, visualization, etc.)"
    )


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str
    timestamp: str


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize inference pipeline on startup."""
    global pipeline
    
    logger.info("🚀 Starting Anomaly Detection Inference Service...")
    
    try:
        # Load configuration
        config = load_inference_config()
        
        # Initialize pipeline
        logger.info("📦 Loading inference pipeline...")
        pipeline = InferencePipeline(config=config)
        
        logger.info(f"✅ Pipeline loaded successfully")
        logger.info(f"   Device: {pipeline.device}")
        logger.info(f"   Model: {pipeline.model.__class__.__name__}")
        logger.info(f"   Threshold: {pipeline.threshold}")
        logger.info("🎉 Service ready for inference!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global pipeline
    logger.info("🛑 Shutting down inference service...")
    
    if pipeline:
        pipeline.cleanup()
    
    logger.info("✅ Shutdown complete")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "Anomaly Detection Inference API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and readiness.
    """
    return HealthCheck(
        status="healthy" if pipeline is not None else "initializing",
        model_loaded=pipeline is not None,
        device=str(pipeline.device) if pipeline else "unknown",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/predict", response_model=AnomalyPrediction)
async def predict_anomaly(request: BatchFramesInput):
    """
    Predict anomaly from a batch of frames.
    
    **Input:**
    - `frames`: List of base64-encoded JPEG/PNG frames (RGB)
    - `timestamp`: Optional ISO 8601 timestamp
    - `save_features`: Whether to persist extracted features
    
    **Output:**
    - `anomaly_score`: Probability of anomaly [0.0-1.0]
    - `is_anomaly`: Binary decision (True/False)
    - `predicted_class`: Anomaly category
    - `confidence`: Classification confidence
    - Processing metadata
    
    **Example:**
    ```bash
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "frames": ["base64_frame_1", "base64_frame_2", ...],
        "timestamp": "2024-01-15T10:30:00Z",
        "save_features": false
      }'
    ```
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Pipeline still initializing."
        )
    
    try:
        # Start timing
        start_time = datetime.utcnow()
        
        # Decode frames
        logger.info(f"📥 Received {len(request.frames)} frames for inference")
        frames = _decode_frames(request.frames)
        
        # Run inference
        result = pipeline.predict(
            frames=frames,
            save_features=request.save_features,
            timestamp=request.timestamp,
            return_visualization=request.return_visualization
        )
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Build response
        response = AnomalyPrediction(
            anomaly_score=float(result['anomaly_score']),
            is_anomaly=bool(result['is_anomaly']),
            predicted_class=str(result['predicted_class']),
            confidence=float(result['confidence']),
            threshold_used=float(result['threshold_used']),
            timestamp=end_time.isoformat() + "Z",
            processing_time_ms=processing_time_ms,
            metadata=result.get('metadata')
        )
        
        logger.info(
            f"✅ Inference complete: score={response.anomaly_score:.3f}, "
            f"class={response.predicted_class}, time={processing_time_ms:.1f}ms"
        )
        
        return response
    
    except ValueError as e:
        logger.error(f"❌ Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/stream", response_model=AnomalyPrediction)
async def predict_stream(frames_file: UploadFile = File(...)):
    """
    Alternative endpoint: upload frames as multipart file.
    
    Useful for large batches or binary transfer.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Read uploaded file
        content = await frames_file.read()
        
        # Decode frames (assumes JSON array of base64 strings)
        import json
        frames_data = json.loads(content)
        frames = _decode_frames(frames_data['frames'])
        
        # Run inference
        result = pipeline.predict(frames=frames)
        
        return AnomalyPrediction(
            anomaly_score=result['anomaly_score'],
            is_anomaly=result['is_anomaly'],
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            threshold_used=result['threshold_used'],
            timestamp=datetime.utcnow().isoformat() + "Z",
            processing_time_ms=0.0  # TODO: Add timing
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Helper Functions
# ============================================================================

def _decode_frames(frames_base64: List[str]) -> List[np.ndarray]:
    """
    Decode base64-encoded frames to numpy arrays.
    
    Parameters
    ----------
    frames_base64 : List[str]
        List of base64-encoded frames (JPEG/PNG)
    
    Returns
    -------
    List[np.ndarray]
        List of RGB frames as numpy arrays (H, W, 3) uint8
    
    Raises
    ------
    ValueError
        If frames cannot be decoded
    """
    decoded_frames = []
    
    for idx, frame_b64 in enumerate(frames_base64):
        try:
            # Decode base64
            frame_bytes = base64.b64decode(frame_b64)
            
            # Open as PIL Image
            image = Image.open(io.BytesIO(frame_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            frame_np = np.array(image)
            
            decoded_frames.append(frame_np)
        
        except Exception as e:
            raise ValueError(f"Failed to decode frame {idx}: {e}")
    
    return decoded_frames


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable in production
        log_level="info"
    )