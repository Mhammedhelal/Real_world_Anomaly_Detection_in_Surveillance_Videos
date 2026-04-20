"""inference_service/src — ML abstraction layer."""
from inference_service.src.inference_pipeline import RealTimeInferencePipeline, InferenceResult
from inference_service.src.spatial_localizer import SpatialLocalizer, LocalisationResult, BoundingBox
from inference_service.src.config_loader import load_inference_config

__all__ = [
    "RealTimeInferencePipeline", "InferenceResult",
    "SpatialLocalizer", "LocalisationResult", "BoundingBox",
    "load_inference_config",
]