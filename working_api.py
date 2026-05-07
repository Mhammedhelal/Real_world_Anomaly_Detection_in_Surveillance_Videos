from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
import time
import numpy as np
from PIL import Image
import torch
import sys
import cv2
import os

sys.path.insert(0, os.getcwd())

from src.models.anomaly_detector import AnomalyDetector
from src.models.video_preprocessor import VideoPreprocessor
from src.models.feature_extractors import R3DFeatureExtractor, YOLOObjectFeatureExtractor, YOLOFeatureAdapter, TwoStreamFeatureExtractor

app = FastAPI()

# Global variables
model = None
preprocessor = None
feature_extractor = None
THRESHOLD = 0.5
device = torch.device("cpu")

@app.on_event("startup")
async def startup():
    global model, preprocessor, feature_extractor
    print("="*50)
    print("Loading Model...")
    print("="*50)
    
    # Load model
    checkpoint_path = "inference_service/models/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AnomalyDetector(input_size=595, hidden_size=256, num_classes=14)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✅ Model loaded (595-dim)")
    
    # Patch YOLO to output only 83-dim (mean only, no std)
    original_extract = YOLOObjectFeatureExtractor.extract_segment_features
    
    def patched_extract(self, frames):
        from collections import defaultdict
        obj_counts = defaultdict(int)
        bbox_stats = []
        for frame in frames:
            results = self.model(frame, verbose=False)[0]
            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls)
                    obj_counts[cls] += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = (x2 - x1).item() if hasattr(x2 - x1, 'item') else (x2 - x1)
                    h = (y2 - y1).item() if hasattr(y2 - y1, 'item') else (y2 - y1)
                    bbox_stats.append([w, h, float(box.conf)])
        
        count_features = np.zeros(80, dtype=np.float32)
        for cls, count in obj_counts.items():
            if cls < 80:
                count_features[cls] = count
        count_features /= max(len(frames), 1)
        
        if bbox_stats:
            bbox_arr = np.array(bbox_stats)
            bbox_features = bbox_arr.mean(axis=0)
        else:
            bbox_features = np.zeros(3, dtype=np.float32)
        
        return np.concatenate([count_features, bbox_features])
    
    YOLOObjectFeatureExtractor.extract_segment_features = patched_extract
    
    # Initialize feature extractor (R3D + YOLO = 595-dim)
    preprocessor = VideoPreprocessor(frame_size=(224, 224), segment_length=16)
    motion = R3DFeatureExtractor(device=str(device), pretrained=True)
    yolo_raw = YOLOObjectFeatureExtractor(device=str(device))
    yolo_adapter = YOLOFeatureAdapter(yolo_raw, device=str(device))
    feature_extractor = TwoStreamFeatureExtractor(motion, yolo_adapter)
    
    print(f"✅ Feature extractor ready (dim: {feature_extractor.feature_dim})")
    print("="*50)

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None, "threshold": THRESHOLD}

class PredictRequest(BaseModel):
    frames: List[str]
    timestamp: Optional[str] = None

@app.post("/predict")
async def predict(request: PredictRequest):
    global model, preprocessor, feature_extractor, THRESHOLD
    start_time = time.time()
    
    try:
        # Decode frames
        frames = []
        for b64 in request.frames:
            img_data = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            frames.append(np.array(img))
        
        # Process frames
        segments = preprocessor.to_segments(frames)
        features = feature_extractor.extract_features(segments)
        
        # Check feature dimension
        if features.shape[1] != 595:
            print(f"Warning: Expected 595 dim, got {features.shape[1]}")
            # Take first 595 dimensions if needed
            features = features[:, :595]
        
        features_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            scores, probs = model(features_t)
        
        video_score = float(scores.max().item())
        
        class_names = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
                       'Explosion', 'Fighting', 'Robbery', 'Shooting', 'Shoplifting',
                       'Stealing', 'Vandalism', 'RoadAccidents']
        class_probs = probs.squeeze(0).mean(dim=0)
        class_id = int(class_probs.argmax().item())
        
        return {
            "anomaly_score": video_score,
            "is_anomaly": video_score >= THRESHOLD,
            "threshold_used": THRESHOLD,
            "predicted_class": class_names[class_id],
            "predicted_class_id": class_id,
            "class_confidence": float(class_probs[class_id].item()),
            "inference_time_ms": (time.time() - start_time) * 1000,
            "timestamp": request.timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)