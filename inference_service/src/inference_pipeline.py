"""
inference_service/src/inference_pipeline.py
--------------------------------------------
Unified real-time inference pipeline — ML abstraction layer.

Architecture (parallel threads)
--------------------------------

    ┌────────────────────────────────────────────────────────────┐
    │  Camera / RTSP source                                      │
    │  _CameraThread  (I/O-bound, GIL released in cap.read())    │
    │        ↓  List[np.ndarray]  — batch_size=segment_length    │
    │  frame_queue  (bounded — drops oldest on overflow)         │
    │        ↓                                                   │
    │  _InferenceThread  (GPU compute)                           │
    │    1. VideoPreprocessor.to_segments()                      │
    │    2. I3D motion features                                  │
    │    3. YOLO object features  ──→ raw detections saved       │
    │    4. AnomalyDetector.forward()                            │
    │    5. SpatialLocalizer.localise()  ← uses saved detections │
    │        ↓  InferenceResult  (score + boxes + heatmap)       │
    │  result_queue  (bounded)                                   │
    │        ↓                                                   │
    │  Caller (FastAPI handler / AlertGenerator)                 │
    └────────────────────────────────────────────────────────────┘

While _InferenceThread processes batch N, _CameraThread is already
capturing batch N+1 — zero idle GPU time between batches.

Latency budget (approximate, NVIDIA RTX-class GPU)
---------------------------------------------------
  Camera capture:  16 frames @ 8 fps  → ~2 s capture wall-time
  Preprocessing:   resize + normalise → ~10 ms
  I3D forward:     1 segment          → ~30–80 ms
  YOLO forward:    16 frames          → ~20–60 ms
  BiGRU forward:   1 segment          → < 1 ms
  SpatialLocalizer: NMS + scoring     → < 2 ms
  ─────────────────────────────────────────────
  Total inference                       ~60–145 ms
  → alert raised < 0.15 s after last frame arrives
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from src.config import Config
from src.data.labels import get_class_name
from src.models.anomaly_detector import AnomalyDetector
from src.models.video_preprocessor import VideoPreprocessor
from src.utils.checkpointing import load_model_from_checkpoint
from src.utils.logging import get_logger
from inference_service.src.spatial_localizer import SpatialLocalizer, LocalisationResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    Complete result for one inference batch (one camera segment).

    Consumed by:
    - FastAPI ``/predict`` endpoint  → serialised to JSON
    - AlertGenerator                 → filtered and dispatched
    """
    # --- Timing ---
    capture_time: float         # Unix timestamp when the batch was captured
    inference_time_ms: float    # Wall-clock time for the full inference pass

    # --- Anomaly score ---
    anomaly_score: float        # Max per-segment score in [0, 1]
    is_anomaly: bool            # True if anomaly_score >= threshold
    threshold: float

    # --- Classification ---
    predicted_class_id: int
    predicted_class: str        # Human-readable UCF-Crime class name
    class_confidence: float     # Softmax probability of predicted class

    # --- Temporal localisation ---
    segment_scores: List[float] = field(default_factory=list)
    # index of segment with max score — the "when"
    peak_segment_idx: int = 0

    # --- Spatial localisation ---
    localisation: Optional[LocalisationResult] = None

    # --- Key frame for overlays (first frame of batch, RGB) ---
    key_frame: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------

class _CameraThread(threading.Thread):
    """
    I/O-bound thread: reads frames from camera, pushes fixed-size batches
    into a bounded queue.

    OpenCV releases the GIL during ``cap.read()``, so this thread runs
    truly concurrently with the GPU inference thread.
    """

    def __init__(
        self,
        source: Union[int, str],
        batch_size: int,
        target_fps: int,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        reconnect_delay: float = 2.0,
        max_reconnects: int = 10,
    ) -> None:
        super().__init__(name="CameraThread", daemon=True)
        self.source = source
        self.batch_size = batch_size
        self.target_fps = target_fps
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self._frames_captured = 0
        self._batches_enqueued = 0
        self._batches_dropped = 0

    @property
    def frames_captured(self)  -> int: return self._frames_captured
    @property
    def batches_enqueued(self) -> int: return self._batches_enqueued
    @property
    def batches_dropped(self)  -> int: return self._batches_dropped

    def run(self) -> None:
        reconnects = 0
        while not self.stop_event.is_set() and reconnects <= self.max_reconnects:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                reconnects += 1
                logger.warning(
                    "CameraThread: cannot open %s (%d/%d), retry in %.1fs",
                    self.source, reconnects, self.max_reconnects, self.reconnect_delay,
                )
                time.sleep(self.reconnect_delay)
                continue

            reconnects = 0
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            skip = max(1, int(round(native_fps / self.target_fps)))
            logger.info(
                "CameraThread: connected %s  %.1f→%.1f fps  batch=%d",
                self.source, native_fps, native_fps / skip, self.batch_size,
            )

            batch: List[np.ndarray] = []
            frame_idx = 0
            capture_ts = time.time()

            try:
                while not self.stop_event.is_set():
                    ret, bgr = cap.read()
                    if not ret:
                        logger.warning("CameraThread: read failed, reconnecting…")
                        break
                    if frame_idx % skip == 0:
                        batch.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        self._frames_captured += 1
                        if len(batch) == self.batch_size:
                            self._enqueue(batch, capture_ts)
                            batch = []
                            capture_ts = time.time()
                    frame_idx += 1
            finally:
                cap.release()
                if batch and not self.stop_event.is_set():
                    self._enqueue(batch, capture_ts)

        logger.info(
            "CameraThread stopped | captured=%d  enqueued=%d  dropped=%d",
            self._frames_captured, self._batches_enqueued, self._batches_dropped,
        )

    def _enqueue(self, batch: List[np.ndarray], ts: float) -> None:
        """Non-blocking put; if full, evict oldest then retry."""
        try:
            self.frame_queue.put_nowait((list(batch), ts))
            self._batches_enqueued += 1
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait((list(batch), ts))
                self._batches_enqueued += 1
            except queue.Full:
                self._batches_dropped += 1


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

class _InferenceThread(threading.Thread):
    """
    GPU-bound thread: drains frame_queue, runs the full ML pipeline
    (preprocess → I3D → YOLO → BiGRU → SpatialLocalizer), pushes
    InferenceResult to result_queue.
    """

    def __init__(
        self,
        preprocessor: VideoPreprocessor,
        feature_extractor,      # TwoStreamFeatureExtractor
        yolo_raw,               # YOLOObjectFeatureExtractor (for raw detections)
        model: AnomalyDetector,
        localizer: SpatialLocalizer,
        device: torch.device,
        threshold: float,
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="InferenceThread", daemon=True)
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.yolo_raw = yolo_raw
        self.model = model
        self.localizer = localizer
        self.device = device
        self.threshold = threshold
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self._batches_processed = 0
        self._total_ms = 0.0

    @property
    def batches_processed(self) -> int:  return self._batches_processed
    @property
    def avg_inference_ms(self) -> float:
        return self._total_ms / max(1, self._batches_processed)

    def run(self) -> None:
        logger.info("InferenceThread started | device=%s  threshold=%.3f",
                    self.device, self.threshold)
        self.model.eval()

        while not self.stop_event.is_set():
            try:
                frames, capture_ts = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            result = self._process(frames, capture_ts)
            if result is not None:
                # Put into result_queue; if full evict oldest
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.result_queue.put_nowait(result)

            self.frame_queue.task_done()

        logger.info("InferenceThread stopped | processed=%d  avg_ms=%.1f",
                    self._batches_processed, self.avg_inference_ms)

    def _process(
        self,
        frames: List[np.ndarray],
        capture_ts: float,
    ) -> Optional[InferenceResult]:
        t0 = time.perf_counter()
        try:
            # 1. Preprocess
            segments = self.preprocessor.to_segments(frames)
            if not segments:
                return None

            # 2. YOLO: collect raw detections for spatial localisation
            #    AND produce the object feature vector for the model.
            #    We run YOLO once and use results for both purposes.
            yolo_detections: List[Dict] = []
            if self.yolo_raw is not None:
                yolo_detections = self._run_yolo_for_detections(frames)

            # 3. Full feature extraction (I3D + YOLO aggregated vector)
            features_np = self.feature_extractor.extract_features(segments)
            # shape: [num_segments, feature_dim]

            # 4. BiGRU anomaly detector
            features_t = (
                torch.from_numpy(features_np)
                .unsqueeze(0)              # [1, S, D]
                .to(self.device, non_blocking=True)
            )
            with torch.no_grad():
                anomaly_scores, class_probs = self.model(features_t)
                # anomaly_scores: [1, S, 1]
                # class_probs:    [1, S, C]

            seg_scores = anomaly_scores.squeeze().cpu().tolist()
            if isinstance(seg_scores, float):
                seg_scores = [seg_scores]

            video_score = float(max(seg_scores))
            peak_idx = int(np.argmax(seg_scores))

            mean_probs = class_probs.squeeze(0).mean(dim=0)   # [C]
            cls_id = int(mean_probs.argmax().item())
            cls_conf = float(mean_probs[cls_id].item())

            # 5. Spatial localisation (only when anomalous and YOLO ran)
            localisation: Optional[LocalisationResult] = None
            if video_score >= self.threshold and yolo_detections:
                h, w = frames[0].shape[:2]
                localisation = self.localizer.localise(
                    yolo_detections=yolo_detections,
                    anomaly_score=video_score,
                    frame_shape=(h, w),
                )

            ms = (time.perf_counter() - t0) * 1000.0
            self._batches_processed += 1
            self._total_ms += ms

            return InferenceResult(
                capture_time      = capture_ts,
                inference_time_ms = ms,
                anomaly_score     = video_score,
                is_anomaly        = video_score >= self.threshold,
                threshold         = self.threshold,
                predicted_class_id= cls_id,
                predicted_class   = get_class_name(cls_id),
                class_confidence  = cls_conf,
                segment_scores    = seg_scores,
                peak_segment_idx  = peak_idx,
                localisation      = localisation,
                key_frame         = frames[0] if frames else None,
            )

        except Exception as exc:
            logger.error("InferenceThread: batch failed: %s", exc, exc_info=True)
            return None

    def _run_yolo_for_detections(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Run YOLO on raw frames and return per-frame detection dicts.
        Used by the spatial localiser — separate from the aggregated
        feature vector that goes into the anomaly detector.
        """
        detections: List[Dict] = []
        try:
            results = self.yolo_raw.model(frames, verbose=False)
            for res in results:
                frame_dets: Dict = {"boxes": []}
                for box in res.boxes:
                    cls = int(box.cls)
                    frame_dets["boxes"].append({
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3]),
                        "confidence": float(box.conf),
                        "class_id": cls,
                        "class_name": res.names.get(cls, str(cls)),
                    })
                detections.append(frame_dets)
        except Exception as exc:
            logger.warning("YOLO detection pass failed: %s", exc)
        return detections


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

class RealTimeInferencePipeline:
    """
    Parallel real-time inference pipeline.

    Typical usage
    -------------
    ::

        # Load from checkpoint (recommended)
        pipe = RealTimeInferencePipeline.from_checkpoint(
            checkpoint_path="/app/models/best_model.pt",
            camera_source=0,           # webcam or "rtsp://..."
        )

        # Context manager — auto start/stop
        with pipe:
            while True:
                result = pipe.get_result(timeout=1.0)
                if result and result.is_anomaly:
                    handle_alert(result)

        # Or blocking run with callbacks
        pipe.run(
            on_alert=lambda r: send_to_api(r),
            on_result=lambda r: log_metrics(r),
        )

    Parameters
    ----------
    model : AnomalyDetector
    preprocessor : VideoPreprocessor
    feature_extractor : TwoStreamFeatureExtractor
    yolo_raw : YOLOObjectFeatureExtractor
        Raw extractor kept for spatial localisation; set to None to disable.
    localizer : SpatialLocalizer
    device : torch.device
    threshold : float
        Anomaly score threshold. Calibrate with ``calibrate_threshold_from_features``.
    camera_source : int | str
    batch_size : int  — must equal config.dataset.segment_length
    target_fps : int
    frame_queue_size : int  — bounded buffer between camera and inference threads
    result_queue_size : int — bounded buffer for caller
    """

    def __init__(
        self,
        model: AnomalyDetector,
        preprocessor: VideoPreprocessor,
        feature_extractor,
        yolo_raw,
        localizer: SpatialLocalizer,
        device: torch.device,
        threshold: float = 0.5,
        camera_source: Union[int, str] = 0,
        batch_size: int = 16,
        target_fps: int = 8,
        frame_queue_size: int = 4,
        result_queue_size: int = 8,
    ) -> None:
        self.model             = model
        self.preprocessor      = preprocessor
        self.feature_extractor = feature_extractor
        self.yolo_raw          = yolo_raw
        self.localizer         = localizer
        self.device            = device
        self.threshold         = threshold
        self.camera_source     = camera_source
        self.batch_size        = batch_size
        self.target_fps        = target_fps

        self._frame_queue  = queue.Queue(maxsize=frame_queue_size)
        self._result_queue = queue.Queue(maxsize=result_queue_size)
        self._stop_event   = threading.Event()
        self._camera_thread    : Optional[_CameraThread]    = None
        self._inference_thread : Optional[_InferenceThread] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        camera_source: Union[int, str] = 0,
        threshold: Optional[float] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        localizer_strategy: str = "object",
    ) -> "RealTimeInferencePipeline":
        """
        Build the complete pipeline from a checkpoint file.

        Reads model architecture from the checkpoint's ``config`` sub-dict
        and all other settings from ``configs/default.yaml``.
        """
        if config_path is None:
            config_path = (
                Path(__file__).resolve().parent.parent.parent
                / "configs" / "default.yaml"
            )
        cfg = Config.from_yaml(config_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _device = torch.device(device)

        if threshold is None:
            inf_cfg = getattr(cfg, "inference", None)
            threshold = float(getattr(inf_cfg, "anomaly_threshold", 0.5))

        logger.info(
            "Building pipeline: device=%s  threshold=%.3f  strategy=%s",
            _device, threshold, localizer_strategy,
        )

        model = load_model_from_checkpoint(checkpoint_path, device=_device)

        preprocessor = VideoPreprocessor(
            frame_size=tuple(cfg.dataset.frame_size),
            segment_length=cfg.dataset.segment_length,
        )

        from src.models.feature_extractors import (
            I3DFeatureExtractor,
            YOLOObjectFeatureExtractor,
            YOLOFeatureAdapter,
            TwoStreamFeatureExtractor,
        )
        motion   = I3DFeatureExtractor(device=str(_device), pretrained=True, freeze=True)
        yolo_raw = YOLOObjectFeatureExtractor(
            model_name=cfg.feature_extraction.yolo_model_name,
            device=str(_device),
        )
        yolo_adapter = YOLOFeatureAdapter(yolo_raw, device=str(_device))
        extractor = TwoStreamFeatureExtractor(motion, yolo_adapter)

        localizer = SpatialLocalizer(strategy=localizer_strategy)

        return cls(
            model             = model,
            preprocessor      = preprocessor,
            feature_extractor = extractor,
            yolo_raw          = yolo_raw,
            localizer         = localizer,
            device            = _device,
            threshold         = threshold,
            camera_source     = camera_source,
            batch_size        = cfg.dataset.segment_length,
            target_fps        = cfg.feature_extraction.target_fps,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._camera_thread and self._camera_thread.is_alive():
            return
        self._stop_event.clear()
        self._camera_thread = _CameraThread(
            source=self.camera_source, batch_size=self.batch_size,
            target_fps=self.target_fps, frame_queue=self._frame_queue,
            stop_event=self._stop_event,
        )
        self._inference_thread = _InferenceThread(
            preprocessor=self.preprocessor,
            feature_extractor=self.feature_extractor,
            yolo_raw=self.yolo_raw,
            model=self.model,
            localizer=self.localizer,
            device=self.device,
            threshold=self.threshold,
            frame_queue=self._frame_queue,
            result_queue=self._result_queue,
            stop_event=self._stop_event,
        )
        self._inference_thread.start()
        self._camera_thread.start()
        logger.info("Pipeline started")

    def stop(self) -> None:
        self._stop_event.set()
        for t in (self._camera_thread, self._inference_thread):
            if t:
                t.join(timeout=5.0)
        cam  = self._camera_thread
        inf  = self._inference_thread
        logger.info(
            "Pipeline stopped | cam: captured=%d enqueued=%d dropped=%d"
            " | inf: processed=%d avg_ms=%.1f",
            cam.frames_captured  if cam else 0,
            cam.batches_enqueued if cam else 0,
            cam.batches_dropped  if cam else 0,
            inf.batches_processed if inf else 0,
            inf.avg_inference_ms  if inf else 0.0,
        )

    def __enter__(self) -> "RealTimeInferencePipeline":
        self.start(); return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Result access
    # ------------------------------------------------------------------

    def get_result(self, timeout: float = 1.0) -> Optional[InferenceResult]:
        """
        Non-blocking poll. Returns the next InferenceResult or None.
        Call from the main thread (or FastAPI background task).
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def run(
        self,
        on_alert:  Optional[Callable[[InferenceResult], None]] = None,
        on_result: Optional[Callable[[InferenceResult], None]] = None,
    ) -> None:
        """
        Blocking run loop. Calls on_result for every batch and on_alert
        for anomalous batches.  Returns on Ctrl-C or camera death.
        """
        self.start()
        logger.info("Running — Ctrl-C to stop")
        try:
            while True:
                result = self.get_result(timeout=1.0)
                if result is None:
                    if self._camera_thread and not self._camera_thread.is_alive():
                        logger.warning("CameraThread died — stopping")
                        break
                    continue
                if on_result:
                    on_result(result)
                if result.is_anomaly and on_alert:
                    on_alert(result)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    def update_threshold(self, new_threshold: float) -> None:
        """Update anomaly threshold without restarting threads."""
        old = self.threshold
        self.threshold = new_threshold
        if self._inference_thread:
            self._inference_thread.threshold = new_threshold
        logger.info("Threshold updated %.3f → %.3f", old, new_threshold)