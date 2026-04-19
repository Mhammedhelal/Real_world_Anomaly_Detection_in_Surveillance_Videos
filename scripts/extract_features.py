"""
scripts/extract_features.py
----------------------------
CLI for extracting video features.

Usage
-----
    python scripts/extract_features.py --video-folder normal --split train
    python scripts/extract_features.py --video-dir /data/UCF-Crime/Assault --split test
    python scripts/extract_features.py --video-folder normal --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data.sources.disk_source import DiskVideoSource
from src.engine.FeatureExtractionPipeline import FeatureExtractionPipeline
from src.models.video_preprocessor import VideoPreprocessor
from src.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract video features.")
    p.add_argument("--config", default="configs/default.yaml")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video-folder", help="Subdirectory name under base video dir")
    src.add_argument("--video-dir", help="Explicit path to video directory")
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--log-dir", default="outputs/logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    setup_logging(log_dir=args.log_dir, run_name="extract_features")
    logger = get_logger(__name__)

    cfg = Config.from_yaml(args.config)

    if args.video_dir:
        input_video_dir = Path(args.video_dir)
        folder_name = input_video_dir.name
    elif args.video_folder:
        base_parent = Path(cfg.dataset.input_video_dir).parent
        input_video_dir = base_parent / args.video_folder
        folder_name = args.video_folder
    else:
        input_video_dir = Path(cfg.dataset.input_video_dir)
        folder_name = input_video_dir.name

    if not input_video_dir.is_dir():
        logger.error("Input directory not found: %s", input_video_dir)
        sys.exit(1)

    features_base = Path(cfg.dataset.output_base_dir) / cfg.dataset.features_dir_name
    features_dir = features_base / f"{folder_name}_{args.split}"
    metadata_dir = features_base / "metadata"

    logger.info("INPUT  : %s", input_video_dir)
    logger.info("OUTPUT : %s", features_dir)
    logger.info("SPLIT  : %s", args.split)

    if args.dry_run:
        source = DiskVideoSource(video_dir=input_video_dir, split=args.split)
        logger.info("Dry-run: would process %d videos", len(source))
        for p in source._video_paths[:10]:
            logger.info("  %s", p.name)
        return

    if not args.yes:
        resp = input("\nProceed with extraction? (yes/no): ")
        if resp.strip().lower() != "yes":
            logger.info("Aborted.")
            sys.exit(0)

    source = DiskVideoSource(
        video_dir=input_video_dir,
        batch_size=cfg.dataset.segment_length,
        target_fps=cfg.feature_extraction.target_fps,
        max_frames=cfg.dataset.max_frames,
        split=args.split,
    )

    preprocessor = VideoPreprocessor(
        frame_size=tuple(cfg.dataset.frame_size),
        segment_length=cfg.dataset.segment_length,
    )

    try:
        from src.models.feature_extractors import (
            I3DFeatureExtractor, YOLOObjectFeatureExtractor,
            YOLOFeatureAdapter, TwoStreamFeatureExtractor,
        )
        motion_extractor = I3DFeatureExtractor(device=cfg.hardware.device)
        yolo_raw = YOLOObjectFeatureExtractor(device=cfg.hardware.device)
        object_extractor = YOLOFeatureAdapter(yolo_raw, device=cfg.hardware.device)
        feature_extractor = TwoStreamFeatureExtractor(motion_extractor, object_extractor)
    except ImportError as exc:
        logger.error("Could not load feature extractor: %s", exc)
        logger.error("Install pytorchvideo and ultralytics first.")
        sys.exit(1)

    pipeline = FeatureExtractionPipeline(
        source=source,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        features_dir=features_dir,
        metadata_dir=metadata_dir,
        device=cfg.hardware.device,
    )

    processed, failed = pipeline.extract_all_features(
        resume=args.resume and not args.force,
        force_reprocess=args.force,
        max_videos=args.max_videos,
    )

    logger.info("Done: %d succeeded, %d failed.", processed, failed)
    pipeline.check_status()


if __name__ == "__main__":
    main()
