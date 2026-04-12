"""
scripts/extract_features.py
----------------------------
CLI for extracting video features.

Uses the new source-abstraction layer:
  DiskVideoSource → VideoPreprocessor → FusionExtractor → .npz files

The pipeline itself is fully decoupled from disk I/O; this script is
the only place that constructs a DiskVideoSource, and it does so by
reading paths from the YAML config or CLI arguments.

Usage
-----
    # Extract normal training videos
    python scripts/extract_features.py --video-folder normal --split train

    # Extract anomalous test videos from an explicit path
    python scripts/extract_features.py --video-dir /data/UCF-Crime/Assault --split test

    # Dry-run: show what would be processed
    python scripts/extract_features.py --video-folder normal --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.sources.disk_source import DiskVideoSource
from src.models.video_preprocessor import VideoPreprocessor
from src.engine.FeatureExtractionPipeline import FeatureExtractionPipeline
from src.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract video features using the decoupled source pipeline."
    )
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )

    source_group = p.add_mutually_exclusive_group()
    source_group.add_argument(
        "--video-folder",
        help="Subdirectory name under the base video dir (e.g. 'normal').",
    )
    source_group.add_argument(
        "--video-dir",
        help="Explicit path to a directory of videos.",
    )

    p.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Split label encoded in output filenames.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Videos per processing batch (for memory management).",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Stop after N videos (useful for testing).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip videos already in the progress log (default: on).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess videos even if already logged.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Index videos and print the plan without extracting.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = Config.from_yaml(args.config)

    # ------------------------------------------------------------------
    # Resolve input directory
    # ------------------------------------------------------------------
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
        print(f"❌ Input directory not found: {input_video_dir}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resolve output directories
    # ------------------------------------------------------------------
    features_base = Path(cfg.dataset.output_base_dir) / cfg.dataset.features_dir_name
    features_dir  = features_base / f"{folder_name}_{args.split}"
    metadata_dir  = features_base / "metadata"

    # ------------------------------------------------------------------
    # Print plan
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"INPUT DIR    : {input_video_dir}")
    print(f"FEATURES DIR : {features_dir}")
    print(f"SPLIT        : {args.split}")
    print(f"BATCH SIZE   : {args.batch_size}")
    print(f"RESUME       : {args.resume and not args.force}")
    print("=" * 70)

    if args.dry_run:
        # Build source just to count and list videos
        source = DiskVideoSource(
            video_dir=input_video_dir,
            split=args.split,
        )
        print(f"\n📋 Dry-run: would process {len(source)} videos")
        for p in source._video_paths[:10]:
            print(f"   {p.name}")
        if len(source) > 10:
            print(f"   … and {len(source)-10} more")
        return

    if not args.yes:
        resp = input("\nProceed with extraction? (yes/no): ")
        if resp.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    # ------------------------------------------------------------------
    # Build the pipeline with a DiskVideoSource
    # ------------------------------------------------------------------
    source = DiskVideoSource(
        video_dir=input_video_dir,
        batch_size=cfg.dataset.segment_length,   # one batch = one segment
        target_fps=cfg.feature_extraction.target_fps,
        max_frames=cfg.dataset.max_frames,
        split=args.split,
    )

    preprocessor = VideoPreprocessor(
        frame_size=tuple(cfg.dataset.frame_size),
        segment_length=cfg.dataset.segment_length,
    )

    # Feature extractor — lazy import so the script can be tested without
    # heavy dependencies installed
    try:
        from src.models.feature_extractors import (
            I3DFeatureExtractor,
            YOLOObjectFeatureExtractor,
            YOLOFeatureAdapter,
            TwoStreamFeatureExtractor,
        )
        motion_extractor = I3DFeatureExtractor(device=cfg.hardware.device)
        yolo_raw         = YOLOObjectFeatureExtractor(device=cfg.hardware.device)
        object_extractor = YOLOFeatureAdapter(yolo_raw, device=cfg.hardware.device)
        feature_extractor = TwoStreamFeatureExtractor(motion_extractor, object_extractor)
    except ImportError as exc:
        print(f"⚠️  Could not load feature extractor: {exc}")
        print("   Install pytorchvideo and ultralytics to use full extraction.")
        sys.exit(1)

    pipeline = FeatureExtractionPipeline(
        source=source,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        features_dir=features_dir,
        metadata_dir=metadata_dir,
        device=cfg.hardware.device,
    )

    # ------------------------------------------------------------------
    # Run extraction
    # ------------------------------------------------------------------
    processed, failed = pipeline.extract_all_features(
        resume=args.resume and not args.force,
        force_reprocess=args.force,
        max_videos=args.max_videos,
    )

    print(f"\n🎉 Done: {processed} succeeded, {failed} failed.")
    pipeline.check_status()


if __name__ == "__main__":
    main()