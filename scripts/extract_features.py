"""
scripts/extract_features.py
----------------------------
CLI for extracting video features.

Usage
-----
    python scripts/extract_features.py --video-folder normal --split train
    python scripts/extract_features.py --video-dir /data/UCF-Crime/Assault --split test
    python scripts/extract_features.py --video-folder normal --dry-run
    python scripts/extract_features.py --video-folder normal --save-dir my/custom/output

Resume behaviour
----------------
By default the script resumes from where it left off:
  • Videos already in extraction_progress.json → skipped.
  • The video that was mid-processing when the pipeline last stopped
    (tracked via ``last_checkpoint``) → re-processed from scratch to
    guarantee a clean .npz.

Pass --no-resume to start completely fresh (keeps existing .npz files
on disk but ignores the progress log).
Pass --force to re-extract every video regardless of prior progress.
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
    p = argparse.ArgumentParser(
        description="Extract video features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to YAML config file")

    # ── Video source (mutually exclusive) ─────────────────────────────────────
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video-folder",
                     help="Subdirectory name under the base video dir "
                          "(e.g. 'normal', 'anomalous')")
    src.add_argument("--video-dir",
                     help="Explicit path to the directory containing video files")

    # ── Output locations ───────────────────────────────────────────────────────
    p.add_argument(
        "--save-dir",
        default=None,
        help=(
            "Directory where extracted .npz feature files are saved. "
            "Defaults to  <output_base_dir>/<features_dir_name>/<folder>_<split>/  "
            "as defined in the YAML config  "
            "(e.g. data/features/extracted/normal_train/)."
        ),
    )
    p.add_argument(
        "--metadata-dir",
        default=None,
        help=(
            "Directory where extraction_progress.json is written. "
            "Defaults to  <save-dir>/metadata/  when --save-dir is given, "
            "otherwise uses  <output_base_dir>/<features_dir_name>/metadata/."
        ),
    )

    # ── Extraction options ─────────────────────────────────────────────────────
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="Split label embedded in output filenames")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Number of videos processed per batch")
    p.add_argument("--max-videos", type=int, default=None,
                   help="Stop after processing this many videos (useful for testing)")

    # ── Resume control ─────────────────────────────────────────────────────────
    resume_grp = p.add_mutually_exclusive_group()
    resume_grp.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help=(
            "Resume from the last checkpoint: skip videos that completed "
            "successfully and re-process the one that was mid-extraction "
            "when the pipeline last stopped. This is the default behaviour."
        ),
    )
    resume_grp.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help=(
            "Ignore extraction_progress.json and process all videos from "
            "the beginning.  Existing .npz files on disk are NOT deleted; "
            "use --force to also re-extract those."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-extract every video even if its .npz already exists and it "
            "appears in the progress log.  Implies --no-resume."
        ),
    )

    p.add_argument("--dry-run", action="store_true",
                   help="List what would be processed without extracting anything")
    p.add_argument("--yes", action="store_true",
                   help="Skip the interactive 'Proceed?' confirmation prompt")
    p.add_argument("--log-dir", default="outputs/logs",
                   help="Directory for log files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --force implies ignoring prior progress entirely
    if args.force:
        args.resume = False

    setup_logging(log_dir=args.log_dir, run_name="extract_features")
    logger = get_logger(__name__)

    cfg = Config.from_yaml(args.config)

    # ── Resolve input video directory ──────────────────────────────────────────
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

    # ── Resolve output directories ─────────────────────────────────────────────
    if args.save_dir:
        features_dir = Path(args.save_dir)
        metadata_dir = (
            Path(args.metadata_dir)
            if args.metadata_dir
            else features_dir / "metadata"
        )
    else:
        features_base = Path(cfg.dataset.output_base_dir) / cfg.dataset.features_dir_name
        features_dir  = features_base / f"{folder_name}_{args.split}"
        metadata_dir  = (
            Path(args.metadata_dir)
            if args.metadata_dir
            else features_base / "metadata"
        )

    logger.info("INPUT        : %s", input_video_dir)
    logger.info("FEATURES DIR : %s", features_dir)
    logger.info("METADATA DIR : %s", metadata_dir)
    logger.info("SPLIT        : %s", args.split)
    logger.info(
        "RESUME       : %s%s",
        args.resume,
        "  (--force overrides, will re-extract everything)" if args.force else "",
    )

    # ── Dry run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        source = DiskVideoSource(video_dir=input_video_dir, split=args.split)
        logger.info("Dry-run: would process %d videos → %s", len(source), features_dir)
        for p in source._video_paths[:10]:
            logger.info("  %s", p.name)
        if len(source) > 10:
            logger.info("  … and %d more", len(source) - 10)
        return

    # ── Confirmation prompt ────────────────────────────────────────────────────
    if not args.yes:
        print(f"\n  Input  : {input_video_dir}")
        print(f"  Output : {features_dir}")
        print(f"  Split  : {args.split}")
        print(f"  Resume : {args.resume}")
        if args.force:
            print("  ⚠  --force: all videos will be re-extracted\n")
        resp = input("Proceed with extraction? (yes/no): ")
        if resp.strip().lower() != "yes":
            logger.info("Aborted.")
            sys.exit(0)

    # ── Build pipeline components ──────────────────────────────────────────────
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
        logger.error("Could not load feature extractor: %s", exc)
        logger.error("Install pytorchvideo and ultralytics first.")
        sys.exit(1)

    # ── Run extraction ─────────────────────────────────────────────────────────
    pipeline = FeatureExtractionPipeline(
        source=source,
        preprocessor=preprocessor,
        feature_extractor=feature_extractor,
        features_dir=features_dir,
        metadata_dir=metadata_dir,
        device=cfg.hardware.device,
    )

    processed, failed = pipeline.extract_all_features(
        resume=args.resume,
        force_reprocess=args.force,
        max_videos=args.max_videos,
    )

    logger.info("Done: %d succeeded, %d failed.", processed, failed)
    pipeline.check_status()


if __name__ == "__main__":
    main()