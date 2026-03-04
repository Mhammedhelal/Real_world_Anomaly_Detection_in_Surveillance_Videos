"""CLI for extracting video features.

This script supports arbitrary subfolders under the video directory and
lets the user specify whether the resulting feature files should be
marked as "train" or "test".  Paths are derived from the YAML config so
that everything stays consistent with the rest of the project.
"""

import argparse
import os
import sys

import numpy as np

from src.engine.FeatureExtractionPipeline import process_in_batches
from src.config import Config


def main():
    parser = argparse.ArgumentParser(
        description="Extract video features and save to the configured directory."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--video-folder",
        type=str,
        help="Subdirectory under the base video directory (e.g. 'normal', 'anomalous').",
    )
    group.add_argument(
        "--video-dir",
        type=str,
        help="Explicit path to a folder containing videos (overrides --video-folder).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Split label to encode in the output filenames.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of videos to process per batch.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Assume yes to confirmation prompt.",
    )

    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    # determine input directory
    if args.video_dir:
        input_video_dir = args.video_dir
    elif args.video_folder:
        base_parent = os.path.dirname(cfg.dataset.input_video_dir)
        input_video_dir = os.path.join(base_parent, args.video_folder)
    else:
        input_video_dir = cfg.dataset.input_video_dir

    if not os.path.isdir(input_video_dir):
        print(f"❌ Input video directory not found: {input_video_dir}")
        sys.exit(1)

    # output directories from config
    features_dir = os.path.join(
        cfg.dataset.output_base_dir, cfg.dataset.features_dir_name
    )
    metadata_dir = os.path.join(features_dir, "metadata")

    # compute subfolder name
    base_name = (
        args.video_folder
        if args.video_folder
        else os.path.basename(os.path.normpath(input_video_dir))
    )
    features_subfolder = f"{base_name}_{args.split}"

    print("\n" + "=" * 70)
    print(f"INPUT VIDEOS   : {input_video_dir}")
    print(f"FEATURES DIR   : {features_dir}")
    print(f"SUBFOLDER      : {features_subfolder}")
    print(f"LABEL SPLIT    : {args.split}")
    print(f"BATCH SIZE     : {args.batch_size}")
    print("=" * 70)

    if not args.yes:
        resp = input("Proceed with extraction? (yes/no): ")
        if resp.lower() != "yes":
            print("Aborted by user.")
            sys.exit(0)

    processed, failed = process_in_batches(
        batch_size=args.batch_size,
        features_subfolder=features_subfolder,
        input_video_dir=input_video_dir,
        features_dir=features_dir,
        metadata_dir=metadata_dir,
    )

    print(f"\n🎉 Extraction finished: {processed} succeeded, {failed} failed.")

    # report
    subfolder_path = os.path.join(features_dir, features_subfolder)
    if os.path.exists(subfolder_path):
        feature_files = [f for f in os.listdir(subfolder_path) if f.endswith(".npz")]
        print(f"✅ {len(feature_files)} feature files in {subfolder_path}")
        train_files = [f for f in feature_files if f.startswith("train_")]
        test_files = [f for f in feature_files if f.startswith("test_")]
        print(f"   • Train: {len(train_files)}")
        print(f"   • Test : {len(test_files)}")
        total_size = sum(os.path.getsize(os.path.join(subfolder_path, f)) for f in feature_files)
        print(f"   • Size : {total_size/(1024*1024):.2f} MB")
    else:
        print(f"❌ Output folder missing: {subfolder_path}")


if __name__ == "__main__":
    main()
