"""
src/data/metadata.py
--------------------
Dataset metadata and split management utilities.
"""

import os
import pickle
from datetime import datetime

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetMetadata:
    """Load and manage custom dataset metadata."""

    VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv')

    @staticmethod
    def get_all_videos(video_dir: str) -> list:
        videos = []
        if not os.path.exists(video_dir):
            logger.error("Video directory not found: %s", video_dir)
            return videos

        for root, _, files in os.walk(video_dir):
            for file in files:
                if file.lower().endswith(DatasetMetadata.VIDEO_EXTENSIONS):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, video_dir)
                    videos.append({
                        'video_path': rel_path,
                        'full_path': full_path,
                        'label': 0,
                        'class': 'Normal',
                        'filename': file,
                        'directory': os.path.basename(root) if root != video_dir else 'root',
                    })

        logger.info("Found %d video files in %s", len(videos), video_dir)
        return videos

    @staticmethod
    def create_single_split(videos: list) -> dict:
        splits = {'train': videos, 'test': []}
        logger.info(
            "Split created: %d training videos (normal only)", len(videos)
        )
        return splits

    @staticmethod
    def save_metadata(splits: dict, metadata_path: str) -> dict:
        metadata = {
            'splits': splits,
            'total_videos': len(splits['train']) + len(splits['test']),
            'train_count': len(splits['train']),
            'test_count': len(splits['test']),
            'created_at': datetime.now().isoformat(),
            'note': 'All videos are NORMAL (label=0) for training normal patterns',
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info("Metadata saved → %s", metadata_path)
        return metadata

    @staticmethod
    def load_metadata(metadata_path: str) -> dict | None:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info("Metadata loaded ← %s", metadata_path)
            return metadata
        return None
