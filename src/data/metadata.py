"""
Dataset metadata and split management utilities.

This module handles filesystem operations for indexing video files,
creating dataset splits, and saving/loading metadata. It is intentionally
separated from label definitions and any OpenCV logic.
"""

import os
import pickle
from datetime import datetime


class DatasetMetadata:
    """Load and manage custom dataset metadata"""

    @staticmethod
    def get_all_videos(video_dir):
        """Get all videos from dataset directory"""
        videos = []

        if not os.path.exists(video_dir):
            print(f"❌ Video directory not found: {video_dir}")
            return videos

        # Get all video files
        video_extensions = ('.avi', '.mp4', '.mov', '.mkv', '.flv', '.wmv')

        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.lower().endswith(video_extensions):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, video_dir)

                    # For normal videos, set label=0
                    videos.append({
                        'video_path': rel_path,
                        'full_path': full_path,
                        'label': 0,  # All are normal videos
                        'class': 'Normal',
                        'filename': file,
                        'directory': os.path.basename(root) if root != video_dir else 'root'
                    })

        print(f"✅ Found {len(videos)} video files in {video_dir}")
        return videos

    @staticmethod
    def create_single_split(videos):
        """Create single split - ALL videos for training"""
        splits = {
            'train': videos,  # All videos for training
            'test': []        # Empty test set
        }

        print(f"✅ Created single split for anomaly detection training:")
        print(f"   Training videos: {len(videos)} (ALL videos for training)")
        print(f"   Testing videos: 0 (Will use UCF-Crime test set later)")
        print(f"   Note: These are NORMAL videos only for training normal patterns")

        return splits

    @staticmethod
    def save_metadata(splits, metadata_path):
        """Save metadata to file"""
        metadata = {
            'splits': splits,
            'total_videos': len(splits['train']) + len(splits['test']),
            'train_count': len(splits['train']),
            'test_count': len(splits['test']),
            'created_at': datetime.now().isoformat(),
            'note': 'All videos are NORMAL (label=0) for training normal patterns'
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadata saved to: {metadata_path}")
        return metadata

    @staticmethod
    def load_metadata(metadata_path):
        """Load metadata from file"""
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"✅ Metadata loaded from: {metadata_path}")
            return metadata
        return None
