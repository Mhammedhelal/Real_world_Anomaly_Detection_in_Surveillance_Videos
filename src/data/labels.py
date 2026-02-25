"""
Dataset labels and metadata management utilities.

From: extractoin_normal_one_Feb_15.ipynb
"""

import os
import json
import pickle
from datetime import datetime
import cv2


# UCF-Crime Dataset Categories
UCF_CRIME_CATEGORIES = {
    0: 'Normal',
    1: 'Abuse',
    2: 'Arrest',
    3: 'Arson',
    4: 'Assault',
    5: 'Burglary',
    6: 'Explosion',
    7: 'Fighting',
    8: 'Robbery',
    9: 'Shooting',
    10: 'Shoplifting',
    11: 'Stealing',
    12: 'Vandalism',
    13: 'RoadAccidents'
}


class CustomDatasetMetadata:
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

    @staticmethod
    def get_video_info(video_path):
        """Get video information using OpenCV"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            cap.release()

            return {
                'fps': fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration,
                'size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
        except:
            return None


def get_class_name(label):
    """Get class name from label index"""
    return UCF_CRIME_CATEGORIES.get(label, 'Unknown')


def get_label_from_name(class_name):
    """Get label index from class name"""
    for label, name in UCF_CRIME_CATEGORIES.items():
        if name.lower() == class_name.lower():
            return label
    return None
