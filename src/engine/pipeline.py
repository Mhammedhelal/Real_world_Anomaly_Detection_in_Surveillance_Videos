"""
Feature extraction pipeline for video processing.

From: extractoin_normal_one_Feb_15.ipynb
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from ..data.transforms import VideoPreprocessor
from ..data.labels import CustomDatasetMetadata
from ..config import get_config


class FeatureExtractionPipeline:
    """Pipeline for extracting and saving features"""

    def __init__(self, input_video_dir, features_dir, metadata_dir, feature_extractor, device='cuda'):
        self.input_video_dir = input_video_dir
        self.features_dir = features_dir
        self.metadata_dir = metadata_dir
        self.device = device
        self.feature_extractor = feature_extractor

        # Get config for preprocessing settings
        config = get_config()
        self.preprocessor = VideoPreprocessor(
            frame_size=config['dataset']['frame_size'],
            max_frames=config['dataset']['max_frames']
        )

        # Progress tracking
        self.progress_file = os.path.join(metadata_dir, 'extraction_progress.json')
        self.progress = self._load_progress()

        # Statistics
        self.stats = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'total_features': 0,
            'total_size_mb': 0
        }

    # ----------------------------
    # Progress utilities
    # ----------------------------
    def _load_progress(self):
        """Load extraction progress"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'processed': [], 'failed': [], 'start_time': datetime.now().isoformat()}

    def _save_progress(self):
        """Save extraction progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    # ----------------------------
    # Feature saving
    # ----------------------------
    def _save_features(self, features, video_info, split):
        """Save extracted features to NPZ file"""
        config = get_config()
        video_name = os.path.splitext(video_info['filename'])[0]
        filename = f"{split}_{video_name}.npz"
        filepath = os.path.join(self.features_dir, filename)

        metadata = {
            'video_path': video_info['video_path'],
            'filename': video_info['filename'],
            'label': video_info['label'],
            'class': video_info['class'],
            'split': split,
            'motion_extractor': config['feature_extraction']['motion_extractor'],
            'feature_dim': features.shape[1],
            'num_segments': features.shape[0],
            'segment_length': config['dataset']['segment_length'],
            'extraction_time': datetime.now().isoformat(),
            'video_info': CustomDatasetMetadata.get_video_info(video_info['full_path']),
            'dataset_type': 'training_normal_only'
        }

        np.savez_compressed(
            filepath,
            features=features.astype(np.float32),
            metadata=metadata
        )

        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        self.stats['total_features'] += 1
        self.stats['total_size_mb'] += file_size

        print(f"  ✅ Features saved: {filename}")
        print(f"     Shape: {features.shape[0]} segments × {features.shape[1]} features")
        print(f"     Size: {file_size:.2f} MB")

        return filepath

    # ----------------------------
    # Video processing
    # ----------------------------
    def process_video(self, video_info, split):
        """Process a single video and extract features"""
        video_path = video_info['full_path']

        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return False

        print(f"\n{'='*60}")
        print(f"Processing: {video_info['filename']}")
        print(f"Split: {split}, Label: {video_info['label']} ({video_info['class']})")
        print(f"Path: {video_path}")
        print(f"{'='*60}")

        try:
            config = get_config()
            
            # Step 1: Read video
            frames, fps, video_metadata = self.preprocessor.read_video(
                video_path,
                target_fps=config['feature_extraction']['target_fps']
            )
            if frames is None:
                print(f"❌ Failed to read video")
                return False

            # Step 2: Create segments
            segments = self.preprocessor.create_segments(
                frames,
                segment_length=config['dataset']['segment_length']
            )
            if len(segments) == 0:
                print(f"❌ No segments created")
                return False

            # Step 3: Extract features
            print("🔍 Extracting features...")
            start_time = time.time()
            features = self.feature_extractor.extract_features(segments)
            extraction_time = time.time() - start_time

            if features.shape[0] == 0:
                print(f"❌ No features extracted")
                return False

            print(f"✅ Features extracted: {features.shape[0]} segments")
            print(f"⏱️  Extraction time: {extraction_time:.2f} seconds")

            # Step 4: Save features
            self._save_features(features, video_info, split)

            # Step 5: Update progress
            self.progress['processed'].append({
                'filename': video_info['filename'],
                'split': split,
                'time': datetime.now().isoformat(),
                'features_shape': features.shape,
                'extraction_time': extraction_time
            })
            self._save_progress()

            # Step 6: Cleanup
            del frames, segments, features
            torch.cuda.empty_cache()

            self.stats['successful'] += 1
            return True

        except Exception as e:
            print(f"❌ Error processing video: {e}")
            import traceback
            traceback.print_exc()

            self.progress['failed'].append({
                'filename': video_info['filename'],
                'split': split,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            self._save_progress()

            self.stats['failed'] += 1
            return False

    # ----------------------------
    # Extract all features
    # ----------------------------
    def extract_all_features(self, splits, max_videos_per_split=None,
                             resume=True, force_reprocess=False):
        print("=" * 70)
        print("FEATURE EXTRACTION PIPELINE")
        print("=" * 70)

        total_processed = 0

        # Only process training videos
        split = 'train'
        print(f"\n📂 Processing {split.upper()} split (normal videos only)")
        print("-" * 40)

        split_videos = splits[split]
        if max_videos_per_split:
            split_videos = split_videos[:max_videos_per_split]

        processed_count = 0

        for i, video_info in enumerate(tqdm(split_videos, desc=f"Processing {split}")):
            # Check if already processed
            already_processed = any(
                item['filename'] == video_info['filename'] and item['split'] == split
                for item in self.progress['processed']
            )

            if resume and already_processed and not force_reprocess:
                print(f"⏭️  Skipping (already processed): {video_info['filename']}")
                processed_count += 1
                continue

            success = self.process_video(video_info, split)
            if success:
                processed_count += 1

            time.sleep(0.1)

            if (i + 1) % 5 == 0:
                print(f"\n📊 Progress: {i+1}/{len(split_videos)} videos")

        total_processed += processed_count
        self.stats['total_videos'] += len(split_videos)

        print(f"\n✅ {split.upper()} split completed:")
        print(f"   Processed: {processed_count}/{len(split_videos)}")

        # Skip test split if empty
        test_videos = splits.get('test', [])
        if len(test_videos) > 0:
            print(f"\n📂 Processing TEST split")
            print("-" * 40)
            print(f"   Test videos: {len(test_videos)}")
        else:
            print(f"\n📂 TEST split: No test videos in dataset")
            print(f"   This is correct for normal training videos")

        print(f"\n{'='*70}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total videos: {self.stats['total_videos']}")
        print(f"Successfully processed: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Feature files created: {self.stats['total_features']}")
        print(f"Total storage used: {self.stats['total_size_mb']:.2f} MB")
        print(f"Progress saved to: {self.progress_file}")

        self._save_progress()
        return self.stats['successful'], self.stats['failed']

    # ----------------------------
    # Analyze features
    # ----------------------------
    def analyze_features(self):
        """Analyze extracted features"""
        feature_files = [f for f in os.listdir(self.features_dir) if f.endswith('.npz')]

        if not feature_files:
            print("❌ No feature files found")
            return

        print(f"\n📊 Found {len(feature_files)} feature files")

        # Collect statistics
        shapes = []
        train_count = 0
        test_count = 0

        for filename in feature_files[:5]:
            filepath = os.path.join(self.features_dir, filename)
            data = np.load(filepath, allow_pickle=True)
            features = data['features']
            metadata = data['metadata'].item()

            shapes.append(features.shape)

            if metadata['split'] == 'train':
                train_count += 1
            else:
                test_count += 1

        print(f"\n📈 Feature Analysis:")
        print(f"   Train files: {train_count}")
        print(f"   Test files: {test_count}")

        if shapes:
            avg_shape = np.mean([s[0] for s in shapes]), np.mean([s[1] for s in shapes])
            print(f"   Average shape: {avg_shape[0]:.1f} segments × {avg_shape[1]:.1f} features")

            sample_file = os.path.join(self.features_dir, feature_files[0])
            sample_data = np.load(sample_file, allow_pickle=True)
            sample_features = sample_data['features']
            sample_metadata = sample_data['metadata'].item()

            print(f"\n📋 Sample file: {feature_files[0]}")
            print(f"   Video: {sample_metadata['filename']}")
            print(f"   Label: {sample_metadata['label']} ({sample_metadata['class']})")
            print(f"   Feature shape: {sample_features.shape}")
            print(f"   Extraction time: {sample_metadata['extraction_time']}")


def process_in_batches(batch_size=50, features_subfolder='normal_training',
                      input_video_dir='', features_dir='', metadata_dir='',
                      feature_extractor=None, device='cuda'):
    """
    Process videos in batches to avoid timeouts

    Args:
        batch_size: Number of videos to process in each batch
        features_subfolder: Subfolder to save features in
        input_video_dir: Path to input video directory
        features_dir: Path to features directory
        metadata_dir: Path to metadata directory
        feature_extractor: Feature extractor object
        device: Device to use ('cuda' or 'cpu')
    """
    print("\n" + "=" * 70)
    print(f"PROCESSING IN BATCHES OF {batch_size} VIDEOS")
    print(f"Saving to: {features_subfolder}")
    print("=" * 70)

    # Create subfolder for features
    subfolder_dir = os.path.join(features_dir, features_subfolder)
    os.makedirs(subfolder_dir, exist_ok=True)
    print(f"✅ Created subfolder: {subfolder_dir}")

    # Load metadata
    metadata_path = os.path.join(metadata_dir, 'dataset_metadata.pkl')
    if not os.path.exists(metadata_path):
        print("❌ Metadata file not found!")
        return

    import pickle
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    all_splits = metadata['splits']

    # Initialize pipeline with new features directory
    pipeline = FeatureExtractionPipeline(
        input_video_dir=input_video_dir,
        features_dir=subfolder_dir,
        metadata_dir=metadata_dir,
        feature_extractor=feature_extractor,
        device=device
    )

    total_processed = 0
    total_failed = 0

    print(f"\n📂 Processing TRAINING split (normal videos only)")
    print("-" * 40)

    train_videos = all_splits['train']
    num_batches = (len(train_videos) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(train_videos))
        batch_videos = train_videos[start_idx:end_idx]

        print(f"\n🔄 Batch {batch_num + 1}/{num_batches} "
              f"(videos {start_idx + 1}-{end_idx})")

        batch_split = {
            'train': batch_videos,
            'test': []
        }

        processed, failed = pipeline.extract_all_features(
            splits=batch_split,
            max_videos_per_split=None,
            resume=True,
            force_reprocess=False
        )

        total_processed += processed
        total_failed += failed

        print(f"\n📊 Batch {batch_num + 1} complete:")
        print(f"   Processed: {processed}")
        print(f"   Failed: {failed}")
        print(f"   Total so far: {total_processed}")

        if batch_num < num_batches - 1:
            print(f"\n⏳ Waiting 30 seconds before next batch...")
            time.sleep(30)

    print("\n" + "=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✅ Total processed: {total_processed}")
    print(f"❌ Total failed: {total_failed}")
    print(f"📁 Features saved to: {subfolder_dir}")

    return total_processed, total_failed


def check_status(features_subfolder='normal_training', features_dir='', metadata_dir=''):
    """Check current processing status"""
    print("\n" + "=" * 70)
    print(f"CURRENT PROCESSING STATUS: {features_subfolder}")
    print("=" * 70)

    subfolder_path = os.path.join(features_dir, features_subfolder)

    if not os.path.exists(subfolder_path):
        print(f"❌ Subfolder not found: {subfolder_path}")
        print(f"   Try: 'normal_one' or 'normal_training'")
        return

    feature_files = [f for f in os.listdir(subfolder_path) if f.endswith('.npz')]

    train_files = [f for f in feature_files if f.startswith('train_')]
    test_files = [f for f in feature_files if f.startswith('test_')]

    print(f"📊 Feature files created: {len(feature_files)}")
    print(f"   • Training files: {len(train_files)} (NORMAL videos)")
    print(f"   • Testing files: {len(test_files)} (should be 0)")

    total_size = 0
    for f in feature_files:
        try:
            total_size += os.path.getsize(os.path.join(subfolder_path, f))
        except:
            pass

    print(f"💾 Total storage used: {total_size / (1024*1024):.2f} MB")

    metadata_path = os.path.join(metadata_dir, 'dataset_metadata.pkl')
    if os.path.exists(metadata_path):
        import pickle
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        print(f"\n📈 Progress against total dataset:")
        print(f"   • Total normal videos: {metadata['total_videos']}")
        print(f"   • Processed: {len(feature_files)}")
        print(f"   • Remaining: {metadata['total_videos'] - len(feature_files)}")
        print(f"   • Completion: {(len(feature_files) / metadata['total_videos']) * 100:.1f}%")

        if 'note' in metadata:
            print(f"   • Note: {metadata['note']}")

    print(f"\n✅ SUMMARY:")
    print(f"   • You are processing NORMAL training videos only")
    print(f"   • These are for learning normal patterns")
    print(f"   • You'll need UCF-Crime anomalous videos next")
