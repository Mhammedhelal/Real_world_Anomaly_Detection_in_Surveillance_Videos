"""
Video preprocessing and transformation utilities.

Extracted from: extractoin_normal_one_Feb_15.ipynb
"""

import os
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.transforms import Transform
from config import Config


cfg = Config.from_yaml('configs/default.yaml')


class VideoPreprocessor:
    """Preprocess video files for feature extraction"""

    def __init__(self, frame_size=cfg.dataset.frame_size, max_frames=3000):
        """
        Initialize video preprocessor.
        
        Args:
            frame_size: Tuple of (height, width) for frame resizing
            max_frames: Maximum frames to read from a video
        """
        self.frame_size = frame_size
        self.max_frames = max_frames

        # Transformation pipeline (ImageNet normalization)
        self.transform = Transform.build_transform(frame_size, mean=cfg.dataset.normalize_mean, std=cfg.datset.normalize_std)

    def read_video(self, video_path: str, target_fps: int = 8):
        """
        Read video file and return preprocessed frames at a target FPS.

        Args:
            video_path (str): Path to video file
            target_fps (int): Desired frames per second for output

        Returns:
            Tuple of (frames, fps, video_info)
            - frames: List of preprocessed frames as tensors
            - fps: Effective FPS after resampling
            - video_info: Dictionary with video metadata
        """
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None, None, None

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                return None, None, None

            # Get original video properties
            original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / original_fps if original_fps > 0 else 0

            # Determine frame sampling interval
            frame_interval = max(1, int(original_fps // target_fps))
            effective_fps = original_fps / frame_interval

            video_info = {
                'fps': effective_fps,
                'original_fps': original_fps,
                'total_frames': total_frames,
                'width': width,
                'height': height,
                'duration': duration,
                'path': video_path
            }

            print(f"🎬 Reading video: {os.path.basename(video_path)}")
            print(f"  Original FPS: {original_fps:.1f}, Target FPS: {effective_fps:.1f}, "
                  f"Frames: {total_frames}, Size: {width}x{height}")

            # Read frames
            frames = []
            frame_count = 0
            read_count = 0

            with tqdm(total=min(total_frames, self.max_frames),
                      desc="Reading frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret or read_count >= self.max_frames:
                        break

                    # Skip frames to match target FPS
                    if frame_count % frame_interval != 0:
                        frame_count += 1
                        continue

                    # Convert BGR → RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Apply transformation to tensor
                    frame_tensor = self.transform(frame_rgb)
                    frames.append(frame_tensor)

                    frame_count += 1
                    read_count += 1
                    pbar.update(1)

            cap.release()

            if len(frames) == 0:
                print(f"❌ No frames read from: {video_path}")
                return None, None, None

            print(f"✅ Successfully read {len(frames)} frames")
            return frames, effective_fps, video_info

        except Exception as e:
            print(f"❌ Error reading video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def create_segments(self, frames, segment_length=16):
        """
        Create fixed-length segments from frames

        Args:
            frames: List of frame tensors
            segment_length: Number of frames per segment

        Returns:
            segments: List of segments [segment_length, C, H, W]
        """
        if len(frames) < segment_length:
            # Pad with last frame
            padding_needed = segment_length - len(frames)
            frames = frames + [frames[-1]] * padding_needed

        segments = []
        for i in range(0, len(frames), segment_length):
            segment_frames = frames[i:i+segment_length]

            # Pad if necessary
            if len(segment_frames) < segment_length:
                padding_needed = segment_length - len(segment_frames)
                segment_frames.extend([segment_frames[-1]] * padding_needed)

            # Stack frames: [segment_length, C, H, W]
            segment = torch.stack(segment_frames)
            segments.append(segment)

        print(f"📊 Created {len(segments)} segments of {segment_length} frames each")
        return segments

    def save_frames(self, frames, output_dir, video_name, max_save=10):
        """Save sample frames for visualization"""
        os.makedirs(output_dir, exist_ok=True)

        # Save a few sample frames
        save_indices = np.linspace(0, len(frames)-1,
                                  min(max_save, len(frames)), dtype=int)

        for i, idx in enumerate(save_indices):
            # Convert tensor to numpy for saving
            frame_np = frames[idx].permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame_np = frame_np * std + mean
            frame_np = np.clip(frame_np, 0, 1)
            frame_np = (frame_np * 255).astype(np.uint8)

            save_path = os.path.join(output_dir,
                                   f"{video_name}_frame_{i:03d}.jpg")
            Image.fromarray(frame_np).save(save_path)

        print(f"💾 Saved {len(save_indices)} sample frames to {output_dir}")
