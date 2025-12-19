#!/usr/bin/env python3
"""
Video Annotation Script for F8 Drill Analysis
Overlays bounding boxes (ball, cones) and pose skeleton on videos.

Usage:
    python annotate_videos.py                      # Process all videos
    python annotate_videos.py abdullah_nasib_f8    # Process single video
    python annotate_videos.py --list               # List available videos
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from dataclasses import dataclass
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AnnotationConfig:
    """Configuration for annotation styles."""
    # Colors (BGR format for OpenCV)
    BALL_COLOR: Tuple[int, int, int] = (0, 255, 0)        # Green
    CONE_COLOR: Tuple[int, int, int] = (0, 165, 255)      # Orange
    POSE_KEYPOINT_COLOR: Tuple[int, int, int] = (255, 0, 255)  # Magenta
    POSE_SKELETON_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)    # White
    TEXT_BG_COLOR: Tuple[int, int, int] = (0, 0, 0)       # Black

    # Line/circle sizes
    BBOX_THICKNESS: int = 2
    SKELETON_THICKNESS: int = 2
    KEYPOINT_RADIUS: int = 4
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 1

    # Confidence thresholds
    MIN_KEYPOINT_CONFIDENCE: float = 0.3
    MIN_BBOX_CONFIDENCE: float = 0.1


# Skeleton connections for pose visualization (COCO-style + extended)
SKELETON_CONNECTIONS = [
    # Face
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),

    # Upper body
    ('nose', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),

    # Torso
    ('neck', 'hip'), ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),

    # Lower body
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),

    # Feet
    ('left_ankle', 'left_heel'), ('right_ankle', 'right_heel'),
    ('left_ankle', 'left_big_toe'), ('right_ankle', 'right_big_toe'),
    ('left_big_toe', 'left_small_toe'), ('right_big_toe', 'right_small_toe'),
]

# Keypoint colors by body part (BGR)
KEYPOINT_COLORS = {
    'head': (255, 200, 200),     # Light pink - nose, eyes, ears
    'torso': (200, 255, 200),    # Light green - shoulders, neck, hip
    'arms': (200, 200, 255),     # Light blue - elbows, wrists
    'legs': (255, 255, 200),     # Light cyan - knees, ankles
    'feet': (255, 200, 255),     # Light magenta - toes, heels
}

KEYPOINT_BODY_PART = {
    'nose': 'head', 'left_eye': 'head', 'right_eye': 'head',
    'left_ear': 'head', 'right_ear': 'head', 'head': 'head',
    'neck': 'torso', 'left_shoulder': 'torso', 'right_shoulder': 'torso',
    'hip': 'torso', 'left_hip': 'torso', 'right_hip': 'torso',
    'left_elbow': 'arms', 'right_elbow': 'arms',
    'left_wrist': 'arms', 'right_wrist': 'arms',
    'left_knee': 'legs', 'right_knee': 'legs',
    'left_ankle': 'legs', 'right_ankle': 'legs',
    'left_big_toe': 'feet', 'right_big_toe': 'feet',
    'left_small_toe': 'feet', 'right_small_toe': 'feet',
    'left_heel': 'feet', 'right_heel': 'feet',
}


# ============================================================================
# Data Loaders
# ============================================================================

def load_ball_data(parquet_path: Path) -> pd.DataFrame:
    """Load ball detection data."""
    df = pd.read_parquet(parquet_path)
    return df[['frame_id', 'x1', 'y1', 'x2', 'y2', 'confidence']].copy()


def load_cone_data(parquet_path: Path) -> pd.DataFrame:
    """Load cone detection data."""
    df = pd.read_parquet(parquet_path)
    return df[['frame_id', 'object_id', 'x1', 'y1', 'x2', 'y2', 'confidence']].copy()


def load_pose_data(parquet_path: Path) -> pd.DataFrame:
    """Load pose keypoint data."""
    df = pd.read_parquet(parquet_path)
    return df[['frame_idx', 'person_id', 'keypoint_name', 'x', 'y', 'confidence']].copy()


def prepare_pose_lookup(pose_df: pd.DataFrame) -> Dict[int, Dict[int, Dict[str, Tuple[float, float, float]]]]:
    """
    Create efficient lookup structure for pose data.
    Returns: {frame_id: {person_id: {keypoint_name: (x, y, confidence)}}}
    """
    lookup = {}
    for _, row in pose_df.iterrows():
        frame_id = int(row['frame_idx'])
        person_id = int(row['person_id'])
        keypoint = row['keypoint_name']

        if frame_id not in lookup:
            lookup[frame_id] = {}
        if person_id not in lookup[frame_id]:
            lookup[frame_id][person_id] = {}

        lookup[frame_id][person_id][keypoint] = (
            float(row['x']),
            float(row['y']),
            float(row['confidence'])
        )
    return lookup


# ============================================================================
# Drawing Functions
# ============================================================================

def draw_bbox(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
              color: Tuple[int, int, int], label: str, config: AnnotationConfig) -> None:
    """Draw a bounding box with label."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

    # Draw label background
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
    )
    cv2.rectangle(frame, (x1, y1 - text_height - 8),
                  (x1 + text_width + 4, y1), color, -1)

    # Draw label text
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.TEXT_BG_COLOR, config.FONT_THICKNESS)


def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, Tuple[float, float, float]],
                  config: AnnotationConfig) -> None:
    """Draw pose skeleton with keypoints."""
    # Draw connections first (so keypoints appear on top)
    for kp1_name, kp2_name in SKELETON_CONNECTIONS:
        if kp1_name in keypoints and kp2_name in keypoints:
            x1, y1, conf1 = keypoints[kp1_name]
            x2, y2, conf2 = keypoints[kp2_name]

            if conf1 >= config.MIN_KEYPOINT_CONFIDENCE and conf2 >= config.MIN_KEYPOINT_CONFIDENCE:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.line(frame, pt1, pt2, config.POSE_SKELETON_COLOR, config.SKELETON_THICKNESS)

    # Draw keypoints
    for kp_name, (x, y, conf) in keypoints.items():
        if conf >= config.MIN_KEYPOINT_CONFIDENCE:
            pt = (int(x), int(y))
            body_part = KEYPOINT_BODY_PART.get(kp_name, 'torso')
            color = KEYPOINT_COLORS.get(body_part, config.POSE_KEYPOINT_COLOR)
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, color, -1)
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, (0, 0, 0), 1)  # Black outline


def draw_frame_info(frame: np.ndarray, frame_id: int,
                    ball_count: int, cone_count: int, person_count: int,
                    config: AnnotationConfig) -> None:
    """Draw frame information overlay."""
    info_lines = [
        f"Frame: {frame_id}",
        f"Ball: {ball_count} | Cones: {cone_count} | Persons: {person_count}"
    ]

    y_offset = 25
    for line in info_lines:
        (text_width, text_height), _ = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
        )
        cv2.rectangle(frame, (5, y_offset - text_height - 5),
                      (10 + text_width, y_offset + 5), config.TEXT_BG_COLOR, -1)
        cv2.putText(frame, line, (8, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE, config.TEXT_COLOR, config.FONT_THICKNESS)
        y_offset += 25


# ============================================================================
# Video Processing
# ============================================================================

def annotate_video(video_path: Path, parquet_dir: Path, output_path: Path,
                   config: AnnotationConfig = None) -> bool:
    """
    Annotate a video with bounding boxes and pose skeleton.

    Args:
        video_path: Path to input video
        parquet_dir: Directory containing parquet files
        output_path: Path for output annotated video
        config: Annotation configuration

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = AnnotationConfig()

    # Get base name for parquet files
    base_name = parquet_dir.name

    # Load data
    ball_path = parquet_dir / f"{base_name}_football.parquet"
    cone_path = parquet_dir / f"{base_name}_cone.parquet"
    pose_path = parquet_dir / f"{base_name}_pose.parquet"

    if not all(p.exists() for p in [ball_path, cone_path, pose_path]):
        print(f"  ❌ Missing parquet files in {parquet_dir}")
        return False

    print(f"  📂 Loading data...")
    ball_df = load_ball_data(ball_path)
    cone_df = load_cone_data(cone_path)
    pose_df = load_pose_data(pose_path)

    # Create lookup structures
    ball_lookup = ball_df.groupby('frame_id').apply(
        lambda g: g[['x1', 'y1', 'x2', 'y2', 'confidence']].to_dict('records')
    ).to_dict()

    cone_lookup = cone_df.groupby('frame_id').apply(
        lambda g: g[['object_id', 'x1', 'y1', 'x2', 'y2', 'confidence']].to_dict('records')
    ).to_dict()

    pose_lookup = prepare_pose_lookup(pose_df)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ Cannot open video: {video_path}")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  📹 Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"  ❌ Cannot create output video: {output_path}")
        cap.release()
        return False

    # Process frames
    print(f"  🎬 Processing frames...")
    for frame_id in tqdm(range(total_frames), desc="  Annotating", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        # Get data for this frame
        balls = ball_lookup.get(frame_id, [])
        cones = cone_lookup.get(frame_id, [])
        persons = pose_lookup.get(frame_id, {})

        # Draw cones (bottom layer)
        for cone in cones:
            if cone['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                label = f"Cone {int(cone['object_id'])}"
                draw_bbox(frame, cone['x1'], cone['y1'], cone['x2'], cone['y2'],
                         config.CONE_COLOR, label, config)

        # Draw ball
        for ball in balls:
            if ball['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                label = f"Ball {ball['confidence']:.2f}"
                draw_bbox(frame, ball['x1'], ball['y1'], ball['x2'], ball['y2'],
                         config.BALL_COLOR, label, config)

        # Draw pose skeletons (top layer)
        for person_id, keypoints in persons.items():
            draw_skeleton(frame, keypoints, config)

        # Draw frame info
        draw_frame_info(frame, frame_id, len(balls), len(cones), len(persons), config)

        # Write frame
        out.write(frame)

    # Cleanup
    cap.release()
    out.release()

    print(f"  ✅ Saved: {output_path}")
    return True


# ============================================================================
# Main
# ============================================================================

def get_available_videos(videos_dir: Path, parquet_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Get list of videos with matching parquet data."""
    available = []

    for video_path in sorted(videos_dir.glob("*.MOV")):
        # Get base name (without extension)
        base_name = video_path.stem
        parquet_path = parquet_dir / base_name

        if parquet_path.is_dir():
            available.append((base_name, video_path, parquet_path))

    return available


def main():
    parser = argparse.ArgumentParser(
        description="Annotate F8 drill videos with bounding boxes and pose skeleton"
    )
    parser.add_argument(
        "video_name",
        nargs="?",
        help="Name of video to process (without extension), or --all for all videos"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process all available videos"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available videos"
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/videos"),
        help="Directory containing source videos"
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/video_detection_pose_ball_cones"),
        help="Directory containing parquet data folders"
    )

    args = parser.parse_args()

    # Get available videos
    available = get_available_videos(args.videos_dir, args.parquet_dir)

    if args.list:
        print("\n📋 Available videos with parquet data:\n")
        for name, video_path, parquet_path in available:
            print(f"  • {name}")
        print(f"\nTotal: {len(available)} videos")
        return 0

    if not available:
        print("❌ No videos found with matching parquet data")
        return 1

    # Determine which videos to process
    to_process = []

    if args.all or args.video_name is None:
        to_process = available
    else:
        # Find matching video
        for name, video_path, parquet_path in available:
            if name == args.video_name or name.startswith(args.video_name):
                to_process.append((name, video_path, parquet_path))
                break

        if not to_process:
            print(f"❌ Video not found: {args.video_name}")
            print("Use --list to see available videos")
            return 1

    # Process videos
    config = AnnotationConfig()
    success_count = 0

    print(f"\n🎬 Processing {len(to_process)} video(s)...\n")

    for name, video_path, parquet_path in to_process:
        print(f"📹 {name}")

        output_path = parquet_path / f"{name}_annotated.mp4"

        if annotate_video(video_path, parquet_path, output_path, config):
            success_count += 1
        print()

    print(f"✅ Completed: {success_count}/{len(to_process)} videos annotated")

    return 0 if success_count == len(to_process) else 1


if __name__ == "__main__":
    sys.exit(main())
