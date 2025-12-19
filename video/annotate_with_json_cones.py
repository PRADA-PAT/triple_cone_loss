#!/usr/bin/env python3
"""
Video Annotation with JSON Cone Annotations for F8 Drill Analysis.

Creates annotated videos using:
- STATIC cone positions from JSON annotations (same position every frame)
- DYNAMIC ball positions from parquet detection
- DYNAMIC pose skeleton from parquet detection

This provides a stable drill layout reference while showing player/ball movement.

Usage:
    python annotate_with_json_cones.py abdullah_nasib_f8
    python annotate_with_json_cones.py --list
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
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
    BALL_COLOR: Tuple[int, int, int] = (0, 255, 0)           # Green
    START_CONE_COLOR: Tuple[int, int, int] = (0, 255, 255)   # Yellow
    GATE1_COLOR: Tuple[int, int, int] = (255, 150, 0)        # Blue
    GATE2_COLOR: Tuple[int, int, int] = (255, 0, 255)        # Magenta
    POSE_KEYPOINT_COLOR: Tuple[int, int, int] = (255, 0, 255)
    POSE_SKELETON_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)       # White
    TEXT_BG_COLOR: Tuple[int, int, int] = (0, 0, 0)          # Black

    # Sizes
    BBOX_THICKNESS: int = 2
    SKELETON_THICKNESS: int = 2
    KEYPOINT_RADIUS: int = 4
    CONE_RADIUS: int = 12
    GATE_LINE_THICKNESS: int = 3
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 1

    # Confidence thresholds
    MIN_KEYPOINT_CONFIDENCE: float = 0.3
    MIN_BBOX_CONFIDENCE: float = 0.1


# Skeleton connections for pose visualization
SKELETON_CONNECTIONS = [
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('nose', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
    ('neck', 'hip'), ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
    ('left_ankle', 'left_heel'), ('right_ankle', 'right_heel'),
    ('left_ankle', 'left_big_toe'), ('right_ankle', 'right_big_toe'),
    ('left_big_toe', 'left_small_toe'), ('right_big_toe', 'right_small_toe'),
]

KEYPOINT_COLORS = {
    'head': (255, 200, 200),
    'torso': (200, 255, 200),
    'arms': (200, 200, 255),
    'legs': (255, 255, 200),
    'feet': (255, 200, 255),
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
# JSON Cone Loading
# ============================================================================

@dataclass
class ConeAnnotation:
    """Single cone from JSON annotation."""
    role: str
    px: float
    py: float


@dataclass
class Figure8Layout:
    """Complete Figure-8 layout from JSON."""
    start: ConeAnnotation
    gate1_left: ConeAnnotation
    gate1_right: ConeAnnotation
    gate2_left: ConeAnnotation
    gate2_right: ConeAnnotation

    @classmethod
    def from_json(cls, json_path: Path) -> 'Figure8Layout':
        """Load from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        cones = data['cones']
        return cls(
            start=ConeAnnotation('start', cones['start']['px'], cones['start']['py']),
            gate1_left=ConeAnnotation('gate1_left', cones['gate1_left']['px'], cones['gate1_left']['py']),
            gate1_right=ConeAnnotation('gate1_right', cones['gate1_right']['px'], cones['gate1_right']['py']),
            gate2_left=ConeAnnotation('gate2_left', cones['gate2_left']['px'], cones['gate2_left']['py']),
            gate2_right=ConeAnnotation('gate2_right', cones['gate2_right']['px'], cones['gate2_right']['py']),
        )

    @property
    def gate1_width(self) -> float:
        return np.sqrt((self.gate1_right.px - self.gate1_left.px)**2 +
                       (self.gate1_right.py - self.gate1_left.py)**2)

    @property
    def gate2_width(self) -> float:
        return np.sqrt((self.gate2_right.px - self.gate2_left.px)**2 +
                       (self.gate2_right.py - self.gate2_left.py)**2)


# ============================================================================
# Data Loaders
# ============================================================================

def load_ball_data(parquet_path: Path) -> pd.DataFrame:
    """Load ball detection data."""
    df = pd.read_parquet(parquet_path)
    return df[['frame_id', 'x1', 'y1', 'x2', 'y2', 'confidence']].copy()


def load_pose_data(parquet_path: Path) -> pd.DataFrame:
    """Load pose keypoint data."""
    df = pd.read_parquet(parquet_path)
    return df[['frame_idx', 'person_id', 'keypoint_name', 'x', 'y', 'confidence']].copy()


def prepare_pose_lookup(pose_df: pd.DataFrame) -> Dict[int, Dict[int, Dict[str, Tuple[float, float, float]]]]:
    """Create efficient lookup for pose data."""
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

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
    )
    cv2.rectangle(frame, (x1, y1 - text_height - 8),
                  (x1 + text_width + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.TEXT_BG_COLOR, config.FONT_THICKNESS)


def draw_cone_marker(frame: np.ndarray, cone: ConeAnnotation, label: str,
                     color: Tuple[int, int, int], config: AnnotationConfig) -> None:
    """Draw a cone marker with label."""
    center = (int(cone.px), int(cone.py))

    # Draw filled circle with white outline
    cv2.circle(frame, center, config.CONE_RADIUS, color, -1)
    cv2.circle(frame, center, config.CONE_RADIUS + 2, (255, 255, 255), 2)

    # Draw label above cone
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    text_x = center[0] - text_width // 2
    text_y = center[1] - config.CONE_RADIUS - 10

    # Background for label
    cv2.rectangle(frame,
                  (text_x - 3, text_y - text_height - 3),
                  (text_x + text_width + 3, text_y + 5),
                  (0, 0, 0), -1)
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)


def draw_gate_line(frame: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float],
                   color: Tuple[int, int, int], label: str, config: AnnotationConfig) -> None:
    """Draw a gate line between two cones."""
    pt1 = (int(p1[0]), int(p1[1]))
    pt2 = (int(p2[0]), int(p2[1]))

    # Draw dashed-style gate line
    cv2.line(frame, pt1, pt2, color, config.GATE_LINE_THICKNESS)

    # Draw label in center of gate
    center_x = (pt1[0] + pt2[0]) // 2
    center_y = (pt1[1] + pt2[1]) // 2 - 25  # Above the line

    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )

    # Background
    cv2.rectangle(frame,
                  (center_x - text_width // 2 - 3, center_y - text_height - 3),
                  (center_x + text_width // 2 + 3, center_y + 3),
                  (0, 0, 0), -1)
    cv2.putText(frame, label,
                (center_x - text_width // 2, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_json_cones(frame: np.ndarray, layout: Figure8Layout, config: AnnotationConfig) -> None:
    """Draw all JSON-annotated cones with gate lines."""
    # Draw Start cone
    draw_cone_marker(frame, layout.start, "START", config.START_CONE_COLOR, config)

    # Draw Gate 1 cones and line
    draw_cone_marker(frame, layout.gate1_left, "G1-L", config.GATE1_COLOR, config)
    draw_cone_marker(frame, layout.gate1_right, "G1-R", config.GATE1_COLOR, config)
    draw_gate_line(frame,
                   (layout.gate1_left.px, layout.gate1_left.py),
                   (layout.gate1_right.px, layout.gate1_right.py),
                   config.GATE1_COLOR, "GATE 1", config)

    # Draw Gate 2 cones and line
    draw_cone_marker(frame, layout.gate2_left, "G2-L", config.GATE2_COLOR, config)
    draw_cone_marker(frame, layout.gate2_right, "G2-R", config.GATE2_COLOR, config)
    draw_gate_line(frame,
                   (layout.gate2_left.px, layout.gate2_left.py),
                   (layout.gate2_right.px, layout.gate2_right.py),
                   config.GATE2_COLOR, "GATE 2", config)


def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, Tuple[float, float, float]],
                  config: AnnotationConfig) -> None:
    """Draw pose skeleton with keypoints."""
    # Draw connections first
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
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, (0, 0, 0), 1)


def draw_frame_info(frame: np.ndarray, frame_id: int,
                    ball_count: int, person_count: int,
                    layout: Figure8Layout,
                    config: AnnotationConfig) -> None:
    """Draw frame information overlay."""
    info_lines = [
        f"Frame: {frame_id}",
        f"Ball: {ball_count} | Persons: {person_count}",
        f"Cones: JSON Annotated (static)",
        f"G1 width: {layout.gate1_width:.0f}px | G2 width: {layout.gate2_width:.0f}px"
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
        y_offset += 22


# ============================================================================
# Video Processing
# ============================================================================

def annotate_video_with_json_cones(video_path: Path, parquet_dir: Path, output_path: Path,
                                   config: AnnotationConfig = None) -> bool:
    """
    Annotate video with JSON cone positions (static) and parquet ball/pose (dynamic).

    Args:
        video_path: Path to input video
        parquet_dir: Directory containing parquet files and cone_annotations.json
        output_path: Path for output annotated video
        config: Annotation configuration

    Returns:
        True if successful
    """
    if config is None:
        config = AnnotationConfig()

    base_name = parquet_dir.name

    # Load JSON cone annotations
    json_path = parquet_dir / "cone_annotations.json"
    if not json_path.exists():
        print(f"  Error: cone_annotations.json not found in {parquet_dir}")
        return False

    print(f"  Loading JSON cone annotations...")
    layout = Figure8Layout.from_json(json_path)
    print(f"    Gate 1 width: {layout.gate1_width:.0f}px")
    print(f"    Gate 2 width: {layout.gate2_width:.0f}px")

    # Load parquet data
    ball_path = parquet_dir / f"{base_name}_football.parquet"
    pose_path = parquet_dir / f"{base_name}_pose.parquet"

    if not ball_path.exists() or not pose_path.exists():
        print(f"  Error: Missing parquet files in {parquet_dir}")
        return False

    print(f"  Loading parquet data...")
    ball_df = load_ball_data(ball_path)
    pose_df = load_pose_data(pose_path)

    # Create lookup structures
    ball_lookup = ball_df.groupby('frame_id').apply(
        lambda g: g[['x1', 'y1', 'x2', 'y2', 'confidence']].to_dict('records')
    ).to_dict()
    pose_lookup = prepare_pose_lookup(pose_df)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open video: {video_path}")
        return False

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"  Error: Cannot create output video: {output_path}")
        cap.release()
        return False

    # Process frames
    print(f"  Processing frames...")
    for frame_id in tqdm(range(total_frames), desc="  Annotating", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        # STATIC: Draw JSON cones (same every frame)
        draw_json_cones(frame, layout, config)

        # DYNAMIC: Draw ball from parquet
        balls = ball_lookup.get(frame_id, [])
        for ball in balls:
            if ball['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                label = f"Ball {ball['confidence']:.2f}"
                draw_bbox(frame, ball['x1'], ball['y1'], ball['x2'], ball['y2'],
                         config.BALL_COLOR, label, config)

        # DYNAMIC: Draw pose skeletons from parquet
        persons = pose_lookup.get(frame_id, {})
        for person_id, keypoints in persons.items():
            draw_skeleton(frame, keypoints, config)

        # Draw frame info
        draw_frame_info(frame, frame_id, len(balls), len(persons), layout, config)

        out.write(frame)

    cap.release()
    out.release()

    print(f"  Saved (mp4v): {output_path}")

    # Convert to H.264 for better compatibility
    h264_path = convert_to_h264(output_path)
    if h264_path:
        return True
    else:
        print(f"  Warning: H.264 conversion failed, keeping mp4v version")
        return True


def convert_to_h264(input_path: Path) -> Path:
    """
    Convert video to H.264 codec using ffmpeg for better compatibility.

    Args:
        input_path: Path to input mp4v video

    Returns:
        Path to H.264 video, or None if conversion failed
    """
    # Create temporary output path
    temp_path = input_path.parent / f"{input_path.stem}_h264_temp.mp4"

    print(f"  Converting to H.264 for compatibility...")

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(temp_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and temp_path.exists():
            # Replace original with H.264 version
            backup_path = input_path.parent / f"{input_path.stem}_mp4v.mp4"
            input_path.rename(backup_path)
            temp_path.rename(input_path)
            print(f"  Converted to H.264: {input_path}")
            print(f"  Backup (mp4v): {backup_path}")
            return input_path
        else:
            print(f"  FFmpeg error: {result.stderr}")
            if temp_path.exists():
                temp_path.unlink()
            return None

    except FileNotFoundError:
        print(f"  FFmpeg not found - keeping mp4v format")
        return None
    except Exception as e:
        print(f"  Conversion error: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return None


# ============================================================================
# Main
# ============================================================================

def get_available_videos(videos_dir: Path, parquet_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Get list of videos with matching parquet data and JSON annotations."""
    available = []

    for parquet_path in sorted(parquet_dir.iterdir()):
        if not parquet_path.is_dir():
            continue

        base_name = parquet_path.name
        json_path = parquet_path / "cone_annotations.json"

        if not json_path.exists():
            continue

        # Look for matching video
        video_path = videos_dir / f"{base_name}.MOV"
        if video_path.exists():
            available.append((base_name, video_path, parquet_path))

    return available


def main():
    parser = argparse.ArgumentParser(
        description="Annotate F8 drill videos with JSON cone annotations"
    )
    parser.add_argument(
        "video_name",
        nargs="?",
        help="Name of video to process (e.g., abdullah_nasib_f8)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available videos with JSON annotations"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process ALL available videos with JSON annotations"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have JSON cone annotated output"
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

    available = get_available_videos(args.videos_dir, args.parquet_dir)

    if args.list:
        print("\n Available videos with JSON cone annotations:\n")
        for name, video_path, parquet_path in available:
            output_path = parquet_path / f"{name}_json_cones.mp4"
            status = "✓" if output_path.exists() else " "
            print(f"  [{status}] {name}")
        print(f"\nTotal: {len(available)} videos")
        print("(✓ = already processed)")
        return 0

    if args.all:
        # Process all available videos
        print(f"\n Processing ALL {len(available)} videos with JSON cone annotations...\n")
        config = AnnotationConfig()

        success_count = 0
        skip_count = 0
        fail_count = 0

        for i, (name, video_path, parquet_path) in enumerate(available, 1):
            output_path = parquet_path / f"{name}_json_cones.mp4"

            # Skip if already exists and flag is set
            if args.skip_existing and output_path.exists():
                print(f"[{i}/{len(available)}] Skipping {name} (already exists)")
                skip_count += 1
                continue

            print(f"\n{'='*60}")
            print(f"[{i}/{len(available)}] Processing: {name}")
            print(f"{'='*60}")

            success = annotate_video_with_json_cones(video_path, parquet_path, output_path, config)

            if success:
                success_count += 1
                print(f"  ✓ Success: {output_path.name}")
            else:
                fail_count += 1
                print(f"  ✗ Failed: {name}")

        print(f"\n{'='*60}")
        print(f" BATCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Processed: {success_count}")
        print(f"  Skipped:   {skip_count}")
        print(f"  Failed:    {fail_count}")
        print(f"  Total:     {len(available)}")

        return 0 if fail_count == 0 else 1

    if not args.video_name:
        print("Error: Please specify a video name, use --list, or use --all")
        return 1

    # Find matching video
    to_process = None
    for name, video_path, parquet_path in available:
        if name == args.video_name or name.startswith(args.video_name):
            to_process = (name, video_path, parquet_path)
            break

    if not to_process:
        print(f"Error: Video not found: {args.video_name}")
        print("Use --list to see available videos")
        return 1

    name, video_path, parquet_path = to_process

    print(f"\n Annotating {name} with JSON cones...\n")

    # Output with different name to preserve original
    output_path = parquet_path / f"{name}_json_cones.mp4"

    config = AnnotationConfig()
    success = annotate_video_with_json_cones(video_path, parquet_path, output_path, config)

    if success:
        print(f"\n Done! Output: {output_path}")
        return 0
    else:
        print("\n Annotation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
