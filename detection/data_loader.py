"""
Data loading module for Ball Control Detection System.

Handles loading parquet files and preprocessing for analysis.
Only uses ankle keypoints (left_ankle, right_ankle) for stability.

Key functions:
- load_triple_cone_layout_from_parquet(): Load 3-cone positions from parquet
- load_parquet_data(): Load parquet files (ball, pose)
- extract_ankle_positions(): Filter pose data to ankles only
- get_closest_ankle_per_frame(): Find nearest ankle to ball per frame
- get_video_fps(): Read actual FPS from video file
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

# OpenCV import for video FPS reading (optional)
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

from .data_structures import TripleConeLayout

logger = logging.getLogger(__name__)

# Only use ankles for ball-foot distance (more stable than toes)
ANKLE_KEYPOINTS = ['left_ankle', 'right_ankle']

# Default FPS fallback when video cannot be read
DEFAULT_FPS = 30.0


# =============================================================================
# VIDEO METADATA
# =============================================================================

def get_video_fps(video_path: str, default_fps: float = DEFAULT_FPS) -> float:
    """
    Read actual FPS from a video file.

    This ensures timestamps are accurate regardless of whether the video
    was recorded at 25fps, 30fps, or other frame rates.

    Args:
        video_path: Path to the video file
        default_fps: Fallback FPS if video cannot be read (default: 30.0)

    Returns:
        FPS value from video, or default_fps if unable to read
    """
    if not HAS_OPENCV:
        logger.warning(
            f"OpenCV not available, using default FPS={default_fps}. "
            "Install opencv-python for accurate FPS detection."
        )
        return default_fps

    path = Path(video_path)
    if not path.exists():
        logger.warning(f"Video file not found: {path}, using default FPS={default_fps}")
        return default_fps

    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            logger.warning(f"Could not open video: {path}, using default FPS={default_fps}")
            return default_fps

        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Sanity check - FPS should be reasonable (10-120)
        if fps < 10 or fps > 120:
            logger.warning(
                f"Unusual FPS={fps} from {path.name}, using default FPS={default_fps}"
            )
            return default_fps

        logger.info(f"Video FPS: {fps:.2f} (from {path.name})")
        return fps

    except Exception as e:
        logger.warning(f"Error reading video FPS: {e}, using default FPS={default_fps}")
        return default_fps


# =============================================================================
# 3-CONE LAYOUT LOADING
# =============================================================================

def load_triple_cone_layout_from_parquet(cone_parquet_path: str) -> TripleConeLayout:
    """
    Load 3-cone positions from parquet file.

    Cones are identified by their mean position across all frames,
    then sorted by X position (left to right) to assign roles:
    - CONE1 (leftmost): HOME cone where player starts
    - CONE2 (center): First turning cone
    - CONE3 (rightmost): Second turning cone

    Args:
        cone_parquet_path: Path to cone parquet file (*_cone.parquet)

    Returns:
        TripleConeLayout with cone1, cone2, cone3 positions

    Raises:
        FileNotFoundError: If parquet file doesn't exist
        ValueError: If fewer than 3 cones detected
    """
    path = Path(cone_parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Cone parquet file not found: {path}")

    logger.info(f"Loading cone positions from parquet: {path.name}")

    cone_df = pd.read_parquet(path)

    if cone_df.empty:
        raise ValueError(f"Cone parquet file is empty: {path}")

    # Group by object_id and compute mean position across all frames
    positions = []
    for obj_id in sorted(cone_df['object_id'].unique()):
        obj_data = cone_df[cone_df['object_id'] == obj_id]
        mean_x = obj_data['center_x'].mean()
        mean_y = obj_data['center_y'].mean()
        positions.append((mean_x, mean_y))

    # Sort by X position (left to right): CONE1, CONE2, CONE3
    positions.sort(key=lambda p: p[0])

    if len(positions) < 3:
        raise ValueError(
            f"Expected at least 3 cones in parquet, found {len(positions)}. "
            f"Positions: {positions}"
        )

    # Take first 3 cones (in case there are extras)
    cone1, cone2, cone3 = positions[:3]

    layout = TripleConeLayout(
        cone1=cone1,
        cone2=cone2,
        cone3=cone3,
    )

    logger.info(
        f"Loaded Triple Cone layout from parquet: "
        f"CONE1=({cone1[0]:.0f}, {cone1[1]:.0f}), "
        f"CONE2=({cone2[0]:.0f}, {cone2[1]:.0f}), "
        f"CONE3=({cone3[0]:.0f}, {cone3[1]:.0f})"
    )

    return layout


# =============================================================================
# PARQUET DATA LOADING
# =============================================================================

def load_parquet_data(path: str) -> pd.DataFrame:
    """
    Load a parquet file with validation.

    Args:
        path: Path to parquet file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Parquet file is empty: {path}")

    logger.info(f"Loaded {len(df)} records from {path.name}")
    return df


def load_all_data(
    cone_path: str,
    ball_path: str,
    pose_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three parquet files.

    Args:
        cone_path: Path to cone parquet
        ball_path: Path to football parquet
        pose_path: Path to pose parquet

    Returns:
        Tuple of (cone_df, ball_df, pose_df)
    """
    logger.info("Loading all parquet files...")

    cone_df = load_parquet_data(cone_path)
    ball_df = load_parquet_data(ball_path)
    pose_df = load_parquet_data(pose_path)

    logger.info(f"Loaded: {len(cone_df)} cones, {len(ball_df)} balls, {len(pose_df)} poses")

    return cone_df, ball_df, pose_df


def extract_ankle_positions(pose_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only ankle keypoints from pose data.

    Args:
        pose_df: Full pose DataFrame with all keypoints

    Returns:
        DataFrame with only ankle keypoints (left_ankle, right_ankle)
    """
    ankle_df = pose_df[pose_df['keypoint_name'].isin(ANKLE_KEYPOINTS)].copy()

    if ankle_df.empty:
        raise ValueError(
            f"No ankle keypoints found. Available keypoints: "
            f"{pose_df['keypoint_name'].unique().tolist()}"
        )

    logger.info(f"Extracted {len(ankle_df)} ankle records from {len(pose_df)} pose records")
    return ankle_df


def get_closest_ankle_per_frame(
    ankle_df: pd.DataFrame,
    ball_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Find the closest ankle to ball for each frame.

    For each frame, computes distance from ball to both ankles
    and returns the closest one.

    NOTE: Uses PIXEL coordinates for distance calculation because
    ball and pose field coordinates use different transformations.

    Args:
        ankle_df: DataFrame with ankle positions (from extract_ankle_positions)
        ball_df: DataFrame with ball positions

    Returns:
        DataFrame with one row per frame containing:
        - frame_id
        - ankle_x, ankle_y (pixel coordinates)
        - ankle_field_x, ankle_field_y (field coordinates)
        - closest_ankle (keypoint name)
        - ball_ankle_distance (PIXEL units - consistent coordinate system)
    """
    results = []
    skipped_frames = 0

    for frame_id in ball_df['frame_id'].unique():
        # Get ball position for this frame
        ball_rows = ball_df[ball_df['frame_id'] == frame_id]
        if ball_rows.empty:
            skipped_frames += 1
            continue

        ball_row = ball_rows.iloc[0]
        # Use PIXEL coordinates for distance (ball and pose have different field transforms)
        ball_px = ball_row['center_x']
        ball_py = ball_row['center_y']

        # Get ankles for this frame
        # Note: pose uses 'frame_idx', ball uses 'frame_id'
        frame_ankles = ankle_df[ankle_df['frame_idx'] == frame_id]

        if frame_ankles.empty:
            skipped_frames += 1
            continue

        # Calculate distance to each ankle using PIXEL coordinates
        frame_ankles = frame_ankles.copy()
        frame_ankles['distance'] = np.sqrt(
            (frame_ankles['x'] - ball_px)**2 +
            (frame_ankles['y'] - ball_py)**2
        )

        # Get closest ankle
        closest_idx = frame_ankles['distance'].idxmin()
        closest = frame_ankles.loc[closest_idx]

        results.append({
            'frame_id': frame_id,
            'ankle_x': closest['x'],
            'ankle_y': closest['y'],
            'ankle_field_x': closest['field_x'],
            'ankle_field_y': closest['field_y'],
            'closest_ankle': closest['keypoint_name'],
            'ball_ankle_distance': closest['distance'],  # Now in pixels
        })

    if skipped_frames > 0:
        logger.warning(f"Skipped {skipped_frames} frames due to missing data")

    result_df = pd.DataFrame(results)
    logger.info(f"Computed closest ankle for {len(result_df)} frames")

    return result_df


def validate_data_alignment(
    cone_df: pd.DataFrame,
    ball_df: pd.DataFrame,
    pose_df: pd.DataFrame
) -> dict:
    """
    Validate that all data files are properly aligned.

    Args:
        cone_df, ball_df, pose_df: Loaded DataFrames

    Returns:
        Dictionary with validation statistics
    """
    stats = {}

    # Frame ranges
    cone_frames = set(cone_df['frame_id'].unique())
    ball_frames = set(ball_df['frame_id'].unique())
    pose_frames = set(pose_df['frame_idx'].unique())

    stats['cone_frame_range'] = (min(cone_frames), max(cone_frames))
    stats['ball_frame_range'] = (min(ball_frames), max(ball_frames))
    stats['pose_frame_range'] = (min(pose_frames), max(pose_frames))

    # Common frames
    common_frames = cone_frames & ball_frames & pose_frames
    stats['common_frames'] = len(common_frames)
    stats['total_unique_frames'] = len(cone_frames | ball_frames | pose_frames)

    # Coverage
    stats['coverage_pct'] = (
        len(common_frames) / stats['total_unique_frames'] * 100
        if stats['total_unique_frames'] > 0 else 0
    )

    # Record counts
    stats['cone_records'] = len(cone_df)
    stats['ball_records'] = len(ball_df)
    stats['pose_records'] = len(pose_df)

    logger.info(f"Data alignment: {stats['coverage_pct']:.1f}% coverage "
                f"({stats['common_frames']} common frames)")

    return stats


def get_frame_data(
    frame_id: int,
    ball_df: pd.DataFrame,
    ankle_df: pd.DataFrame,
    cone_df: pd.DataFrame
) -> Optional[dict]:
    """
    Get all data for a single frame.

    Args:
        frame_id: Frame number
        ball_df: Ball DataFrame
        ankle_df: Ankle DataFrame (filtered)
        cone_df: Cone DataFrame

    Returns:
        Dictionary with frame data or None if missing
    """
    # Ball
    ball_row = ball_df[ball_df['frame_id'] == frame_id]
    if ball_row.empty:
        return None
    ball_row = ball_row.iloc[0]

    # Ankles
    frame_ankles = ankle_df[ankle_df['frame_idx'] == frame_id]
    if frame_ankles.empty:
        return None

    # Calculate closest ankle
    ball_x = ball_row['field_center_x']
    ball_y = ball_row['field_center_y']

    frame_ankles = frame_ankles.copy()
    frame_ankles['distance'] = np.sqrt(
        (frame_ankles['field_x'] - ball_x)**2 +
        (frame_ankles['field_y'] - ball_y)**2
    )

    closest = frame_ankles.loc[frame_ankles['distance'].idxmin()]

    # Cones
    frame_cones = cone_df[cone_df['frame_id'] == frame_id]

    return {
        'frame_id': frame_id,
        'ball': {
            'x': ball_row['center_x'],
            'y': ball_row['center_y'],
            'field_x': ball_x,
            'field_y': ball_y,
        },
        'ankle': {
            'x': closest['x'],
            'y': closest['y'],
            'field_x': closest['field_x'],
            'field_y': closest['field_y'],
            'name': closest['keypoint_name'],
            'distance': closest['distance'],
        },
        'cones': frame_cones[['object_id', 'center_x', 'center_y',
                              'field_center_x', 'field_center_y']].to_dict('records'),
    }
