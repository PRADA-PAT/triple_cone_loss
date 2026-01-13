#!/usr/bin/env python3
"""
Video Annotation for Triple Cone Drill Analysis.

Creates annotated videos using:
- STATIC cone positions from parquet detection (mean positions per player)
- DYNAMIC ball positions from parquet detection
- DYNAMIC pose skeleton from parquet detection
- LEFT SIDEBAR showing all object coordinates in real-time
- Turn detection and ball-behind tracking

Triple Cone Drill Pattern:
    CONE1 (HOME) -> CONE2 (turn) -> CONE1 (turn) -> CONE3 (turn) -> CONE1 (turn) -> repeat

Usage:
    python annotate_triple_cone.py "Alex mochar"
    python annotate_triple_cone.py --list
    python annotate_triple_cone.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import cv2
from dataclasses import dataclass
from collections import deque
from tqdm import tqdm

# Import turning zones from detection module
sys.path.insert(0, str(Path(__file__).parent.parent))
from detection.turning_zones import (
    TripleConeZoneSet, TripleConeZoneConfig, create_triple_cone_zones,
    draw_triple_cone_zones, CONE1_ZONE_COLOR, CONE2_ZONE_COLOR, CONE3_ZONE_COLOR,
    ZONE_HIGHLIGHT_COLOR
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TripleConeAnnotationConfig:
    """Configuration for Triple Cone annotation styles."""
    # Sidebar settings
    SIDEBAR_WIDTH: int = 300
    SIDEBAR_BG_COLOR: Tuple[int, int, int] = (25, 25, 25)  # Dark gray
    SIDEBAR_HEADER_COLOR: Tuple[int, int, int] = (80, 80, 80)  # Lighter gray
    SIDEBAR_LINE_HEIGHT: int = 24
    SIDEBAR_FONT_SCALE: float = 0.55
    SIDEBAR_PADDING: int = 12

    # Colors (BGR format for OpenCV)
    BALL_COLOR: Tuple[int, int, int] = (0, 255, 0)           # Green
    CONE1_COLOR: Tuple[int, int, int] = (200, 200, 0)        # Teal (HOME)
    CONE2_COLOR: Tuple[int, int, int] = (200, 100, 200)      # Purple (CENTER)
    CONE3_COLOR: Tuple[int, int, int] = (100, 200, 200)      # Orange (RIGHT)
    POSE_KEYPOINT_COLOR: Tuple[int, int, int] = (255, 0, 255)
    POSE_SKELETON_COLOR: Tuple[int, int, int] = (255, 255, 0)  # Cyan
    TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)       # White
    TEXT_BG_COLOR: Tuple[int, int, int] = (0, 0, 0)          # Black

    # Turning zone colors (use same as detection module)
    CONE1_ZONE_COLOR: Tuple[int, int, int] = (200, 200, 0)     # Teal (HOME)
    CONE2_ZONE_COLOR: Tuple[int, int, int] = (200, 100, 200)   # Purple (CENTER)
    CONE3_ZONE_COLOR: Tuple[int, int, int] = (100, 200, 200)   # Orange (RIGHT)
    ZONE_HIGHLIGHT_COLOR: Tuple[int, int, int] = (0, 255, 255) # Yellow (ball inside)
    ZONE_ALPHA: float = 0.25                                    # Zone transparency

    # Sizes
    BBOX_THICKNESS: int = 2
    SKELETON_THICKNESS: int = 2
    KEYPOINT_RADIUS: int = 4
    # CONE_MARKER_SIZE removed - now using actual bbox dimensions from parquet
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 1

    # Confidence thresholds
    MIN_KEYPOINT_CONFIDENCE: float = 0.3
    MIN_BBOX_CONFIDENCE: float = 0.1

    # Momentum arrow settings
    DRAW_MOMENTUM_ARROW: bool = True
    MOMENTUM_THICKNESS: int = 8
    MOMENTUM_LOOKBACK_FRAMES: int = 10
    MOMENTUM_SCALE: float = 3.0
    MOMENTUM_MAX_LENGTH: int = 150
    MOMENTUM_MIN_LENGTH: int = 5

    # Momentum color gradient (light blue=slow, blue=medium, dark blue=fast)
    MOMENTUM_COLOR_LOW: Tuple[int, int, int] = (255, 255, 100)   # Light cyan (BGR)
    MOMENTUM_COLOR_MID: Tuple[int, int, int] = (255, 150, 0)     # Blue (BGR)
    MOMENTUM_COLOR_HIGH: Tuple[int, int, int] = (200, 50, 50)    # Dark blue (BGR)
    MOMENTUM_SPEED_LOW: float = 5.0
    MOMENTUM_SPEED_HIGH: float = 80.0

    # Ball momentum arrow settings
    DRAW_BALL_MOMENTUM_ARROW: bool = True
    BALL_MOMENTUM_THICKNESS: int = 6  # Slightly thinner than player arrow (8)
    BALL_MOMENTUM_SCALE: float = 2.5  # Different scale for ball movement
    BALL_MOMENTUM_MAX_LENGTH: int = 120
    BALL_MOMENTUM_MIN_LENGTH: int = 3

    # Ball momentum colors (orange palette - distinct from player's blue)
    BALL_MOMENTUM_COLOR_LOW: Tuple[int, int, int] = (150, 200, 255)   # Light orange (BGR)
    BALL_MOMENTUM_COLOR_MID: Tuple[int, int, int] = (0, 165, 255)     # Orange (BGR)
    BALL_MOMENTUM_COLOR_HIGH: Tuple[int, int, int] = (0, 100, 200)    # Dark orange (BGR)
    BALL_MOMENTUM_SPEED_LOW: float = 5.0
    BALL_MOMENTUM_SPEED_HIGH: float = 120.0  # Ball moves faster than player

    # Ball position relative to player settings
    # NOTE: Must match detection/ball_control_detector.py thresholds
    DRAW_BALL_POSITION: bool = True
    BALL_POSITION_FRONT_COLOR: Tuple[int, int, int] = (0, 255, 0)    # Green
    BALL_POSITION_BEHIND_COLOR: Tuple[int, int, int] = (0, 0, 255)   # Red
    BALL_POSITION_ALIGNED_COLOR: Tuple[int, int, int] = (0, 255, 255) # Yellow
    BALL_POSITION_NEUTRAL_COLOR: Tuple[int, int, int] = (180, 180, 180) # Gray
    BALL_HIP_LINE_THICKNESS: int = 2
    BALL_POSITION_THRESHOLD: float = 20.0  # Pixels for "aligned" detection (auto-scaled)
    MOVEMENT_THRESHOLD: float = 3.0  # Min movement to determine direction (auto-scaled)
    DIVIDER_LINE_HEIGHT: int = 100

    # Ball-behind duration counter
    BEHIND_COUNTER_PERSIST_SECONDS: float = 3.0
    BEHIND_COUNTER_FONT_SCALE: float = 1.2
    BEHIND_COUNTER_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Red when active
    BEHIND_COUNTER_PERSIST_COLOR: Tuple[int, int, int] = (0, 200, 255)  # Yellow persist
    BEHIND_COUNTER_POS_X: int = 50
    BEHIND_COUNTER_POS_Y: int = 100

    # Edge zone visualization (auto-scaled)
    EDGE_MARGIN: int = 50
    EDGE_ZONE_COLOR: Tuple[int, int, int] = (0, 0, 255)      # Red danger zone
    EDGE_ZONE_ALPHA: float = 0.15
    EDGE_COUNTER_FONT_SCALE: float = 1.2
    EDGE_COUNTER_COLOR: Tuple[int, int, int] = (0, 0, 255)
    EDGE_COUNTER_PERSIST_COLOR: Tuple[int, int, int] = (0, 200, 255)
    EDGE_COUNTER_PERSIST_SECONDS: float = 3.0
    EDGE_COUNTER_POS_X: int = 50
    EDGE_COUNTER_POS_Y: int = 150

    # Ball off-screen visualization settings (when ball is interpolated / not detected)
    DRAW_OFF_SCREEN_INDICATOR: bool = True
    OFF_SCREEN_TEXT: str = "BALL OFF-SCREEN"
    OFF_SCREEN_COLOR: Tuple[int, int, int] = (0, 0, 255)  # Red
    OFF_SCREEN_FONT_SCALE: float = 1.5
    OFF_SCREEN_POS_X: int = 50   # Position on video area
    OFF_SCREEN_POS_Y: int = 250  # Below intention counter

    # Return counter settings (shows "WAS GONE: Xf" after ball returns)
    RETURN_COUNTER_PERSIST_SECONDS: float = 3.0  # Show for 3 sec after ball returns
    RETURN_COUNTER_COLOR: Tuple[int, int, int] = (0, 200, 255)  # Yellow-orange
    RETURN_COUNTER_FONT_SCALE: float = 1.2
    RETURN_COUNTER_POS_X: int = 50
    RETURN_COUNTER_POS_Y: int = 300  # Below off-screen indicator

    # Unified edge+off-screen tracking (replaces separate edge/off-screen counters)
    UNIFIED_TRACKING_ENABLED: bool = True  # Use new unified state machine
    UNIFIED_COUNTER_PERSIST_SECONDS: float = 3.0  # Persist counter after ball returns to normal
    UNIFIED_COUNTER_FONT_SCALE: float = 1.2
    UNIFIED_COUNTER_POS_X: int = 50
    UNIFIED_COUNTER_POS_Y: int = 150  # Same position as old edge counter
    # Colors for different states
    UNIFIED_EDGE_COLOR: Tuple[int, int, int] = (0, 200, 255)       # Yellow - ball in edge zone
    UNIFIED_OFF_SCREEN_COLOR: Tuple[int, int, int] = (0, 0, 255)   # Red - ball off-screen via edge
    UNIFIED_PERSIST_COLOR: Tuple[int, int, int] = (0, 165, 255)    # Orange - post-return display
    # Mid-field disappearance (detection failure)
    SHOW_MID_FIELD_DISAPPEAR: bool = True
    MID_FIELD_DISAPPEAR_TEXT: str = "DETECTION LOST"
    MID_FIELD_DISAPPEAR_COLOR: Tuple[int, int, int] = (128, 128, 128)  # Gray
    MID_FIELD_DISAPPEAR_POS_X: int = 50
    MID_FIELD_DISAPPEAR_POS_Y: int = 250

    # Torso facing direction visualization (intention arrow above head)
    DRAW_TORSO_FACING: bool = True
    NOSE_HIP_FACING_THRESHOLD: float = 15.0  # min nose-hip X diff for facing direction (auto-scaled)
    TORSO_FACING_COLOR_RIGHT: Tuple[int, int, int] = (0, 255, 255)   # Yellow (BGR) - facing right
    TORSO_FACING_COLOR_LEFT: Tuple[int, int, int] = (255, 0, 255)   # Magenta (BGR) - facing left
    # Intention arrow settings (horizontal arrow above head)
    INTENTION_ARROW_LENGTH: int = 60          # Arrow length in pixels
    INTENTION_ARROW_THICKNESS: int = 4        # Arrow line thickness
    INTENTION_ARROW_OFFSET_Y: int = 50        # Pixels above nose/head
    INTENTION_ARROW_TIP_LENGTH: float = 0.35  # Arrow tip proportion

    # Intention-based ball position settings (ball vs torso facing)
    DRAW_BALL_POSITION_INTENTION: bool = True
    INTENTION_FRONT_COLOR: Tuple[int, int, int] = (255, 255, 0)    # Cyan (BGR)
    INTENTION_BEHIND_COLOR: Tuple[int, int, int] = (255, 0, 255)   # Magenta (BGR)
    INTENTION_ALIGNED_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
    INTENTION_LINE_DASH_LENGTH: int = 8       # Dash length for dashed line
    INTENTION_LINE_GAP_LENGTH: int = 6        # Gap length for dashed line
    INTENTION_LABEL_OFFSET_Y: int = 45        # Pixels below ball for label
    INTENTION_BEHIND_COUNTER_POS_Y: int = 200 # Below momentum counter

    # Debug axes (always on per user request)
    DRAW_DEBUG_AXES: bool = True
    DEBUG_AXES_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    DEBUG_AXES_THICKNESS: int = 1

    # Turn event log
    DRAW_EVENT_LOG: bool = True
    EVENT_LOG_MAX_EVENTS: int = 8

    # Resolution scaling (set automatically based on video width)
    RESOLUTION_SCALE: float = 1.0  # Linear scale for positions (1.0 = 2816px reference)
    FONT_SCALE_FACTOR: float = 1.0  # Sqrt scale for fonts (gentler, stays readable)


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

# Keypoints to track in sidebar
TRACKED_KEYPOINTS = [
    ('left_ankle', 'L_ANKLE'),
    ('right_ankle', 'R_ANKLE'),
    ('left_big_toe', 'L_TOE'),
    ('right_big_toe', 'R_TOE'),
    ('hip', 'HIP'),
]


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class BallPositionResult:
    """Result of ball position analysis relative to player's MOMENTUM (movement direction)."""
    position: str  # "FRONT", "BEHIND", "ALIGNED", or "UNKNOWN"
    movement_direction: Optional[str]  # "LEFT", "RIGHT", or None
    ball_hip_delta_x: float  # ball_x - hip_x
    color: Tuple[int, int, int]


@dataclass
class IntentionPositionResult:
    """Result of ball position analysis relative to player's INTENTION (torso facing)."""
    position: str  # "I-FRONT", "I-BEHIND", "I-ALIGNED", or "UNKNOWN"
    facing_direction: Optional[str]  # "LEFT", "RIGHT", or None (from nose-hip)
    ball_hip_delta_x: float  # ball_x - hip_x
    color: Tuple[int, int, int]


@dataclass
class EdgeZoneStatus:
    """Status of ball relative to screen edges."""
    in_edge_zone: bool
    edge_side: str  # "LEFT", "RIGHT", or "NONE"
    distance_to_edge: float


from enum import Enum

class BallTrackingState(Enum):
    """State machine states for unified edge+off-screen tracking."""
    NORMAL = "NORMAL"                      # Ball visible, not in edge zone
    EDGE_LEFT = "EDGE_LEFT"                # Ball in left edge zone
    EDGE_RIGHT = "EDGE_RIGHT"              # Ball in right edge zone
    OFF_SCREEN_LEFT = "OFF_SCREEN_LEFT"    # Ball left via left edge
    OFF_SCREEN_RIGHT = "OFF_SCREEN_RIGHT"  # Ball left via right edge
    DISAPPEARED_MID = "DISAPPEARED_MID"    # Ball disappeared without edge (detection failure)


@dataclass
class TurnEvent:
    """A detected turn at a cone."""
    frame_id: int
    timestamp: float
    zone: str  # "CONE1", "CONE2", "CONE3"
    from_direction: str
    to_direction: str


# ============================================================================
# Cone Loading from Parquet
# ============================================================================

def read_parquet_safe(parquet_path: Path) -> pd.DataFrame:
    """
    Read parquet file with fallback for uint32 dictionary encoding issue.

    Some parquet files use uint32 dictionary indices which pandas/pyarrow
    doesn't support directly. This function handles that case by using
    pyarrow to decode dictionaries first.
    """
    try:
        # Try normal pandas read first
        return pd.read_parquet(parquet_path)
    except Exception as e:
        if "unsigned dictionary indices" in str(e) or "uint32" in str(e):
            # Fallback: use pyarrow and decode dictionaries manually
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_path)

            # Decode dictionary columns to regular columns
            new_columns = []
            for i, field in enumerate(table.schema):
                col = table.column(i)
                if pa.types.is_dictionary(field.type):
                    # For ChunkedArray, decode each chunk and combine
                    decoded_chunks = [chunk.dictionary_decode() for chunk in col.chunks]
                    new_col = pa.chunked_array(decoded_chunks)
                    new_columns.append(new_col)
                else:
                    new_columns.append(col)

            # Rebuild table with decoded columns
            new_table = pa.table(
                {field.name: new_columns[i] for i, field in enumerate(table.schema)}
            )
            return new_table.to_pandas()
        else:
            raise


@dataclass
class ConeData:
    """Cone detection data with position and bounding box dimensions."""
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)


def load_cone_positions_from_parquet(parquet_path: Path) -> Tuple[ConeData, ConeData, ConeData]:
    """
    Load cone positions and bounding box dimensions from parquet file.

    Returns (cone1, cone2, cone3) sorted by X (left to right).
    CONE1 = leftmost (HOME), CONE2 = center, CONE3 = rightmost.
    Each cone includes center position and mean bbox dimensions.
    """
    cone_df = read_parquet_safe(parquet_path)

    # Filter out NaN object_ids
    cone_df = cone_df[cone_df['object_id'].notna()]

    # Group by object_id and get mean position + dimensions
    cones = []
    for obj_id in sorted(cone_df['object_id'].unique()):
        obj_data = cone_df[cone_df['object_id'] == obj_id]
        mean_x = obj_data['center_x'].mean()
        mean_y = obj_data['center_y'].mean()
        mean_width = obj_data['width'].mean() if 'width' in obj_data.columns else 15.0
        mean_height = obj_data['height'].mean() if 'height' in obj_data.columns else 15.0

        # Skip positions with NaN values
        if pd.notna(mean_x) and pd.notna(mean_y):
            cones.append(ConeData(
                center_x=mean_x,
                center_y=mean_y,
                width=mean_width if pd.notna(mean_width) else 15.0,
                height=mean_height if pd.notna(mean_height) else 15.0,
            ))

    # Sort by X position (left to right = CONE1, CONE2, CONE3)
    cones.sort(key=lambda c: c.center_x)

    if len(cones) < 3:
        raise ValueError(f"Expected 3 cones, found {len(cones)}")

    return (cones[0], cones[1], cones[2])


# ============================================================================
# Data Loaders
# ============================================================================

def load_ball_data(parquet_path: Path) -> pd.DataFrame:
    """Load ball detection data including interpolated flag for off-screen detection."""
    df = read_parquet_safe(parquet_path)
    cols = ['frame_id', 'x1', 'y1', 'x2', 'y2', 'confidence']
    if 'interpolated' in df.columns:
        cols.append('interpolated')
    return df[cols].copy()


def load_pose_data(parquet_path: Path) -> pd.DataFrame:
    """Load pose keypoint data."""
    df = read_parquet_safe(parquet_path)
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
# Turn Tracker
# ============================================================================

class TripleConeTurnTracker:
    """Track turns at each cone based on movement direction change."""

    def __init__(self):
        self.turn_events: List[TurnEvent] = []
        self.prev_direction: Optional[str] = None
        self.prev_zone: Optional[str] = None
        self.in_zone_frames: int = 0  # Debounce counter

    def update(
        self,
        frame_id: int,
        timestamp: float,
        current_zone: Optional[str],
        movement_direction: Optional[str]
    ) -> Optional[TurnEvent]:
        """
        Detect turn when:
        1. Ball is in a turning zone
        2. Movement direction reverses
        """
        turn_event = None

        # Track time in zone for debouncing
        if current_zone:
            self.in_zone_frames += 1
        else:
            self.in_zone_frames = 0

        # Detect direction reversal while in zone (with debounce)
        if current_zone and movement_direction and self.prev_direction:
            if movement_direction != self.prev_direction and self.in_zone_frames >= 3:
                turn_event = TurnEvent(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    zone=current_zone,
                    from_direction=self.prev_direction,
                    to_direction=movement_direction,
                )
                self.turn_events.append(turn_event)
                # Reset in_zone_frames to avoid duplicate events
                self.in_zone_frames = 0

        # Update state
        if movement_direction:
            self.prev_direction = movement_direction
        self.prev_zone = current_zone

        return turn_event

    def get_recent_events(self, max_events: int = 8) -> List[TurnEvent]:
        """Get the most recent turn events."""
        return self.turn_events[-max_events:]


# ============================================================================
# Sidebar Drawing Functions
# ============================================================================

def draw_sidebar_section_header(frame: np.ndarray, y: int, text: str,
                                color: Tuple[int, int, int], config: TripleConeAnnotationConfig) -> int:
    """Draw a section header in the sidebar. Returns next y position."""
    cv2.line(frame, (config.SIDEBAR_PADDING, y),
             (config.SIDEBAR_WIDTH - config.SIDEBAR_PADDING, y),
             config.SIDEBAR_HEADER_COLOR, 1)

    y += 18
    cv2.putText(frame, text, (config.SIDEBAR_PADDING, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                color, 1, cv2.LINE_AA)

    return y + 8


def draw_sidebar_row(frame: np.ndarray, y: int, label: str,
                     x_coord: Optional[float], y_coord: Optional[float],
                     color: Tuple[int, int, int], config: TripleConeAnnotationConfig) -> int:
    """Draw a coordinate row in the sidebar. Returns next y position."""
    # Check for both None and NaN values
    x_valid = x_coord is not None and pd.notna(x_coord)
    y_valid = y_coord is not None and pd.notna(y_coord)
    if x_valid and y_valid:
        coord_str = f"({int(x_coord):4d}, {int(y_coord):4d})"
    else:
        coord_str = "   --"

    label_x = config.SIDEBAR_PADDING + max(2, int(5 * config.FONT_SCALE_FACTOR))
    cv2.putText(frame, f"{label}:", (label_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.05 * config.FONT_SCALE_FACTOR,
                color, 1, cv2.LINE_AA)

    # Coordinate column offset proportional to sidebar width (130/300 = 43% from right edge)
    coord_offset = int(130 * config.RESOLUTION_SCALE)  # Keep linear for position
    coord_x = config.SIDEBAR_WIDTH - coord_offset
    cv2.putText(frame, coord_str, (coord_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.05 * config.FONT_SCALE_FACTOR,
                (200, 200, 200), 1, cv2.LINE_AA)

    return y + config.SIDEBAR_LINE_HEIGHT


def draw_sidebar(frame: np.ndarray, frame_id: int,
                 cone1: ConeData,
                 cone2: ConeData,
                 cone3: ConeData,
                 ball_center: Optional[Tuple[float, float]],
                 pose_keypoints: Dict[str, Tuple[float, float, float]],
                 config: TripleConeAnnotationConfig,
                 active_zone: Optional[str] = None,
                 ball_position_result: Optional[BallPositionResult] = None,
                 intention_result: Optional[IntentionPositionResult] = None,
                 turn_events: Optional[List[TurnEvent]] = None) -> None:
    """Draw the sidebar with all object coordinates for Triple Cone drill."""
    # Fill sidebar background
    frame[:, :config.SIDEBAR_WIDTH] = config.SIDEBAR_BG_COLOR

    y = 25

    # Frame number header
    cv2.putText(frame, f"FRAME: {frame_id}", (config.SIDEBAR_PADDING, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y += 35

    # ===== CONES SECTION (3 cones) =====
    y = draw_sidebar_section_header(frame, y, "CONES (static)",
                                    config.CONE1_COLOR, config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    y = draw_sidebar_row(frame, y, "CONE1", cone1.center_x, cone1.center_y,
                         config.CONE1_COLOR, config)
    y = draw_sidebar_row(frame, y, "CONE2", cone2.center_x, cone2.center_y,
                         config.CONE2_COLOR, config)
    y = draw_sidebar_row(frame, y, "CONE3", cone3.center_x, cone3.center_y,
                         config.CONE3_COLOR, config)

    y += max(4, int(10 * config.FONT_SCALE_FACTOR))

    # ===== BALL SECTION =====
    y = draw_sidebar_section_header(frame, y, "BALL (dynamic)",
                                    config.BALL_COLOR, config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    ball_x = ball_center[0] if ball_center else None
    ball_y = ball_center[1] if ball_center else None
    y = draw_sidebar_row(frame, y, "BALL", ball_x, ball_y,
                         config.BALL_COLOR, config)

    y += max(4, int(10 * config.FONT_SCALE_FACTOR))

    # ===== POSE SECTION =====
    y = draw_sidebar_section_header(frame, y, "POSE (dynamic)",
                                    config.POSE_SKELETON_COLOR, config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    for kp_name, display_name in TRACKED_KEYPOINTS:
        if kp_name in pose_keypoints:
            x, ycoord, conf = pose_keypoints[kp_name]
            if conf >= config.MIN_KEYPOINT_CONFIDENCE:
                body_part = KEYPOINT_BODY_PART.get(kp_name, 'torso')
                color = KEYPOINT_COLORS.get(body_part, config.POSE_KEYPOINT_COLOR)
                y = draw_sidebar_row(frame, y, display_name, x, ycoord, color, config)
            else:
                y = draw_sidebar_row(frame, y, display_name, None, None,
                                     (100, 100, 100), config)
        else:
            y = draw_sidebar_row(frame, y, display_name, None, None,
                                 (100, 100, 100), config)

    y += max(6, int(15 * config.FONT_SCALE_FACTOR))

    # ===== TURNING ZONE SECTION (3 zones) =====
    y = draw_sidebar_section_header(frame, y, "TURNING ZONE",
                                    (150, 150, 150), config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    if active_zone == "CONE1":
        zone_text = "Ball in: CONE1 (HOME)"
        zone_color = config.CONE1_ZONE_COLOR
    elif active_zone == "CONE2":
        zone_text = "Ball in: CONE2"
        zone_color = config.CONE2_ZONE_COLOR
    elif active_zone == "CONE3":
        zone_text = "Ball in: CONE3"
        zone_color = config.CONE3_ZONE_COLOR
    else:
        zone_text = "Ball in: --"
        zone_color = (100, 100, 100)

    cv2.putText(frame, zone_text,
                (config.SIDEBAR_PADDING + max(2, int(5 * config.FONT_SCALE_FACTOR)), y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                zone_color, 1, cv2.LINE_AA)

    y += config.SIDEBAR_LINE_HEIGHT + max(4, int(10 * config.FONT_SCALE_FACTOR))

    # ===== BALL POSITION SECTION =====
    y = draw_sidebar_section_header(frame, y, "BALL POSITION",
                                    (150, 150, 150), config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    text_indent = config.SIDEBAR_PADDING + max(2, int(5 * config.FONT_SCALE_FACTOR))

    # Momentum-based position (existing)
    if ball_position_result is not None:
        momentum_text = f"Momentum: {ball_position_result.position}"
        cv2.putText(frame, momentum_text,
                    (text_indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    ball_position_result.color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Momentum: --",
                    (text_indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    (100, 100, 100), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT

    # Intention-based position (new)
    if intention_result is not None:
        intention_text = f"Intention: {intention_result.position}"
        cv2.putText(frame, intention_text,
                    (text_indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    intention_result.color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Intention: --",
                    (text_indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    (100, 100, 100), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT

    # Delta X (shared)
    delta_x = ball_position_result.ball_hip_delta_x if ball_position_result else 0.0
    delta_text = f"Delta X: {delta_x:+.0f}px"
    cv2.putText(frame, delta_text,
                (text_indent, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1 * config.FONT_SCALE_FACTOR,
                (180, 180, 180), 1, cv2.LINE_AA)

    y += config.SIDEBAR_LINE_HEIGHT + max(4, int(10 * config.FONT_SCALE_FACTOR))

    # ===== TURN EVENTS SECTION =====
    if config.DRAW_EVENT_LOG and turn_events:
        y = draw_sidebar_section_header(frame, y, "TURN EVENTS",
                                        (150, 150, 150), config)
        y += max(2, int(5 * config.FONT_SCALE_FACTOR))

        recent = turn_events[-config.EVENT_LOG_MAX_EVENTS:]
        for event in recent:
            # Format: "12.5s CONE2: L->R"
            event_text = f"{event.timestamp:5.1f}s {event.zone}: {event.from_direction[0]}->{event.to_direction[0]}"

            # Color by zone
            if event.zone == "CONE1":
                event_color = config.CONE1_ZONE_COLOR
            elif event.zone == "CONE2":
                event_color = config.CONE2_ZONE_COLOR
            elif event.zone == "CONE3":
                event_color = config.CONE3_ZONE_COLOR
            else:
                event_color = (150, 150, 150)

            cv2.putText(frame, event_text,
                        (config.SIDEBAR_PADDING + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1,
                        event_color, 1, cv2.LINE_AA)
            y += config.SIDEBAR_LINE_HEIGHT - 4

    # ===== ARROW LEGEND SECTION =====
    y += max(6, int(15 * config.FONT_SCALE_FACTOR))
    y = draw_sidebar_section_header(frame, y, "ARROW LEGEND",
                                    (150, 150, 150), config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    # Scale legend layout values (with minimum values for readability)
    scale = config.FONT_SCALE_FACTOR  # Use font scale for legend
    legend_font_scale = max(0.35, 0.45 * scale)  # Minimum 0.35 for readability
    legend_height = max(8, int(12 * scale))
    swatch_width = max(12, int(18 * scale))
    text_offset = max(16, int(22 * scale))
    col2_offset = max(45, int(60 * scale))
    col3_offset = max(90, int(115 * scale))

    # Momentum arrow legend (at hip)
    cv2.putText(frame, "MOMENTUM (at hip):",
                (config.SIDEBAR_PADDING + max(2, int(5 * scale)), y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1 * scale,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT - max(1, int(2 * scale))

    # Draw small color samples for momentum
    sample_x = config.SIDEBAR_PADDING + max(4, int(10 * scale))
    cv2.rectangle(frame, (sample_x, y - legend_height), (sample_x + swatch_width, y), config.MOMENTUM_COLOR_LOW, -1)
    cv2.putText(frame, "Slow", (sample_x + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.MOMENTUM_COLOR_LOW, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + col2_offset, y - legend_height), (sample_x + col2_offset + swatch_width, y), config.MOMENTUM_COLOR_MID, -1)
    cv2.putText(frame, "Med", (sample_x + col2_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.MOMENTUM_COLOR_MID, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + col3_offset, y - legend_height), (sample_x + col3_offset + swatch_width, y), config.MOMENTUM_COLOR_HIGH, -1)
    cv2.putText(frame, "Fast", (sample_x + col3_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.MOMENTUM_COLOR_HIGH, 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT + max(2, int(5 * scale))

    # Intention arrow legend (above head)
    cv2.putText(frame, "INTENTION (above head):",
                (config.SIDEBAR_PADDING + max(2, int(5 * scale)), y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1 * scale,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT - max(1, int(2 * scale))

    # Draw small color samples for intention (2 columns only)
    intent_col2_offset = max(50, int(70 * scale))
    cv2.rectangle(frame, (sample_x, y - legend_height), (sample_x + swatch_width, y), config.TORSO_FACING_COLOR_RIGHT, -1)
    cv2.putText(frame, "Right", (sample_x + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.TORSO_FACING_COLOR_RIGHT, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + intent_col2_offset, y - legend_height), (sample_x + intent_col2_offset + swatch_width, y), config.TORSO_FACING_COLOR_LEFT, -1)
    cv2.putText(frame, "Left", (sample_x + intent_col2_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.TORSO_FACING_COLOR_LEFT, 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT + max(2, int(5 * scale))

    # Ball momentum arrow legend (at ball)
    cv2.putText(frame, "BALL MOMENTUM:",
                (config.SIDEBAR_PADDING + max(2, int(5 * scale)), y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1 * scale,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT - max(1, int(2 * scale))

    # Draw small color samples for ball momentum (orange gradient)
    cv2.rectangle(frame, (sample_x, y - legend_height), (sample_x + swatch_width, y), config.BALL_MOMENTUM_COLOR_LOW, -1)
    cv2.putText(frame, "Slow", (sample_x + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.BALL_MOMENTUM_COLOR_LOW, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + col2_offset, y - legend_height), (sample_x + col2_offset + swatch_width, y), config.BALL_MOMENTUM_COLOR_MID, -1)
    cv2.putText(frame, "Med", (sample_x + col2_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.BALL_MOMENTUM_COLOR_MID, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + col3_offset, y - legend_height), (sample_x + col3_offset + swatch_width, y), config.BALL_MOMENTUM_COLOR_HIGH, -1)
    cv2.putText(frame, "Fast", (sample_x + col3_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.BALL_MOMENTUM_COLOR_HIGH, 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT

    # Vertical separator line
    cv2.line(frame, (config.SIDEBAR_WIDTH - 1, 0),
             (config.SIDEBAR_WIDTH - 1, frame.shape[0]),
             (60, 60, 60), 2)


# ============================================================================
# Video Drawing Functions
# ============================================================================

def draw_bbox(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
              color: Tuple[int, int, int], label: str, config: TripleConeAnnotationConfig,
              x_offset: int = 0) -> None:
    """Draw a bounding box with label."""
    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [x1, y1, x2, y2]):
        return
    x1, y1, x2, y2 = int(x1) + x_offset, int(y1), int(x2) + x_offset, int(y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
    )
    cv2.rectangle(frame, (x1, y1 - text_height - 8),
                  (x1 + text_width + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.TEXT_BG_COLOR, config.FONT_THICKNESS)


def draw_triple_cone_markers(
    frame: np.ndarray,
    cone1: ConeData,
    cone2: ConeData,
    cone3: ConeData,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw 3 cone markers with actual bounding boxes from detection."""
    cones = [
        ("CONE1 (HOME)", cone1, config.CONE1_COLOR),
        ("CONE2", cone2, config.CONE2_COLOR),
        ("CONE3", cone3, config.CONE3_COLOR),
    ]

    for label, cone, color in cones:
        # Draw actual bounding box from detection
        half_w = cone.width / 2
        half_h = cone.height / 2
        x1 = int(cone.center_x - half_w) + x_offset
        y1 = int(cone.center_y - half_h)
        x2 = int(cone.center_x + half_w) + x_offset
        y2 = int(cone.center_y + half_h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label above
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_x = x1
        label_y = y1 - 8

        # Background for label
        cv2.rectangle(frame,
                      (label_x - 2, label_y - text_height - 2),
                      (label_x + text_width + 2, label_y + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, Tuple[float, float, float]],
                  config: TripleConeAnnotationConfig, x_offset: int = 0) -> None:
    """Draw pose skeleton with keypoints."""
    # Draw connections first
    for kp1_name, kp2_name in SKELETON_CONNECTIONS:
        if kp1_name in keypoints and kp2_name in keypoints:
            x1, y1, conf1 = keypoints[kp1_name]
            x2, y2, conf2 = keypoints[kp2_name]

            if conf1 >= config.MIN_KEYPOINT_CONFIDENCE and conf2 >= config.MIN_KEYPOINT_CONFIDENCE:
                pt1 = (int(x1) + x_offset, int(y1))
                pt2 = (int(x2) + x_offset, int(y2))
                cv2.line(frame, pt1, pt2, config.POSE_SKELETON_COLOR, config.SKELETON_THICKNESS)

    # Draw keypoints
    for kp_name, (x, y, conf) in keypoints.items():
        if conf >= config.MIN_KEYPOINT_CONFIDENCE:
            pt = (int(x) + x_offset, int(y))
            body_part = KEYPOINT_BODY_PART.get(kp_name, 'torso')
            color = KEYPOINT_COLORS.get(body_part, config.POSE_KEYPOINT_COLOR)
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, color, -1)
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, (0, 0, 0), 1)


def get_momentum_color(magnitude: float, config: TripleConeAnnotationConfig) -> Tuple[int, int, int]:
    """Calculate momentum arrow color based on speed magnitude."""
    if magnitude <= config.MOMENTUM_SPEED_LOW:
        t = 0.0
    elif magnitude >= config.MOMENTUM_SPEED_HIGH:
        t = 1.0
    else:
        t = (magnitude - config.MOMENTUM_SPEED_LOW) / (config.MOMENTUM_SPEED_HIGH - config.MOMENTUM_SPEED_LOW)

    if t <= 0.5:
        ratio = t * 2
        b = int(config.MOMENTUM_COLOR_LOW[0] + ratio * (config.MOMENTUM_COLOR_MID[0] - config.MOMENTUM_COLOR_LOW[0]))
        g = int(config.MOMENTUM_COLOR_LOW[1] + ratio * (config.MOMENTUM_COLOR_MID[1] - config.MOMENTUM_COLOR_LOW[1]))
        r = int(config.MOMENTUM_COLOR_LOW[2] + ratio * (config.MOMENTUM_COLOR_MID[2] - config.MOMENTUM_COLOR_LOW[2]))
    else:
        ratio = (t - 0.5) * 2
        b = int(config.MOMENTUM_COLOR_MID[0] + ratio * (config.MOMENTUM_COLOR_HIGH[0] - config.MOMENTUM_COLOR_MID[0]))
        g = int(config.MOMENTUM_COLOR_MID[1] + ratio * (config.MOMENTUM_COLOR_HIGH[1] - config.MOMENTUM_COLOR_MID[1]))
        r = int(config.MOMENTUM_COLOR_MID[2] + ratio * (config.MOMENTUM_COLOR_HIGH[2] - config.MOMENTUM_COLOR_MID[2]))

    return (b, g, r)


def draw_momentum_arrow(
    frame: np.ndarray,
    current_hip: Tuple[float, float],
    previous_hip: Tuple[float, float],
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw thick horizontal momentum arrow with color gradient."""
    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [current_hip[0], current_hip[1], previous_hip[0], previous_hip[1]]):
        return

    dx = current_hip[0] - previous_hip[0]
    horizontal_magnitude = abs(dx)

    if horizontal_magnitude < config.MOMENTUM_MIN_LENGTH:
        return

    color = get_momentum_color(horizontal_magnitude, config)
    scaled_length = min(horizontal_magnitude * config.MOMENTUM_SCALE, config.MOMENTUM_MAX_LENGTH)

    direction = 1 if dx > 0 else -1
    arrow_dx = direction * scaled_length

    start_x = int(current_hip[0]) + x_offset
    start_y = int(current_hip[1])
    end_x = int(start_x + arrow_dx)
    end_y = start_y

    cv2.arrowedLine(
        frame,
        (start_x, start_y),
        (end_x, end_y),
        color,
        config.MOMENTUM_THICKNESS,
        tipLength=0.25,
        line_type=cv2.LINE_AA
    )


def get_ball_momentum_color(magnitude: float, config: TripleConeAnnotationConfig) -> Tuple[int, int, int]:
    """Calculate ball momentum arrow color based on speed magnitude (orange gradient)."""
    if magnitude <= config.BALL_MOMENTUM_SPEED_LOW:
        t = 0.0
    elif magnitude >= config.BALL_MOMENTUM_SPEED_HIGH:
        t = 1.0
    else:
        t = (magnitude - config.BALL_MOMENTUM_SPEED_LOW) / (config.BALL_MOMENTUM_SPEED_HIGH - config.BALL_MOMENTUM_SPEED_LOW)

    if t <= 0.5:
        ratio = t * 2
        b = int(config.BALL_MOMENTUM_COLOR_LOW[0] + ratio * (config.BALL_MOMENTUM_COLOR_MID[0] - config.BALL_MOMENTUM_COLOR_LOW[0]))
        g = int(config.BALL_MOMENTUM_COLOR_LOW[1] + ratio * (config.BALL_MOMENTUM_COLOR_MID[1] - config.BALL_MOMENTUM_COLOR_LOW[1]))
        r = int(config.BALL_MOMENTUM_COLOR_LOW[2] + ratio * (config.BALL_MOMENTUM_COLOR_MID[2] - config.BALL_MOMENTUM_COLOR_LOW[2]))
    else:
        ratio = (t - 0.5) * 2
        b = int(config.BALL_MOMENTUM_COLOR_MID[0] + ratio * (config.BALL_MOMENTUM_COLOR_HIGH[0] - config.BALL_MOMENTUM_COLOR_MID[0]))
        g = int(config.BALL_MOMENTUM_COLOR_MID[1] + ratio * (config.BALL_MOMENTUM_COLOR_HIGH[1] - config.BALL_MOMENTUM_COLOR_MID[1]))
        r = int(config.BALL_MOMENTUM_COLOR_MID[2] + ratio * (config.BALL_MOMENTUM_COLOR_HIGH[2] - config.BALL_MOMENTUM_COLOR_MID[2]))

    return (b, g, r)


def draw_ball_momentum_arrow(
    frame: np.ndarray,
    current_ball: Tuple[float, float],
    previous_ball: Tuple[float, float],
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw momentum arrow for ball showing movement direction and speed (orange gradient)."""
    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [current_ball[0], current_ball[1], previous_ball[0], previous_ball[1]]):
        return

    dx = current_ball[0] - previous_ball[0]
    dy = current_ball[1] - previous_ball[1]
    magnitude = np.sqrt(dx * dx + dy * dy)

    if magnitude < config.BALL_MOMENTUM_MIN_LENGTH:
        return

    color = get_ball_momentum_color(magnitude, config)
    scaled_length = min(magnitude * config.BALL_MOMENTUM_SCALE, config.BALL_MOMENTUM_MAX_LENGTH)

    # Normalize direction
    norm_dx = dx / magnitude
    norm_dy = dy / magnitude

    start_x = int(current_ball[0]) + x_offset
    start_y = int(current_ball[1])
    end_x = int(start_x + norm_dx * scaled_length)
    end_y = int(start_y + norm_dy * scaled_length)

    cv2.arrowedLine(
        frame,
        (start_x, start_y),
        (end_x, end_y),
        color,
        config.BALL_MOMENTUM_THICKNESS,
        tipLength=0.25,
        line_type=cv2.LINE_AA
    )


# ============================================================================
# Ball Position Detection (FRONT/BEHIND)
# ============================================================================

def determine_ball_position_relative_to_player(
    ball_center: Optional[Tuple[float, float]],
    current_hip: Optional[Tuple[float, float]],
    previous_hip: Optional[Tuple[float, float]],
    config: TripleConeAnnotationConfig
) -> BallPositionResult:
    """
    Determine if the ball is in front of or behind the player.

    For Triple Cone drill (horizontal movement):
    - Player moving LEFT (toward CONE1/HOME): FRONT = ball to left of hip
    - Player moving RIGHT (toward CONE3): FRONT = ball to right of hip
    """
    if ball_center is None or current_hip is None:
        return BallPositionResult(
            position="UNKNOWN",
            movement_direction=None,
            ball_hip_delta_x=0.0,
            color=config.BALL_POSITION_NEUTRAL_COLOR
        )

    ball_x = ball_center[0]
    hip_x = current_hip[0]
    delta_x = ball_x - hip_x

    # Determine player movement direction from hip history
    movement_direction: Optional[str] = None
    if previous_hip is not None:
        dx_movement = current_hip[0] - previous_hip[0]
        if dx_movement > config.MOVEMENT_THRESHOLD:
            movement_direction = "RIGHT"  # Moving toward CONE3
        elif dx_movement < -config.MOVEMENT_THRESHOLD:
            movement_direction = "LEFT"   # Moving toward CONE1

    # Check if ball is aligned with player
    if abs(delta_x) < config.BALL_POSITION_THRESHOLD:
        return BallPositionResult(
            position="ALIGNED",
            movement_direction=movement_direction,
            ball_hip_delta_x=delta_x,
            color=config.BALL_POSITION_ALIGNED_COLOR
        )

    # If player is stationary, just report left/right position
    if movement_direction is None:
        return BallPositionResult(
            position="LEFT" if delta_x < 0 else "RIGHT",
            movement_direction=None,
            ball_hip_delta_x=delta_x,
            color=config.BALL_POSITION_NEUTRAL_COLOR
        )

    # Determine front/behind based on movement direction
    if movement_direction == "LEFT":
        # Moving left: FRONT = ball to left (negative delta)
        if delta_x < 0:
            position = "FRONT"
            color = config.BALL_POSITION_FRONT_COLOR
        else:
            position = "BEHIND"
            color = config.BALL_POSITION_BEHIND_COLOR
    else:  # RIGHT
        # Moving right: FRONT = ball to right (positive delta)
        if delta_x > 0:
            position = "FRONT"
            color = config.BALL_POSITION_FRONT_COLOR
        else:
            position = "BEHIND"
            color = config.BALL_POSITION_BEHIND_COLOR

    return BallPositionResult(
        position=position,
        movement_direction=movement_direction,
        ball_hip_delta_x=delta_x,
        color=color
    )


def determine_torso_facing(
    persons: dict,
    config: TripleConeAnnotationConfig
) -> Optional[str]:
    """
    Determine if player torso is facing LEFT or RIGHT.

    Uses nose position relative to hip:
    - Facing RIGHT: nose.x > hip.x (head is to the right of body center)
    - Facing LEFT: nose.x < hip.x (head is to the left of body center)

    This captures turn intention since head turns before body.

    Returns 'RIGHT', 'LEFT', or None if data unreliable.
    """
    if not persons:
        return None

    # Get first person's keypoints (tuple format: x, y, confidence)
    first_person_id = min(persons.keys())
    keypoints = persons[first_person_id]

    nose = keypoints.get('nose')
    hip = keypoints.get('hip')

    # Fallback: if central hip not available, average left/right hip
    if not hip or hip[2] < config.MIN_KEYPOINT_CONFIDENCE:
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')
        if left_hip and right_hip:
            if left_hip[2] >= config.MIN_KEYPOINT_CONFIDENCE and right_hip[2] >= config.MIN_KEYPOINT_CONFIDENCE:
                hip = ((left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2,
                       (left_hip[2] + right_hip[2]) / 2)

    if not nose or not hip:
        return None

    # Check confidence
    if nose[2] < config.MIN_KEYPOINT_CONFIDENCE:
        return None
    if isinstance(hip, tuple) and len(hip) >= 3:
        if hip[2] < config.MIN_KEYPOINT_CONFIDENCE:
            return None

    # Compare X positions: nose relative to hip
    diff = nose[0] - hip[0]

    if diff > config.NOSE_HIP_FACING_THRESHOLD:
        return "RIGHT"
    elif diff < -config.NOSE_HIP_FACING_THRESHOLD:
        return "LEFT"
    else:
        return None  # Aligned/neutral - don't display


def draw_intention_arrow(
    frame: np.ndarray,
    persons: dict,
    facing_direction: str,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw a horizontal arrow above the player's head showing intention (torso facing).

    The arrow is parallel to the video frame, pointing LEFT or RIGHT based on
    nose-hip orientation. This captures turn intention since head turns before body.

    Args:
        frame: Canvas to draw on
        persons: Dict of person_id -> keypoints dict
        facing_direction: "LEFT" or "RIGHT"
        config: Annotation config
        x_offset: X offset for sidebar
    """
    if not persons or facing_direction not in ("LEFT", "RIGHT"):
        return

    # Get head position (prefer nose, fallback to head keypoint)
    first_person_id = min(persons.keys())
    keypoints = persons[first_person_id]

    # Try nose first, then head keypoint
    head_pos = None
    for kp_name in ['nose', 'head']:
        kp = keypoints.get(kp_name)
        if kp and kp[2] >= config.MIN_KEYPOINT_CONFIDENCE:
            head_pos = (kp[0], kp[1])
            break

    if head_pos is None:
        return

    # Arrow center point (above head)
    arrow_center_x = int(head_pos[0]) + x_offset
    arrow_center_y = int(head_pos[1]) - config.INTENTION_ARROW_OFFSET_Y

    # Arrow direction and endpoints
    half_length = config.INTENTION_ARROW_LENGTH // 2

    if facing_direction == "RIGHT":
        start_x = arrow_center_x - half_length
        end_x = arrow_center_x + half_length
        color = config.TORSO_FACING_COLOR_RIGHT
    else:  # LEFT
        start_x = arrow_center_x + half_length
        end_x = arrow_center_x - half_length
        color = config.TORSO_FACING_COLOR_LEFT

    # Draw the arrow (horizontal, parallel to video)
    cv2.arrowedLine(
        frame,
        (start_x, arrow_center_y),
        (end_x, arrow_center_y),
        color,
        config.INTENTION_ARROW_THICKNESS,
        tipLength=config.INTENTION_ARROW_TIP_LENGTH,
        line_type=cv2.LINE_AA
    )


def determine_ball_position_vs_intention(
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    facing_direction: Optional[str],
    config: TripleConeAnnotationConfig
) -> IntentionPositionResult:
    """
    Determine if ball is in front of or behind player's FACING direction (intention).

    Unlike momentum-based detection, this uses where the player is LOOKING
    (nose-hip orientation) rather than where they're MOVING.

    Args:
        ball_center: Ball (x, y) position
        hip_position: Player hip (x, y) position
        facing_direction: "LEFT", "RIGHT", or None from determine_torso_facing()
        config: Annotation config

    Returns:
        IntentionPositionResult with position relative to facing direction
    """
    if ball_center is None or hip_position is None:
        return IntentionPositionResult(
            position="UNKNOWN",
            facing_direction=facing_direction,
            ball_hip_delta_x=0.0,
            color=(180, 180, 180)  # Gray
        )

    ball_x = ball_center[0]
    hip_x = hip_position[0]
    delta_x = ball_x - hip_x

    # If no facing direction detected, return unknown
    if facing_direction is None:
        return IntentionPositionResult(
            position="UNKNOWN",
            facing_direction=None,
            ball_hip_delta_x=delta_x,
            color=(180, 180, 180)  # Gray
        )

    # Check if ball is aligned with player (same threshold as momentum-based)
    if abs(delta_x) < config.BALL_POSITION_THRESHOLD:
        return IntentionPositionResult(
            position="I-ALIGNED",
            facing_direction=facing_direction,
            ball_hip_delta_x=delta_x,
            color=config.INTENTION_ALIGNED_COLOR
        )

    # Determine front/behind based on facing direction
    if facing_direction == "LEFT":
        # Facing left: FRONT = ball to left (negative delta)
        if delta_x < 0:
            position = "I-FRONT"
            color = config.INTENTION_FRONT_COLOR
        else:
            position = "I-BEHIND"
            color = config.INTENTION_BEHIND_COLOR
    else:  # RIGHT
        # Facing right: FRONT = ball to right (positive delta)
        if delta_x > 0:
            position = "I-FRONT"
            color = config.INTENTION_FRONT_COLOR
        else:
            position = "I-BEHIND"
            color = config.INTENTION_BEHIND_COLOR

    return IntentionPositionResult(
        position=position,
        facing_direction=facing_direction,
        ball_hip_delta_x=delta_x,
        color=color
    )


def check_edge_zone_status(
    ball_x: Optional[float],
    video_width: int,
    config: TripleConeAnnotationConfig
) -> EdgeZoneStatus:
    """Check if ball is in edge zone."""
    if ball_x is None:
        return EdgeZoneStatus(False, "NONE", float('inf'))

    left_distance = ball_x
    right_distance = video_width - ball_x

    if right_distance < config.EDGE_MARGIN:
        return EdgeZoneStatus(True, "RIGHT", right_distance)

    if left_distance < config.EDGE_MARGIN:
        return EdgeZoneStatus(True, "LEFT", left_distance)

    return EdgeZoneStatus(False, "NONE", min(left_distance, right_distance))


def update_ball_tracking_state(
    current_state: BallTrackingState,
    ball_visible: bool,
    edge_status: EdgeZoneStatus,
) -> Tuple[BallTrackingState, bool]:
    """
    State machine for unified edge+off-screen tracking.

    Returns:
        (new_state, should_reset_counter)
        - should_reset_counter: True when transitioning to NORMAL or switching sides

    State transitions:
        NORMAL + ball in left edge → EDGE_LEFT (reset counter)
        NORMAL + ball in right edge → EDGE_RIGHT (reset counter)
        NORMAL + ball disappears → DISAPPEARED_MID (no counter)

        EDGE_LEFT + ball disappears → OFF_SCREEN_LEFT (continue counter)
        EDGE_LEFT + ball leaves edge → NORMAL (reset counter, trigger persist)
        EDGE_LEFT + ball enters right edge → EDGE_RIGHT (reset counter)

        EDGE_RIGHT + ball disappears → OFF_SCREEN_RIGHT (continue counter)
        EDGE_RIGHT + ball leaves edge → NORMAL (reset counter, trigger persist)
        EDGE_RIGHT + ball enters left edge → EDGE_LEFT (reset counter)

        OFF_SCREEN_LEFT + ball returns to left edge → EDGE_LEFT (continue counter)
        OFF_SCREEN_LEFT + ball returns outside edge → NORMAL (reset, trigger persist)
        OFF_SCREEN_LEFT + ball returns to right edge → EDGE_RIGHT (reset counter)

        OFF_SCREEN_RIGHT + ball returns to right edge → EDGE_RIGHT (continue counter)
        OFF_SCREEN_RIGHT + ball returns outside edge → NORMAL (reset, trigger persist)
        OFF_SCREEN_RIGHT + ball returns to left edge → EDGE_LEFT (reset counter)

        DISAPPEARED_MID + ball returns → check edge status and transition accordingly
    """
    # Ball not visible
    if not ball_visible:
        if current_state == BallTrackingState.EDGE_LEFT:
            return (BallTrackingState.OFF_SCREEN_LEFT, False)  # Continue counter
        elif current_state == BallTrackingState.EDGE_RIGHT:
            return (BallTrackingState.OFF_SCREEN_RIGHT, False)  # Continue counter
        elif current_state in (BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
            return (current_state, False)  # Stay in off-screen state
        elif current_state == BallTrackingState.DISAPPEARED_MID:
            return (current_state, False)  # Stay in disappeared mid
        else:
            # NORMAL → disappeared mid-field (detection failure)
            return (BallTrackingState.DISAPPEARED_MID, True)  # Reset counter (no counting for mid-field)

    # Ball is visible
    in_edge = edge_status.in_edge_zone
    edge_side = edge_status.edge_side

    if current_state == BallTrackingState.NORMAL:
        if in_edge:
            if edge_side == "LEFT":
                return (BallTrackingState.EDGE_LEFT, True)  # Start new counter
            else:  # RIGHT
                return (BallTrackingState.EDGE_RIGHT, True)  # Start new counter
        return (BallTrackingState.NORMAL, False)  # Stay normal

    elif current_state == BallTrackingState.EDGE_LEFT:
        if in_edge:
            if edge_side == "LEFT":
                return (BallTrackingState.EDGE_LEFT, False)  # Continue counter
            else:  # Switched to RIGHT
                return (BallTrackingState.EDGE_RIGHT, True)  # Reset and start right counter
        else:
            return (BallTrackingState.NORMAL, True)  # Left edge, trigger persist display

    elif current_state == BallTrackingState.EDGE_RIGHT:
        if in_edge:
            if edge_side == "RIGHT":
                return (BallTrackingState.EDGE_RIGHT, False)  # Continue counter
            else:  # Switched to LEFT
                return (BallTrackingState.EDGE_LEFT, True)  # Reset and start left counter
        else:
            return (BallTrackingState.NORMAL, True)  # Left edge, trigger persist display

    elif current_state == BallTrackingState.OFF_SCREEN_LEFT:
        # Ball returned after going off-screen via left edge
        if in_edge:
            if edge_side == "LEFT":
                return (BallTrackingState.EDGE_LEFT, False)  # Continue counter (still in danger)
            else:  # RIGHT edge
                return (BallTrackingState.EDGE_RIGHT, True)  # Reset, start new right sequence
        else:
            return (BallTrackingState.NORMAL, True)  # Fully recovered, trigger persist

    elif current_state == BallTrackingState.OFF_SCREEN_RIGHT:
        # Ball returned after going off-screen via right edge
        if in_edge:
            if edge_side == "RIGHT":
                return (BallTrackingState.EDGE_RIGHT, False)  # Continue counter (still in danger)
            else:  # LEFT edge
                return (BallTrackingState.EDGE_LEFT, True)  # Reset, start new left sequence
        else:
            return (BallTrackingState.NORMAL, True)  # Fully recovered, trigger persist

    elif current_state == BallTrackingState.DISAPPEARED_MID:
        # Ball returned after mid-field disappearance
        if in_edge:
            if edge_side == "LEFT":
                return (BallTrackingState.EDGE_LEFT, True)  # Start new edge sequence
            else:  # RIGHT
                return (BallTrackingState.EDGE_RIGHT, True)  # Start new edge sequence
        else:
            return (BallTrackingState.NORMAL, True)  # Return to normal

    return (current_state, False)  # Default: no change


def draw_ball_position_indicator(
    frame: np.ndarray,
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    result: BallPositionResult,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw visual indicators showing ball position relative to player."""
    if ball_center is None or hip_position is None:
        return

    if result.position == "UNKNOWN":
        return

    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [ball_center[0], ball_center[1], hip_position[0], hip_position[1]]):
        return

    hip_x = int(hip_position[0]) + x_offset
    hip_y = int(hip_position[1])
    ball_x = int(ball_center[0]) + x_offset
    ball_y = int(ball_center[1])

    # 1. Dashed vertical line through hip (use config dash/gap values)
    half_height = config.DIVIDER_LINE_HEIGHT // 2
    line_color = (100, 100, 100)

    dash_length = config.INTENTION_LINE_DASH_LENGTH  # Already scaled
    gap_length = config.INTENTION_LINE_GAP_LENGTH    # Already scaled
    y_start = hip_y - half_height
    y_end = hip_y + half_height
    y = y_start
    while y < y_end:
        y_next = min(y + dash_length, y_end)
        cv2.line(frame, (hip_x, y), (hip_x, y_next), line_color, 1, cv2.LINE_AA)
        y = y_next + gap_length

    # Hip marker (scale radius)
    hip_radius = max(2, int(5 * config.FONT_SCALE_FACTOR))
    cv2.circle(frame, (hip_x, hip_y), hip_radius, result.color, -1)
    cv2.circle(frame, (hip_x, hip_y), hip_radius, (255, 255, 255), 1)

    # 2. Connecting line from hip to ball
    cv2.line(frame, (hip_x, hip_y), (ball_x, ball_y),
             result.color, config.BALL_HIP_LINE_THICKNESS, cv2.LINE_AA)

    # 3. Position label above ball (scale font and positioning)
    label = result.position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * config.FONT_SCALE_FACTOR
    font_thickness = max(1, int(2 * config.FONT_SCALE_FACTOR))

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    label_x = ball_x - text_width // 2
    label_y = ball_y - int(30 * config.FONT_SCALE_FACTOR)

    padding = max(2, int(3 * config.FONT_SCALE_FACTOR))
    cv2.rectangle(
        frame,
        (label_x - padding, label_y - text_height - padding),
        (label_x + text_width + padding, label_y + padding),
        (0, 0, 0), -1
    )
    cv2.putText(
        frame, label,
        (label_x, label_y),
        font, font_scale, result.color, font_thickness, cv2.LINE_AA
    )

    # Movement direction arrow removed - redundant with momentum arrow at hip


def draw_dashed_line(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: int = 8,
    gap_length: int = 6
) -> None:
    """Draw a dashed line from pt1 to pt2."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Calculate total distance and direction
    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx * dx + dy * dy)

    if distance < 1:
        return

    # Normalize direction
    dx /= distance
    dy /= distance

    # Draw dashes
    segment_length = dash_length + gap_length
    current_pos = 0.0

    while current_pos < distance:
        # Start of dash
        start_x = int(x1 + dx * current_pos)
        start_y = int(y1 + dy * current_pos)

        # End of dash (limited by total distance)
        end_pos = min(current_pos + dash_length, distance)
        end_x = int(x1 + dx * end_pos)
        end_y = int(y1 + dy * end_pos)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)

        current_pos += segment_length


def draw_intention_position_indicator(
    frame: np.ndarray,
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    result: IntentionPositionResult,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw visual indicator for ball position vs intention (dashed line style).

    Shows where ball is relative to player's FACING direction (intention).
    Uses dashed line and label BELOW ball to distinguish from momentum indicator.
    """
    if ball_center is None or hip_position is None:
        return

    if result.position == "UNKNOWN":
        return

    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [ball_center[0], ball_center[1], hip_position[0], hip_position[1]]):
        return

    hip_x = int(hip_position[0]) + x_offset
    hip_y = int(hip_position[1])
    ball_x = int(ball_center[0]) + x_offset
    ball_y = int(ball_center[1])

    # 1. Dashed line from hip to ball (distinct from solid momentum line)
    draw_dashed_line(
        frame,
        (hip_x, hip_y),
        (ball_x, ball_y),
        result.color,
        config.BALL_HIP_LINE_THICKNESS,
        config.INTENTION_LINE_DASH_LENGTH,
        config.INTENTION_LINE_GAP_LENGTH
    )

    # 2. Position label BELOW ball (to distinguish from momentum label above)
    label = result.position  # "I-FRONT", "I-BEHIND", "I-ALIGNED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55  # Slightly smaller than momentum label
    font_thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    label_x = ball_x - text_width // 2
    label_y = ball_y + config.INTENTION_LABEL_OFFSET_Y  # Below ball (momentum is above)

    padding = 3
    cv2.rectangle(
        frame,
        (label_x - padding, label_y - text_height - padding),
        (label_x + text_width + padding, label_y + padding),
        (0, 0, 0), -1
    )
    cv2.putText(
        frame, label,
        (label_x, label_y),
        font, font_scale, result.color, font_thickness, cv2.LINE_AA
    )


def draw_behind_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw ball-behind duration counter."""
    if count <= 0:
        return

    text = f"BEHIND: {count}f"
    x = x_offset + config.BEHIND_COUNTER_POS_X
    y = config.BEHIND_COUNTER_POS_Y

    color = config.BEHIND_COUNTER_COLOR if is_active else config.BEHIND_COUNTER_PERSIST_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * config.FONT_SCALE_FACTOR))
    (tw, th), _ = cv2.getTextSize(text, font, config.BEHIND_COUNTER_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * config.FONT_SCALE_FACTOR))
    pad_y = max(4, int(10 * config.FONT_SCALE_FACTOR))
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font,
                config.BEHIND_COUNTER_FONT_SCALE, color, thickness, cv2.LINE_AA)


def draw_intention_behind_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw intention-based ball-behind duration counter."""
    if count <= 0:
        return

    text = f"BEHIND (I): {count}f"
    x = x_offset + config.BEHIND_COUNTER_POS_X
    y = config.INTENTION_BEHIND_COUNTER_POS_Y

    # Use intention colors (magenta active, lighter persist)
    color = config.INTENTION_BEHIND_COLOR if is_active else (255, 150, 255)  # Lighter magenta

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * config.FONT_SCALE_FACTOR))
    (tw, th), _ = cv2.getTextSize(text, font, config.BEHIND_COUNTER_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * config.FONT_SCALE_FACTOR))
    pad_y = max(4, int(10 * config.FONT_SCALE_FACTOR))
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font,
                config.BEHIND_COUNTER_FONT_SCALE, color, thickness, cv2.LINE_AA)


def draw_edge_zones(
    frame: np.ndarray,
    video_width: int,
    video_height: int,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> np.ndarray:
    """Draw semi-transparent edge zone overlays."""
    overlay = frame.copy()

    # Left edge zone
    cv2.rectangle(
        overlay,
        (x_offset, 0),
        (x_offset + config.EDGE_MARGIN, video_height),
        config.EDGE_ZONE_COLOR, -1
    )
    # Right edge zone
    cv2.rectangle(
        overlay,
        (x_offset + video_width - config.EDGE_MARGIN, 0),
        (x_offset + video_width, video_height),
        config.EDGE_ZONE_COLOR, -1
    )

    return cv2.addWeighted(overlay, config.EDGE_ZONE_ALPHA, frame,
                           1 - config.EDGE_ZONE_ALPHA, 0)


def draw_edge_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    edge_side: str,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw edge zone frame counter."""
    if count <= 0:
        return

    text = f"EDGE ({edge_side}): {count}f"
    x = x_offset + config.EDGE_COUNTER_POS_X
    y = config.EDGE_COUNTER_POS_Y

    color = config.EDGE_COUNTER_COLOR if is_active else config.EDGE_COUNTER_PERSIST_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * config.FONT_SCALE_FACTOR))
    (tw, th), _ = cv2.getTextSize(text, font, config.EDGE_COUNTER_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * config.FONT_SCALE_FACTOR))
    pad_y = max(4, int(10 * config.FONT_SCALE_FACTOR))
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font,
                config.EDGE_COUNTER_FONT_SCALE, color, thickness, cv2.LINE_AA)


def draw_off_screen_indicator(
    frame: np.ndarray,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw 'BALL OFF-SCREEN' indicator when ball is not detected.

    Args:
        frame: Video frame to draw on
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    text = config.OFF_SCREEN_TEXT
    x = x_offset + config.OFF_SCREEN_POS_X
    y = config.OFF_SCREEN_POS_Y

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    (tw, th), _ = cv2.getTextSize(text, font, config.OFF_SCREEN_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    pad_y = max(4, int(10 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))

    # Draw background box
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font,
                config.OFF_SCREEN_FONT_SCALE, config.OFF_SCREEN_COLOR, thickness, cv2.LINE_AA)


def draw_return_counter(
    frame: np.ndarray,
    frames_gone: int,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw counter showing how long ball was off-screen (after it returns).

    Shows "WAS GONE: Xf" to indicate how many frames the ball was not detected.

    Args:
        frame: Video frame to draw on
        frames_gone: Number of frames the ball was off-screen
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    text = f"WAS GONE: {frames_gone}f"
    x = x_offset + config.RETURN_COUNTER_POS_X
    y = config.RETURN_COUNTER_POS_Y

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    (tw, th), _ = cv2.getTextSize(text, font, config.RETURN_COUNTER_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    pad_y = max(4, int(10 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))

    # Draw background box
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font,
                config.RETURN_COUNTER_FONT_SCALE, config.RETURN_COUNTER_COLOR, thickness, cv2.LINE_AA)


def draw_unified_tracking_indicator(
    frame: np.ndarray,
    state: BallTrackingState,
    counter: int,
    is_persisting: bool,
    persist_side: str,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw unified edge+off-screen tracking indicator.

    Displays different text/color based on state:
    - EDGE_LEFT/RIGHT: "EDGE LEFT: Xf" / "EDGE RIGHT: Xf" (yellow)
    - OFF_SCREEN_LEFT/RIGHT: "LEFT EDGE + OFF: Xf" / "RIGHT EDGE + OFF: Xf" (red)
    - Persisting after return: "WAS LEFT: Xf" / "WAS RIGHT: Xf" (orange)
    - DISAPPEARED_MID: "DETECTION LOST" (gray) - no counter

    Args:
        frame: Video frame to draw on
        state: Current BallTrackingState
        counter: Number of frames in current state sequence
        is_persisting: True when showing persist display after returning to normal
        persist_side: "LEFT" or "RIGHT" - which side the ball exited from (for persist display)
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    # Determine text and color based on state
    text = ""
    color = config.UNIFIED_EDGE_COLOR
    pos_x = x_offset + config.UNIFIED_COUNTER_POS_X
    pos_y = config.UNIFIED_COUNTER_POS_Y

    if state == BallTrackingState.EDGE_LEFT:
        text = f"EDGE LEFT: {counter}f"
        color = config.UNIFIED_EDGE_COLOR
    elif state == BallTrackingState.EDGE_RIGHT:
        text = f"EDGE RIGHT: {counter}f"
        color = config.UNIFIED_EDGE_COLOR
    elif state == BallTrackingState.OFF_SCREEN_LEFT:
        text = f"LEFT EDGE + OFF: {counter}f"
        color = config.UNIFIED_OFF_SCREEN_COLOR
    elif state == BallTrackingState.OFF_SCREEN_RIGHT:
        text = f"RIGHT EDGE + OFF: {counter}f"
        color = config.UNIFIED_OFF_SCREEN_COLOR
    elif state == BallTrackingState.DISAPPEARED_MID:
        # Mid-field disappearance - show different indicator
        text = config.MID_FIELD_DISAPPEAR_TEXT
        color = config.MID_FIELD_DISAPPEAR_COLOR
        pos_x = x_offset + config.MID_FIELD_DISAPPEAR_POS_X
        pos_y = config.MID_FIELD_DISAPPEAR_POS_Y
    elif is_persisting and counter > 0:
        # Persist display after returning to normal
        text = f"WAS {persist_side}: {counter}f"
        color = config.UNIFIED_PERSIST_COLOR
    else:
        return  # Nothing to draw (NORMAL state with no persist)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.UNIFIED_COUNTER_FONT_SCALE
    thickness = max(1, int(2 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    pad_y = max(4, int(10 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))

    # Draw background box
    cv2.rectangle(frame, (pos_x - pad_x, pos_y - th - pad_y),
                  (pos_x + tw + pad_x * 2, pos_y + pad_y), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (pos_x, pos_y), font,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_debug_axes(
    frame: np.ndarray,
    ball_center: Optional[Tuple[float, float]],
    config: TripleConeAnnotationConfig,
    video_width: int,
    video_height: int,
    x_offset: int = 0
) -> None:
    """Draw debug axes through ball position (always on)."""
    if ball_center is None:
        return
    # Skip if coordinates are NaN
    if pd.isna(ball_center[0]) or pd.isna(ball_center[1]):
        return

    ball_x = int(ball_center[0]) + x_offset
    ball_y = int(ball_center[1])

    # Vertical line through ball X
    cv2.line(frame, (ball_x, 0), (ball_x, video_height),
             config.DEBUG_AXES_COLOR, config.DEBUG_AXES_THICKNESS)

    # Horizontal line through ball Y
    cv2.line(frame, (x_offset, ball_y), (x_offset + video_width, ball_y),
             config.DEBUG_AXES_COLOR, config.DEBUG_AXES_THICKNESS)


# ============================================================================
# Video Processing
# ============================================================================

def annotate_triple_cone_video(video_path: Path, parquet_dir: Path, output_path: Path,
                               config: TripleConeAnnotationConfig = None) -> bool:
    """
    Annotate Triple Cone drill video with cone positions, ball/pose tracking,
    and debug visualization.
    """
    if config is None:
        config = TripleConeAnnotationConfig()

    base_name = parquet_dir.name

    # Load cone positions from parquet
    cone_parquet = list(parquet_dir.glob("*_cone.parquet"))
    if not cone_parquet:
        print(f"  Error: No cone parquet found in {parquet_dir}")
        return False

    print(f"  Loading cone positions from parquet...")
    try:
        cone1, cone2, cone3 = load_cone_positions_from_parquet(cone_parquet[0])
    except Exception as e:
        print(f"  Error loading cones: {e}")
        return False

    print(f"    CONE1 (HOME): ({cone1.center_x:.0f}, {cone1.center_y:.0f}) [{cone1.width:.0f}x{cone1.height:.0f}]")
    print(f"    CONE2 (CENTER): ({cone2.center_x:.0f}, {cone2.center_y:.0f}) [{cone2.width:.0f}x{cone2.height:.0f}]")
    print(f"    CONE3 (RIGHT): ({cone3.center_x:.0f}, {cone3.center_y:.0f}) [{cone3.width:.0f}x{cone3.height:.0f}]")

    # Create turning zones (scaling will be applied after video resolution is known)
    # Zone config is created here but will be updated after we know the resolution
    print(f"  Creating turning zones...")
    zone_config = TripleConeZoneConfig.default()
    # Note: turning_zones will be recreated after resolution detection if needed
    turning_zones_placeholder = (cone1, cone2, cone3, zone_config)

    # Load parquet data
    ball_parquets = list(parquet_dir.glob("*_football.parquet"))
    pose_parquets = list(parquet_dir.glob("*_pose.parquet"))

    if not ball_parquets or not pose_parquets:
        print(f"  Error: Missing parquet files in {parquet_dir}")
        return False

    print(f"  Loading parquet data...")
    ball_df = load_ball_data(ball_parquets[0])
    pose_df = load_pose_data(pose_parquets[0])

    # Create lookup structures (include interpolated if available for off-screen detection)
    ball_cols = ['x1', 'y1', 'x2', 'y2', 'confidence']
    if 'interpolated' in ball_df.columns:
        ball_cols.append('interpolated')
    ball_lookup = ball_df.groupby('frame_id').apply(
        lambda g: g[ball_cols].to_dict('records')
    ).to_dict()
    pose_lookup = prepare_pose_lookup(pose_df)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open video: {video_path}")
        return False

    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Resolution auto-detection and config scaling
    REFERENCE_WIDTH = 1920  # Full HD baseline (more reasonable for most videos)
    resolution_scale = orig_width / REFERENCE_WIDTH

    if abs(resolution_scale - 1.0) > 0.01:  # Not original resolution
        # Use sqrt for font scaling (gentler) vs linear for positions
        import math
        font_scale = math.sqrt(resolution_scale)  # 0.45 -> 0.67 (gentler)

        print(f"  [AUTO-SCALE] Detected {orig_width}px width")
        print(f"    Position scale: {resolution_scale:.3f}, Font scale: {font_scale:.3f}")

        # Store scale factors for use in drawing functions
        config.RESOLUTION_SCALE = resolution_scale
        config.FONT_SCALE_FACTOR = font_scale  # For inline font scaling

        # Scale pixel-based thresholds (linear - these are distances)
        config.BALL_POSITION_THRESHOLD *= resolution_scale
        config.MOVEMENT_THRESHOLD *= resolution_scale
        config.NOSE_HIP_FACING_THRESHOLD *= resolution_scale
        config.EDGE_MARGIN = int(config.EDGE_MARGIN * resolution_scale)
        config.BEHIND_COUNTER_POS_X = int(config.BEHIND_COUNTER_POS_X * resolution_scale)
        config.BEHIND_COUNTER_POS_Y = int(config.BEHIND_COUNTER_POS_Y * resolution_scale)
        config.EDGE_COUNTER_POS_X = int(config.EDGE_COUNTER_POS_X * resolution_scale)
        config.EDGE_COUNTER_POS_Y = int(config.EDGE_COUNTER_POS_Y * resolution_scale)
        config.INTENTION_BEHIND_COUNTER_POS_Y = int(config.INTENTION_BEHIND_COUNTER_POS_Y * resolution_scale)
        config.OFF_SCREEN_POS_X = int(config.OFF_SCREEN_POS_X * resolution_scale)
        config.OFF_SCREEN_POS_Y = int(config.OFF_SCREEN_POS_Y * resolution_scale)
        config.RETURN_COUNTER_POS_X = int(config.RETURN_COUNTER_POS_X * resolution_scale)
        config.RETURN_COUNTER_POS_Y = int(config.RETURN_COUNTER_POS_Y * resolution_scale)

        # Scale sidebar dimensions (with minimum values for readability)
        config.SIDEBAR_WIDTH = max(200, int(config.SIDEBAR_WIDTH * resolution_scale))
        config.SIDEBAR_LINE_HEIGHT = max(18, int(config.SIDEBAR_LINE_HEIGHT * font_scale))
        config.SIDEBAR_PADDING = max(10, int(config.SIDEBAR_PADDING * font_scale))

        # Scale font sizes (sqrt scaling - stay readable, with minimums)
        config.SIDEBAR_FONT_SCALE = max(0.40, config.SIDEBAR_FONT_SCALE * font_scale)
        config.FONT_SCALE = max(0.35, config.FONT_SCALE * font_scale)
        config.BEHIND_COUNTER_FONT_SCALE *= font_scale
        config.EDGE_COUNTER_FONT_SCALE *= font_scale
        config.OFF_SCREEN_FONT_SCALE *= font_scale
        config.RETURN_COUNTER_FONT_SCALE *= font_scale

        # Scale line/arrow thicknesses (use font_scale for visual consistency)
        config.MOMENTUM_THICKNESS = max(3, int(config.MOMENTUM_THICKNESS * font_scale))
        config.BALL_MOMENTUM_THICKNESS = max(2, int(config.BALL_MOMENTUM_THICKNESS * font_scale))
        config.INTENTION_ARROW_THICKNESS = max(2, int(config.INTENTION_ARROW_THICKNESS * font_scale))
        config.SKELETON_THICKNESS = max(1, int(config.SKELETON_THICKNESS * font_scale))
        config.BBOX_THICKNESS = max(1, int(config.BBOX_THICKNESS * font_scale))
        config.BALL_HIP_LINE_THICKNESS = max(1, int(config.BALL_HIP_LINE_THICKNESS * font_scale))

        # Scale visual elements
        config.INTENTION_LINE_DASH_LENGTH = max(3, int(config.INTENTION_LINE_DASH_LENGTH * resolution_scale))
        config.INTENTION_LINE_GAP_LENGTH = max(2, int(config.INTENTION_LINE_GAP_LENGTH * resolution_scale))
        config.INTENTION_LABEL_OFFSET_Y = int(config.INTENTION_LABEL_OFFSET_Y * resolution_scale)
        config.MOMENTUM_MAX_LENGTH = int(config.MOMENTUM_MAX_LENGTH * resolution_scale)
        config.DIVIDER_LINE_HEIGHT = int(config.DIVIDER_LINE_HEIGHT * resolution_scale)
        config.INTENTION_ARROW_LENGTH = int(config.INTENTION_ARROW_LENGTH * resolution_scale)
        config.INTENTION_ARROW_OFFSET_Y = int(config.INTENTION_ARROW_OFFSET_Y * resolution_scale)
        # CONE_MARKER_SIZE scaling removed - using actual bbox from parquet
        config.KEYPOINT_RADIUS = max(2, int(config.KEYPOINT_RADIUS * resolution_scale))

        # NOTE: Zone radii are NOT scaled here because they must match
        # the parquet data coordinate space (720p), not video resolution.
        # The 720p parquet data uses 720p-scaled zone radii (68.0 default).

    # Now create turning zones with potentially scaled config
    cone1, cone2, cone3, zone_config = turning_zones_placeholder
    turning_zones = create_triple_cone_zones(cone1.center, cone2.center, cone3.center, zone_config)
    print(f"    Zone radius: {zone_config.cone1_zone_radius:.0f}px, stretch_y: {zone_config.stretch_y}")

    canvas_width = config.SIDEBAR_WIDTH + orig_width
    canvas_height = orig_height

    print(f"  Video: {orig_width}x{orig_height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"  Output canvas: {canvas_width}x{canvas_height}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_width, canvas_height))

    if not out.isOpened():
        print(f"  Error: Cannot create output video: {output_path}")
        cap.release()
        return False

    print(f"  Processing frames...")

    # Initialize trackers
    hip_history: deque = deque(maxlen=config.MOMENTUM_LOOKBACK_FRAMES + 1)
    ball_history: deque = deque(maxlen=config.MOMENTUM_LOOKBACK_FRAMES + 1)
    turn_tracker = TripleConeTurnTracker()

    # Behind counter (momentum-based)
    behind_counter: int = 0
    behind_display_value: int = 0
    behind_display_timer: int = 0
    behind_persist_frames = int(config.BEHIND_COUNTER_PERSIST_SECONDS * fps)

    # Behind counter (intention-based)
    intention_behind_counter: int = 0
    intention_behind_display_value: int = 0
    intention_behind_display_timer: int = 0

    # Legacy edge counter (kept for backward compatibility when UNIFIED_TRACKING_ENABLED=False)
    edge_counter: int = 0
    edge_display_value: int = 0
    edge_display_timer: int = 0
    edge_last_side: str = "NONE"
    edge_persist_frames = int(config.EDGE_COUNTER_PERSIST_SECONDS * fps)

    # Legacy ball off-screen tracking (kept for backward compatibility)
    off_screen_counter: int = 0
    return_display_value: int = 0
    return_display_timer: int = 0
    return_persist_frames = int(config.RETURN_COUNTER_PERSIST_SECONDS * fps)

    # Unified edge+off-screen tracking (new state machine)
    unified_tracking_state: BallTrackingState = BallTrackingState.NORMAL
    unified_counter: int = 0                    # Continuous counter for edge+off-screen sequence
    unified_persist_value: int = 0              # Counter value to show during persist display
    unified_persist_timer: int = 0              # Frames remaining for persist display
    unified_persist_side: str = "NONE"          # Which side ball exited from (for persist text)
    unified_persist_frames = int(config.UNIFIED_COUNTER_PERSIST_SECONDS * fps)

    for frame_id in tqdm(range(total_frames), desc="  Annotating", unit="frame"):
        ret, video_frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps

        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:, config.SIDEBAR_WIDTH:] = video_frame

        # Get ball data
        balls = ball_lookup.get(frame_id, [])
        ball_center = None
        ball_is_off_screen = True  # Assume off-screen until we find a real detection
        for ball in balls:
            if ball['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                # Check if this is an interpolated (not real) detection
                is_interpolated = ball.get('interpolated', False)
                if not is_interpolated:
                    ball_is_off_screen = False  # Real detection found

                center_x = (ball['x1'] + ball['x2']) / 2
                center_y = (ball['y1'] + ball['y2']) / 2
                ball_center = (center_x, center_y)
                break

        # Get pose keypoints
        persons = pose_lookup.get(frame_id, {})
        pose_keypoints = {}
        if persons:
            first_person_id = min(persons.keys())
            pose_keypoints = persons[first_person_id]

        # Get hip position
        current_hip = None
        for _, keypoints in persons.items():
            hip_data = keypoints.get('hip')
            if hip_data and hip_data[2] >= config.MIN_KEYPOINT_CONFIDENCE:
                current_hip = (hip_data[0], hip_data[1])
                break

        # Update hip history
        if current_hip:
            hip_history.append(current_hip)

        previous_hip = hip_history[0] if len(hip_history) >= 2 else None

        # Update ball history
        if ball_center and not any(pd.isna(v) for v in ball_center):
            ball_history.append(ball_center)

        previous_ball = ball_history[0] if len(ball_history) >= 2 else None

        # Get movement direction for turn tracking
        movement_direction = None
        if current_hip and previous_hip:
            dx = current_hip[0] - previous_hip[0]
            if dx > config.MOVEMENT_THRESHOLD:
                movement_direction = "RIGHT"
            elif dx < -config.MOVEMENT_THRESHOLD:
                movement_direction = "LEFT"

        # Check turning zone
        active_zone = None
        if ball_center and not any(pd.isna(v) for v in ball_center):
            active_zone = turning_zones.get_zone_at_point(ball_center[0], ball_center[1])

        # Update turn tracker
        turn_tracker.update(frame_id, timestamp, active_zone, movement_direction)

        # Ball position detection
        ball_position_result = None
        if config.DRAW_BALL_POSITION:
            ball_position_result = determine_ball_position_relative_to_player(
                ball_center, current_hip, previous_hip, config
            )

        # Torso facing detection
        torso_facing = None
        if config.DRAW_TORSO_FACING:
            torso_facing = determine_torso_facing(persons, config)

        # Intention-based ball position detection
        intention_result = None
        if config.DRAW_BALL_POSITION_INTENTION and torso_facing:
            intention_result = determine_ball_position_vs_intention(
                ball_center, current_hip, torso_facing, config
            )

        # Update momentum behind counter
        if ball_position_result and ball_position_result.position == "BEHIND":
            behind_counter += 1
            behind_display_value = behind_counter
            behind_display_timer = behind_persist_frames
        else:
            if behind_counter > 0:
                behind_display_value = behind_counter
                behind_display_timer = behind_persist_frames
            behind_counter = 0

        if behind_display_timer > 0:
            behind_display_timer -= 1

        # Update intention behind counter
        if intention_result and intention_result.position == "I-BEHIND":
            intention_behind_counter += 1
            intention_behind_display_value = intention_behind_counter
            intention_behind_display_timer = behind_persist_frames
        else:
            if intention_behind_counter > 0:
                intention_behind_display_value = intention_behind_counter
                intention_behind_display_timer = behind_persist_frames
            intention_behind_counter = 0

        if intention_behind_display_timer > 0:
            intention_behind_display_timer -= 1

        # Edge zone detection
        ball_x = ball_center[0] if ball_center else None
        edge_status = check_edge_zone_status(ball_x, orig_width, config)

        if config.UNIFIED_TRACKING_ENABLED:
            # === NEW UNIFIED STATE MACHINE ===
            # Get previous state for detecting transitions
            prev_state = unified_tracking_state

            # Update state machine
            # ball_visible = not ball_is_off_screen (ball was actually detected, not interpolated)
            ball_visible = not ball_is_off_screen
            new_state, should_reset = update_ball_tracking_state(
                unified_tracking_state, ball_visible, edge_status
            )

            # Handle counter updates based on state transition
            if should_reset:
                # Check if we're transitioning TO normal (trigger persist)
                if new_state == BallTrackingState.NORMAL and prev_state != BallTrackingState.NORMAL:
                    # Save counter value and which side for persist display
                    unified_persist_value = unified_counter
                    unified_persist_timer = unified_persist_frames
                    # Determine which side we came from
                    if prev_state in (BallTrackingState.EDGE_LEFT, BallTrackingState.OFF_SCREEN_LEFT):
                        unified_persist_side = "LEFT"
                    elif prev_state in (BallTrackingState.EDGE_RIGHT, BallTrackingState.OFF_SCREEN_RIGHT):
                        unified_persist_side = "RIGHT"
                # Reset counter for new sequence
                unified_counter = 0

            # Increment counter for active states (edge or off-screen)
            if new_state in (BallTrackingState.EDGE_LEFT, BallTrackingState.EDGE_RIGHT,
                            BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
                unified_counter += 1

            unified_tracking_state = new_state

            # Decrement persist timer when in NORMAL state
            if unified_tracking_state == BallTrackingState.NORMAL and unified_persist_timer > 0:
                unified_persist_timer -= 1

        else:
            # === LEGACY SEPARATE COUNTERS (backward compatibility) ===
            if edge_status.in_edge_zone:
                if edge_last_side != "NONE" and edge_last_side != edge_status.edge_side:
                    edge_counter = 1
                else:
                    edge_counter += 1
                edge_display_value = edge_counter
                edge_last_side = edge_status.edge_side
                edge_display_timer = edge_persist_frames
            else:
                if edge_counter > 0:
                    edge_display_timer = edge_persist_frames
                edge_counter = 0
                if edge_display_timer > 0:
                    edge_display_timer -= 1

            # Ball off-screen tracking (interpolated detection)
            if ball_is_off_screen:
                off_screen_counter += 1
            else:
                if off_screen_counter > 0:
                    # Ball just came back - save how long it was gone
                    return_display_value = off_screen_counter
                    return_display_timer = return_persist_frames
                off_screen_counter = 0

            # Decrement return display timer (only when ball is on screen)
            if return_display_timer > 0 and not ball_is_off_screen:
                return_display_timer -= 1

        # === DRAW EVERYTHING ===

        # 1. Turning zones (3 zones) - draw first so edge zones appear on top
        draw_triple_cone_zones(
            canvas, turning_zones, ball_center,
            x_offset=config.SIDEBAR_WIDTH,
            cone1_color=config.CONE1_ZONE_COLOR,
            cone2_color=config.CONE2_ZONE_COLOR,
            cone3_color=config.CONE3_ZONE_COLOR,
            highlight_color=config.ZONE_HIGHLIGHT_COLOR,
            alpha=config.ZONE_ALPHA,
        )

        # 2. Edge zones (on top of turning zones so they're visible)
        canvas = draw_edge_zones(
            canvas, orig_width, orig_height, config,
            x_offset=config.SIDEBAR_WIDTH
        )

        # 3. Sidebar
        draw_sidebar(
            canvas, frame_id, cone1, cone2, cone3, ball_center, pose_keypoints,
            config, active_zone, ball_position_result,
            intention_result=intention_result,
            turn_events=turn_tracker.get_recent_events(config.EVENT_LOG_MAX_EVENTS)
        )

        # 4. Cone markers
        draw_triple_cone_markers(canvas, cone1, cone2, cone3, config,
                                 x_offset=config.SIDEBAR_WIDTH)

        # 5. Ball bbox
        for ball in balls:
            if ball['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                label = f"Ball {ball['confidence']:.2f}"
                draw_bbox(canvas, ball['x1'], ball['y1'], ball['x2'], ball['y2'],
                         config.BALL_COLOR, label, config, x_offset=config.SIDEBAR_WIDTH)

        # 5.5. Ball momentum arrow
        if config.DRAW_BALL_MOMENTUM_ARROW and ball_center and previous_ball:
            draw_ball_momentum_arrow(
                canvas, ball_center, previous_ball,
                config, x_offset=config.SIDEBAR_WIDTH
            )

        # 6. Debug axes (always on)
        if config.DRAW_DEBUG_AXES:
            draw_debug_axes(canvas, ball_center, config, orig_width, orig_height,
                           x_offset=config.SIDEBAR_WIDTH)

        # 7. Pose skeleton
        for _, keypoints in persons.items():
            draw_skeleton(canvas, keypoints, config, x_offset=config.SIDEBAR_WIDTH)

        # 8. Ball position indicator
        if config.DRAW_BALL_POSITION and ball_position_result:
            draw_ball_position_indicator(
                canvas, ball_center, current_hip, ball_position_result,
                config, x_offset=config.SIDEBAR_WIDTH
            )

        # 8.5. Intention arrow above head (torso facing direction)
        # Shows where player is looking/facing - captures turn intention
        if config.DRAW_TORSO_FACING and torso_facing:
            draw_intention_arrow(
                canvas, persons, torso_facing,
                config, x_offset=config.SIDEBAR_WIDTH
            )

        # 8.6. Intention position indicator (dashed line, label below ball)
        # Shows ball position relative to facing direction
        if config.DRAW_BALL_POSITION_INTENTION and intention_result:
            draw_intention_position_indicator(
                canvas, ball_center, current_hip, intention_result,
                config, x_offset=config.SIDEBAR_WIDTH
            )

        # 9. Momentum arrow
        if config.DRAW_MOMENTUM_ARROW and current_hip and previous_hip:
            draw_momentum_arrow(
                canvas, current_hip, previous_hip,
                config, x_offset=config.SIDEBAR_WIDTH
            )

        # 10. Behind counter
        if behind_display_timer > 0 or behind_counter > 0:
            draw_behind_counter(
                canvas, behind_display_value,
                is_active=(behind_counter > 0),
                config=config,
                x_offset=config.SIDEBAR_WIDTH
            )

        # 11. Edge counter / Unified tracking indicator
        if config.UNIFIED_TRACKING_ENABLED:
            # New unified tracking indicator
            # Show if: in active state, or persisting after return
            is_active_state = unified_tracking_state in (
                BallTrackingState.EDGE_LEFT, BallTrackingState.EDGE_RIGHT,
                BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT,
                BallTrackingState.DISAPPEARED_MID
            )
            is_persisting = unified_tracking_state == BallTrackingState.NORMAL and unified_persist_timer > 0

            if is_active_state or is_persisting:
                draw_unified_tracking_indicator(
                    canvas,
                    state=unified_tracking_state,
                    counter=unified_counter if is_active_state else unified_persist_value,
                    is_persisting=is_persisting,
                    persist_side=unified_persist_side,
                    config=config,
                    x_offset=config.SIDEBAR_WIDTH
                )
        else:
            # Legacy edge counter
            if edge_display_timer > 0 or edge_counter > 0:
                draw_edge_counter(
                    canvas, edge_display_value,
                    is_active=(edge_counter > 0),
                    edge_side=edge_last_side,
                    config=config,
                    x_offset=config.SIDEBAR_WIDTH
                )

        # 12. Intention behind counter
        if intention_behind_display_timer > 0 or intention_behind_counter > 0:
            draw_intention_behind_counter(
                canvas, intention_behind_display_value,
                is_active=(intention_behind_counter > 0),
                config=config,
                x_offset=config.SIDEBAR_WIDTH
            )

        # 13. Ball off-screen indicator (legacy - only when unified tracking disabled)
        if not config.UNIFIED_TRACKING_ENABLED:
            if config.DRAW_OFF_SCREEN_INDICATOR and ball_is_off_screen:
                draw_off_screen_indicator(canvas, config, x_offset=config.SIDEBAR_WIDTH)

        # 14. Return counter (legacy - only when unified tracking disabled)
        if not config.UNIFIED_TRACKING_ENABLED:
            if return_display_timer > 0 and not ball_is_off_screen:
                draw_return_counter(canvas, return_display_value, config, x_offset=config.SIDEBAR_WIDTH)

        out.write(canvas)

    cap.release()
    out.release()

    print(f"  Saved (mp4v): {output_path}")

    # Convert to H.264
    h264_path = convert_to_h264(output_path)
    if h264_path:
        return True
    else:
        print(f"  Warning: H.264 conversion failed, keeping mp4v version")
        return True


def convert_to_h264(input_path: Path) -> Optional[Path]:
    """Convert video to H.264 codec using ffmpeg."""
    temp_path = input_path.parent / f"{input_path.stem}_h264_temp.mp4"

    print(f"  Converting to H.264 for compatibility...")

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(temp_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and temp_path.exists():
            backup_path = input_path.parent / f"{input_path.stem}_mp4v.mp4"
            input_path.rename(backup_path)
            temp_path.rename(input_path)
            print(f"  Converted to H.264: {input_path}")
            if backup_path.exists():
                backup_path.unlink()
            return input_path
        else:
            print(f"  FFmpeg error: {result.stderr}")
            if temp_path.exists():
                temp_path.unlink()
            return None

    except FileNotFoundError:
        print(f"  FFmpeg not found - keeping mp4v format")
        return None
    except subprocess.TimeoutExpired:
        print(f"  FFmpeg timeout (>5min) - keeping mp4v format")
        if temp_path.exists():
            temp_path.unlink()
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
    """Get list of videos with matching parquet data."""
    available = []

    for parquet_path in sorted(parquet_dir.iterdir()):
        if not parquet_path.is_dir():
            continue

        base_name = parquet_path.name

        # Check for required parquet files
        cone_files = list(parquet_path.glob("*_cone.parquet"))
        ball_files = list(parquet_path.glob("*_football.parquet"))
        pose_files = list(parquet_path.glob("*_pose.parquet"))

        if not (cone_files and ball_files and pose_files):
            continue

        # Look for matching video (try .MOV first, then .mp4 for 720p)
        video_path = videos_dir / f"{base_name}.MOV"
        if not video_path.exists():
            video_path = videos_dir / f"{base_name}.mp4"
        if video_path.exists():
            available.append((base_name, video_path, parquet_path))

    return available


def main():
    parser = argparse.ArgumentParser(
        description="Annotate Triple Cone drill videos with debug visualization"
    )
    parser.add_argument(
        "video_name",
        nargs="?",
        help="Name of video to process (partial match supported)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available videos"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process ALL available videos"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have annotated output"
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(__file__).parent.parent / "videos",
        help="Directory containing source videos"
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path(__file__).parent.parent / "video_detection_pose_ball_cones",
        help="Directory containing parquet data folders"
    )
    parser.add_argument(
        "--resolution", "-r",
        choices=["original", "720p"],
        default="720p",
        help="Which resolution to process (720p uses downscaled videos/parquets)"
    )

    args = parser.parse_args()

    # Override directories for 720p resolution
    if args.resolution == "720p":
        base_dir = Path(__file__).parent.parent
        args.videos_dir = base_dir / "videos_720p"
        args.parquet_dir = base_dir / "video_detection_pose_ball_cones_720p"
        print(f"\n[720p MODE] Using downscaled videos and parquets\n")

    available = get_available_videos(args.videos_dir, args.parquet_dir)

    if args.list:
        print("\n Available Triple Cone videos:\n")
        for name, video_path, parquet_path in available:
            output_path = parquet_path / f"{name}_triple_cone.mp4"
            status = "[done]" if output_path.exists() else "[    ]"
            print(f"  {status} {name}")
        print(f"\nTotal: {len(available)} videos")
        return 0

    if args.all:
        print(f"\n Processing ALL {len(available)} Triple Cone videos...\n")

        success_count = 0
        skip_count = 0
        fail_count = 0

        for i, (name, video_path, parquet_path) in enumerate(available, 1):
            output_path = parquet_path / f"{name}_triple_cone.mp4"

            if args.skip_existing and output_path.exists():
                print(f"[{i}/{len(available)}] Skipping {name} (already exists)")
                skip_count += 1
                continue

            print(f"\n{'='*60}")
            print(f"[{i}/{len(available)}] Processing: {name}")
            print(f"{'='*60}")

            # Create fresh config for each video to avoid cumulative scaling
            config = TripleConeAnnotationConfig()
            success = annotate_triple_cone_video(video_path, parquet_path, output_path, config)

            if success:
                success_count += 1
            else:
                fail_count += 1

        print(f"\n{'='*60}")
        print(f" BATCH COMPLETE")
        print(f"{'='*60}")
        print(f"  Processed: {success_count}")
        print(f"  Skipped:   {skip_count}")
        print(f"  Failed:    {fail_count}")

        return 0 if fail_count == 0 else 1

    if not args.video_name:
        print("Error: Please specify a video name, use --list, or use --all")
        return 1

    # Find matching video (partial match - flexible with spaces/underscores)
    to_process = None
    search_term = args.video_name.lower()
    for name, video_path, parquet_path in available:
        name_lower = name.lower()
        # Match if search term is in name (with either space or underscore)
        if search_term in name_lower or search_term.replace(" ", "_") in name_lower or search_term.replace("_", " ") in name_lower:
            to_process = (name, video_path, parquet_path)
            break

    if not to_process:
        print(f"Error: Video not found matching: {args.video_name}")
        print("Use --list to see available videos")
        return 1

    name, video_path, parquet_path = to_process

    print(f"\n Annotating {name}...\n")

    output_path = parquet_path / f"{name}_triple_cone.mp4"
    config = TripleConeAnnotationConfig()
    success = annotate_triple_cone_video(video_path, parquet_path, output_path, config)

    if success:
        print(f"\n Done! Output: {output_path}")
        return 0
    else:
        print("\n Annotation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
