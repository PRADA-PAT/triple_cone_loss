"""
Configuration for Triple Cone video annotation.

Contains all configuration parameters, colors, thresholds, and skeleton constants.
"""

import math
from dataclasses import dataclass
from typing import Tuple


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
    FONT_SCALE: float = 0.5
    FONT_THICKNESS: int = 1

    # Data source settings
    USE_POSTPROCESSED_BALL: bool = True  # Use _pp columns (smoothed/stabilized) vs raw

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

    # Ball vertical deviation detection (for detecting ball going up/down instead of left/right)
    DETECT_BALL_VERTICAL_DEVIATION: bool = True
    VERTICAL_DEVIATION_THRESHOLD: float = 30.0  # Degrees from horizontal (sensitive - catches slight deviations)
    VERTICAL_DEVIATION_SUSTAINED_FRAMES: int = 10  # ~0.33s at 30fps
    VERTICAL_DEVIATION_MIN_SPEED: float = 5.0  # Min speed to detect (ignore stationary)

    # Vertical deviation visualization
    VERTICAL_DEVIATION_UP_COLOR: Tuple[int, int, int] = (255, 0, 255)    # Magenta for UP
    VERTICAL_DEVIATION_DOWN_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Yellow for DOWN
    VERTICAL_DEVIATION_PERSIST_COLOR: Tuple[int, int, int] = (0, 165, 255)  # Orange persist
    VERTICAL_DEVIATION_COUNTER_POS_X: int = 50
    VERTICAL_DEVIATION_COUNTER_POS_Y: int = 350  # Below other counters
    VERTICAL_DEVIATION_COUNTER_FONT_SCALE: float = 1.2
    VERTICAL_DEVIATION_PERSIST_SECONDS: float = 3.0

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


# Color palette for N-cone drills (BGR format for OpenCV)
# Used by annotate_video.py for generic multi-drill support
CONE_COLOR_PALETTE = [
    (200, 200, 0),    # Teal (index 0)
    (200, 100, 200),  # Purple (index 1)
    (100, 200, 200),  # Orange (index 2)
    (0, 200, 0),      # Green (index 3)
    (200, 0, 200),    # Magenta (index 4)
    (0, 200, 200),    # Yellow (index 5)
    (200, 100, 100),  # Light blue (index 6)
    (100, 100, 200),  # Salmon (index 7)
    (150, 200, 100),  # Lime (index 8)
    (100, 150, 200),  # Peach (index 9)
]


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


def scale_config_for_resolution(config: TripleConeAnnotationConfig, video_width: int) -> None:
    """
    Scale config values based on video resolution.

    Uses 1920px (Full HD) as reference. Linear scaling for positions,
    sqrt scaling for fonts (gentler, stays readable).
    """
    REFERENCE_WIDTH = 1920
    resolution_scale = video_width / REFERENCE_WIDTH

    if abs(resolution_scale - 1.0) <= 0.01:
        return  # No scaling needed

    font_scale = math.sqrt(resolution_scale)

    print(f"  [AUTO-SCALE] Detected {video_width}px width")
    print(f"    Position scale: {resolution_scale:.3f}, Font scale: {font_scale:.3f}")

    # Store scale factors
    config.RESOLUTION_SCALE = resolution_scale
    config.FONT_SCALE_FACTOR = font_scale

    # Scale pixel-based thresholds (linear)
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
    config.VERTICAL_DEVIATION_COUNTER_POS_X = int(config.VERTICAL_DEVIATION_COUNTER_POS_X * resolution_scale)
    config.VERTICAL_DEVIATION_COUNTER_POS_Y = int(config.VERTICAL_DEVIATION_COUNTER_POS_Y * resolution_scale)

    # Scale sidebar dimensions
    config.SIDEBAR_WIDTH = max(200, int(config.SIDEBAR_WIDTH * resolution_scale))
    config.SIDEBAR_LINE_HEIGHT = max(18, int(config.SIDEBAR_LINE_HEIGHT * font_scale))
    config.SIDEBAR_PADDING = max(10, int(config.SIDEBAR_PADDING * font_scale))

    # Scale font sizes (sqrt scaling)
    config.SIDEBAR_FONT_SCALE = max(0.40, config.SIDEBAR_FONT_SCALE * font_scale)
    config.FONT_SCALE = max(0.35, config.FONT_SCALE * font_scale)
    config.BEHIND_COUNTER_FONT_SCALE *= font_scale
    config.EDGE_COUNTER_FONT_SCALE *= font_scale
    config.OFF_SCREEN_FONT_SCALE *= font_scale
    config.RETURN_COUNTER_FONT_SCALE *= font_scale
    config.VERTICAL_DEVIATION_COUNTER_FONT_SCALE *= font_scale

    # Scale line/arrow thicknesses
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
    config.KEYPOINT_RADIUS = max(2, int(config.KEYPOINT_RADIUS * resolution_scale))
