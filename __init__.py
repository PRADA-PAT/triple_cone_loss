"""
Ball Control Detection System for Triple Cone Drills.

A modular system for detecting when a player loses control of the ball
during Triple Cone drill exercises.

Package Structure:
    triple_cone_loss/
    ├── detection/     # Loss of control calculation logic
    ├── annotation/    # Cone annotation and visualization tools
    └── video/         # Video generation with loss events marked

Cone Setup (3-cone layout):
    [CONE1/HOME] ---- [CONE2/CENTER] ---- [CONE3/RIGHT]

    - CONE1 (HOME): Where player starts and returns
    - CONE2 (CENTER): Middle turning cone
    - CONE3 (RIGHT): Far right turning cone

Drill Pattern:
    CONE1 → CONE2(turn) → CONE1(turn) → CONE3(turn) → CONE1(turn) → repeat

Quick Start:
    from triple_cone_loss import detect_ball_control, load_parquet_data, export_to_csv

    # Load data
    ball_df = load_parquet_data("ball.parquet")
    pose_df = load_parquet_data("pose.parquet")
    cone_df = load_parquet_data("cone.parquet")

    # Detect
    result = detect_ball_control(ball_df, pose_df, cone_df)

    # Export
    export_to_csv(result, "output.csv")

Classes:
    - AppConfig: Main configuration container
    - TripleConeDrillConfig: Triple Cone specific drill settings
    - BallControlDetector: Core detection class
    - TripleConeDetector: 3-cone phase tracking and turn detection
    - CSVExporter: CSV export functionality

Data Structures:
    - FrameData: Per-frame analysis data
    - LossEvent: A detected loss-of-control event
    - DetectionResult: Complete detection output
    - ControlState: Ball control state enum
    - TripleConeDrillPhase: Current phase in drill (AT_CONE1, AT_CONE2, etc.)
    - DrillDirection: Forward/backward direction
    - TripleConeLayout: 3-cone positions
"""

# =============================================================================
# Re-export from detection module
# =============================================================================

# Configuration
from .detection.config import (
    AppConfig,
    TripleConeDrillConfig,
    DetectionConfig,
    PathConfig,
    VisualizationConfig,
    DetectionMode,
)

# Data structures
from .detection.data_structures import (
    ControlState,
    EventType,
    FrameData,
    LossEvent,
    DetectionResult,
    TripleConeDrillPhase,
    DrillDirection,
    TripleConeLayout,
)

# Data loading
from .detection.data_loader import (
    load_parquet_data,
    load_all_data,
    extract_ankle_positions,
    get_closest_ankle_per_frame,
    validate_data_alignment,
    ANKLE_KEYPOINTS,
    # 3-cone loading
    load_triple_cone_layout_from_parquet,
    load_triple_cone_annotations,
    EXPECTED_CONE_ROLES,
    # Video metadata
    get_video_fps,
)

# Triple Cone detection
from .detection.triple_cone_detector import TripleConeDetector, TurnEvent, DrillState

# Detection
from .detection.ball_control_detector import (
    BallControlDetector,
    detect_ball_control,
)

# Export
from .detection.csv_exporter import (
    CSVExporter,
    export_to_csv,
)

# Turning zones
from .detection.turning_zones import (
    TurningZone,
    TripleConeZoneConfig,
    TripleConeZoneSet,
    create_triple_cone_zones,
    draw_turning_zone,
    draw_triple_cone_zones,
    CONE1_ZONE_COLOR,
    CONE2_ZONE_COLOR,
    CONE3_ZONE_COLOR,
    ZONE_HIGHLIGHT_COLOR,
)

# =============================================================================
# Re-export from annotation module (optional - handles missing OpenCV)
# =============================================================================
try:
    from .annotation.drill_visualizer import DrillVisualizer
    from .annotation.cone_annotator import ConeAnnotator
    _HAS_VISUALIZER = True
except ImportError:
    _HAS_VISUALIZER = False
    DrillVisualizer = None
    ConeAnnotator = None

# =============================================================================
# Re-export from video module (optional - handles missing OpenCV)
# =============================================================================
try:
    from .video.annotate_with_json_cones import (
        annotate_video_with_json_cones,
        convert_to_h264,
        get_available_videos,
    )
    from .video.annotate_videos import annotate_video
    from .video.annotate_triple_cone import annotate_triple_cone_video
    _HAS_VIDEO = True
except ImportError:
    _HAS_VIDEO = False
    annotate_video_with_json_cones = None
    convert_to_h264 = None
    get_available_videos = None
    annotate_video = None
    annotate_triple_cone_video = None


__version__ = "0.4.0"  # Major update: 3-cone architecture

__all__ = [
    # Version
    '__version__',
    # Config
    'AppConfig',
    'TripleConeDrillConfig',
    'DetectionConfig',
    'PathConfig',
    'VisualizationConfig',
    'DetectionMode',
    # Data structures
    'ControlState',
    'EventType',
    'FrameData',
    'LossEvent',
    'DetectionResult',
    'TripleConeDrillPhase',
    'DrillDirection',
    'TripleConeLayout',
    # Data loading
    'load_parquet_data',
    'load_all_data',
    'extract_ankle_positions',
    'get_closest_ankle_per_frame',
    'validate_data_alignment',
    'ANKLE_KEYPOINTS',
    # 3-cone loading
    'load_triple_cone_layout_from_parquet',
    'load_triple_cone_annotations',
    'EXPECTED_CONE_ROLES',
    # Video metadata
    'get_video_fps',
    # Triple Cone detection
    'TripleConeDetector',
    'TurnEvent',
    'DrillState',
    # Detection
    'BallControlDetector',
    'detect_ball_control',
    # Export
    'CSVExporter',
    'export_to_csv',
    # Turning zones
    'TurningZone',
    'TripleConeZoneConfig',
    'TripleConeZoneSet',
    'create_triple_cone_zones',
    'draw_turning_zone',
    'draw_triple_cone_zones',
    'CONE1_ZONE_COLOR',
    'CONE2_ZONE_COLOR',
    'CONE3_ZONE_COLOR',
    'ZONE_HIGHLIGHT_COLOR',
    # Visualization (optional)
    'DrillVisualizer',
    'ConeAnnotator',
    # Video (optional)
    'annotate_video_with_json_cones',
    'convert_to_h264',
    'get_available_videos',
    'annotate_video',
    'annotate_triple_cone_video',
]
