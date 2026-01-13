# Detection module - Loss of control calculation logic
"""
Core detection logic for ball control analysis.

Supports:
- Triple Cone drills (3 cones in horizontal line)

This module contains:
- BallControlDetector: Main detection engine with detect_loss() method
- TripleConeDetector: 3-cone phase tracking and turn detection
- Data structures: ControlState, LossEvent, FrameData, etc.
- Configuration: AppConfig, DetectionConfig, TripleConeDrillConfig
- Data loading: load_triple_cone_layout_from_parquet, load_parquet_data
- Export: CSVExporter
- Turning zones: TripleConeZoneSet (3 zones for CONE1, CONE2, CONE3)
"""

from .ball_control_detector import BallControlDetector, detect_ball_control
from .triple_cone_detector import TripleConeDetector, TripleConeConeDetector, TurnEvent, DrillState
from .data_structures import (
    # Core enums and states
    ControlState, EventType, DrillDirection, BallTrackingState,
    # Triple Cone structures (3-cone)
    TripleConeDrillPhase, TripleConeLayout,
    # Common structures
    FrameData, LossEvent, DetectionResult
)
from .config import (
    # Triple Cone config
    AppConfig, DetectionConfig, TripleConeDrillConfig, DrillType,
    # Common config
    PathConfig, VisualizationConfig, DetectionMode
)
from .data_loader import (
    # 3-cone loading (parquet only)
    load_triple_cone_layout_from_parquet,
    # Parquet loading
    load_parquet_data, load_all_data,
    extract_ankle_positions, get_closest_ankle_per_frame,
    validate_data_alignment, get_frame_data, get_video_fps,
)
from .csv_exporter import CSVExporter, export_to_csv
from .turning_zones import (
    # Base zone class
    TurningZone,
    # 3-cone zones
    TripleConeZoneConfig, TripleConeZoneSet, create_triple_cone_zones,
    # Drawing functions
    draw_turning_zone, draw_triple_cone_zones,
    # Zone colors
    ZONE_HIGHLIGHT_COLOR,
    CONE1_ZONE_COLOR, CONE2_ZONE_COLOR, CONE3_ZONE_COLOR,
)

__all__ = [
    # Detector classes
    'BallControlDetector', 'detect_ball_control',
    'TripleConeDetector', 'TripleConeConeDetector', 'TurnEvent', 'DrillState',
    # Data structures - Core
    'ControlState', 'EventType', 'DrillDirection', 'BallTrackingState',
    'FrameData', 'LossEvent', 'DetectionResult',
    # Data structures - Triple Cone (3-cone)
    'TripleConeDrillPhase', 'TripleConeLayout',
    # Configuration
    'AppConfig', 'TripleConeDrillConfig', 'DrillType',
    'DetectionConfig', 'PathConfig', 'VisualizationConfig', 'DetectionMode',
    # Data loading (parquet only)
    'load_triple_cone_layout_from_parquet',
    'load_parquet_data', 'load_all_data',
    'extract_ankle_positions', 'get_closest_ankle_per_frame',
    'validate_data_alignment', 'get_frame_data', 'get_video_fps',
    # Export
    'CSVExporter', 'export_to_csv',
    # Turning zones
    'TurningZone',
    'TripleConeZoneConfig', 'TripleConeZoneSet', 'create_triple_cone_zones',
    # Drawing functions
    'draw_turning_zone', 'draw_triple_cone_zones',
    # Zone colors
    'ZONE_HIGHLIGHT_COLOR',
    'CONE1_ZONE_COLOR', 'CONE2_ZONE_COLOR', 'CONE3_ZONE_COLOR',
]
