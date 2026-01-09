"""
Configuration module for Ball Control Detection System.

Supports:
- Triple Cone drills (3 cones in horizontal line)

Defines all configuration parameters using dataclasses for type safety
and easy modification.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


class DrillType(Enum):
    """Type of drill being analyzed."""
    TRIPLE_CONE = "triple_cone"


class DetectionMode(Enum):
    """Detection sensitivity modes."""
    STANDARD = "standard"
    STRICT = "strict"
    LENIENT = "lenient"


@dataclass
class TripleConeDrillConfig:
    """
    Triple Cone drill setup parameters.

    Cone arrangement (horizontal line, left to right):
    [CONE1/LEFT/HOME] ---- [CONE2/CENTER] ---- [CONE3/RIGHT]

    Drill pattern (one repetition):
    CONE1 → CONE2(turn) → CONE1(turn) → CONE3(turn) → CONE1(turn) → repeat

    All 3 cones are turn points where player makes tight turns.
    """
    expected_cone_count: int = 3
    cone_layout: str = "triple_cone"

    # Mean cone positions from parquet analysis (pixels)
    # These are populated by analyzing cone parquet data
    cone1_position: Optional[Tuple[float, float]] = None  # LEFT/HOME (px, py)
    cone2_position: Optional[Tuple[float, float]] = None  # CENTER (px, py)
    cone3_position: Optional[Tuple[float, float]] = None  # RIGHT (px, py)

    # Turning zone configuration
    zone_radius: float = 150.0  # Base radius for all turning zones (pixels)
    zone_stretch_x: float = 1.0  # Horizontal stretch factor
    zone_stretch_y: float = 5.0  # Vertical compression for side-view camera

    # Expected cone spacing (from analysis: ~926px between adjacent cones)
    expected_cone_spacing: float = 926.0  # pixels
    cone_spacing_tolerance: float = 50.0  # pixels

    # Cone detection from parquet
    cone_y_tolerance: float = 30.0  # Max Y deviation for horizontal line validation

    def set_cone_positions(
        self,
        cone1: Tuple[float, float],
        cone2: Tuple[float, float],
        cone3: Tuple[float, float]
    ) -> None:
        """Set cone positions from analysis."""
        self.cone1_position = cone1
        self.cone2_position = cone2
        self.cone3_position = cone3

    def validate_layout(self) -> bool:
        """Validate that cones form expected horizontal line pattern."""
        if not all([self.cone1_position, self.cone2_position, self.cone3_position]):
            return False

        # Check Y coordinates are roughly aligned (horizontal line)
        y_coords = [self.cone1_position[1], self.cone2_position[1], self.cone3_position[1]]
        y_range = max(y_coords) - min(y_coords)
        if y_range > self.cone_y_tolerance:
            return False

        # Check X ordering (left to right)
        if not (self.cone1_position[0] < self.cone2_position[0] < self.cone3_position[0]):
            return False

        return True


@dataclass
class DetectionConfig:
    """Ball control detection thresholds."""
    # Ball-foot proximity (field units)
    control_radius: float = 120.0
    loss_distance_threshold: float = 200.0
    loss_duration_frames: int = 5

    # Velocity thresholds
    high_velocity_threshold: float = 50.0
    loss_velocity_spike: float = 100.0

    # Control scoring
    min_control_score: float = 0.45

    mode: DetectionMode = DetectionMode.STANDARD

    # Intention-based (face direction) detection settings
    # NOTE: Must match thresholds in video/annotate_triple_cone.py
    use_intention_detection: bool = True  # Enable face-direction-based detection
    nose_hip_facing_threshold: float = 15.0  # Min nose-hip X diff for facing direction
    intention_sustained_frames: int = 10  # Frames to confirm intention-based loss


@dataclass
class PathConfig:
    """
    File path configuration.

    Primary inputs:
    - cone_json: Required - JSON file with annotated cone positions and roles
    - football_parquet: Required - Ball detection parquet
    - pose_parquet: Required - Pose keypoints parquet

    Optional inputs:
    - cone_parquet: Optional - Only for visualization
    - video_path: Optional - For debug video generation
    """
    # Required inputs
    cone_json: Optional[Path] = None  # cone_annotations.json (required)
    football_parquet: Optional[Path] = None
    pose_parquet: Optional[Path] = None

    # Optional inputs
    cone_parquet: Optional[Path] = None  # Optional, for visualization only
    video_path: Optional[Path] = None

    # Outputs
    output_dir: Optional[Path] = None
    output_csv: Optional[Path] = None
    output_video: Optional[Path] = None


@dataclass
class VisualizationConfig:
    """Visualization settings (debug only)."""
    show_ball_trajectory: bool = True
    show_player_trajectory: bool = True
    show_cone_positions: bool = True
    show_event_markers: bool = True
    show_metrics_overlay: bool = True
    show_gate_zones: bool = True  # Triple Cone specific

    # Colors (BGR for OpenCV)
    ball_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    player_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    cone_color: Tuple[int, int, int] = (0, 165, 255)  # Orange
    loss_event_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    gate_color: Tuple[int, int, int] = (255, 0, 255)  # Magenta
    start_cone_color: Tuple[int, int, int] = (0, 255, 0)  # Green

    trail_length: int = 30
    output_fps: float = 30.0


@dataclass
class AppConfig:
    """
    Main application configuration for Triple Cone drill.

    Triple cone drill has 3 cones in a horizontal line:
    [CONE1/LEFT/HOME] ---- [CONE2/CENTER] ---- [CONE3/RIGHT]

    All 3 cones are turn points where player makes tight turns.
    """
    drill: TripleConeDrillConfig = field(default_factory=TripleConeDrillConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    fps: float = 30.0
    verbose: bool = False
    drill_type: DrillType = DrillType.TRIPLE_CONE

    @classmethod
    def for_triple_cone(cls) -> 'AppConfig':
        """Create config for Triple Cone drill (default)."""
        return cls(drill_type=DrillType.TRIPLE_CONE)

    @classmethod
    def default(cls) -> 'AppConfig':
        """Create default config for Triple Cone drill."""
        return cls()

    @classmethod
    def with_strict_detection(cls) -> 'AppConfig':
        """Create config with strict detection thresholds."""
        config = cls()
        config.detection.mode = DetectionMode.STRICT
        config.detection.min_control_score = 0.55
        config.detection.loss_distance_threshold = 150.0
        return config

    @classmethod
    def with_lenient_detection(cls) -> 'AppConfig':
        """Create config with lenient detection thresholds."""
        config = cls()
        config.detection.mode = DetectionMode.LENIENT
        config.detection.min_control_score = 0.35
        config.detection.loss_distance_threshold = 250.0
        return config

    def set_cone_positions_from_parquet(
        self,
        cone1: Tuple[float, float],
        cone2: Tuple[float, float],
        cone3: Tuple[float, float]
    ) -> None:
        """Set cone positions from parquet analysis."""
        self.drill.set_cone_positions(cone1, cone2, cone3)
