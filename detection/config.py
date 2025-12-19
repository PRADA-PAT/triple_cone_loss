"""
Configuration module for Figure-8 Ball Control Detection System.

Defines all configuration parameters using dataclasses for type safety
and easy modification.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum


class DetectionMode(Enum):
    """Detection sensitivity modes."""
    STANDARD = "standard"
    STRICT = "strict"
    LENIENT = "lenient"


@dataclass
class Figure8DrillConfig:
    """
    Figure-8 drill setup parameters.

    Cone arrangement (left to right on horizontal line):
    [START] ---- [PAIR1_L] [PAIR1_R] ---- [PAIR2_L] [PAIR2_R]

    - START: Single cone where player starts
    - PAIR1 (Gate 1): First pair of cones with gap between them
    - PAIR2 (Gate 2): Second pair of cones with gap between them

    Player pattern: Start → G1 → G2 → Turn → G2 → G1 → (repeat)
    """
    expected_cone_count: int = 5  # 1 start + 2 pair1 + 2 pair2
    cone_layout: str = "figure_8"

    # Cone role mapping (auto-detected from positions or manually set)
    # These are populated by analyze_cone_positions()
    start_cone_id: Optional[int] = None
    gate1_cone_ids: Optional[Tuple[int, int]] = None  # First pair
    gate2_cone_ids: Optional[Tuple[int, int]] = None  # Second pair

    # Gate definitions (populated after cone role detection)
    gate_definitions: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "G1": None,  # Will be set to gate1_cone_ids
            "G2": None,  # Will be set to gate2_cone_ids
        }
    )

    # Geometry thresholds
    gate_width_threshold: float = 200.0  # Max distance to be "within gate"
    gate_passage_margin: float = 50.0    # Extra margin around gate center

    # Expected drill pattern for validation
    forward_gate_sequence: List[str] = field(
        default_factory=lambda: ["G1", "G2"]
    )
    backward_gate_sequence: List[str] = field(
        default_factory=lambda: ["G2", "G1"]
    )

    # Clustering thresholds for auto-detection
    pair_distance_threshold: float = 300.0  # Max distance for cones to be a "pair"

    def update_gate_definitions(self):
        """Update gate_definitions based on detected cone IDs."""
        if self.gate1_cone_ids:
            self.gate_definitions["G1"] = self.gate1_cone_ids
        if self.gate2_cone_ids:
            self.gate_definitions["G2"] = self.gate2_cone_ids


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
    show_gate_zones: bool = True  # Figure-8 specific

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
    """Main application configuration for Figure-8 drill."""
    drill: Figure8DrillConfig = field(default_factory=Figure8DrillConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    fps: float = 30.0
    verbose: bool = False

    @classmethod
    def for_figure8(cls) -> 'AppConfig':
        """Create config for Figure-8 drill (default)."""
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
