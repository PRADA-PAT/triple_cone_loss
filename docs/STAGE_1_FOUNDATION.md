# STAGE 1: Foundation (Types & Config)

**Duration**: ~30 minutes
**Prerequisites**: None (first stage)
**Outcome**: Core type system and configuration ready

---

## Project Context

**Project Path**: `/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/`

**Purpose**: Ball Control Detection System for Figure-8 cone drills - detecting when a player loses control of the ball.

**Key Design Decisions**:
- Flat structure (not nested package)
- Ankle keypoints only for ball-foot distance (more stable than toe tracking)
- Modular design with placeholders for detection logic

---

## Files to Create

```
f8_loss/
├── __init__.py              # Stub (completed in Stage 4)
├── config.py                # ← CREATE THIS
├── data_structures.py       # ← CREATE THIS
├── requirements.txt         # ← CREATE THIS
└── tests/
    ├── __init__.py          # ← CREATE THIS (empty)
    └── test_stage1.py       # ← CREATE THIS
```

---

## 1. Create Directory Structure

```bash
mkdir -p "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/tests"
touch "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/__init__.py"
touch "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/tests/__init__.py"
```

---

## 2. requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
opencv-python>=4.8.0
pyarrow>=12.0.0
tqdm>=4.65.0
pytest>=7.0.0
```

---

## 3. config.py - Configuration Module

```python
"""
Configuration module for Ball Control Detection System.

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
class DrillConfig:
    """Figure-8 drill setup parameters."""
    expected_cone_count: int = 7
    cone_layout: str = "horizontal"

    # Cone order left to right (by object_id)
    cone_order_left_to_right: List[int] = field(
        default_factory=lambda: [7, 6, 5, 4, 3, 2, 1]
    )

    # Gate definitions (pairs of cone IDs forming gates)
    gate_definitions: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "G1": (7, 6),
            "G2": (6, 5),
            "G3": (5, 4),
            "G4": (4, 3),
            "G5": (3, 2),
            "G6": (2, 1),
        }
    )


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
    """File path configuration."""
    cone_parquet: Optional[Path] = None
    football_parquet: Optional[Path] = None
    pose_parquet: Optional[Path] = None
    video_path: Optional[Path] = None
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

    # Colors (BGR for OpenCV)
    ball_color: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    player_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    cone_color: Tuple[int, int, int] = (0, 165, 255)  # Orange
    loss_event_color: Tuple[int, int, int] = (0, 0, 255)  # Red

    trail_length: int = 30
    output_fps: float = 30.0


@dataclass
class AppConfig:
    """Main application configuration."""
    drill: DrillConfig = field(default_factory=DrillConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    fps: float = 30.0
    verbose: bool = False
```

---

## 4. data_structures.py - Data Models

```python
"""
Data structures for Ball Control Detection System.

Defines all data models for detection results including:
- ControlState: Ball control state machine
- EventType: Types of loss events
- FrameData: Per-frame analysis data
- LossEvent: A detected loss-of-control event
- DetectionResult: Complete detection output
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class ControlState(Enum):
    """Ball control states."""
    CONTROLLED = "controlled"
    TRANSITION = "transition"
    LOST = "lost"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class EventType(Enum):
    """Loss-of-control event types."""
    LOSS_DISTANCE = "loss_distance"
    LOSS_VELOCITY = "loss_velocity"
    LOSS_DIRECTION = "loss_direction"
    RECOVERY = "recovery"
    BOUNDARY_VIOLATION = "boundary"


@dataclass
class FrameData:
    """Data for a single frame analysis."""
    frame_id: int
    timestamp: float

    # Ball position
    ball_x: float
    ball_y: float
    ball_field_x: float
    ball_field_y: float
    ball_velocity: float

    # Player ankle position (closest)
    ankle_x: float
    ankle_y: float
    ankle_field_x: float
    ankle_field_y: float
    closest_ankle: str  # "left_ankle" or "right_ankle"

    # Context
    nearest_cone_id: int
    nearest_cone_distance: float
    current_gate: Optional[str]

    # Computed metrics
    ball_ankle_distance: float
    control_score: float
    control_state: ControlState

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'ball_x': self.ball_x,
            'ball_y': self.ball_y,
            'ball_field_x': self.ball_field_x,
            'ball_field_y': self.ball_field_y,
            'ball_velocity': self.ball_velocity,
            'ankle_x': self.ankle_x,
            'ankle_y': self.ankle_y,
            'ankle_field_x': self.ankle_field_x,
            'ankle_field_y': self.ankle_field_y,
            'closest_ankle': self.closest_ankle,
            'nearest_cone_id': self.nearest_cone_id,
            'nearest_cone_distance': self.nearest_cone_distance,
            'current_gate': self.current_gate,
            'ball_ankle_distance': self.ball_ankle_distance,
            'control_score': self.control_score,
            'control_state': self.control_state.value,
        }


@dataclass
class LossEvent:
    """A ball control loss event."""
    event_id: int
    event_type: EventType
    start_frame: int
    end_frame: Optional[int]
    start_timestamp: float
    end_timestamp: Optional[float]

    # Position at loss
    ball_position: Tuple[float, float]
    player_position: Tuple[float, float]
    distance_at_loss: float
    velocity_at_loss: float

    # Context
    nearest_cone_id: int
    gate_context: Optional[str]

    # Recovery
    recovered: bool = False
    recovery_frame: Optional[int] = None
    severity: str = "medium"
    notes: str = ""

    @property
    def duration_frames(self) -> int:
        if self.end_frame is None:
            return 0
        return self.end_frame - self.start_frame

    @property
    def duration_seconds(self) -> float:
        if self.end_timestamp is None:
            return 0.0
        return self.end_timestamp - self.start_timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'duration_frames': self.duration_frames,
            'duration_seconds': self.duration_seconds,
            'ball_x': self.ball_position[0],
            'ball_y': self.ball_position[1],
            'player_x': self.player_position[0],
            'player_y': self.player_position[1],
            'distance_at_loss': self.distance_at_loss,
            'velocity_at_loss': self.velocity_at_loss,
            'nearest_cone_id': self.nearest_cone_id,
            'gate_context': self.gate_context,
            'recovered': self.recovered,
            'recovery_frame': self.recovery_frame,
            'severity': self.severity,
            'notes': self.notes,
        }


@dataclass
class DetectionResult:
    """Complete result from ball control detection."""
    success: bool
    total_frames: int
    events: List[LossEvent]
    frame_data: List[FrameData]

    # Summary
    total_loss_events: int = 0
    total_loss_duration_frames: int = 0
    control_percentage: float = 0.0

    error: Optional[str] = None

    def __post_init__(self):
        """Calculate summary statistics."""
        self.total_loss_events = len(self.events)
        self.total_loss_duration_frames = sum(
            e.duration_frames for e in self.events if e.end_frame
        )
        if self.total_frames > 0:
            controlled = self.total_frames - self.total_loss_duration_frames
            self.control_percentage = (controlled / self.total_frames) * 100
```

---

## 5. __init__.py (Stub for Stage 1)

```python
"""
Ball Control Detection System for Figure-8 Cone Drills.
Stage 1: Foundation - Core types and configuration.
"""

from .config import (
    AppConfig,
    DrillConfig,
    DetectionConfig,
    PathConfig,
    VisualizationConfig,
    DetectionMode,
)

from .data_structures import (
    ControlState,
    EventType,
    FrameData,
    LossEvent,
    DetectionResult,
)

__all__ = [
    # Config
    'AppConfig',
    'DrillConfig',
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
]
```

---

## 6. tests/test_stage1.py - Stage 1 Tests

```python
"""Stage 1 Tests: Foundation - Types and Configuration."""
import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    """Test all imports work correctly."""

    def test_import_config_module(self):
        """Test config module imports."""
        from f8_loss.config import (
            AppConfig,
            DrillConfig,
            DetectionConfig,
            PathConfig,
            VisualizationConfig,
            DetectionMode,
        )
        assert AppConfig is not None
        assert DetectionMode is not None

    def test_import_data_structures(self):
        """Test data structures imports."""
        from f8_loss.data_structures import (
            ControlState,
            EventType,
            FrameData,
            LossEvent,
            DetectionResult,
        )
        assert ControlState is not None
        assert FrameData is not None

    def test_import_from_package(self):
        """Test importing from package __init__."""
        from f8_loss import (
            AppConfig,
            FrameData,
            LossEvent,
            DetectionResult,
            ControlState,
        )
        assert AppConfig is not None


class TestConfigDataclasses:
    """Test configuration dataclasses."""

    def test_app_config_defaults(self):
        """Test AppConfig has sensible defaults."""
        from f8_loss.config import AppConfig

        config = AppConfig()
        assert config.fps == 30.0
        assert config.verbose is False
        assert config.drill.expected_cone_count == 7
        assert config.detection.control_radius == 120.0

    def test_drill_config_gates(self):
        """Test DrillConfig gate definitions."""
        from f8_loss.config import DrillConfig

        drill = DrillConfig()
        assert len(drill.gate_definitions) == 6
        assert drill.gate_definitions["G1"] == (7, 6)
        assert drill.cone_order_left_to_right == [7, 6, 5, 4, 3, 2, 1]

    def test_detection_config_thresholds(self):
        """Test DetectionConfig threshold values."""
        from f8_loss.config import DetectionConfig, DetectionMode

        config = DetectionConfig()
        assert config.loss_distance_threshold == 200.0
        assert config.mode == DetectionMode.STANDARD

    def test_visualization_config_colors(self):
        """Test VisualizationConfig color tuples."""
        from f8_loss.config import VisualizationConfig

        config = VisualizationConfig()
        assert len(config.ball_color) == 3  # BGR tuple
        assert config.trail_length == 30


class TestDataStructures:
    """Test data structure classes."""

    def test_control_state_enum(self):
        """Test ControlState enum values."""
        from f8_loss.data_structures import ControlState

        assert ControlState.CONTROLLED.value == "controlled"
        assert ControlState.LOST.value == "lost"
        assert len(list(ControlState)) == 5

    def test_event_type_enum(self):
        """Test EventType enum values."""
        from f8_loss.data_structures import EventType

        assert EventType.LOSS_DISTANCE.value == "loss_distance"
        assert EventType.RECOVERY.value == "recovery"

    def test_frame_data_creation(self):
        """Test FrameData instantiation."""
        from f8_loss.data_structures import FrameData, ControlState

        frame = FrameData(
            frame_id=0,
            timestamp=0.0,
            ball_x=100.0,
            ball_y=200.0,
            ball_field_x=50.0,
            ball_field_y=100.0,
            ball_velocity=10.0,
            ankle_x=90.0,
            ankle_y=190.0,
            ankle_field_x=45.0,
            ankle_field_y=95.0,
            closest_ankle="right_ankle",
            nearest_cone_id=4,
            nearest_cone_distance=85.0,
            current_gate="G3",
            ball_ankle_distance=15.0,
            control_score=0.85,
            control_state=ControlState.CONTROLLED,
        )

        assert frame.frame_id == 0
        assert frame.closest_ankle == "right_ankle"

        # Test to_dict
        d = frame.to_dict()
        assert d['frame_id'] == 0
        assert d['control_state'] == "controlled"

    def test_loss_event_creation(self):
        """Test LossEvent instantiation and properties."""
        from f8_loss.data_structures import LossEvent, EventType

        event = LossEvent(
            event_id=1,
            event_type=EventType.LOSS_DISTANCE,
            start_frame=100,
            end_frame=120,
            start_timestamp=3.33,
            end_timestamp=4.0,
            ball_position=(200.0, 150.0),
            player_position=(180.0, 160.0),
            distance_at_loss=25.0,
            velocity_at_loss=45.0,
            nearest_cone_id=3,
            gate_context="G2",
        )

        assert event.duration_frames == 20
        assert abs(event.duration_seconds - 0.67) < 0.01

        # Test to_dict
        d = event.to_dict()
        assert d['event_type'] == "loss_distance"
        assert d['ball_x'] == 200.0

    def test_detection_result_summary(self):
        """Test DetectionResult summary calculation."""
        from f8_loss.data_structures import (
            DetectionResult, LossEvent, EventType
        )

        events = [
            LossEvent(
                event_id=1,
                event_type=EventType.LOSS_DISTANCE,
                start_frame=100,
                end_frame=120,
                start_timestamp=3.33,
                end_timestamp=4.0,
                ball_position=(200.0, 150.0),
                player_position=(180.0, 160.0),
                distance_at_loss=25.0,
                velocity_at_loss=45.0,
                nearest_cone_id=3,
                gate_context="G2",
            )
        ]

        result = DetectionResult(
            success=True,
            total_frames=1000,
            events=events,
            frame_data=[],
        )

        assert result.total_loss_events == 1
        assert result.total_loss_duration_frames == 20
        assert result.control_percentage == 98.0  # (1000-20)/1000 * 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Validation Commands

Run these commands to verify Stage 1 is complete:

```bash
# Navigate to project
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"

# Install dependencies
pip install -r requirements.txt

# Run Stage 1 tests
pytest tests/test_stage1.py -v

# Quick import validation
python -c "
from f8_loss import AppConfig, FrameData, LossEvent, DetectionResult, ControlState
config = AppConfig()
print(f'Config FPS: {config.fps}')
print(f'Cone count: {config.drill.expected_cone_count}')
print(f'Control radius: {config.detection.control_radius}')
print('Stage 1 COMPLETE')
"
```

---

## Ready for Stage 2 Checklist

- [ ] All 4 files created: `config.py`, `data_structures.py`, `requirements.txt`, `__init__.py`
- [ ] `tests/test_stage1.py` passes all tests
- [ ] Can import from `f8_loss` package without errors
- [ ] Dataclasses instantiate with default values
- [ ] Enums have correct values
- [ ] `to_dict()` methods work on FrameData and LossEvent

---

## Next Stage Preview

**Stage 2: Data Layer** will implement:
- `data_loader.py` - Parquet loading and ankle extraction
- Tests with real data from `/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/`

**Context needed for Stage 2**:
- Parquet schema definitions from original guide
- Data file paths
- Ankle keypoint constants: `['left_ankle', 'right_ankle']`
