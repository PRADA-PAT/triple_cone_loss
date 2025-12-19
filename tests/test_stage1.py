"""Stage 1 Tests: Foundation - Types and Configuration."""
import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestImports:
    """Test all imports work correctly."""

    def test_import_config_module(self):
        """Test config module imports from detection subpackage."""
        from f8_loss.detection.config import (
            AppConfig,
            Figure8DrillConfig,
            DetectionConfig,
            PathConfig,
            VisualizationConfig,
            DetectionMode,
        )
        assert AppConfig is not None
        assert DetectionMode is not None

    def test_import_data_structures(self):
        """Test data structures imports from detection subpackage."""
        from f8_loss.detection.data_structures import (
            ControlState,
            EventType,
            FrameData,
            LossEvent,
            DetectionResult,
        )
        assert ControlState is not None
        assert FrameData is not None

    def test_import_from_package(self):
        """Test importing from package __init__ (backwards compatibility)."""
        from f8_loss import (
            AppConfig,
            FrameData,
            LossEvent,
            DetectionResult,
            ControlState,
        )
        assert AppConfig is not None

    def test_import_from_detection_subpackage(self):
        """Test importing from detection subpackage."""
        from f8_loss.detection import (
            BallControlDetector,
            Figure8ConeDetector,
            AppConfig,
            ControlState,
        )
        assert BallControlDetector is not None
        assert Figure8ConeDetector is not None


class TestConfigDataclasses:
    """Test configuration dataclasses."""

    def test_app_config_defaults(self):
        """Test AppConfig has sensible defaults."""
        from f8_loss.detection.config import AppConfig

        config = AppConfig()
        assert config.fps == 30.0
        assert config.verbose is False
        assert config.drill.expected_cone_count == 5  # Figure-8 has 5 cones
        assert config.detection.control_radius == 120.0

    def test_drill_config_gates(self):
        """Test Figure8DrillConfig gate definitions."""
        from f8_loss.detection.config import Figure8DrillConfig

        drill = Figure8DrillConfig()
        assert drill.expected_cone_count == 5

    def test_detection_config_thresholds(self):
        """Test DetectionConfig threshold values."""
        from f8_loss.detection.config import DetectionConfig, DetectionMode

        config = DetectionConfig()
        assert config.loss_distance_threshold == 200.0
        assert config.mode == DetectionMode.STANDARD

    def test_visualization_config_colors(self):
        """Test VisualizationConfig color tuples."""
        from f8_loss.detection.config import VisualizationConfig

        config = VisualizationConfig()
        assert len(config.ball_color) == 3  # BGR tuple
        assert config.trail_length == 30


class TestDataStructures:
    """Test data structure classes."""

    def test_control_state_enum(self):
        """Test ControlState enum values."""
        from f8_loss.detection.data_structures import ControlState

        assert ControlState.CONTROLLED.value == "controlled"
        assert ControlState.LOST.value == "lost"
        assert len(list(ControlState)) == 5

    def test_event_type_enum(self):
        """Test EventType enum values."""
        from f8_loss.detection.data_structures import EventType

        assert EventType.LOSS_DISTANCE.value == "loss_distance"
        assert EventType.RECOVERY.value == "recovery"

    def test_frame_data_creation(self):
        """Test FrameData instantiation."""
        from f8_loss.detection.data_structures import FrameData, ControlState

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
        from f8_loss.detection.data_structures import LossEvent, EventType

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
        from f8_loss.detection.data_structures import (
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
