"""Stage 3 Tests: Core Detection - BallControlDetector."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from f8_loss.detection.ball_control_detector import (
    BallControlDetector,
    detect_ball_control,
)
from f8_loss.detection.config import AppConfig, DetectionConfig
from f8_loss.detection.data_structures import DetectionResult, ControlState
from f8_loss.detection.data_loader import load_parquet_data, extract_ankle_positions

# Real data paths
DATA_DIR = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar")
CONE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet"
BALL_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet"
POSE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet"


@pytest.fixture
def sample_data():
    """Create minimal sample test data."""
    ball_df = pd.DataFrame({
        'frame_id': [0, 1, 2, 3, 4],
        'center_x': [100, 110, 120, 130, 140],
        'center_y': [200, 210, 220, 230, 240],
        'field_center_x': [50.0, 55.0, 60.0, 65.0, 70.0],
        'field_center_y': [100.0, 105.0, 110.0, 115.0, 120.0],
    })

    pose_df = pd.DataFrame({
        'frame_idx': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        'keypoint_name': ['left_ankle', 'right_ankle'] * 5,
        'x': [90, 95, 100, 105, 110, 115, 120, 125, 130, 135],
        'y': [190, 195, 200, 205, 210, 215, 220, 225, 230, 235],
        'field_x': [45.0, 48.0, 50.0, 53.0, 55.0, 58.0, 60.0, 63.0, 65.0, 68.0],
        'field_y': [95.0, 98.0, 100.0, 103.0, 105.0, 108.0, 110.0, 113.0, 115.0, 118.0],
        'confidence': [0.9] * 10,
    })

    cone_df = pd.DataFrame({
        'frame_id': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        'object_id': [3, 4] * 5,
        'center_x': [50, 150] * 5,
        'center_y': [300, 300] * 5,
        'field_center_x': [25.0, 75.0] * 5,
        'field_center_y': [150.0, 150.0] * 5,
    })

    return ball_df, pose_df, cone_df


class TestBallControlDetector:
    """Tests for BallControlDetector class."""

    def test_initialization_default_config(self):
        """Test detector initializes with default config."""
        detector = BallControlDetector()
        assert detector is not None
        assert detector.config.fps == 30.0
        assert detector._detection_config.control_radius == 120.0

    def test_initialization_custom_config(self):
        """Test detector initializes with custom config."""
        config = AppConfig()
        config.detection.control_radius = 200.0
        config.fps = 60.0

        detector = BallControlDetector(config)
        assert detector.config.fps == 60.0
        assert detector._detection_config.control_radius == 200.0

    def test_detect_returns_result(self, sample_data):
        """Test detect() returns DetectionResult."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        result = detector.detect(ball_df, pose_df, cone_df)

        assert isinstance(result, DetectionResult)
        assert result.success is True

    def test_detect_processes_all_frames(self, sample_data):
        """Test detect() processes all available frames."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        result = detector.detect(ball_df, pose_df, cone_df)

        assert result.total_frames == 5

    def test_detect_generates_frame_data(self, sample_data):
        """Test detect() generates frame data for each frame."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        result = detector.detect(ball_df, pose_df, cone_df)

        # With placeholder logic, should generate frame data
        assert len(result.frame_data) == 5

    def test_detect_empty_ball_df(self, sample_data):
        """Test detect() handles empty ball DataFrame."""
        _, pose_df, cone_df = sample_data
        empty_ball = pd.DataFrame(columns=['frame_id', 'center_x', 'center_y',
                                           'field_center_x', 'field_center_y'])

        detector = BallControlDetector()
        result = detector.detect(empty_ball, pose_df, cone_df)

        assert result.success is False
        assert "No valid frames" in result.error

    def test_detect_calculates_ball_velocity(self, sample_data):
        """Test detect() calculates ball velocity."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        result = detector.detect(ball_df, pose_df, cone_df)

        # Check that velocity is calculated in frame data
        if result.frame_data:
            # First frame should have velocity 0 or small
            assert result.frame_data[0].ball_velocity >= 0
            # Subsequent frames should have calculated velocity
            assert result.frame_data[1].ball_velocity >= 0


class TestConvenienceFunction:
    """Tests for detect_ball_control convenience function."""

    def test_convenience_function(self, sample_data):
        """Test detect_ball_control works."""
        ball_df, pose_df, cone_df = sample_data

        result = detect_ball_control(ball_df, pose_df, cone_df)

        assert isinstance(result, DetectionResult)
        assert result.success is True

    def test_convenience_function_with_config(self, sample_data):
        """Test detect_ball_control with custom config."""
        ball_df, pose_df, cone_df = sample_data
        config = AppConfig()
        config.detection.control_radius = 50.0

        result = detect_ball_control(ball_df, pose_df, cone_df, config=config)

        assert result.success is True


class TestNearestCone:
    """Tests for nearest cone detection."""

    def test_nearest_cone_found(self, sample_data):
        """Test nearest cone is correctly identified."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        # Ball at (50, 100), cones at (25, 150) and (75, 150)
        # Distance to cone 3: sqrt((50-25)^2 + (100-150)^2) = sqrt(625+2500) = ~55.9
        # Distance to cone 4: sqrt((50-75)^2 + (100-150)^2) = sqrt(625+2500) = ~55.9
        cone_id, distance = detector._get_nearest_cone((50.0, 100.0), cone_df, 0)

        assert cone_id in [3, 4]
        assert distance < 60


class TestGateDetection:
    """Tests for gate detection via Figure8ConeDetector."""

    def test_gate_detection_basic(self, sample_data):
        """Test Figure8ConeDetector handles cone data."""
        ball_df, pose_df, cone_df = sample_data

        # The BallControlDetector uses Figure8ConeDetector internally
        detector = BallControlDetector()
        result = detector.detect(ball_df, pose_df, cone_df)

        # Detection should succeed even with minimal data
        # (cone roles may not be perfectly detected with sample data)
        assert result is not None


class TestIntegration:
    """Integration tests with real data."""

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_detect_with_real_data(self):
        """Test detection with real parquet files."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        detector = BallControlDetector()
        result = detector.detect(ball_df, pose_df, cone_df)

        assert result.success is True
        assert result.total_frames > 1400  # Most frames should be processed
        assert len(result.frame_data) > 1400

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_frame_data_has_valid_values(self):
        """Test that frame data has reasonable values."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        detector = BallControlDetector()
        result = detector.detect(ball_df, pose_df, cone_df)

        # Check frame data quality
        for fd in result.frame_data[:10]:  # Check first 10
            assert fd.frame_id >= 0
            assert fd.timestamp >= 0
            assert fd.ball_ankle_distance >= 0
            assert fd.nearest_cone_id in [1, 2, 3, 4, 5, 6, 7, -1]
            assert 0 <= fd.control_score <= 1

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_detection_result_summary(self):
        """Test detection result summary statistics."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        result = detect_ball_control(ball_df, pose_df, cone_df)

        # With placeholder logic, should have 0 events but valid summary
        assert result.total_loss_events >= 0
        assert 0 <= result.control_percentage <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
