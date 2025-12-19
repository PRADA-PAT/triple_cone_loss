"""Stage 4 Tests: Export & Public API - CSV export functionality."""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from f8_loss.detection.csv_exporter import CSVExporter, export_to_csv
from f8_loss.detection.data_structures import (
    DetectionResult, LossEvent, FrameData,
    EventType, ControlState
)
from f8_loss import (
    detect_ball_control, load_parquet_data,
    AppConfig, BallControlDetector
)

# Real data paths
DATA_DIR = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar")
CONE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet"
BALL_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet"
POSE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet"


@pytest.fixture
def sample_result():
    """Create sample detection result."""
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
            recovered=True,
            recovery_frame=120,
        )
    ]

    frame_data = [
        FrameData(
            frame_id=i,
            timestamp=i / 30.0,
            ball_x=100.0 + i,
            ball_y=200.0 + i,
            ball_field_x=50.0 + i * 0.5,
            ball_field_y=100.0 + i * 0.5,
            ball_velocity=10.0,
            ankle_x=90.0 + i,
            ankle_y=190.0 + i,
            ankle_field_x=45.0 + i * 0.5,
            ankle_field_y=95.0 + i * 0.5,
            closest_ankle="right_ankle",
            nearest_cone_id=4,
            nearest_cone_distance=85.0,
            current_gate="G3",
            ball_ankle_distance=15.0,
            control_score=0.85,
            control_state=ControlState.CONTROLLED,
        )
        for i in range(10)
    ]

    return DetectionResult(
        success=True,
        total_frames=1000,
        events=events,
        frame_data=frame_data,
    )


@pytest.fixture
def empty_result():
    """Create empty detection result."""
    return DetectionResult(
        success=True,
        total_frames=100,
        events=[],
        frame_data=[],
    )


class TestCSVExporter:
    """Tests for CSVExporter class."""

    def test_export_events_creates_file(self, sample_result, tmp_path):
        """Test export_events creates CSV file."""
        exporter = CSVExporter()
        output_path = tmp_path / "events.csv"

        success = exporter.export_events(sample_result, str(output_path))

        assert success
        assert output_path.exists()

    def test_export_events_has_metadata_header(self, sample_result, tmp_path):
        """Test export_events includes metadata header."""
        exporter = CSVExporter()
        output_path = tmp_path / "events.csv"
        exporter.export_events(sample_result, str(output_path))

        content = output_path.read_text()

        assert "# Ball Control Loss Events" in content
        assert "# Total frames:" in content
        assert "# Total events:" in content
        assert "# Control percentage:" in content

    def test_export_events_correct_columns(self, sample_result, tmp_path):
        """Test export_events has correct columns."""
        exporter = CSVExporter()
        output_path = tmp_path / "events.csv"
        exporter.export_events(sample_result, str(output_path))

        # Read CSV skipping comment lines
        df = pd.read_csv(output_path, comment='#')

        expected_cols = ['event_id', 'event_type', 'start_frame', 'end_frame']
        for col in expected_cols:
            assert col in df.columns

    def test_export_empty_events(self, empty_result, tmp_path):
        """Test export_events handles empty event list."""
        exporter = CSVExporter()
        output_path = tmp_path / "events.csv"

        success = exporter.export_events(empty_result, str(output_path))

        assert success
        assert output_path.exists()

        # Should have headers but no data rows
        df = pd.read_csv(output_path, comment='#')
        assert len(df) == 0
        assert 'event_id' in df.columns

    def test_export_frame_analysis(self, sample_result, tmp_path):
        """Test export_frame_analysis creates CSV."""
        exporter = CSVExporter()
        output_path = tmp_path / "frames.csv"

        success = exporter.export_frame_analysis(sample_result, str(output_path))

        assert success
        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 10  # 10 frame_data entries
        assert 'frame_id' in df.columns
        assert 'control_score' in df.columns

    def test_export_frame_analysis_empty(self, empty_result, tmp_path):
        """Test export_frame_analysis handles empty frame data."""
        exporter = CSVExporter()
        output_path = tmp_path / "frames.csv"

        success = exporter.export_frame_analysis(empty_result, str(output_path))

        assert success is False  # Should return False for empty

    def test_export_all(self, sample_result, tmp_path):
        """Test export_all exports both files."""
        exporter = CSVExporter()

        results = exporter.export_all(sample_result, str(tmp_path))

        assert results['events']['success']
        assert results['frames']['success']
        assert Path(results['events']['path']).exists()
        assert Path(results['frames']['path']).exists()

    def test_decimal_precision(self, sample_result, tmp_path):
        """Test decimal precision is applied."""
        exporter = CSVExporter(decimal_precision=2)
        output_path = tmp_path / "frames.csv"
        exporter.export_frame_analysis(sample_result, str(output_path))

        df = pd.read_csv(output_path)
        # Check that floats are rounded
        # (this is approximate due to float representation)
        assert df['control_score'].iloc[0] == 0.85


class TestExportToCSV:
    """Tests for export_to_csv convenience function."""

    def test_convenience_function(self, sample_result, tmp_path):
        """Test export_to_csv convenience function."""
        output_path = tmp_path / "output.csv"

        success = export_to_csv(sample_result, str(output_path))

        assert success
        assert output_path.exists()

    def test_convenience_function_with_frames(self, sample_result, tmp_path):
        """Test export_to_csv with include_frames=True."""
        output_path = tmp_path / "output.csv"
        frames_path = tmp_path / "output_frames.csv"

        success = export_to_csv(sample_result, str(output_path), include_frames=True)

        assert success
        assert output_path.exists()
        assert frames_path.exists()


class TestIntegration:
    """Integration tests with real data."""

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_full_pipeline_to_csv(self, tmp_path):
        """Test complete pipeline: load -> detect -> export."""
        # Load
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        # Detect
        result = detect_ball_control(ball_df, pose_df, cone_df)

        # Export
        exporter = CSVExporter()
        export_results = exporter.export_all(result, str(tmp_path))

        assert export_results['events']['success']
        assert export_results['frames']['success']

        # Verify CSVs are readable
        events_df = pd.read_csv(export_results['events']['path'], comment='#')
        frames_df = pd.read_csv(export_results['frames']['path'])

        assert 'event_id' in events_df.columns
        assert len(frames_df) > 1000

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_csv_content_quality(self, tmp_path):
        """Test that CSV content has reasonable values."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        result = detect_ball_control(ball_df, pose_df, cone_df)

        exporter = CSVExporter()
        exporter.export_all(result, str(tmp_path))

        frames_df = pd.read_csv(tmp_path / "frame_analysis.csv")

        # Check data quality
        assert frames_df['frame_id'].min() >= 0
        assert frames_df['control_score'].min() >= 0
        assert frames_df['control_score'].max() <= 1
        assert frames_df['ball_ankle_distance'].min() >= 0


class TestPublicAPI:
    """Test the public API (__init__.py exports)."""

    def test_import_all_exports(self):
        """Test all expected exports are available."""
        import f8_loss

        # Config
        assert hasattr(f8_loss, 'AppConfig')
        assert hasattr(f8_loss, 'Figure8DrillConfig')  # Updated from DrillConfig
        assert hasattr(f8_loss, 'DetectionConfig')

        # Data structures
        assert hasattr(f8_loss, 'FrameData')
        assert hasattr(f8_loss, 'LossEvent')
        assert hasattr(f8_loss, 'DetectionResult')
        assert hasattr(f8_loss, 'ControlState')

        # Functions
        assert hasattr(f8_loss, 'load_parquet_data')
        assert hasattr(f8_loss, 'detect_ball_control')
        assert hasattr(f8_loss, 'export_to_csv')

        # Classes
        assert hasattr(f8_loss, 'BallControlDetector')
        assert hasattr(f8_loss, 'CSVExporter')

    def test_version(self):
        """Test version is defined."""
        import f8_loss
        assert hasattr(f8_loss, '__version__')
        # Version should be a valid semver string
        assert isinstance(f8_loss.__version__, str)
        assert len(f8_loss.__version__.split('.')) == 3  # Major.Minor.Patch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
