"""Stage 5 Tests: CLI & Visualization."""
import pytest
import subprocess
import sys
from pathlib import Path
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from triple_cone_loss.main import main
from triple_cone_loss.annotation.drill_visualizer import DrillVisualizer
from triple_cone_loss.detection.config import VisualizationConfig
from triple_cone_loss import detect_ball_control, load_parquet_data
from triple_cone_loss.detection.data_structures import DetectionResult, FrameData, ControlState

# Real data paths
DATA_DIR = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar")
VIDEO_PATH = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone/Drill_1_7 Cone_dubaiacademy_Alex Mochar.MOV")
CONE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet"
BALL_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet"
POSE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet"


class TestCLI:
    """Tests for CLI main function."""

    def test_cli_help(self):
        """Test CLI help works."""
        # Should not raise
        with pytest.raises(SystemExit) as exc_info:
            main(['--help'])
        assert exc_info.value.code == 0

    def test_cli_missing_required_args(self):
        """Test CLI fails without required args."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_cli_basic_detection(self, tmp_path):
        """Test CLI runs basic detection."""
        result = main([
            '--ball', str(BALL_PATH),
            '--pose', str(POSE_PATH),
            '--cone', str(CONE_PATH),
            '--output-dir', str(tmp_path),
        ])

        assert result == 0
        assert (tmp_path / 'loss_events.csv').exists()
        assert (tmp_path / 'frame_analysis.csv').exists()

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_cli_with_custom_fps(self, tmp_path):
        """Test CLI with custom FPS."""
        result = main([
            '--ball', str(BALL_PATH),
            '--pose', str(POSE_PATH),
            '--cone', str(CONE_PATH),
            '--output-dir', str(tmp_path),
            '--fps', '60.0',
        ])

        assert result == 0

    def test_cli_visualize_without_video(self, tmp_path):
        """Test CLI fails when --visualize without --video."""
        with pytest.raises(SystemExit):
            main([
                '--ball', 'ball.parquet',
                '--pose', 'pose.parquet',
                '--cone', 'cone.parquet',
                '--visualize',
            ])


class TestDrillVisualizer:
    """Tests for DrillVisualizer class."""

    @pytest.fixture
    def sample_result(self):
        """Create sample detection result."""
        frame_data = [
            FrameData(
                frame_id=i,
                timestamp=i / 30.0,
                ball_x=100.0 + i,
                ball_y=200.0 + i,
                ball_field_x=50.0,
                ball_field_y=100.0,
                ball_velocity=10.0,
                ankle_x=90.0 + i,
                ankle_y=190.0 + i,
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
            for i in range(10)
        ]

        return DetectionResult(
            success=True,
            total_frames=10,
            events=[],
            frame_data=frame_data,
        )

    def test_visualizer_initialization(self):
        """Test visualizer initializes with default config."""
        viz = DrillVisualizer()
        assert viz.config is not None
        assert viz.config.ball_color == (0, 255, 255)

    def test_visualizer_custom_config(self):
        """Test visualizer with custom config."""
        config = VisualizationConfig()
        config.trail_length = 50
        config.ball_color = (255, 0, 0)

        viz = DrillVisualizer(config)
        assert viz.config.trail_length == 50
        assert viz.config.ball_color == (255, 0, 0)

    @pytest.mark.skipif(not VIDEO_PATH.exists(), reason="Video not available")
    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_create_annotated_video(self, tmp_path):
        """Test creating annotated video (integration test)."""
        # Load data
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        # Run detection
        result = detect_ball_control(ball_df, pose_df, cone_df)

        # Create video (only first 30 frames for speed)
        viz = DrillVisualizer()
        output_path = tmp_path / "test_video.mp4"

        # Note: Full video creation is slow, so we just test it starts
        # In real tests, you might want to use a shorter clip
        success = viz.create_annotated_video(
            str(VIDEO_PATH),
            str(output_path),
            result,
            cone_df,
            ball_df,
            pose_df
        )

        # Video creation should succeed
        assert success
        assert output_path.exists()


class TestIntegration:
    """Full integration tests."""

    @pytest.mark.skipif(not VIDEO_PATH.exists(), reason="Video not available")
    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_full_cli_with_visualization(self, tmp_path):
        """Test complete CLI with visualization."""
        result = main([
            '--ball', str(BALL_PATH),
            '--pose', str(POSE_PATH),
            '--cone', str(CONE_PATH),
            '--video', str(VIDEO_PATH),
            '--output-dir', str(tmp_path),
            '--visualize',
            '--verbose',
        ])

        assert result == 0

        # Check all outputs exist
        assert (tmp_path / 'loss_events.csv').exists()
        assert (tmp_path / 'frame_analysis.csv').exists()
        assert (tmp_path / 'annotated_video.mp4').exists()

        # Verify video has content
        video_size = (tmp_path / 'annotated_video.mp4').stat().st_size
        assert video_size > 1000  # Should be more than 1KB


class TestModuleExecution:
    """Test running as module."""

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_run_as_module(self, tmp_path):
        """Test running with python -m triple_cone_loss.main."""
        result = subprocess.run([
            sys.executable, '-m', 'triple_cone_loss.main',
            '--ball', str(BALL_PATH),
            '--pose', str(POSE_PATH),
            '--cone', str(CONE_PATH),
            '--output-dir', str(tmp_path),
        ], capture_output=True, text=True, cwd=str(Path(__file__).parent.parent.parent))

        assert result.returncode == 0, f"stderr: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
