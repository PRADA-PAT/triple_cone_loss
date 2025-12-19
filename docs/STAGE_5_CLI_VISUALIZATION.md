# STAGE 5: CLI & Visualization

**Duration**: ~30 minutes
**Prerequisites**: Stages 1-4 complete (full detection pipeline)
**Outcome**: CLI tool and annotated video output

---

## Project Context

**Project Path**: `/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/`

**Data Path**:
```
/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/
```

**Video Path**:
```
/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone/Drill_1_7 Cone_dubaiacademy_Alex Mochar.MOV
```

---

## Prerequisites Check

```bash
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"

# Verify Stages 1-4 are complete
python -c "
from f8_loss import detect_ball_control, load_parquet_data, export_to_csv, CSVExporter
print('Prerequisites OK')
"
```

---

## Files to Create

```
f8_loss/
├── main.py               # ← CREATE THIS
├── drill_visualizer.py   # ← CREATE THIS
└── tests/
    └── test_cli.py       # ← CREATE THIS
```

---

## Important Notes

**Visualization is SEPARATE from core detection logic.**

The `DrillVisualizer` class:
- Is only for debugging purposes
- Can be removed without affecting detection functionality
- Requires OpenCV and video file
- Creates annotated MP4 video

---

## 1. drill_visualizer.py - Visualization Module

```python
"""
Drill visualization for debugging.

THIS MODULE IS FOR DEBUGGING ONLY.
It is completely separate from core detection logic.
Can be removed without affecting detection functionality.

Creates annotated videos showing:
- Ball position and trajectory
- Player ankle positions
- Cone locations
- Loss event indicators
- Metrics overlay
"""
import logging
from pathlib import Path
from typing import Optional, List, Set
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import VisualizationConfig
from .data_structures import DetectionResult, FrameData

logger = logging.getLogger(__name__)


class DrillVisualizer:
    """Create annotated debug videos for ball control analysis."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()

    def create_annotated_video(
        self,
        video_path: str,
        output_path: str,
        detection_result: DetectionResult,
        cone_df: pd.DataFrame,
        ball_df: pd.DataFrame,
        pose_df: pd.DataFrame
    ) -> bool:
        """
        Create annotated video with detection overlays.

        Args:
            video_path: Input video path
            output_path: Output video path
            detection_result: Detection results for annotations
            cone_df: Cone detection DataFrame
            ball_df: Ball detection DataFrame
            pose_df: Pose keypoint DataFrame

        Returns:
            True if successful
        """
        try:
            # Open input video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return False

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                logger.error(f"Cannot create output video: {output_path}")
                cap.release()
                return False

            # Build lookups for fast access
            frame_lookup = {fd.frame_id: fd for fd in detection_result.frame_data}

            # Get frames with loss events
            event_frames: Set[int] = set()
            for event in detection_result.events:
                if event.end_frame:
                    event_frames.update(range(event.start_frame, event.end_frame + 1))
                else:
                    event_frames.add(event.start_frame)

            # Process frames
            pbar = tqdm(total=total_frames, desc="Creating video")
            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw annotations
                if self.config.show_cone_positions:
                    self._draw_cones(frame, cone_df, frame_id)

                if self.config.show_ball_trajectory:
                    self._draw_ball(frame, ball_df, frame_id)

                if self.config.show_player_trajectory:
                    self._draw_player(frame, pose_df, frame_id)

                if self.config.show_event_markers and frame_id in event_frames:
                    self._draw_loss_marker(frame)

                if self.config.show_metrics_overlay:
                    self._draw_metrics(frame, frame_lookup.get(frame_id), frame_id, fps)

                out.write(frame)
                frame_id += 1
                pbar.update(1)

            pbar.close()
            cap.release()
            out.release()

            logger.info(f"Created video: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Visualization failed: {e}", exc_info=True)
            return False

    def _draw_cones(self, frame: np.ndarray, cone_df: pd.DataFrame, frame_id: int):
        """Draw cone markers on frame."""
        cones = cone_df[cone_df['frame_id'] == frame_id]

        for _, cone in cones.iterrows():
            center = (int(cone['center_x']), int(cone['center_y']))

            # Draw cone circle
            cv2.circle(frame, center, 10, self.config.cone_color, -1)
            cv2.circle(frame, center, 12, (255, 255, 255), 2)

            # Draw cone ID label
            label = str(int(cone['object_id']))
            cv2.putText(
                frame, label,
                (center[0] + 15, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                self.config.cone_color, 2
            )

    def _draw_ball(self, frame: np.ndarray, ball_df: pd.DataFrame, frame_id: int):
        """Draw ball position and trajectory trail."""
        # Get recent ball positions for trail
        trail_start = max(0, frame_id - self.config.trail_length)
        recent = ball_df[
            (ball_df['frame_id'] >= trail_start) &
            (ball_df['frame_id'] <= frame_id)
        ].sort_values('frame_id')

        if recent.empty:
            return

        # Draw trail
        points = [(int(r['center_x']), int(r['center_y']))
                  for _, r in recent.iterrows()]

        for i in range(1, len(points)):
            alpha = i / len(points)
            thickness = max(1, int(3 * alpha))
            color = tuple(int(c * alpha) for c in self.config.ball_color)
            cv2.line(frame, points[i-1], points[i], color, thickness)

        # Draw current ball position
        if points:
            cv2.circle(frame, points[-1], 12, self.config.ball_color, -1)
            cv2.circle(frame, points[-1], 14, (255, 255, 255), 2)

    def _draw_player(self, frame: np.ndarray, pose_df: pd.DataFrame, frame_id: int):
        """Draw player ankle positions."""
        frame_pose = pose_df[
            (pose_df['frame_idx'] == frame_id) &
            (pose_df['keypoint_name'].isin(['left_ankle', 'right_ankle']))
        ]

        for _, kp in frame_pose.iterrows():
            center = (int(kp['x']), int(kp['y']))

            # Draw ankle marker
            cv2.circle(frame, center, 6, self.config.player_color, -1)

            # Label
            label = "L" if "left" in kp['keypoint_name'] else "R"
            cv2.putText(
                frame, label,
                (center[0] + 8, center[1] + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                self.config.player_color, 1
            )

    def _draw_loss_marker(self, frame: np.ndarray):
        """Draw loss-of-control indicator."""
        h, w = frame.shape[:2]

        # Red border
        cv2.rectangle(frame, (0, 0), (w-1, h-1), self.config.loss_event_color, 6)

        # Warning text
        text = "BALL CONTROL LOSS"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 40

        # Background for text
        cv2.rectangle(
            frame,
            (text_x - 10, text_y - 30),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 0), -1
        )

        cv2.putText(
            frame, text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            self.config.loss_event_color, 2
        )

    def _draw_metrics(
        self,
        frame: np.ndarray,
        frame_data: Optional[FrameData],
        frame_id: int,
        fps: float
    ):
        """Draw metrics overlay panel."""
        # Semi-transparent background
        overlay = frame.copy()
        panel_width = 280
        panel_height = 140
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Panel border
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (100, 100, 100), 1)

        # Metrics text
        lines = [
            f"Frame: {frame_id}",
            f"Time: {frame_id / fps:.2f}s",
        ]

        if frame_data:
            lines.extend([
                f"Ball-Ankle: {frame_data.ball_ankle_distance:.1f}",
                f"Control Score: {frame_data.control_score:.2f}",
                f"State: {frame_data.control_state.value}",
                f"Gate: {frame_data.current_gate or 'N/A'}",
            ])
        else:
            lines.extend([
                "Ball-Ankle: N/A",
                "Control Score: N/A",
                "State: N/A",
            ])

        y = 32
        for line in lines:
            cv2.putText(
                frame, line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1
            )
            y += 18


def create_debug_video(
    video_path: str,
    output_path: str,
    detection_result: DetectionResult,
    cone_df: pd.DataFrame,
    ball_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    config: Optional[VisualizationConfig] = None
) -> bool:
    """
    Convenience function to create debug video.

    Args:
        video_path: Input video
        output_path: Output video
        detection_result: Detection results
        cone_df, ball_df, pose_df: Data for annotations
        config: Visualization config

    Returns:
        True if successful
    """
    visualizer = DrillVisualizer(config)
    return visualizer.create_annotated_video(
        video_path, output_path, detection_result,
        cone_df, ball_df, pose_df
    )
```

---

## 2. main.py - CLI Entry Point

```python
"""
CLI entry point for Ball Control Detection System.

Usage:
    python -m f8_loss.main --ball ball.parquet --pose pose.parquet --cone cone.parquet

    # With visualization
    python -m f8_loss.main --ball ball.parquet --pose pose.parquet --cone cone.parquet \
        --video video.mov --visualize --output-dir ./output
"""
import argparse
import logging
import sys
from pathlib import Path

from .config import AppConfig
from .data_loader import load_parquet_data
from .ball_control_detector import BallControlDetector
from .csv_exporter import CSVExporter

# Conditional import for visualization (optional dependency)
try:
    from .drill_visualizer import DrillVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main(args=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ball Control Detection for Figure-8 Cone Drills',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection
  python -m f8_loss.main -b ball.parquet -p pose.parquet -c cone.parquet

  # With visualization
  python -m f8_loss.main -b ball.parquet -p pose.parquet -c cone.parquet \\
      -v video.mov --visualize -o ./output

  # Full example with actual paths
  python -m f8_loss.main \\
      -b "path/to/football.parquet" \\
      -p "path/to/pose.parquet" \\
      -c "path/to/cone.parquet" \\
      -v "path/to/video.MOV" \\
      --visualize --verbose
        """
    )

    # Required arguments
    parser.add_argument(
        '--ball', '-b', required=True,
        help='Ball detection parquet file'
    )
    parser.add_argument(
        '--pose', '-p', required=True,
        help='Pose keypoints parquet file'
    )
    parser.add_argument(
        '--cone', '-c', required=True,
        help='Cone detection parquet file'
    )

    # Optional arguments
    parser.add_argument(
        '--video', '-v',
        help='Video file (required for --visualize)'
    )
    parser.add_argument(
        '--output-dir', '-o', default='./output',
        help='Output directory (default: ./output)'
    )
    parser.add_argument(
        '--fps', type=float, default=30.0,
        help='Video FPS (default: 30.0)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Create visualization video (requires --video)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output'
    )

    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(parsed_args.verbose)
    logger = logging.getLogger(__name__)

    # Validate arguments
    if parsed_args.visualize and not parsed_args.video:
        parser.error("--visualize requires --video")

    if parsed_args.visualize and not HAS_VISUALIZER:
        logger.warning("Visualization disabled: OpenCV not available")
        parsed_args.visualize = False

    # Create output directory
    output_dir = Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        logger.info("Loading data...")
        ball_df = load_parquet_data(parsed_args.ball)
        pose_df = load_parquet_data(parsed_args.pose)
        cone_df = load_parquet_data(parsed_args.cone)

        logger.info(f"  Balls: {len(ball_df)} records")
        logger.info(f"  Poses: {len(pose_df)} records")
        logger.info(f"  Cones: {len(cone_df)} records")

        # Run detection
        logger.info("Running detection...")
        config = AppConfig()
        config.fps = parsed_args.fps
        config.verbose = parsed_args.verbose

        detector = BallControlDetector(config)
        result = detector.detect(ball_df, pose_df, cone_df, parsed_args.fps)

        if not result.success:
            logger.error(f"Detection failed: {result.error}")
            return 1

        # Export results
        logger.info("Exporting results...")
        exporter = CSVExporter()

        events_path = output_dir / 'loss_events.csv'
        exporter.export_events(result, str(events_path))

        frames_path = output_dir / 'frame_analysis.csv'
        exporter.export_frame_analysis(result, str(frames_path))

        # Create visualization
        if parsed_args.visualize:
            logger.info("Creating visualization...")
            visualizer = DrillVisualizer(config.visualization)
            video_output = output_dir / 'annotated_video.mp4'

            success = visualizer.create_annotated_video(
                parsed_args.video,
                str(video_output),
                result,
                cone_df,
                ball_df,
                pose_df
            )

            if success:
                logger.info(f"Video created: {video_output}")
            else:
                logger.error("Video creation failed")

        # Print summary
        print("\n" + "=" * 60)
        print("BALL CONTROL DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total frames analyzed: {result.total_frames}")
        print(f"Loss events detected: {result.total_loss_events}")
        print(f"Total loss duration: {result.total_loss_duration_frames} frames")
        print(f"Control percentage: {result.control_percentage:.1f}%")
        print()
        print("Output files:")
        print(f"  Events:  {events_path}")
        print(f"  Frames:  {frames_path}")
        if parsed_args.visualize:
            print(f"  Video:   {output_dir / 'annotated_video.mp4'}")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=parsed_args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

---

## 3. tests/test_cli.py - CLI Tests

```python
"""Stage 5 Tests: CLI & Visualization."""
import pytest
import subprocess
import sys
from pathlib import Path
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from f8_loss.main import main
from f8_loss.drill_visualizer import DrillVisualizer, create_debug_video
from f8_loss.config import VisualizationConfig
from f8_loss import detect_ball_control, load_parquet_data
from f8_loss.data_structures import DetectionResult, FrameData, ControlState

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
        """Test running with python -m f8_loss.main."""
        result = subprocess.run([
            sys.executable, '-m', 'f8_loss.main',
            '--ball', str(BALL_PATH),
            '--pose', str(POSE_PATH),
            '--cone', str(CONE_PATH),
            '--output-dir', str(tmp_path),
        ], capture_output=True, text=True, cwd=str(Path(__file__).parent.parent.parent))

        assert result.returncode == 0, f"stderr: {result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 4. Update __init__.py (Add visualization exports)

Add to existing `__init__.py`:

```python
# Add visualization import (optional - handles missing OpenCV)
try:
    from .drill_visualizer import DrillVisualizer, create_debug_video
    _HAS_VISUALIZER = True
except ImportError:
    _HAS_VISUALIZER = False
    DrillVisualizer = None
    create_debug_video = None

# Add to __all__
__all__ = [
    # ... existing exports ...
    # Visualization (optional)
    'DrillVisualizer',
    'create_debug_video',
]
```

---

## Validation Commands

```bash
# Navigate to project
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"

# Run Stage 5 tests (skip video tests for speed)
pytest tests/test_cli.py -v -k "not video"

# Test CLI help
python -m f8_loss.main --help

# Test basic CLI
python -m f8_loss.main \
    --ball "/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet" \
    --pose "/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet" \
    --cone "/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet" \
    --output-dir ./output \
    --verbose

# Test with visualization (takes a few minutes)
python -m f8_loss.main \
    --ball "/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet" \
    --pose "/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet" \
    --cone "/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet" \
    --video "/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone/Drill_1_7 Cone_dubaiacademy_Alex Mochar.MOV" \
    --output-dir ./output \
    --visualize \
    --verbose

# Verify outputs
ls -la output/
```

---

## Final Checklist - All Stages Complete

### Stage 1: Foundation
- [ ] `config.py` - Configuration dataclasses
- [ ] `data_structures.py` - Data models
- [ ] `requirements.txt` - Dependencies

### Stage 2: Data Layer
- [ ] `data_loader.py` - Parquet loading

### Stage 3: Core Detection
- [ ] `ball_control_detector.py` - Detector with placeholders

### Stage 4: Export & API
- [ ] `csv_exporter.py` - CSV export
- [ ] `__init__.py` - Complete public API

### Stage 5: CLI & Visualization
- [ ] `main.py` - CLI entry point
- [ ] `drill_visualizer.py` - Debug video
- [ ] All tests pass

---

## Project Complete Structure

```
f8_loss/
├── __init__.py              # Public API
├── config.py                # Configuration
├── data_structures.py       # Data models
├── data_loader.py           # Parquet loading
├── ball_control_detector.py # Core detection
├── csv_exporter.py          # CSV export
├── drill_visualizer.py      # Debug visualization
├── main.py                  # CLI entry point
├── requirements.txt         # Dependencies
├── IMPLEMENTATION_GUIDE.md  # Original guide
├── STAGE_1_FOUNDATION.md
├── STAGE_2_DATA_LAYER.md
├── STAGE_3_CORE_DETECTION.md
├── STAGE_4_EXPORT_API.md
├── STAGE_5_CLI_VISUALIZATION.md
└── tests/
    ├── __init__.py
    ├── test_stage1.py
    ├── test_data_loader.py
    ├── test_detector.py
    ├── test_exporter.py
    └── test_cli.py
```

---

## Next Steps (After All Stages)

1. **Implement Detection Logic** - Fill in placeholder methods:
   - `_analyze_frame()` - Full frame analysis
   - `_calculate_control_score()` - Multi-factor scoring
   - `_detect_loss_condition()` - Event detection
   - `_check_state_transition()` - State machine

2. **Tune Thresholds** - Adjust detection parameters in `DetectionConfig`

3. **Add More Tests** - Edge cases, error handling

4. **Performance Optimization** - Vectorize operations, parallel processing
