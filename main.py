"""
CLI entry point for Triple Cone Ball Control Detection System.

Usage:
    python -m triple_cone_loss.main --ball ball.parquet --pose pose.parquet --cone cone.parquet

    # With visualization
    python -m triple_cone_loss.main --ball ball.parquet --pose pose.parquet --cone cone.parquet \
        --video video.mov --visualize --output-dir ./output
"""
import argparse
import logging
import sys
from pathlib import Path

from .detection.config import AppConfig, DetectionMode
from .detection.data_loader import load_parquet_data
from .detection.ball_control_detector import BallControlDetector
from .detection.csv_exporter import CSVExporter

# Conditional import for visualization (optional dependency)
try:
    from .annotation.drill_visualizer import DrillVisualizer
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
        description='Triple Cone Drill Ball Control Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection
  python -m triple_cone_loss.main -b ball.parquet -p pose.parquet -c cone.parquet

  # With visualization
  python -m triple_cone_loss.main -b ball.parquet -p pose.parquet -c cone.parquet \\
      -v video.mov --visualize -o ./output

  # With strict detection (fewer false positives)
  python -m triple_cone_loss.main -b ball.parquet -p pose.parquet -c cone.parquet --mode strict

  # With lenient detection (catches more events)
  python -m triple_cone_loss.main -b ball.parquet -p pose.parquet -c cone.parquet --mode lenient
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
        '--mode', choices=['standard', 'strict', 'lenient'], default='standard',
        help='Detection mode (default: standard)'
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

        # Create config based on mode
        if parsed_args.mode == 'strict':
            config = AppConfig.with_strict_detection()
        elif parsed_args.mode == 'lenient':
            config = AppConfig.with_lenient_detection()
        else:
            config = AppConfig.for_triple_cone()

        config.fps = parsed_args.fps
        config.verbose = parsed_args.verbose

        # Run detection
        logger.info(f"Running Triple Cone drill detection (mode: {parsed_args.mode})...")
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
        print("TRIPLE CONE DRILL BALL CONTROL DETECTION SUMMARY")
        print("=" * 60)
        print(f"Detection mode: {parsed_args.mode.upper()}")
        print()
        print("DRILL METRICS:")
        print(f"  Total frames analyzed: {result.total_frames}")
        print(f"  Repetitions completed: {result.total_laps}")
        print()
        print("BALL CONTROL:")
        print(f"  Loss events detected: {result.total_loss_events}")
        print(f"  Total loss duration: {result.total_loss_duration_frames} frames")
        print(f"  Control percentage: {result.control_percentage:.1f}%")
        print()
        print("OUTPUT FILES:")
        print(f"  Events: {events_path}")
        print(f"  Frames: {frames_path}")
        if parsed_args.visualize:
            print(f"  Video:  {output_dir / 'annotated_video.mp4'}")
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
