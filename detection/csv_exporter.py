"""
CSV export for Ball Control Detection results.

Exports:
- Loss events CSV with metadata header
- Frame-by-frame analysis CSV
"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from .data_structures import DetectionResult

logger = logging.getLogger(__name__)


class CSVExporter:
    """Export detection results to CSV files."""

    def __init__(self, decimal_precision: int = 3):
        """
        Initialize exporter.

        Args:
            decimal_precision: Number of decimal places for float columns
        """
        self.decimal_precision = decimal_precision

    def export_events(
        self,
        result: DetectionResult,
        output_path: str
    ) -> bool:
        """
        Export loss events to CSV.

        Creates a CSV with metadata header containing:
        - Total frames processed
        - Total events detected
        - Control percentage

        Args:
            result: Detection result
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if not result.events:
                # Create empty CSV with headers
                columns = [
                    'event_id', 'event_type', 'start_frame', 'end_frame',
                    'start_timestamp', 'end_timestamp', 'duration_frames',
                    'duration_seconds', 'ball_x', 'ball_y', 'player_x', 'player_y',
                    'distance_at_loss', 'velocity_at_loss', 'nearest_cone_id',
                    'gate_context', 'recovered', 'recovery_frame', 'severity', 'notes'
                ]
                df = pd.DataFrame(columns=columns)
            else:
                df = pd.DataFrame([e.to_dict() for e in result.events])

            # Round numeric columns
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].round(self.decimal_precision)

            # Write with metadata header
            with open(output_path, 'w') as f:
                f.write(f"# Ball Control Loss Events\n")
                f.write(f"# Total frames: {result.total_frames}\n")
                f.write(f"# Total events: {result.total_loss_events}\n")
                f.write(f"# Control percentage: {result.control_percentage:.1f}%\n")
                f.write("#\n")
                df.to_csv(f, index=False)

            logger.info(f"Exported {len(result.events)} events to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def export_frame_analysis(
        self,
        result: DetectionResult,
        output_path: str
    ) -> bool:
        """
        Export frame-by-frame analysis to CSV.

        Args:
            result: Detection result
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            if not result.frame_data:
                logger.warning("No frame data to export")
                return False

            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame([f.to_dict() for f in result.frame_data])

            # Round numeric columns
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].round(self.decimal_precision)

            df.to_csv(output_path, index=False)

            logger.info(f"Exported {len(df)} frames to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def export_all(
        self,
        result: DetectionResult,
        output_dir: str,
        events_filename: str = "loss_events.csv",
        frames_filename: str = "frame_analysis.csv"
    ) -> dict:
        """
        Export all available data to CSVs.

        Args:
            result: Detection result
            output_dir: Output directory
            events_filename: Events CSV filename
            frames_filename: Frames CSV filename

        Returns:
            Dict with export status for each file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        events_path = output_dir / events_filename
        frames_path = output_dir / frames_filename

        results = {
            'events': {
                'success': self.export_events(result, str(events_path)),
                'path': str(events_path),
            },
            'frames': {
                'success': self.export_frame_analysis(result, str(frames_path)),
                'path': str(frames_path),
            }
        }

        return results


def export_to_csv(
    result: DetectionResult,
    path: str,
    include_frames: bool = False
) -> bool:
    """
    Convenience function to export detection results to CSV.

    Args:
        result: Detection result
        path: Output path for events CSV
        include_frames: If True, also export frame analysis to *_frames.csv

    Returns:
        True if successful
    """
    exporter = CSVExporter()
    success = exporter.export_events(result, path)

    if include_frames and success:
        frames_path = path.replace('.csv', '_frames.csv')
        exporter.export_frame_analysis(result, frames_path)

    return success
