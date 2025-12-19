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

from ..detection.config import VisualizationConfig
from ..detection.data_structures import DetectionResult, FrameData

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
