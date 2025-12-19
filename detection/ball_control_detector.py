"""
Ball Control Detector - Simplified for Figure-8 drill.

Detects ball control loss events during Figure-8 cone drill.
Tracks gate passages, drill phases, and lap completion.

DETECTION LOGIC LOCATION:
========================
All loss detection logic is in the `detect_loss()` method.
Modify ONLY that method to implement your detection algorithm.

Current implementation: Simple distance-based detection.
"""
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .config import AppConfig, Figure8DrillConfig
from .data_structures import (
    ControlState, EventType, FrameData,
    LossEvent, DetectionResult,
    DrillPhase, DrillDirection, GatePassage
)
from .data_loader import (
    extract_ankle_positions,
    get_closest_ankle_per_frame
)
from .figure8_cone_detector import Figure8ConeDetector

logger = logging.getLogger(__name__)


class BallControlDetector:
    """
    Main class for detecting ball control loss events in Figure-8 drill.

    Usage:
        config = AppConfig.for_figure8()
        detector = BallControlDetector(config)
        result = detector.detect(ball_df, pose_df, cone_df)
    """

    def __init__(self, config: Optional[AppConfig] = None, parquet_dir: Optional[str] = None):
        """
        Initialize detector with Figure-8 config.

        Args:
            config: Application configuration (defaults to Figure-8 config)
            parquet_dir: Path to parquet directory for loading manual cone annotations
        """
        self.config = config or AppConfig.for_figure8()
        self.parquet_dir = parquet_dir

        if not isinstance(self.config.drill, Figure8DrillConfig):
            self.config.drill = Figure8DrillConfig()

        self._detection_config = self.config.detection
        self._drill_config: Figure8DrillConfig = self.config.drill

        # Figure-8 specific detector
        self._f8_detector: Optional[Figure8ConeDetector] = None

        # State tracking
        self._current_state = ControlState.CONTROLLED
        self._current_direction = DrillDirection.STATIONARY
        self._current_phase = DrillPhase.AT_START
        self._events: List[LossEvent] = []
        self._frame_data: List[FrameData] = []
        self._event_counter = 0

        # Previous frame data
        self._prev_ball_pos: Optional[Tuple[float, float]] = None

        logger.info("BallControlDetector initialized (simplified)")

    def detect(
        self,
        ball_df: pd.DataFrame,
        pose_df: pd.DataFrame,
        cone_df: pd.DataFrame,
        fps: float = 30.0
    ) -> DetectionResult:
        """
        Run ball control detection for Figure-8 drill.

        Args:
            ball_df: Ball detection DataFrame
            pose_df: Pose keypoint DataFrame
            cone_df: Cone detection DataFrame
            fps: Video FPS for timestamps

        Returns:
            DetectionResult with events and frame data
        """
        try:
            logger.info("Starting Figure-8 drill detection...")
            logger.info(f"  Ball frames: {len(ball_df)}")
            logger.info(f"  Pose records: {len(pose_df)}")
            logger.info(f"  Cone records: {len(cone_df)}")

            # Reset state
            self._reset_state()

            # Initialize Figure-8 detector and identify cone roles
            # Passing parquet_dir enables loading manual cone annotations if available
            self._f8_detector = Figure8ConeDetector(self._drill_config, parquet_dir=self.parquet_dir)
            cone_roles = self._f8_detector.identify_cone_roles(cone_df, frame_id=0)
            self._f8_detector.setup_gates(cone_roles)

            logger.info(f"Cone roles identified: {len(cone_roles)} cones")

            # Extract ankles and find closest per frame
            ankle_df = extract_ankle_positions(pose_df)
            merged_df = get_closest_ankle_per_frame(ankle_df, ball_df)

            if merged_df.empty:
                return DetectionResult(
                    success=False,
                    total_frames=0,
                    events=[],
                    frame_data=[],
                    error="No valid frames after merging"
                )

            # Merge with ball data
            ball_cols = ['frame_id', 'center_x', 'center_y',
                        'field_center_x', 'field_center_y']
            available_cols = [c for c in ball_cols if c in ball_df.columns]
            merged_df = merged_df.merge(ball_df[available_cols], on='frame_id')

            merged_df.rename(columns={
                'center_x': 'ball_x',
                'center_y': 'ball_y',
                'field_center_x': 'ball_field_x',
                'field_center_y': 'ball_field_y',
            }, inplace=True)

            # Calculate ball velocity
            merged_df = merged_df.sort_values('frame_id')
            merged_df['ball_velocity'] = np.sqrt(
                merged_df['ball_field_x'].diff()**2 +
                merged_df['ball_field_y'].diff()**2
            ).fillna(0)

            # Process each frame
            total_frames = len(merged_df)

            for _, row in merged_df.iterrows():
                frame_id = int(row['frame_id'])
                timestamp = frame_id / fps

                frame_result = self._analyze_frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    row=row,
                    cone_df=cone_df
                )

                if frame_result:
                    self._frame_data.append(frame_result)

            # Finalize any open events
            self._finalize_events()

            result = DetectionResult(
                success=True,
                total_frames=total_frames,
                events=self._events,
                frame_data=self._frame_data,
                gate_passages=self._f8_detector.gate_passages if self._f8_detector else [],
                cone_roles=cone_roles,
                total_laps=self._f8_detector.lap_count if self._f8_detector else 0,
            )

            logger.info(f"Detection complete:")
            logger.info(f"  Processed frames: {total_frames}")
            logger.info(f"  Loss events: {result.total_loss_events}")
            logger.info(f"  Control percentage: {result.control_percentage:.1f}%")

            return result

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return DetectionResult(
                success=False,
                total_frames=0,
                events=[],
                frame_data=[],
                error=str(e)
            )

    def _reset_state(self):
        """Reset internal state for new detection run."""
        self._current_state = ControlState.CONTROLLED
        self._current_direction = DrillDirection.STATIONARY
        self._current_phase = DrillPhase.AT_START
        self._events = []
        self._frame_data = []
        self._event_counter = 0
        self._prev_ball_pos = None

        if self._f8_detector:
            self._f8_detector.reset()

    def _analyze_frame(
        self,
        frame_id: int,
        timestamp: float,
        row: pd.Series,
        cone_df: pd.DataFrame
    ) -> Optional[FrameData]:
        """
        Analyze a single frame.

        Gathers data and calls detect_loss() to determine control state.
        """
        # Get positions
        ball_pos = (row['ball_field_x'], row['ball_field_y'])
        ankle_pos = (row['ankle_field_x'], row['ankle_field_y'])
        distance = row['ball_ankle_distance']
        velocity = row['ball_velocity']

        # ============================================================
        # DETECTION LOGIC - calls detect_loss()
        # ============================================================
        is_loss, loss_type = self.detect_loss(
            ball_pos=ball_pos,
            ankle_pos=ankle_pos,
            distance=distance,
            velocity=velocity,
            frame_id=frame_id,
            timestamp=timestamp,
            history=self._frame_data
        )

        # Determine control state from detection result
        if is_loss:
            control_state = ControlState.LOST
        else:
            control_state = ControlState.CONTROLLED

        # Handle state transitions (create/close events)
        self._handle_state_change(
            new_state=control_state,
            loss_type=loss_type,
            frame_id=frame_id,
            timestamp=timestamp,
            ball_pos=ball_pos,
            ankle_pos=ankle_pos,
            distance=distance,
            velocity=velocity,
            row=row
        )

        # Get nearest cone
        nearest_cone_id, nearest_cone_dist = self._get_nearest_cone(
            ball_pos, cone_df, frame_id
        )

        # Figure-8 tracking (gate passages, direction, phase)
        drill_phase = None
        drill_direction = None
        current_gate = None

        if self._f8_detector:
            if self._prev_ball_pos:
                dx = ball_pos[0] - self._prev_ball_pos[0]
                if abs(dx) > 5:
                    drill_direction = DrillDirection.FORWARD if dx > 0 else DrillDirection.BACKWARD
                else:
                    drill_direction = DrillDirection.STATIONARY
                self._current_direction = drill_direction

                # Detect gate passage
                passage = self._f8_detector.detect_gate_passage(
                    self._prev_ball_pos,
                    ball_pos,
                    frame_id,
                    timestamp,
                    ankle_pos,
                    control_state == ControlState.CONTROLLED
                )

                if passage:
                    current_gate = passage.gate_id
                    self._f8_detector.update_lap_count(passage)
            else:
                drill_direction = DrillDirection.STATIONARY

            drill_phase = self._f8_detector.get_current_phase(
                ball_pos, self._current_direction
            )
            self._current_phase = drill_phase

        self._prev_ball_pos = ball_pos

        # Simple control score (just for reporting, not used in detection)
        control_score = max(0.0, 1.0 - (distance / self._detection_config.loss_distance_threshold))

        return FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            ball_x=row['ball_x'],
            ball_y=row['ball_y'],
            ball_field_x=row['ball_field_x'],
            ball_field_y=row['ball_field_y'],
            ball_velocity=velocity,
            ankle_x=row['ankle_x'],
            ankle_y=row['ankle_y'],
            ankle_field_x=row['ankle_field_x'],
            ankle_field_y=row['ankle_field_y'],
            closest_ankle=row['closest_ankle'],
            nearest_cone_id=nearest_cone_id,
            nearest_cone_distance=nearest_cone_dist,
            current_gate=current_gate,
            ball_ankle_distance=distance,
            control_score=control_score,
            control_state=control_state,
            drill_phase=drill_phase,
            drill_direction=drill_direction,
            lap_count=self._f8_detector.lap_count if self._f8_detector else 0,
        )

    # ================================================================
    # DETECTION LOGIC - IMPLEMENT YOUR ALGORITHM HERE
    # ================================================================
    def detect_loss(
        self,
        ball_pos: Tuple[float, float],
        ankle_pos: Tuple[float, float],
        distance: float,
        velocity: float,
        frame_id: int,
        timestamp: float,
        history: List[FrameData]
    ) -> Tuple[bool, Optional[EventType]]:
        """
        Detect if ball control is lost.

        TODO: Implement detection logic here.

        Args:
            ball_pos: Ball position (field_x, field_y)
            ankle_pos: Closest ankle position (field_x, field_y)
            distance: Ball-to-ankle distance (pre-calculated)
            velocity: Ball velocity this frame
            frame_id: Current frame number
            timestamp: Current timestamp in seconds
            history: List of previous FrameData (for temporal analysis)

        Returns:
            Tuple of:
            - is_loss: True if control is lost, False otherwise
            - loss_type: EventType if loss detected, None otherwise
        """
        # No detection logic - always returns no loss
        return False, None

    # ================================================================
    # EVENT LIFECYCLE (keep as-is)
    # ================================================================
    def _handle_state_change(
        self,
        new_state: ControlState,
        loss_type: Optional[EventType],
        frame_id: int,
        timestamp: float,
        ball_pos: Tuple[float, float],
        ankle_pos: Tuple[float, float],
        distance: float,
        velocity: float,
        row: pd.Series
    ):
        """Handle state transitions and create/close events."""
        prev_state = self._current_state

        # Transition to LOST - create new event
        if new_state == ControlState.LOST and prev_state != ControlState.LOST:
            event = LossEvent(
                event_id=self._event_counter,
                event_type=loss_type or EventType.LOSS_DISTANCE,
                start_frame=frame_id,
                end_frame=None,
                start_timestamp=timestamp,
                end_timestamp=None,
                ball_position=ball_pos,
                player_position=ankle_pos,
                distance_at_loss=distance,
                velocity_at_loss=velocity,
                nearest_cone_id=int(row.get('nearest_cone_id', -1)) if 'nearest_cone_id' in row else -1,
                gate_context=None,
                severity=self._get_severity(),
            )
            self._events.append(event)
            self._event_counter += 1
            logger.debug(f"Loss event started at frame {frame_id}")

        # Transition from LOST - close event
        elif prev_state == ControlState.LOST and new_state != ControlState.LOST:
            for event in reversed(self._events):
                if event.end_frame is None:
                    event.end_frame = frame_id
                    event.end_timestamp = timestamp
                    event.recovered = True
                    event.recovery_frame = frame_id
                    logger.debug(f"Loss event ended at frame {frame_id}")
                    break

        self._current_state = new_state

    def _get_severity(self) -> str:
        """Get severity based on current drill phase."""
        if self._current_phase in [DrillPhase.PASSING_G1, DrillPhase.PASSING_G2]:
            return "high"
        elif self._current_phase == DrillPhase.AT_TURN:
            return "high"
        elif self._current_phase == DrillPhase.BETWEEN_GATES:
            return "medium"
        return "low"

    def _finalize_events(self):
        """Close any open events at end of detection."""
        for event in self._events:
            if event.end_frame is None:
                event.notes += " [Unclosed at end of video]"

    def _get_nearest_cone(
        self,
        ball_pos: Tuple[float, float],
        cone_df: pd.DataFrame,
        frame_id: int
    ) -> Tuple[int, float]:
        """Get nearest cone to ball position."""
        frame_cones = cone_df[cone_df['frame_id'] == frame_id]

        if frame_cones.empty:
            return -1, float('inf')

        distances = np.sqrt(
            (frame_cones['field_center_x'] - ball_pos[0])**2 +
            (frame_cones['field_center_y'] - ball_pos[1])**2
        )

        idx = distances.idxmin()
        return int(frame_cones.loc[idx, 'object_id']), float(distances.min())


# Convenience function
def detect_ball_control(
    ball_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    cone_df: pd.DataFrame,
    config: Optional[AppConfig] = None,
    fps: float = 30.0,
    parquet_dir: Optional[str] = None
) -> DetectionResult:
    """
    Convenience function for Figure-8 drill detection.

    Args:
        ball_df: Ball detection DataFrame
        pose_df: Pose keypoint DataFrame
        cone_df: Cone detection DataFrame
        config: Optional AppConfig (defaults to Figure-8 config)
        fps: Video FPS
        parquet_dir: Path to parquet directory for loading manual cone annotations

    Returns:
        DetectionResult with Figure-8 specific metrics
    """
    if config is None:
        config = AppConfig.for_figure8()
    detector = BallControlDetector(config, parquet_dir=parquet_dir)
    return detector.detect(ball_df, pose_df, cone_df, fps)
