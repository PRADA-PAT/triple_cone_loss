"""
Ball Control Detector - Simplified for Triple Cone drill.

Detects ball control loss events during Triple Cone drill.
Tracks gate passages, drill phases, and lap completion.

DETECTION LOGIC LOCATION:
========================
All loss detection logic is in the `detect_loss()` method.
Modify ONLY that method to implement your detection algorithm.

Current implementation detects:
1. BOUNDARY_VIOLATION - Ball exits video frame
2. BALL_BEHIND_PLAYER - Ball stays behind player for sustained period
"""
import logging
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .config import AppConfig, TripleConeDrillConfig
from .data_structures import (
    ControlState, EventType, FrameData,
    LossEvent, DetectionResult,
    TripleConeDrillPhase, DrillDirection, TripleConeLayout,
    BallTrackingState
)
from .data_loader import (
    extract_ankle_positions,
    get_closest_ankle_per_frame,
    load_triple_cone_layout_from_parquet
)
from .triple_cone_detector import TripleConeDetector, TurnEvent
from .turning_zones import TripleConeZoneSet, TripleConeZoneConfig, create_triple_cone_zones

logger = logging.getLogger(__name__)


class BallControlDetector:
    """
    Main class for detecting ball control loss events in Triple Cone drill.

    Usage:
        config = AppConfig.for_triple_cone()
        detector = BallControlDetector(config)
        result = detector.detect(ball_df, pose_df, cone_df)
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        parquet_dir: Optional[str] = None,
        video_path: Optional[str] = None
    ):
        """
        Initialize detector with Triple Cone config.

        Args:
            config: Application configuration (defaults to Triple Cone config)
            parquet_dir: Path to parquet directory for loading manual cone annotations
            video_path: Path to video file for getting actual video dimensions
        """
        self.config = config or AppConfig.for_triple_cone()
        self.parquet_dir = parquet_dir
        self.video_path = video_path

        if not isinstance(self.config.drill, TripleConeDrillConfig):
            self.config.drill = TripleConeDrillConfig()

        self._detection_config = self.config.detection
        self._drill_config: TripleConeDrillConfig = self.config.drill

        # Triple Cone specific detector (3-cone mode)
        self._cone_detector: Optional[TripleConeDetector] = None

        # Cone layout (3 cones from parquet)
        self._cone_layout: Optional[TripleConeLayout] = None

        # Video dimensions (will be populated from video file)
        self._video_width: int = 1920  # Default fallback
        self._video_height: int = 1080
        if video_path:
            self._load_video_dimensions(video_path)

        # State tracking
        self._current_state = ControlState.CONTROLLED
        self._current_direction = DrillDirection.STATIONARY
        self._current_phase = TripleConeDrillPhase.AT_CONE1
        self._events: List[LossEvent] = []
        self._frame_data: List[FrameData] = []
        self._event_counter = 0

        # Previous frame data
        self._prev_ball_pos: Optional[Tuple[float, float]] = None

        # Hip tracking for ball-behind detection
        self._hip_history: deque = deque(maxlen=15)  # 0.5 sec at 30fps
        self._current_hip_pos: Optional[Tuple[float, float]] = None

        # Direction history for fallback when player stops (STATIC direction)
        self._direction_history: deque = deque(maxlen=15)  # Track recent directions

        # Turning zones (3 zones for CONE1, CONE2, CONE3)
        self._turning_zones: Optional[TripleConeZoneSet] = None

        # Ball-behind detection config (momentum-based) - scaled for 720p
        # NOTE: Must match BALL_POSITION_THRESHOLD in video/annotate_with_json_cones.py
        self._behind_threshold = 9.0  # Pixels - ball must be this far behind hip (720p)
        self._behind_sustained_frames = 10  # ~0.33 sec at 30fps to confirm loss
        self._movement_threshold = 1.4  # Min hip movement to determine direction (720p)

        # Intention-based (face direction) detection config
        # NOTE: Must match thresholds in video/annotate_triple_cone.py
        self._nose_hip_facing_threshold = self._detection_config.nose_hip_facing_threshold
        self._intention_sustained_frames = self._detection_config.intention_sustained_frames
        self._use_intention_detection = self._detection_config.use_intention_detection
        self._min_keypoint_confidence = 0.3  # Same as video/annotate_triple_cone.py

        # Unified boundary tracking state machine
        # Uses interpolated flag to detect when ball actually disappears (off-screen)
        # rather than inferring from position/velocity
        self._ball_tracking_state: BallTrackingState = BallTrackingState.NORMAL
        self._boundary_counter: int = 0
        self._boundary_event_start_frame: Optional[int] = None
        self._boundary_sustained_frames: int = 15  # ~0.5s at 30fps
        self._edge_margin: int = 50  # Matches visualization EDGE_MARGIN

        logger.info(f"BallControlDetector initialized (video: {self._video_width}x{self._video_height})")
        if self._use_intention_detection:
            logger.info("  Intention-based (face direction) detection: ENABLED")

    def _load_video_dimensions(self, video_path: str) -> None:
        """
        Load video dimensions using ffprobe.

        Args:
            video_path: Path to video file
        """
        try:
            result = subprocess.run(
                [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height',
                    '-of', 'csv=p=0',
                    video_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    self._video_width = int(parts[0])
                    self._video_height = int(parts[1])
                    logger.info(f"Loaded video dimensions: {self._video_width}x{self._video_height}")
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
            logger.warning(f"Could not get video dimensions: {e}. Using default 1920x1080")

    def detect(
        self,
        ball_df: pd.DataFrame,
        pose_df: pd.DataFrame,
        cone_df: pd.DataFrame,
        fps: float = 30.0
    ) -> DetectionResult:
        """
        Run ball control detection for Triple Cone drill.

        Args:
            ball_df: Ball detection DataFrame
            pose_df: Pose keypoint DataFrame
            cone_df: Cone detection DataFrame
            fps: Video FPS for timestamps

        Returns:
            DetectionResult with events and frame data
        """
        try:
            logger.info("Starting Triple Cone drill detection...")
            logger.info(f"  Ball frames: {len(ball_df)}")
            logger.info(f"  Pose records: {len(pose_df)}")
            logger.info(f"  Cone records: {len(cone_df)}")

            # Log video dimensions and ball data range for debugging
            # NOTE: Do NOT infer video dimensions from ball data - ball can go off-screen
            # during boundary violations, which would incorrectly suggest larger dimensions.
            if 'center_x' in ball_df.columns:
                logger.info(f"  Video dimensions: {self._video_width}x{self._video_height}")
                logger.info(f"  Ball X range: {ball_df['center_x'].min():.0f} - {ball_df['center_x'].max():.0f}")

            # Reset state
            self._reset_state()

            # Initialize Triple Cone detector (3-cone mode)
            zone_config = TripleConeZoneConfig.default()
            self._cone_detector = TripleConeDetector(self._drill_config, zone_config)

            # Load cone positions from parquet data
            if self.parquet_dir:
                cone_parquet_files = list(Path(self.parquet_dir).glob("*_cone.parquet"))
                if cone_parquet_files:
                    cone_parquet_path = str(cone_parquet_files[0])
                    self._cone_layout = self._cone_detector.setup_from_parquet(cone_parquet_path)
                    self._turning_zones = self._cone_detector.turning_zones
                    logger.info(
                        f"3-cone layout loaded: CONE1=({self._cone_layout.cone1_x:.0f}, {self._cone_layout.cone1_y:.0f}), "
                        f"CONE2=({self._cone_layout.cone2_x:.0f}, {self._cone_layout.cone2_y:.0f}), "
                        f"CONE3=({self._cone_layout.cone3_x:.0f}, {self._cone_layout.cone3_y:.0f})"
                    )
                else:
                    logger.warning(f"No cone parquet found in {self.parquet_dir}")
            else:
                logger.warning("No parquet_dir provided, cannot load cone positions")

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

            # Merge with ball data (include interpolated flag for filtering)
            # Check if field coordinates exist in ball data
            has_ball_field_coords = (
                'field_center_x' in ball_df.columns and
                'field_center_y' in ball_df.columns
            )

            if has_ball_field_coords:
                ball_cols = ['frame_id', 'center_x', 'center_y',
                            'field_center_x', 'field_center_y', 'interpolated']
            else:
                # Fallback: use pixel coordinates when field coords are missing
                logger.info("Ball field coordinates not found - using pixel coordinates")
                ball_cols = ['frame_id', 'center_x', 'center_y', 'interpolated']

            available_cols = [c for c in ball_cols if c in ball_df.columns]
            merged_df = merged_df.merge(ball_df[available_cols], on='frame_id')

            merged_df.rename(columns={
                'center_x': 'ball_x',
                'center_y': 'ball_y',
                'field_center_x': 'ball_field_x',
                'field_center_y': 'ball_field_y',
            }, inplace=True)

            # Create ball_field_x/y from pixel coords if field coords missing
            if not has_ball_field_coords:
                merged_df['ball_field_x'] = merged_df['ball_x']
                merged_df['ball_field_y'] = merged_df['ball_y']

            # Check if ankle field coordinates are all NaN (720p data issue)
            # If so, fall back to pixel coordinates
            if merged_df['ankle_field_x'].isna().all():
                logger.info("Ankle field coordinates are all NaN - using pixel coordinates")
                merged_df['ankle_field_x'] = merged_df['ankle_x']
                merged_df['ankle_field_y'] = merged_df['ankle_y']

            # Calculate ball velocity (field coordinates - for general detection)
            merged_df = merged_df.sort_values('frame_id')
            merged_df['ball_velocity'] = np.sqrt(
                merged_df['ball_field_x'].diff()**2 +
                merged_df['ball_field_y'].diff()**2
            ).fillna(0)

            # Calculate pixel velocity (for boundary stuck detection)
            # Pixel velocity is needed because boundary thresholds are in pixel units
            merged_df['ball_velocity_pixel'] = np.sqrt(
                merged_df['ball_x'].diff()**2 +
                merged_df['ball_y'].diff()**2
            ).fillna(0)

            # Process each frame
            total_frames = len(merged_df)
            processed_frames = set()

            for _, row in merged_df.iterrows():
                frame_id = int(row['frame_id'])
                timestamp = frame_id / fps
                processed_frames.add(frame_id)

                frame_result = self._analyze_frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    row=row,
                    pose_df=pose_df  # Pass pose data for hip extraction
                )

                if frame_result:
                    self._frame_data.append(frame_result)

            # Check ball-only frames for boundary violations
            # (handles cases where player is off-screen but ball is stuck at edge)
            self._check_ball_only_boundary_violations(ball_df, processed_frames, fps)

            # Finalize any open events
            self._finalize_events()

            result = DetectionResult(
                success=True,
                total_frames=total_frames,
                events=self._events,
                frame_data=self._frame_data,
                total_laps=self._cone_detector.rep_count if self._cone_detector else 0,
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
        self._current_phase = TripleConeDrillPhase.AT_CONE1
        self._events = []
        self._frame_data = []
        self._event_counter = 0
        self._prev_ball_pos = None

        # Reset hip tracking
        self._hip_history.clear()
        self._current_hip_pos = None

        # Reset direction history
        self._direction_history.clear()

        if self._cone_detector:
            self._cone_detector.reset()

    def _analyze_frame(
        self,
        frame_id: int,
        timestamp: float,
        row: pd.Series,
        pose_df: Optional[pd.DataFrame] = None
    ) -> Optional[FrameData]:
        """
        Analyze a single frame.

        Gathers data and calls detect_loss() to determine control state.
        Uses static cone positions from JSON annotations (stored in self._cone_roles).
        """
        # Get positions
        ball_pos = (row['ball_field_x'], row['ball_field_y'])
        ball_pixel_pos = (row['ball_x'], row['ball_y'])
        ankle_pos = (row['ankle_field_x'], row['ankle_field_y'])
        distance = row['ball_ankle_distance']
        velocity = row['ball_velocity']
        velocity_pixel = row.get('ball_velocity_pixel', velocity)  # Pixel velocity for boundary detection

        # Extract hip and nose positions for ball-behind detection
        hip_pixel_pos: Optional[Tuple[float, float]] = None
        nose_pixel_pos: Optional[Tuple[float, float]] = None
        if pose_df is not None:
            hip_pixel_pos = self._extract_hip_position(frame_id, pose_df)
            if self._use_intention_detection:
                nose_pixel_pos = self._extract_nose_position(frame_id, pose_df)

        # Update hip history and calculate player movement direction
        player_direction: Optional[str] = None
        if hip_pixel_pos is not None:
            self._hip_history.append(hip_pixel_pos)
            self._current_hip_pos = hip_pixel_pos

            # Calculate direction from hip movement
            if len(self._hip_history) >= 2:
                prev_hip = self._hip_history[0]
                dx = hip_pixel_pos[0] - prev_hip[0]
                if dx > self._movement_threshold:
                    player_direction = "RIGHT"  # Moving toward start cone
                elif dx < -self._movement_threshold:
                    player_direction = "LEFT"  # Moving toward gate 2

        # Store direction in history (including None for STATIC)
        self._direction_history.append(player_direction)

        # If direction is None (STATIC), fall back to last known direction
        # This prevents ball-behind detection from being skipped when player stops
        if player_direction is None:
            for d in reversed(self._direction_history):
                if d is not None:
                    player_direction = d
                    break

        # Check if ball is in turning zone
        in_turning_zone: Optional[str] = None
        if self._turning_zones is not None:
            in_turning_zone = self._turning_zones.get_zone_at_point(
                ball_pixel_pos[0], ball_pixel_pos[1]
            )

        # Calculate intention-based (face direction) ball position
        facing_direction: Optional[str] = None
        ball_behind_intention: Optional[bool] = None
        ball_intention_position: Optional[str] = None
        if self._use_intention_detection and nose_pixel_pos is not None:
            facing_direction = self._determine_facing_direction(nose_pixel_pos, hip_pixel_pos)
            ball_behind_intention, ball_intention_position = self._is_ball_behind_intention(
                ball_pixel_pos, hip_pixel_pos, facing_direction
            )

        # Extract ball interpolated flag for boundary tracking
        # ball_interpolated = True means ball wasn't actually detected (position was filled in)
        # Note: column is 'interpolated' from ball_df merge, not 'ball_interpolated'
        ball_interpolated = bool(row.get('interpolated', False))

        # ============================================================
        # DETECTION LOGIC - calls detect_loss()
        # ============================================================
        is_loss, loss_type = self.detect_loss(
            ball_pos=ball_pos,
            ball_pixel_pos=ball_pixel_pos,
            ankle_pos=ankle_pos,
            distance=distance,
            velocity=velocity,
            velocity_pixel=velocity_pixel,
            frame_id=frame_id,
            timestamp=timestamp,
            history=self._frame_data,
            hip_pixel_pos=hip_pixel_pos,
            player_direction=player_direction,
            in_turning_zone=in_turning_zone,
            # Intention-based parameters
            facing_direction=facing_direction,
            ball_behind_intention=ball_behind_intention,
            # Boundary tracking parameter
            ball_interpolated=ball_interpolated
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

        # Get nearest cone (uses 3-cone layout)
        ball_pixel_pos = (row['ball_x'], row['ball_y'])
        nearest_cone_id, nearest_cone_dist = self._get_nearest_cone(ball_pixel_pos)

        # Triple Cone tracking (3-cone phase and direction)
        drill_phase = None
        drill_direction = None
        current_gate = None  # Legacy field, not used in 3-cone drill

        if self._cone_detector:
            # Calculate direction from ball movement
            if self._prev_ball_pos:
                drill_direction = self._cone_detector.get_direction_from_movement(
                    self._prev_ball_pos[0], ball_pos[0]
                )
                self._current_direction = drill_direction
            else:
                drill_direction = DrillDirection.STATIONARY

            # Update detector state (tracks phase, detects turns)
            state = self._cone_detector.update(
                ball_pixel_pos,  # Use pixel coords for zone detection
                drill_direction,
                frame_id,
                timestamp
            )
            drill_phase = state.phase
            self._current_phase = drill_phase

        self._prev_ball_pos = ball_pos

        # Simple control score (just for reporting, not used in detection)
        control_score = max(0.0, 1.0 - (distance / self._detection_config.loss_distance_threshold))

        # Calculate ball-behind status for this frame
        ball_behind = self._is_ball_behind(
            ball_pixel_pos, hip_pixel_pos, player_direction
        )

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
            lap_count=self._cone_detector.rep_count if self._cone_detector else 0,
            # New fields for ball-behind detection (momentum-based)
            hip_x=hip_pixel_pos[0] if hip_pixel_pos else None,
            hip_y=hip_pixel_pos[1] if hip_pixel_pos else None,
            player_movement_direction=player_direction,
            ball_behind_player=ball_behind,
            in_turning_zone=in_turning_zone,
            # Intention-based (face direction) fields
            nose_x=nose_pixel_pos[0] if nose_pixel_pos else None,
            nose_y=nose_pixel_pos[1] if nose_pixel_pos else None,
            player_facing_direction=facing_direction,
            ball_behind_intention=ball_behind_intention,
            ball_intention_position=ball_intention_position,
            # Ball tracking quality
            ball_interpolated=ball_interpolated,
            ball_tracking_state=self._ball_tracking_state,
        )

    # ================================================================
    # HELPER METHODS FOR BALL-BEHIND DETECTION
    # ================================================================

    def _extract_hip_position(
        self,
        frame_id: int,
        pose_df: pd.DataFrame
    ) -> Optional[Tuple[float, float]]:
        """
        Extract hip position from pose data for a given frame.

        Args:
            frame_id: Frame number to extract hip for
            pose_df: Full pose DataFrame

        Returns:
            (hip_x, hip_y) in pixels, or None if not found
        """
        # Look for 'hip' keypoint in this frame
        frame_pose = pose_df[pose_df['frame_idx'] == frame_id]
        if frame_pose.empty:
            return None

        # Find hip keypoint (try 'hip' first, then calculate from left/right hip)
        hip_row = frame_pose[frame_pose['keypoint_name'] == 'hip']
        if not hip_row.empty:
            hip = hip_row.iloc[0]
            if hip['confidence'] >= 0.3:  # Minimum confidence threshold
                return (float(hip['x']), float(hip['y']))

        # Fallback: calculate from left_hip and right_hip
        left_hip = frame_pose[frame_pose['keypoint_name'] == 'left_hip']
        right_hip = frame_pose[frame_pose['keypoint_name'] == 'right_hip']

        if not left_hip.empty and not right_hip.empty:
            lh = left_hip.iloc[0]
            rh = right_hip.iloc[0]
            if lh['confidence'] >= 0.3 and rh['confidence'] >= 0.3:
                hip_x = (lh['x'] + rh['x']) / 2
                hip_y = (lh['y'] + rh['y']) / 2
                return (float(hip_x), float(hip_y))

        return None

    def _is_ball_behind(
        self,
        ball_pixel_pos: Tuple[float, float],
        hip_pixel_pos: Optional[Tuple[float, float]],
        player_direction: Optional[str]
    ) -> Optional[bool]:
        """
        Determine if ball is behind player relative to movement direction.

        Args:
            ball_pixel_pos: Ball position in pixels
            hip_pixel_pos: Hip position in pixels (or None)
            player_direction: "LEFT", "RIGHT", or None

        Returns:
            True if ball is behind, False if in front, None if can't determine
        """
        if hip_pixel_pos is None or player_direction is None:
            return None

        delta_x = ball_pixel_pos[0] - hip_pixel_pos[0]

        # Ball is "behind" if it's on the opposite side of movement direction
        # Simple rule: ball should be on the same side as where player is heading
        if player_direction == "LEFT":
            # Moving LEFT: ball to RIGHT of hip = BEHIND
            return delta_x > self._behind_threshold
        elif player_direction == "RIGHT":
            # Moving RIGHT: ball to LEFT of hip = BEHIND
            return delta_x < -self._behind_threshold

        return None

    def _check_edge_zone(self, ball_x: float) -> Tuple[bool, str]:
        """
        Check if ball is in edge zone (near screen edge).

        Args:
            ball_x: Ball X position in pixels

        Returns:
            (in_edge_zone, edge_side) where edge_side is "LEFT", "RIGHT", or "NONE"
        """
        left_distance = ball_x
        right_distance = self._video_width - ball_x

        if right_distance < self._edge_margin:
            return True, "RIGHT"
        if left_distance < self._edge_margin:
            return True, "LEFT"
        return False, "NONE"

    def _update_ball_tracking_state(
        self,
        ball_visible: bool,
        in_edge_zone: bool,
        edge_side: str
    ) -> Tuple[BallTrackingState, bool]:
        """
        Update state machine for unified boundary tracking.

        This state machine tracks whether the ball has exited the video frame
        by using the interpolated flag (ball_visible = not interpolated).

        State transitions:
            NORMAL → EDGE_LEFT/RIGHT (ball enters edge zone)
            EDGE_LEFT/RIGHT → OFF_SCREEN_LEFT/RIGHT (ball disappears)
            OFF_SCREEN_LEFT/RIGHT → NORMAL (ball returns)
            NORMAL → DISAPPEARED_MID (ball disappears mid-field)

        Args:
            ball_visible: True if ball was actually detected (not interpolated)
            in_edge_zone: True if ball is in edge zone
            edge_side: "LEFT", "RIGHT", or "NONE"

        Returns:
            (new_state, should_reset_counter)
        """
        current = self._ball_tracking_state

        # Ball not visible (interpolated)
        if not ball_visible:
            if current == BallTrackingState.EDGE_LEFT:
                return BallTrackingState.OFF_SCREEN_LEFT, False
            elif current == BallTrackingState.EDGE_RIGHT:
                return BallTrackingState.OFF_SCREEN_RIGHT, False
            elif current in (BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
                return current, False  # Stay in off-screen
            elif current == BallTrackingState.DISAPPEARED_MID:
                return current, False
            else:  # NORMAL → mid-field disappearance
                return BallTrackingState.DISAPPEARED_MID, True

        # Ball is visible
        if current == BallTrackingState.NORMAL:
            if in_edge_zone:
                if edge_side == "LEFT":
                    return BallTrackingState.EDGE_LEFT, True
                else:
                    return BallTrackingState.EDGE_RIGHT, True
            return BallTrackingState.NORMAL, False

        elif current == BallTrackingState.EDGE_LEFT:
            if in_edge_zone:
                if edge_side == "LEFT":
                    return BallTrackingState.EDGE_LEFT, False
                else:  # Switched to RIGHT
                    return BallTrackingState.EDGE_RIGHT, True
            else:
                return BallTrackingState.NORMAL, True

        elif current == BallTrackingState.EDGE_RIGHT:
            if in_edge_zone:
                if edge_side == "RIGHT":
                    return BallTrackingState.EDGE_RIGHT, False
                else:  # Switched to LEFT
                    return BallTrackingState.EDGE_LEFT, True
            else:
                return BallTrackingState.NORMAL, True

        elif current == BallTrackingState.OFF_SCREEN_LEFT:
            if in_edge_zone:
                if edge_side == "LEFT":
                    return BallTrackingState.EDGE_LEFT, False  # Still in danger
                else:
                    return BallTrackingState.EDGE_RIGHT, True
            else:
                return BallTrackingState.NORMAL, True

        elif current == BallTrackingState.OFF_SCREEN_RIGHT:
            if in_edge_zone:
                if edge_side == "RIGHT":
                    return BallTrackingState.EDGE_RIGHT, False  # Still in danger
                else:
                    return BallTrackingState.EDGE_LEFT, True
            else:
                return BallTrackingState.NORMAL, True

        elif current == BallTrackingState.DISAPPEARED_MID:
            if in_edge_zone:
                if edge_side == "LEFT":
                    return BallTrackingState.EDGE_LEFT, True
                else:
                    return BallTrackingState.EDGE_RIGHT, True
            else:
                return BallTrackingState.NORMAL, True

        return current, False

    def _extract_nose_position(
        self,
        frame_id: int,
        pose_df: pd.DataFrame
    ) -> Optional[Tuple[float, float]]:
        """
        Extract nose position from pose data for a given frame.

        Args:
            frame_id: Frame number to extract nose for
            pose_df: Full pose DataFrame

        Returns:
            (nose_x, nose_y) in pixels, or None if not found
        """
        frame_pose = pose_df[pose_df['frame_idx'] == frame_id]
        if frame_pose.empty:
            return None

        nose_row = frame_pose[frame_pose['keypoint_name'] == 'nose']
        if not nose_row.empty:
            nose = nose_row.iloc[0]
            if nose['confidence'] >= self._min_keypoint_confidence:
                return (float(nose['x']), float(nose['y']))

        return None

    def _determine_facing_direction(
        self,
        nose_pos: Optional[Tuple[float, float]],
        hip_pos: Optional[Tuple[float, float]]
    ) -> Optional[str]:
        """
        Determine if player torso is facing LEFT or RIGHT.

        Uses nose position relative to hip:
        - Facing RIGHT: nose.x > hip.x (head is to the right of body center)
        - Facing LEFT: nose.x < hip.x (head is to the left of body center)

        This captures turn intention since head turns before body.

        Args:
            nose_pos: Nose position in pixels
            hip_pos: Hip position in pixels

        Returns:
            'RIGHT', 'LEFT', or None if data unreliable
        """
        if nose_pos is None or hip_pos is None:
            return None

        diff = nose_pos[0] - hip_pos[0]

        if diff > self._nose_hip_facing_threshold:
            return "RIGHT"
        elif diff < -self._nose_hip_facing_threshold:
            return "LEFT"
        else:
            return None  # Aligned/neutral

    def _is_ball_behind_intention(
        self,
        ball_pixel_pos: Tuple[float, float],
        hip_pixel_pos: Optional[Tuple[float, float]],
        facing_direction: Optional[str]
    ) -> Tuple[Optional[bool], Optional[str]]:
        """
        Determine if ball is behind player relative to FACING direction (intention).

        Unlike momentum-based detection, this uses where the player is LOOKING
        (nose-hip orientation) rather than where they're MOVING.

        Args:
            ball_pixel_pos: Ball position in pixels
            hip_pixel_pos: Hip position in pixels (or None)
            facing_direction: "LEFT", "RIGHT", or None (from nose-hip)

        Returns:
            Tuple of:
            - is_behind: True if ball is behind, False if in front, None if can't determine
            - position: "I-FRONT", "I-BEHIND", "I-ALIGNED", or None
        """
        if hip_pixel_pos is None or facing_direction is None:
            return None, None

        delta_x = ball_pixel_pos[0] - hip_pixel_pos[0]

        # Check if ball is aligned with player
        if abs(delta_x) < self._behind_threshold:
            return False, "I-ALIGNED"

        # Determine front/behind based on facing direction
        if facing_direction == "LEFT":
            # Facing left: FRONT = ball to left (negative delta)
            if delta_x < 0:
                return False, "I-FRONT"
            else:
                return True, "I-BEHIND"
        else:  # RIGHT
            # Facing right: FRONT = ball to right (positive delta)
            if delta_x > 0:
                return False, "I-FRONT"
            else:
                return True, "I-BEHIND"

    # ================================================================
    # DETECTION LOGIC - CATEGORIZED LOSS DETECTION
    # ================================================================
    def detect_loss(
        self,
        ball_pos: Tuple[float, float],
        ball_pixel_pos: Tuple[float, float],
        ankle_pos: Tuple[float, float],
        distance: float,
        velocity: float,
        frame_id: int,
        timestamp: float,
        history: List[FrameData],
        hip_pixel_pos: Optional[Tuple[float, float]] = None,
        player_direction: Optional[str] = None,
        in_turning_zone: Optional[str] = None,
        velocity_pixel: Optional[float] = None,
        # Intention-based parameters
        facing_direction: Optional[str] = None,
        ball_behind_intention: Optional[bool] = None,
        # Boundary tracking parameter
        ball_interpolated: bool = False
    ) -> Tuple[bool, Optional[EventType]]:
        """
        Detect if ball control is lost.

        Detects three types of loss events:
        1. BOUNDARY_VIOLATION - Ball exits video frame (via unified state machine)
        2. BALL_BEHIND_PLAYER - Ball stays behind player for sustained period (momentum-based)
        3. BALL_BEHIND_INTENTION - Ball stays behind player's facing direction (intention-based)

        Args:
            ball_pos: Ball position (field_x, field_y)
            ball_pixel_pos: Ball position in pixels (x, y)
            ankle_pos: Closest ankle position (field_x, field_y)
            distance: Ball-to-ankle distance (pre-calculated)
            velocity: Ball velocity this frame (field coordinates)
            frame_id: Current frame number
            timestamp: Current timestamp in seconds
            history: List of previous FrameData (for temporal analysis)
            hip_pixel_pos: Player hip position in pixels (for ball-behind detection)
            player_direction: "LEFT", "RIGHT", or None (movement-based)
            in_turning_zone: "CONE1", "CONE2", "CONE3", or None (suppress ball-behind in zones)
            velocity_pixel: Ball velocity in pixels/frame (unused, kept for compatibility)
            facing_direction: "LEFT", "RIGHT", or None (from nose-hip orientation)
            ball_behind_intention: True if ball is behind facing direction
            ball_interpolated: True if ball position was interpolated (not detected)

        Returns:
            Tuple of:
            - is_loss: True if control is lost, False otherwise
            - loss_type: EventType if loss detected, None otherwise
        """
        # ============================================================
        # 1. BOUNDARY VIOLATION - Ball exits video frame (unified state machine)
        # ============================================================
        # Uses interpolated flag to detect when ball actually disappears,
        # rather than inferring from position/velocity. This is more accurate
        # because the ball detection actually disappears when off-screen.
        #
        # State machine: NORMAL → EDGE → OFF_SCREEN → NORMAL
        # Only triggers violation when ball goes OFF_SCREEN (via edge zone first)

        MIN_TIMESTAMP = 3.0

        # Skip early frames
        if timestamp < MIN_TIMESTAMP:
            return False, None

        # Determine if ball was actually detected (not interpolated)
        ball_visible = not ball_interpolated

        # Check edge zone status
        in_edge_zone, edge_side = self._check_edge_zone(ball_pixel_pos[0])

        # Update state machine
        new_state, should_reset = self._update_ball_tracking_state(
            ball_visible, in_edge_zone, edge_side
        )

        # Handle counter reset
        if should_reset:
            self._boundary_counter = 0
            self._boundary_event_start_frame = None

        self._ball_tracking_state = new_state

        # Increment counter and check for loss in off-screen states
        if new_state in (BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
            self._boundary_counter += 1
            if self._boundary_event_start_frame is None:
                self._boundary_event_start_frame = frame_id

            # Trigger boundary violation after sustained off-screen
            if self._boundary_counter >= self._boundary_sustained_frames:
                side = "LEFT" if new_state == BallTrackingState.OFF_SCREEN_LEFT else "RIGHT"
                logger.debug(
                    f"Frame {frame_id}: BOUNDARY_VIOLATION at {side} edge "
                    f"(off-screen for {self._boundary_counter} frames, state={new_state.value})"
                )
                return True, EventType.BOUNDARY_VIOLATION
        elif new_state in (BallTrackingState.EDGE_LEFT, BallTrackingState.EDGE_RIGHT):
            # Increment counter while in edge zone (preparing for potential exit)
            self._boundary_counter += 1

        # ============================================================
        # 2. BALL_BEHIND_INTENTION - Ball behind player's facing direction (PRIMARY)
        # ============================================================
        # Uses nose-hip orientation (intention) instead of movement direction.
        # This is MORE ACCURATE than momentum-based detection because it captures
        # where the player is LOOKING, not just where they're moving.
        # Checked FIRST before momentum-based detection.

        if self._use_intention_detection:
            # NOTE: Unlike BALL_BEHIND_PLAYER, we do NOT skip turning zones for intention detection.
            # Intention (facing direction) should be tracked even during turns.

            # Only proceed if we have facing direction information
            if facing_direction is not None and ball_behind_intention is not None:
                # Check if ball is currently behind facing direction
                if not ball_behind_intention:
                    # Ball is in front of facing direction - not a loss condition
                    # Do not continue LOST state even if ball is far from hip
                    # (being far but IN FRONT means player is chasing successfully)
                    pass
                else:
                    # Ball IS behind intention - check for sustained pattern
                    if len(history) >= self._intention_sustained_frames:
                        recent = history[-self._intention_sustained_frames:]

                        # Count consecutive frames where ball was behind intention
                        behind_count = 0
                        for frame in recent:
                            if frame.ball_behind_intention is True:
                                behind_count += 1
                            else:
                                behind_count = 0

                        # Only trigger if ball was behind for entire sustained period
                        if behind_count >= self._intention_sustained_frames - 1:
                            # Verify player facing direction was consistent
                            facings = [f.player_facing_direction for f in recent if f.player_facing_direction]
                            if len(facings) >= self._intention_sustained_frames // 2:
                                dominant_facing = max(set(facings), key=facings.count)
                                same_facing_ratio = facings.count(dominant_facing) / len(facings)

                                if same_facing_ratio >= 0.7:  # 70% consistency threshold
                                    logger.debug(
                                        f"Frame {frame_id}: BALL_BEHIND_INTENTION detected "
                                        f"(behind for {behind_count} frames, facing={dominant_facing})"
                                    )
                                    return True, EventType.BALL_BEHIND_INTENTION

        # ============================================================
        # 3. BALL_BEHIND_PLAYER - Ball stays behind for sustained period (FALLBACK)
        # ============================================================
        # Only trigger if:
        # - Intention-based detection didn't trigger (above)
        # - Player has clear movement direction
        # - Ball is behind player (opposite to movement direction)
        # - NOT in a turning zone (where "behind" is expected briefly)
        # - Sustained for N consecutive frames

        # Skip if in turning zone (turning zones are where "behind" is expected)
        if in_turning_zone is not None:
            return False, None

        # Skip if no direction information
        if hip_pixel_pos is None or player_direction is None:
            return False, None

        # Check if ball is currently behind player
        is_behind = self._is_ball_behind(ball_pixel_pos, hip_pixel_pos, player_direction)

        # If ball is NOT behind, it's not a loss condition
        # Do not continue LOST state even if ball is far from hip
        # (being far but IN FRONT means player is chasing successfully)
        if not is_behind:
            return False, None

        # Check for sustained "behind" pattern in history
        if len(history) >= self._behind_sustained_frames:
            recent = history[-self._behind_sustained_frames:]

            # Count consecutive frames where ball was behind
            behind_count = 0
            for frame in recent:
                # Check if this frame had ball behind (using stored value)
                if frame.ball_behind_player is True:
                    behind_count += 1
                else:
                    # Reset counter if we find a "not behind" frame
                    behind_count = 0

            # Only trigger if ball was behind for entire sustained period
            if behind_count >= self._behind_sustained_frames - 1:
                # Also verify player was moving consistently (not turning)
                directions = [f.player_movement_direction for f in recent if f.player_movement_direction]
                if len(directions) >= self._behind_sustained_frames // 2:
                    # Check if mostly same direction
                    dominant_direction = max(set(directions), key=directions.count)
                    same_direction_ratio = directions.count(dominant_direction) / len(directions)

                    if same_direction_ratio >= 0.7:  # 70% consistency threshold
                        logger.debug(
                            f"Frame {frame_id}: BALL_BEHIND_PLAYER detected "
                            f"(behind for {behind_count} frames, direction={dominant_direction})"
                        )
                        return True, EventType.BALL_BEHIND_PLAYER

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
        """Get severity based on current drill phase (3-cone)."""
        # High severity at turning cones
        if self._current_phase in [
            TripleConeDrillPhase.AT_CONE2,
            TripleConeDrillPhase.AT_CONE3,
        ]:
            return "high"
        # Medium severity while moving between cones
        elif self._current_phase in [
            TripleConeDrillPhase.GOING_TO_CONE2,
            TripleConeDrillPhase.GOING_TO_CONE3,
            TripleConeDrillPhase.RETURNING_FROM_CONE2,
            TripleConeDrillPhase.RETURNING_FROM_CONE3,
        ]:
            return "medium"
        # Low severity at home cone
        return "low"

    def _check_ball_only_boundary_violations(
        self,
        ball_df: pd.DataFrame,
        processed_frames: set,
        fps: float
    ):
        """
        Check for boundary violations in frames that have ball data but no ankle data.

        Uses unified edge+off-screen tracking state machine (same logic as visualization):
        - Ball enters edge zone → start counting
        - Ball disappears (NaN position) while in edge zone → continue counting (OFF_SCREEN)
        - Counter exceeds threshold → boundary violation

        This handles cases where the player kicks the ball off-screen - the ball
        crosses the edge zone and then disappears (object detection loses it).

        Args:
            ball_df: Ball detection DataFrame with center_x, center_y
            processed_frames: Set of frame IDs already processed in main loop
            fps: Video FPS for timestamp calculation
        """
        # Thresholds - must match visualization (annotate_triple_cone.py)
        EDGE_MARGIN = self._edge_margin  # Usually 50px
        MIN_SUSTAINED_FRAMES = 15  # ~0.5s at 30fps to confirm boundary violation
        MIN_TIMESTAMP = 3.0  # Skip first 3 seconds

        # Get all frames (not just ball-only) for continuous state machine tracking
        # We need to track state transitions across ALL frames
        all_frames = sorted(ball_df['frame_id'].unique())

        if not all_frames:
            return

        logger.debug(f"Checking {len(all_frames)} frames for boundary violations (ball-only check)")

        # Create lookup for ball positions
        ball_lookup = {}
        for _, row in ball_df.iterrows():
            frame_id = int(row['frame_id'])
            ball_x = row['center_x']
            ball_y = row['center_y']
            interpolated = row.get('interpolated', False)
            ball_lookup[frame_id] = {
                'x': ball_x,
                'y': ball_y,
                'interpolated': interpolated,
                'visible': not (pd.isna(ball_x) or interpolated)
            }

        # State machine tracking (mirrors visualization logic)
        current_state = BallTrackingState.NORMAL
        counter = 0
        event_start_frame = None
        event_edge_side = None
        went_off_screen = False  # Track if ball actually went off-screen during this sequence

        for frame_id in all_frames:
            timestamp = frame_id / fps

            # Skip early frames
            if timestamp < MIN_TIMESTAMP:
                continue

            # NOTE: We process ALL frames here, not just ball-only frames
            # This is because boundary violations can start in frames with ankle data
            # (ball enters edge zone) and continue into frames without ankle data
            # (ball goes off-screen, player may also leave frame)
            # The main loop doesn't reliably detect these because it requires
            # ankle data for each frame, causing gaps in tracking.

            ball_info = ball_lookup.get(frame_id, {'x': np.nan, 'visible': False})
            ball_x = ball_info['x']
            ball_visible = ball_info['visible']

            # Determine edge zone status
            in_edge_zone = False
            edge_side = "NONE"
            if not pd.isna(ball_x):
                if ball_x < EDGE_MARGIN:
                    in_edge_zone = True
                    edge_side = "LEFT"
                elif ball_x > (self._video_width - EDGE_MARGIN):
                    in_edge_zone = True
                    edge_side = "RIGHT"

            # Update state machine (same logic as _update_ball_tracking_state)
            prev_state = current_state
            should_reset = False

            if not ball_visible:
                # Ball not visible - check if we were in edge zone
                if current_state == BallTrackingState.EDGE_LEFT:
                    current_state = BallTrackingState.OFF_SCREEN_LEFT
                    went_off_screen = True  # Ball actually went off-screen via left edge
                elif current_state == BallTrackingState.EDGE_RIGHT:
                    current_state = BallTrackingState.OFF_SCREEN_RIGHT
                    went_off_screen = True  # Ball actually went off-screen via right edge
                elif current_state in (BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
                    pass  # Stay in off-screen state (already went_off_screen)
                elif current_state == BallTrackingState.DISAPPEARED_MID:
                    pass  # Stay in disappeared mid
                else:
                    # NORMAL → disappeared mid-field
                    current_state = BallTrackingState.DISAPPEARED_MID
                    should_reset = True
            else:
                # Ball is visible
                if current_state == BallTrackingState.NORMAL:
                    if in_edge_zone:
                        if edge_side == "LEFT":
                            current_state = BallTrackingState.EDGE_LEFT
                        else:
                            current_state = BallTrackingState.EDGE_RIGHT
                        should_reset = True
                elif current_state == BallTrackingState.EDGE_LEFT:
                    if not in_edge_zone:
                        current_state = BallTrackingState.NORMAL
                        should_reset = True
                    elif edge_side == "RIGHT":
                        current_state = BallTrackingState.EDGE_RIGHT
                        should_reset = True
                elif current_state == BallTrackingState.EDGE_RIGHT:
                    if not in_edge_zone:
                        current_state = BallTrackingState.NORMAL
                        should_reset = True
                    elif edge_side == "LEFT":
                        current_state = BallTrackingState.EDGE_LEFT
                        should_reset = True
                elif current_state == BallTrackingState.OFF_SCREEN_LEFT:
                    if in_edge_zone and edge_side == "LEFT":
                        current_state = BallTrackingState.EDGE_LEFT
                    elif in_edge_zone and edge_side == "RIGHT":
                        current_state = BallTrackingState.EDGE_RIGHT
                        should_reset = True
                    else:
                        current_state = BallTrackingState.NORMAL
                        should_reset = True
                elif current_state == BallTrackingState.OFF_SCREEN_RIGHT:
                    if in_edge_zone and edge_side == "RIGHT":
                        current_state = BallTrackingState.EDGE_RIGHT
                    elif in_edge_zone and edge_side == "LEFT":
                        current_state = BallTrackingState.EDGE_LEFT
                        should_reset = True
                    else:
                        current_state = BallTrackingState.NORMAL
                        should_reset = True
                elif current_state == BallTrackingState.DISAPPEARED_MID:
                    if in_edge_zone:
                        if edge_side == "LEFT":
                            current_state = BallTrackingState.EDGE_LEFT
                        else:
                            current_state = BallTrackingState.EDGE_RIGHT
                        should_reset = True
                    else:
                        current_state = BallTrackingState.NORMAL
                        should_reset = True

            # Handle counter and event creation
            if should_reset:
                # Check if we're transitioning OUT of a tracked state
                if prev_state in (BallTrackingState.EDGE_LEFT, BallTrackingState.EDGE_RIGHT,
                                  BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
                    # Only create event if ball actually went OFF_SCREEN (not just edge zone entry)
                    # This prevents false positives from brief edge zone touches
                    if counter >= MIN_SUSTAINED_FRAMES and event_start_frame is not None and went_off_screen:
                        self._create_ball_only_boundary_event(
                            event_start_frame, frame_id - 1, event_edge_side, fps
                        )
                counter = 0
                event_start_frame = None
                event_edge_side = None
                went_off_screen = False  # Reset for next sequence

            # Increment counter for tracked states
            if current_state in (BallTrackingState.EDGE_LEFT, BallTrackingState.EDGE_RIGHT,
                                BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
                if event_start_frame is None:
                    event_start_frame = frame_id
                    event_edge_side = "LEFT" if current_state in (BallTrackingState.EDGE_LEFT, BallTrackingState.OFF_SCREEN_LEFT) else "RIGHT"
                counter += 1

        # Handle sequence that extends to end of frames
        # Only create event if ball actually went OFF_SCREEN
        if counter >= MIN_SUSTAINED_FRAMES and event_start_frame is not None and went_off_screen:
            self._create_ball_only_boundary_event(
                event_start_frame, all_frames[-1], event_edge_side, fps
            )

    def _create_ball_only_boundary_event(
        self,
        start_frame: int,
        end_frame: int,
        edge_type: str,
        fps: float
    ):
        """Create a boundary violation event detected via edge+off-screen tracking."""
        self._event_counter += 1

        # Placeholder positions for ball-only events (player may be off-screen)
        edge_x = self._video_width if edge_type == "RIGHT" else 0.0
        placeholder_pos = (edge_x, self._video_height / 2)

        duration_frames = end_frame - start_frame + 1

        event = LossEvent(
            event_id=self._event_counter,
            event_type=EventType.BOUNDARY_VIOLATION,
            start_frame=start_frame,
            end_frame=end_frame,
            start_timestamp=start_frame / fps,
            end_timestamp=end_frame / fps,
            ball_position=placeholder_pos,
            player_position=placeholder_pos,  # Unknown - player may be off-screen
            distance_at_loss=0.0,  # Unknown - no ankle data
            velocity_at_loss=0.0,
            nearest_cone_id=-1,  # Not relevant for boundary violations
            gate_context=None,
            severity="high",  # Boundary violations are always high severity
            notes=f"Ball exited via {edge_type} edge ({duration_frames}f)"
        )

        self._events.append(event)
        logger.info(
            f"Boundary violation: frames {start_frame}-{end_frame} "
            f"({event.start_timestamp:.2f}s-{event.end_timestamp:.2f}s) via {edge_type} edge"
        )

    def _finalize_events(self):
        """Close any open events, merge overlapping, and filter short events."""
        MIN_EVENT_FRAMES = 15  # Minimum 0.5 seconds at 30fps

        for event in self._events:
            if event.end_frame is None:
                event.notes += " [Unclosed at end of video]"

        # Merge overlapping events of the same type
        self._events = self._merge_overlapping_events(self._events)

        # Filter out very short events (likely false positives)
        original_count = len(self._events)
        self._events = [e for e in self._events if e.duration_frames >= MIN_EVENT_FRAMES]
        filtered_count = original_count - len(self._events)
        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} short events (<{MIN_EVENT_FRAMES} frames)")

    def _merge_overlapping_events(self, events: List[LossEvent]) -> List[LossEvent]:
        """
        Merge events that overlap in time and are of the same type.

        This handles cases where ball-only boundary detection creates events
        that overlap with regular boundary detection events.
        """
        if len(events) <= 1:
            return events

        # Sort events by start frame
        sorted_events = sorted(events, key=lambda e: e.start_frame)
        merged = []

        for event in sorted_events:
            if not merged:
                merged.append(event)
                continue

            last = merged[-1]

            # Check if events overlap or are adjacent (within 5 frames)
            # and are the same type
            overlap_threshold = 5  # frames
            events_overlap = (
                event.event_type == last.event_type and
                event.start_frame <= (last.end_frame or last.start_frame) + overlap_threshold
            )

            if events_overlap:
                # Merge: extend last event's end frame
                new_end = max(last.end_frame or last.start_frame, event.end_frame or event.start_frame)
                last.end_frame = new_end
                last.end_timestamp = new_end / 30.0  # Assume 30fps
                # Combine notes if different
                if event.notes and event.notes not in (last.notes or ""):
                    last.notes = f"{last.notes or ''} {event.notes}".strip()
                logger.debug(
                    f"Merged overlapping {event.event_type.name} events: "
                    f"{last.start_frame}-{last.end_frame}"
                )
            else:
                merged.append(event)

        if len(merged) < len(events):
            logger.info(f"Merged {len(events) - len(merged)} overlapping events")

        return merged

    def _get_nearest_cone(
        self,
        ball_pos: Tuple[float, float]
    ) -> Tuple[int, float]:
        """
        Get nearest cone to ball position using 3-cone layout.

        Uses self._cone_layout which contains 3 cone positions from parquet data.
        Returns (cone_id, distance) where cone_id is 1, 2, or 3.
        """
        if self._cone_layout is None:
            return -1, float('inf')

        # Calculate distance to each cone
        cones = [
            (1, self._cone_layout.cone1),
            (2, self._cone_layout.cone2),
            (3, self._cone_layout.cone3),
        ]

        min_dist = float('inf')
        nearest_id = -1

        for cone_id, (cone_x, cone_y) in cones:
            dist = np.sqrt(
                (cone_x - ball_pos[0])**2 +
                (cone_y - ball_pos[1])**2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_id = cone_id

        return nearest_id, min_dist


# Convenience function
def detect_ball_control(
    ball_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    cone_df: pd.DataFrame,
    config: Optional[AppConfig] = None,
    fps: float = 30.0,
    parquet_dir: Optional[str] = None,
    video_path: Optional[str] = None
) -> DetectionResult:
    """
    Convenience function for Triple Cone drill detection.

    Args:
        ball_df: Ball detection DataFrame
        pose_df: Pose keypoint DataFrame
        cone_df: Cone detection DataFrame
        config: Optional AppConfig (defaults to Triple Cone config)
        fps: Video FPS
        parquet_dir: Path to parquet directory for loading manual cone annotations
        video_path: Path to video file for getting actual video dimensions

    Returns:
        DetectionResult with Triple Cone specific metrics
    """
    if config is None:
        config = AppConfig.for_triple_cone()
    detector = BallControlDetector(config, parquet_dir=parquet_dir, video_path=video_path)
    return detector.detect(ball_df, pose_df, cone_df, fps)
