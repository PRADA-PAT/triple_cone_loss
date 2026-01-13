"""
Data structures for Ball Control Detection System.

Defines all data models for detection results including:
- ControlState: Ball control state machine
- EventType: Types of loss events
- FrameData: Per-frame analysis data
- LossEvent: A detected loss-of-control event
- DetectionResult: Complete detection output

Triple Cone (3-cone) structures:
- TripleConeDrillPhase: Current phase in drill (AT_CONE1, AT_CONE2, AT_CONE3, etc.)
- TripleConeLayout: 3-cone positions (CONE1/HOME, CONE2/CENTER, CONE3/RIGHT)
- DrillDirection: Direction of travel (forward/backward)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# DETECTION STATE ENUMS
# =============================================================================


class ControlState(Enum):
    """Ball control states."""
    CONTROLLED = "controlled"
    TRANSITION = "transition"
    LOST = "lost"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class EventType(Enum):
    """Loss-of-control event types."""
    LOSS_DISTANCE = "loss_distance"
    LOSS_VELOCITY = "loss_velocity"
    LOSS_DIRECTION = "loss_direction"
    RECOVERY = "recovery"
    BOUNDARY_VIOLATION = "boundary"
    BALL_BEHIND_PLAYER = "ball_behind"  # Ball stays behind player relative to movement
    BALL_BEHIND_INTENTION = "ball_behind_intention"  # Ball behind relative to facing direction


class DrillDirection(Enum):
    """Direction of travel in drill."""
    FORWARD = "forward"    # CONE1 → CONE2/CONE3 direction (increasing X)
    BACKWARD = "backward"  # Returning to CONE1 (decreasing X)
    STATIONARY = "stationary"


class BallTrackingState(Enum):
    """
    State machine states for unified edge/off-screen tracking.

    Used for boundary violation detection. The key insight is that ball
    detection disappears (interpolated=True) when ball goes off-screen,
    rather than being "stuck" at the edge.

    State transitions:
        NORMAL → EDGE_LEFT/RIGHT (ball enters edge zone)
        EDGE_LEFT/RIGHT → OFF_SCREEN_LEFT/RIGHT (ball disappears)
        OFF_SCREEN_LEFT/RIGHT → NORMAL (ball returns)
        NORMAL → DISAPPEARED_MID (ball disappears mid-field - detection failure)
    """
    NORMAL = "NORMAL"                      # Ball visible, not in edge zone
    EDGE_LEFT = "EDGE_LEFT"                # Ball in left edge zone
    EDGE_RIGHT = "EDGE_RIGHT"              # Ball in right edge zone
    OFF_SCREEN_LEFT = "OFF_SCREEN_LEFT"    # Ball left via left edge
    OFF_SCREEN_RIGHT = "OFF_SCREEN_RIGHT"  # Ball left via right edge
    DISAPPEARED_MID = "DISAPPEARED_MID"    # Ball disappeared mid-field (detection failure)


# =============================================================================
# TRIPLE CONE DRILL STRUCTURES (3-CONE)
# =============================================================================

class TripleConeDrillPhase(Enum):
    """
    Current phase in Triple Cone drill.

    Drill pattern (one repetition):
    CONE1 → CONE2(turn) → CONE1(turn) → CONE3(turn) → CONE1(turn) → repeat

    Cone layout:
    [CONE1/LEFT/HOME] ---- [CONE2/CENTER] ---- [CONE3/RIGHT]
    """
    AT_CONE1 = "at_cone1"                       # At home cone (LEFT)
    GOING_TO_CONE2 = "going_to_cone2"           # Moving toward center cone
    AT_CONE2 = "at_cone2"                       # At center cone, turning
    RETURNING_FROM_CONE2 = "returning_from_cone2"  # Returning to home from center
    GOING_TO_CONE3 = "going_to_cone3"           # Moving toward right cone
    AT_CONE3 = "at_cone3"                       # At right cone, turning
    RETURNING_FROM_CONE3 = "returning_from_cone3"  # Returning to home from right
    COMPLETED = "completed"                     # Drill finished
    UNKNOWN = "unknown"                         # Phase not determined


@dataclass
class TripleConeLayout:
    """
    Triple Cone drill layout with 3 cones in a horizontal line.

    Layout (pixel x increases left to right):
    [CONE1/LEFT/HOME] ---- [CONE2/CENTER] ---- [CONE3/RIGHT]

    Cone positions are extracted from parquet data (mean positions across frames).

    Attributes:
        cone1: Left/HOME cone position (where player starts and returns)
        cone2: Center cone position
        cone3: Right cone position
    """
    cone1: Tuple[float, float]  # (px, py) - LEFT/HOME
    cone2: Tuple[float, float]  # (px, py) - CENTER
    cone3: Tuple[float, float]  # (px, py) - RIGHT

    @property
    def cone1_x(self) -> float:
        """X position of CONE1 (HOME/LEFT)."""
        return self.cone1[0]

    @property
    def cone1_y(self) -> float:
        """Y position of CONE1 (HOME/LEFT)."""
        return self.cone1[1]

    @property
    def cone2_x(self) -> float:
        """X position of CONE2 (CENTER)."""
        return self.cone2[0]

    @property
    def cone2_y(self) -> float:
        """Y position of CONE2 (CENTER)."""
        return self.cone2[1]

    @property
    def cone3_x(self) -> float:
        """X position of CONE3 (RIGHT)."""
        return self.cone3[0]

    @property
    def cone3_y(self) -> float:
        """Y position of CONE3 (RIGHT)."""
        return self.cone3[1]

    @property
    def cone1_to_cone2_distance(self) -> float:
        """Distance from CONE1 to CONE2 in pixels."""
        import math
        return math.sqrt(
            (self.cone2[0] - self.cone1[0])**2 +
            (self.cone2[1] - self.cone1[1])**2
        )

    @property
    def cone2_to_cone3_distance(self) -> float:
        """Distance from CONE2 to CONE3 in pixels."""
        import math
        return math.sqrt(
            (self.cone3[0] - self.cone2[0])**2 +
            (self.cone3[1] - self.cone2[1])**2
        )

    @property
    def total_span(self) -> float:
        """Total horizontal span from CONE1 to CONE3 in pixels."""
        import math
        return math.sqrt(
            (self.cone3[0] - self.cone1[0])**2 +
            (self.cone3[1] - self.cone1[1])**2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'cone1': {'px': self.cone1[0], 'py': self.cone1[1]},
            'cone2': {'px': self.cone2[0], 'py': self.cone2[1]},
            'cone3': {'px': self.cone3[0], 'py': self.cone3[1]},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TripleConeLayout':
        """Create from dictionary."""
        return cls(
            cone1=(data['cone1']['px'], data['cone1']['py']),
            cone2=(data['cone2']['px'], data['cone2']['py']),
            cone3=(data['cone3']['px'], data['cone3']['py']),
        )

    @classmethod
    def from_mean_positions(
        cls,
        cone1_x: float, cone1_y: float,
        cone2_x: float, cone2_y: float,
        cone3_x: float, cone3_y: float
    ) -> 'TripleConeLayout':
        """Create from mean cone positions (from parquet analysis)."""
        return cls(
            cone1=(cone1_x, cone1_y),
            cone2=(cone2_x, cone2_y),
            cone3=(cone3_x, cone3_y),
        )


# =============================================================================
# FRAME AND EVENT STRUCTURES
# =============================================================================

@dataclass
class FrameData:
    """Data for a single frame analysis."""
    frame_id: int
    timestamp: float

    # Ball position
    ball_x: float
    ball_y: float
    ball_field_x: float
    ball_field_y: float
    ball_velocity: float

    # Player ankle position (closest)
    ankle_x: float
    ankle_y: float
    ankle_field_x: float
    ankle_field_y: float
    closest_ankle: str  # "left_ankle" or "right_ankle"

    # Context
    nearest_cone_id: int  # 1, 2, or 3 for CONE1, CONE2, CONE3
    nearest_cone_distance: float
    current_gate: Optional[str]  # Legacy field, always None in 3-cone mode

    # Computed metrics
    ball_ankle_distance: float
    control_score: float
    control_state: ControlState

    # Triple Cone specific fields
    drill_phase: Optional[TripleConeDrillPhase] = None
    drill_direction: Optional[DrillDirection] = None
    lap_count: int = 0  # Number of completed reps

    # Hip position (for ball-behind detection)
    hip_x: Optional[float] = None  # Hip pixel X coordinate
    hip_y: Optional[float] = None  # Hip pixel Y coordinate

    # Ball position relative to player (for ball-behind detection)
    player_movement_direction: Optional[str] = None  # "LEFT", "RIGHT", or None
    ball_behind_player: Optional[bool] = None  # True if ball is behind player
    in_turning_zone: Optional[str] = None  # "CONE1"/"CONE2"/"CONE3" or None

    # Ball tracking quality (for filtering false positives)
    ball_interpolated: bool = False  # True if ball position is interpolated

    # Ball tracking state (for boundary violation detection)
    ball_tracking_state: Optional['BallTrackingState'] = None

    # Intention-based (face direction) ball position detection
    nose_x: Optional[float] = None  # Nose pixel X coordinate
    nose_y: Optional[float] = None  # Nose pixel Y coordinate
    player_facing_direction: Optional[str] = None  # "LEFT", "RIGHT", or None (from nose-hip)
    ball_behind_intention: Optional[bool] = None   # True if ball is behind facing direction
    ball_intention_position: Optional[str] = None  # "I-FRONT", "I-BEHIND", "I-ALIGNED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame."""
        result = {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'ball_x': self.ball_x,
            'ball_y': self.ball_y,
            'ball_field_x': self.ball_field_x,
            'ball_field_y': self.ball_field_y,
            'ball_velocity': self.ball_velocity,
            'ankle_x': self.ankle_x,
            'ankle_y': self.ankle_y,
            'ankle_field_x': self.ankle_field_x,
            'ankle_field_y': self.ankle_field_y,
            'closest_ankle': self.closest_ankle,
            'nearest_cone_id': self.nearest_cone_id,
            'nearest_cone_distance': self.nearest_cone_distance,
            'current_gate': self.current_gate,
            'ball_ankle_distance': self.ball_ankle_distance,
            'control_score': self.control_score,
            'control_state': self.control_state.value,
        }
        # Add Triple Cone specific fields if present
        if self.drill_phase is not None:
            result['drill_phase'] = self.drill_phase.value
        if self.drill_direction is not None:
            result['drill_direction'] = self.drill_direction.value
        result['lap_count'] = self.lap_count

        # Add hip/ball-behind fields if present
        if self.hip_x is not None:
            result['hip_x'] = self.hip_x
        if self.hip_y is not None:
            result['hip_y'] = self.hip_y
        if self.player_movement_direction is not None:
            result['player_movement_direction'] = self.player_movement_direction
        if self.ball_behind_player is not None:
            result['ball_behind_player'] = self.ball_behind_player
        if self.in_turning_zone is not None:
            result['in_turning_zone'] = self.in_turning_zone

        # Add intention-based fields if present
        if self.nose_x is not None:
            result['nose_x'] = self.nose_x
        if self.nose_y is not None:
            result['nose_y'] = self.nose_y
        if self.player_facing_direction is not None:
            result['player_facing_direction'] = self.player_facing_direction
        if self.ball_behind_intention is not None:
            result['ball_behind_intention'] = self.ball_behind_intention
        if self.ball_intention_position is not None:
            result['ball_intention_position'] = self.ball_intention_position

        return result


@dataclass
class LossEvent:
    """A ball control loss event."""
    event_id: int
    event_type: EventType
    start_frame: int
    end_frame: Optional[int]
    start_timestamp: float
    end_timestamp: Optional[float]

    # Position at loss
    ball_position: Tuple[float, float]
    player_position: Tuple[float, float]
    distance_at_loss: float
    velocity_at_loss: float

    # Context
    nearest_cone_id: int
    gate_context: Optional[str]  # Legacy field, always None in 3-cone mode

    # Recovery
    recovered: bool = False
    recovery_frame: Optional[int] = None
    severity: str = "medium"
    notes: str = ""

    @property
    def duration_frames(self) -> int:
        if self.end_frame is None:
            return 0
        return self.end_frame - self.start_frame

    @property
    def duration_seconds(self) -> float:
        if self.end_timestamp is None:
            return 0.0
        return self.end_timestamp - self.start_timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'duration_frames': self.duration_frames,
            'duration_seconds': self.duration_seconds,
            'ball_x': self.ball_position[0],
            'ball_y': self.ball_position[1],
            'player_x': self.player_position[0],
            'player_y': self.player_position[1],
            'distance_at_loss': self.distance_at_loss,
            'velocity_at_loss': self.velocity_at_loss,
            'nearest_cone_id': self.nearest_cone_id,
            'gate_context': self.gate_context,
            'recovered': self.recovered,
            'recovery_frame': self.recovery_frame,
            'severity': self.severity,
            'notes': self.notes,
        }


@dataclass
class DetectionResult:
    """Complete result from ball control detection."""
    success: bool
    total_frames: int
    events: List[LossEvent]
    frame_data: List[FrameData]

    # Summary
    total_loss_events: int = 0
    total_loss_duration_frames: int = 0
    control_percentage: float = 0.0

    error: Optional[str] = None

    # Triple Cone specific results
    total_laps: int = 0  # Number of full repetitions (CONE1→CONE2→CONE1→CONE3→CONE1)

    def __post_init__(self):
        """Calculate summary statistics."""
        self.total_loss_events = len(self.events)
        self.total_loss_duration_frames = sum(
            e.duration_frames for e in self.events if e.end_frame
        )
        if self.total_frames > 0:
            controlled = self.total_frames - self.total_loss_duration_frames
            self.control_percentage = (controlled / self.total_frames) * 100
