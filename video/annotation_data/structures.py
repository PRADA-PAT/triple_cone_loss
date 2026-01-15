"""
Data structures for Triple Cone annotation.

Contains dataclasses and enums used throughout the annotation system.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


@dataclass
class BallPositionResult:
    """Result of ball position analysis relative to player's MOMENTUM (movement direction)."""
    position: str  # "FRONT", "BEHIND", "ALIGNED", or "UNKNOWN"
    movement_direction: Optional[str]  # "LEFT", "RIGHT", or None
    ball_hip_delta_x: float  # ball_x - hip_x
    color: Tuple[int, int, int]


@dataclass
class IntentionPositionResult:
    """Result of ball position analysis relative to player's INTENTION (torso facing)."""
    position: str  # "I-FRONT", "I-BEHIND", "I-ALIGNED", or "UNKNOWN"
    facing_direction: Optional[str]  # "LEFT", "RIGHT", or None (from nose-hip)
    ball_hip_delta_x: float  # ball_x - hip_x
    color: Tuple[int, int, int]


@dataclass
class EdgeZoneStatus:
    """Status of ball relative to screen edges."""
    in_edge_zone: bool
    edge_side: str  # "LEFT", "RIGHT", or "NONE"
    distance_to_edge: float


class BallTrackingState(Enum):
    """State machine states for unified edge+off-screen tracking."""
    NORMAL = "NORMAL"                      # Ball visible, not in edge zone
    EDGE_LEFT = "EDGE_LEFT"                # Ball in left edge zone
    EDGE_RIGHT = "EDGE_RIGHT"              # Ball in right edge zone
    OFF_SCREEN_LEFT = "OFF_SCREEN_LEFT"    # Ball left via left edge
    OFF_SCREEN_RIGHT = "OFF_SCREEN_RIGHT"  # Ball left via right edge
    DISAPPEARED_MID = "DISAPPEARED_MID"    # Ball disappeared without edge (detection failure)


@dataclass
class TurnEvent:
    """A detected turn at a cone."""
    frame_id: int
    timestamp: float
    zone: str  # "CONE1", "CONE2", "CONE3"
    from_direction: str
    to_direction: str


@dataclass
class ConeData:
    """Cone detection data with position and bounding box dimensions."""
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)
