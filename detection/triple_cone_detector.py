"""
Triple Cone Drill Detection - Phase Tracking and Turn Detection.

This module handles:
1. Loading 3-cone positions from parquet data
2. Creating turning zones around each cone
3. Tracking drill phase based on ball position
4. Detecting turns at cones (direction reversal)

Triple Cone Drill Layout (left to right):
[CONE1/HOME] ---- [CONE2/CENTER] ---- [CONE3/RIGHT]

Drill Pattern:
CONE1 → CONE2 (turn) → CONE1 (turn) → CONE3 (turn) → CONE1 (turn) → repeat
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from .config import TripleConeDrillConfig
from .data_structures import (
    TripleConeLayout, TripleConeDrillPhase, DrillDirection
)
from .turning_zones import (
    TripleConeZoneSet, TripleConeZoneConfig, create_triple_cone_zones
)
from .data_loader import load_triple_cone_layout_from_parquet

logger = logging.getLogger(__name__)


@dataclass
class TurnEvent:
    """A detected turn at a cone."""
    frame_id: int
    timestamp: float
    cone: str  # "CONE1", "CONE2", "CONE3"
    from_direction: DrillDirection
    to_direction: DrillDirection

    def to_dict(self) -> Dict:
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp,
            'cone': self.cone,
            'from_direction': self.from_direction.value,
            'to_direction': self.to_direction.value,
        }


@dataclass
class DrillState:
    """Current state of the drill."""
    phase: TripleConeDrillPhase = TripleConeDrillPhase.UNKNOWN
    direction: DrillDirection = DrillDirection.STATIONARY
    current_zone: Optional[str] = None  # "CONE1", "CONE2", "CONE3", or None
    rep_count: int = 0  # Full repetition count (CONE1→CONE2→CONE1→CONE3→CONE1)


class TripleConeDetector:
    """
    Detects drill phases and turns for Triple Cone drill (3 cones).

    Usage:
        detector = TripleConeDetector(config)
        detector.setup_from_parquet(cone_parquet_path)

        # Per-frame tracking
        state = detector.update(ball_pos, frame_id, timestamp)
        turn = detector.check_turn(current_direction)
    """

    def __init__(
        self,
        config: TripleConeDrillConfig,
        zone_config: Optional[TripleConeZoneConfig] = None
    ):
        """
        Initialize Triple Cone detector.

        Args:
            config: Triple Cone drill configuration
            zone_config: Optional zone configuration (defaults to TripleConeZoneConfig.default())
        """
        self.config = config
        self.zone_config = zone_config or TripleConeZoneConfig.default()

        # Cone layout
        self._cone_layout: Optional[TripleConeLayout] = None
        self._turning_zones: Optional[TripleConeZoneSet] = None

        # State tracking
        self._state = DrillState()
        self._turn_events: List[TurnEvent] = []

        # History for direction detection
        self._direction_history: List[DrillDirection] = []
        self._zone_history: List[Optional[str]] = []
        self._frames_in_current_zone: int = 0

        logger.info("TripleConeDetector initialized (3-cone mode)")

    def setup_from_parquet(self, cone_parquet_path: str) -> TripleConeLayout:
        """
        Set up detector from cone parquet file.

        Args:
            cone_parquet_path: Path to cone parquet file

        Returns:
            TripleConeLayout with cone positions
        """
        self._cone_layout = load_triple_cone_layout_from_parquet(cone_parquet_path)

        # Create turning zones
        self._turning_zones = create_triple_cone_zones(
            self._cone_layout.cone1,
            self._cone_layout.cone2,
            self._cone_layout.cone3,
            self.zone_config,
        )

        logger.info(
            f"Detector setup complete: "
            f"CONE1=({self._cone_layout.cone1_x:.0f}, {self._cone_layout.cone1_y:.0f}), "
            f"CONE2=({self._cone_layout.cone2_x:.0f}, {self._cone_layout.cone2_y:.0f}), "
            f"CONE3=({self._cone_layout.cone3_x:.0f}, {self._cone_layout.cone3_y:.0f})"
        )

        return self._cone_layout

    def setup_from_layout(self, layout: TripleConeLayout) -> None:
        """
        Set up detector from existing TripleConeLayout.

        Args:
            layout: Pre-configured cone layout
        """
        self._cone_layout = layout

        # Create turning zones
        self._turning_zones = create_triple_cone_zones(
            layout.cone1,
            layout.cone2,
            layout.cone3,
            self.zone_config,
        )

        logger.info(
            f"Detector setup from layout: "
            f"CONE1=({layout.cone1_x:.0f}, {layout.cone1_y:.0f}), "
            f"CONE2=({layout.cone2_x:.0f}, {layout.cone2_y:.0f}), "
            f"CONE3=({layout.cone3_x:.0f}, {layout.cone3_y:.0f})"
        )

    def get_zone_at_point(self, x: float, y: float) -> Optional[str]:
        """
        Get which turning zone a point is in.

        Args:
            x, y: Point coordinates (pixel space)

        Returns:
            "CONE1", "CONE2", "CONE3", or None if not in any zone
        """
        if self._turning_zones is None:
            return None
        return self._turning_zones.get_zone_at_point(x, y)

    def is_in_turning_zone(self, x: float, y: float) -> bool:
        """Check if point is in any turning zone."""
        if self._turning_zones is None:
            return False
        return self._turning_zones.is_in_turning_zone(x, y)

    def update(
        self,
        ball_pos: Tuple[float, float],
        direction: DrillDirection,
        frame_id: int,
        timestamp: float
    ) -> DrillState:
        """
        Update drill state based on current ball position.

        Args:
            ball_pos: Ball position (pixel coords)
            direction: Current movement direction
            frame_id: Current frame number
            timestamp: Current timestamp

        Returns:
            Updated DrillState
        """
        if self._cone_layout is None:
            return self._state

        # Get current zone
        current_zone = self.get_zone_at_point(ball_pos[0], ball_pos[1])

        # Update zone tracking
        if current_zone == self._state.current_zone:
            self._frames_in_current_zone += 1
        else:
            self._frames_in_current_zone = 1

        prev_zone = self._state.current_zone
        self._state.current_zone = current_zone

        # Update direction
        prev_direction = self._state.direction
        self._state.direction = direction

        # Determine phase
        self._state.phase = self._determine_phase(ball_pos, current_zone, direction)

        # Check for turn (direction reversal in zone)
        if current_zone and self._frames_in_current_zone >= 3:
            turn = self._check_turn(
                current_zone, prev_direction, direction,
                frame_id, timestamp
            )
            if turn:
                self._turn_events.append(turn)
                logger.debug(f"Turn detected at {turn.cone}: {turn.from_direction.value} → {turn.to_direction.value}")

        # Update history
        self._direction_history.append(direction)
        self._zone_history.append(current_zone)
        if len(self._direction_history) > 30:
            self._direction_history.pop(0)
            self._zone_history.pop(0)

        return self._state

    def _determine_phase(
        self,
        ball_pos: Tuple[float, float],
        current_zone: Optional[str],
        direction: DrillDirection
    ) -> TripleConeDrillPhase:
        """
        Determine current drill phase based on ball position and zone.

        Phase logic:
        - If in CONE1 zone: AT_CONE1
        - If in CONE2 zone: AT_CONE2
        - If in CONE3 zone: AT_CONE3
        - Between zones: use direction and X position
        """
        if current_zone == "CONE1":
            return TripleConeDrillPhase.AT_CONE1
        elif current_zone == "CONE2":
            return TripleConeDrillPhase.AT_CONE2
        elif current_zone == "CONE3":
            return TripleConeDrillPhase.AT_CONE3

        # Not in any zone - determine by position and direction
        ball_x = ball_pos[0]
        cone1_x = self._cone_layout.cone1_x
        cone2_x = self._cone_layout.cone2_x
        cone3_x = self._cone_layout.cone3_x

        # Check which "segment" the ball is in
        # CONE1 -- (segment A) -- CONE2 -- (segment B) -- CONE3

        midpoint_1_2 = (cone1_x + cone2_x) / 2
        midpoint_2_3 = (cone2_x + cone3_x) / 2

        if ball_x < midpoint_1_2:
            # Between CONE1 and midpoint
            if direction == DrillDirection.FORWARD:
                return TripleConeDrillPhase.GOING_TO_CONE2
            elif direction == DrillDirection.BACKWARD:
                return TripleConeDrillPhase.RETURNING_FROM_CONE2
            else:
                return TripleConeDrillPhase.GOING_TO_CONE2  # Default

        elif ball_x < midpoint_2_3:
            # Between midpoint and CONE2, or CONE2 and midpoint_2_3
            if direction == DrillDirection.FORWARD:
                return TripleConeDrillPhase.GOING_TO_CONE3
            elif direction == DrillDirection.BACKWARD:
                return TripleConeDrillPhase.RETURNING_FROM_CONE2
            else:
                # Near CONE2
                return TripleConeDrillPhase.AT_CONE2
        else:
            # Between CONE2-CONE3 midpoint and CONE3
            if direction == DrillDirection.FORWARD:
                return TripleConeDrillPhase.GOING_TO_CONE3
            elif direction == DrillDirection.BACKWARD:
                return TripleConeDrillPhase.RETURNING_FROM_CONE3
            else:
                return TripleConeDrillPhase.GOING_TO_CONE3

    def _check_turn(
        self,
        zone: str,
        prev_direction: DrillDirection,
        curr_direction: DrillDirection,
        frame_id: int,
        timestamp: float
    ) -> Optional[TurnEvent]:
        """
        Check if a turn occurred (direction reversal in a zone).

        A turn is detected when:
        1. Ball is in a turning zone
        2. Direction changes from FORWARD to BACKWARD or vice versa
        """
        # Skip if no direction change
        if prev_direction == curr_direction:
            return None

        # Skip stationary
        if curr_direction == DrillDirection.STATIONARY:
            return None
        if prev_direction == DrillDirection.STATIONARY:
            return None

        # Valid turn: direction reversed
        return TurnEvent(
            frame_id=frame_id,
            timestamp=timestamp,
            cone=zone,
            from_direction=prev_direction,
            to_direction=curr_direction,
        )

    def get_direction_from_movement(
        self,
        prev_x: float,
        curr_x: float,
        threshold: float = 3.0
    ) -> DrillDirection:
        """
        Determine movement direction from X coordinate change.

        For Triple Cone drill:
        - Moving RIGHT (increasing X) = FORWARD (toward CONE3)
        - Moving LEFT (decreasing X) = BACKWARD (toward CONE1)

        Args:
            prev_x: Previous X position
            curr_x: Current X position
            threshold: Minimum movement to count as non-stationary

        Returns:
            DrillDirection
        """
        dx = curr_x - prev_x
        if dx > threshold:
            return DrillDirection.FORWARD
        elif dx < -threshold:
            return DrillDirection.BACKWARD
        else:
            return DrillDirection.STATIONARY

    @property
    def cone_layout(self) -> Optional[TripleConeLayout]:
        """Get the cone layout."""
        return self._cone_layout

    @property
    def turning_zones(self) -> Optional[TripleConeZoneSet]:
        """Get the turning zones."""
        return self._turning_zones

    @property
    def turn_events(self) -> List[TurnEvent]:
        """Get all detected turn events."""
        return self._turn_events

    @property
    def state(self) -> DrillState:
        """Get current drill state."""
        return self._state

    @property
    def rep_count(self) -> int:
        """Get repetition count."""
        return self._state.rep_count

    def reset(self):
        """Reset tracking state for new detection run."""
        self._state = DrillState()
        self._turn_events = []
        self._direction_history = []
        self._zone_history = []
        self._frames_in_current_zone = 0
        logger.info("TripleConeDetector reset")


# Legacy alias for backwards compatibility
TripleConeConeDetector = TripleConeDetector
