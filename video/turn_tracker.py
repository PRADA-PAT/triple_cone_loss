"""
Turn tracker for Triple Cone drill.

Tracks direction changes at turning zones to detect turns.
"""

from typing import List, Optional

try:
    from .annotation_data.structures import TurnEvent
except ImportError:
    from annotation_data.structures import TurnEvent


class TripleConeTurnTracker:
    """Track turns at each cone based on movement direction change."""

    def __init__(self):
        self.turn_events: List[TurnEvent] = []
        self.prev_direction: Optional[str] = None
        self.prev_zone: Optional[str] = None
        self.in_zone_frames: int = 0  # Debounce counter

    def update(
        self,
        frame_id: int,
        timestamp: float,
        current_zone: Optional[str],
        movement_direction: Optional[str]
    ) -> Optional[TurnEvent]:
        """
        Detect turn when:
        1. Ball is in a turning zone
        2. Movement direction reverses
        """
        turn_event = None

        # Track time in zone for debouncing
        if current_zone:
            self.in_zone_frames += 1
        else:
            self.in_zone_frames = 0

        # Detect direction reversal while in zone (with debounce)
        if current_zone and movement_direction and self.prev_direction:
            if movement_direction != self.prev_direction and self.in_zone_frames >= 3:
                turn_event = TurnEvent(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    zone=current_zone,
                    from_direction=self.prev_direction,
                    to_direction=movement_direction,
                )
                self.turn_events.append(turn_event)
                # Reset in_zone_frames to avoid duplicate events
                self.in_zone_frames = 0

        # Update state
        if movement_direction:
            self.prev_direction = movement_direction
        self.prev_zone = current_zone

        return turn_event

    def get_recent_events(self, max_events: int = 8) -> List[TurnEvent]:
        """Get the most recent turn events."""
        return self.turn_events[-max_events:]
