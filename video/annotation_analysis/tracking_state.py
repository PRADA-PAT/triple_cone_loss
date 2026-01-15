"""
Ball tracking state machine for Triple Cone annotation.

Handles edge zone detection and unified state tracking for ball visibility.
"""

from typing import Optional, Tuple

try:
    from ..annotation_config import TripleConeAnnotationConfig
    from ..annotation_data.structures import EdgeZoneStatus, BallTrackingState
except ImportError:
    from annotation_config import TripleConeAnnotationConfig
    from annotation_data.structures import EdgeZoneStatus, BallTrackingState


def check_edge_zone_status(
    ball_x: Optional[float],
    video_width: int,
    config: TripleConeAnnotationConfig
) -> EdgeZoneStatus:
    """Check if ball is in edge zone."""
    if ball_x is None:
        return EdgeZoneStatus(False, "NONE", float('inf'))

    left_distance = ball_x
    right_distance = video_width - ball_x

    if right_distance < config.EDGE_MARGIN:
        return EdgeZoneStatus(True, "RIGHT", right_distance)

    if left_distance < config.EDGE_MARGIN:
        return EdgeZoneStatus(True, "LEFT", left_distance)

    return EdgeZoneStatus(False, "NONE", min(left_distance, right_distance))


def update_ball_tracking_state(
    current_state: BallTrackingState,
    ball_visible: bool,
    edge_status: EdgeZoneStatus,
) -> Tuple[BallTrackingState, bool]:
    """
    State machine for unified edge+off-screen tracking.

    Returns:
        (new_state, should_reset_counter)
        - should_reset_counter: True when transitioning to NORMAL or switching sides

    State transitions:
        NORMAL + ball in left edge -> EDGE_LEFT (reset counter)
        NORMAL + ball in right edge -> EDGE_RIGHT (reset counter)
        NORMAL + ball disappears -> DISAPPEARED_MID (no counter)

        EDGE_LEFT + ball disappears -> OFF_SCREEN_LEFT (continue counter)
        EDGE_LEFT + ball leaves edge -> NORMAL (reset counter, trigger persist)
        EDGE_LEFT + ball enters right edge -> EDGE_RIGHT (reset counter)

        EDGE_RIGHT + ball disappears -> OFF_SCREEN_RIGHT (continue counter)
        EDGE_RIGHT + ball leaves edge -> NORMAL (reset counter, trigger persist)
        EDGE_RIGHT + ball enters left edge -> EDGE_LEFT (reset counter)

        OFF_SCREEN_LEFT + ball returns to left edge -> EDGE_LEFT (continue counter)
        OFF_SCREEN_LEFT + ball returns outside edge -> NORMAL (reset, trigger persist)
        OFF_SCREEN_LEFT + ball returns to right edge -> EDGE_RIGHT (reset counter)

        OFF_SCREEN_RIGHT + ball returns to right edge -> EDGE_RIGHT (continue counter)
        OFF_SCREEN_RIGHT + ball returns outside edge -> NORMAL (reset, trigger persist)
        OFF_SCREEN_RIGHT + ball returns to left edge -> EDGE_LEFT (reset counter)

        DISAPPEARED_MID + ball returns -> check edge status and transition accordingly
    """
    # Ball not visible
    if not ball_visible:
        if current_state == BallTrackingState.EDGE_LEFT:
            return (BallTrackingState.OFF_SCREEN_LEFT, False)  # Continue counter
        elif current_state == BallTrackingState.EDGE_RIGHT:
            return (BallTrackingState.OFF_SCREEN_RIGHT, False)  # Continue counter
        elif current_state in (BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
            return (current_state, False)  # Stay in off-screen state
        elif current_state == BallTrackingState.DISAPPEARED_MID:
            return (current_state, False)  # Stay in disappeared mid
        else:
            # NORMAL -> disappeared mid-field (detection failure)
            return (BallTrackingState.DISAPPEARED_MID, True)  # Reset counter (no counting for mid-field)

    # Ball is visible
    in_edge = edge_status.in_edge_zone
    edge_side = edge_status.edge_side

    if current_state == BallTrackingState.NORMAL:
        if in_edge:
            if edge_side == "LEFT":
                return (BallTrackingState.EDGE_LEFT, True)  # Start new counter
            else:  # RIGHT
                return (BallTrackingState.EDGE_RIGHT, True)  # Start new counter
        return (BallTrackingState.NORMAL, False)  # Stay normal

    elif current_state == BallTrackingState.EDGE_LEFT:
        if in_edge:
            if edge_side == "LEFT":
                return (BallTrackingState.EDGE_LEFT, False)  # Continue counter
            else:  # Switched to RIGHT
                return (BallTrackingState.EDGE_RIGHT, True)  # Reset and start right counter
        else:
            return (BallTrackingState.NORMAL, True)  # Left edge, trigger persist display

    elif current_state == BallTrackingState.EDGE_RIGHT:
        if in_edge:
            if edge_side == "RIGHT":
                return (BallTrackingState.EDGE_RIGHT, False)  # Continue counter
            else:  # Switched to LEFT
                return (BallTrackingState.EDGE_LEFT, True)  # Reset and start left counter
        else:
            return (BallTrackingState.NORMAL, True)  # Left edge, trigger persist display

    elif current_state == BallTrackingState.OFF_SCREEN_LEFT:
        # Ball returned after going off-screen via left edge
        if in_edge:
            if edge_side == "LEFT":
                return (BallTrackingState.EDGE_LEFT, False)  # Continue counter (still in danger)
            else:  # RIGHT edge
                return (BallTrackingState.EDGE_RIGHT, True)  # Reset, start new right sequence
        else:
            return (BallTrackingState.NORMAL, True)  # Fully recovered, trigger persist

    elif current_state == BallTrackingState.OFF_SCREEN_RIGHT:
        # Ball returned after going off-screen via right edge
        if in_edge:
            if edge_side == "RIGHT":
                return (BallTrackingState.EDGE_RIGHT, False)  # Continue counter (still in danger)
            else:  # LEFT edge
                return (BallTrackingState.EDGE_LEFT, True)  # Reset, start new left sequence
        else:
            return (BallTrackingState.NORMAL, True)  # Fully recovered, trigger persist

    elif current_state == BallTrackingState.DISAPPEARED_MID:
        # Ball returned after mid-field disappearance
        if in_edge:
            if edge_side == "LEFT":
                return (BallTrackingState.EDGE_LEFT, True)  # Start new edge sequence
            else:  # RIGHT
                return (BallTrackingState.EDGE_RIGHT, True)  # Start new edge sequence
        else:
            return (BallTrackingState.NORMAL, True)  # Return to normal

    return (current_state, False)  # Default: no change
