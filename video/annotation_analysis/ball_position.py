"""
Ball position analysis for Triple Cone annotation.

Determines ball position relative to player movement and facing direction.
"""

from typing import Optional, Tuple

try:
    from ..annotation_config import TripleConeAnnotationConfig
    from ..annotation_data.structures import BallPositionResult, IntentionPositionResult
except ImportError:
    from annotation_config import TripleConeAnnotationConfig
    from annotation_data.structures import BallPositionResult, IntentionPositionResult


def determine_ball_position_relative_to_player(
    ball_center: Optional[Tuple[float, float]],
    current_hip: Optional[Tuple[float, float]],
    previous_hip: Optional[Tuple[float, float]],
    config: TripleConeAnnotationConfig
) -> BallPositionResult:
    """
    Determine if the ball is in front of or behind the player.

    For Triple Cone drill (horizontal movement):
    - Player moving LEFT (toward CONE1/HOME): FRONT = ball to left of hip
    - Player moving RIGHT (toward CONE3): FRONT = ball to right of hip
    """
    if ball_center is None or current_hip is None:
        return BallPositionResult(
            position="UNKNOWN",
            movement_direction=None,
            ball_hip_delta_x=0.0,
            color=config.BALL_POSITION_NEUTRAL_COLOR
        )

    ball_x = ball_center[0]
    hip_x = current_hip[0]
    delta_x = ball_x - hip_x

    # Determine player movement direction from hip history
    movement_direction: Optional[str] = None
    if previous_hip is not None:
        dx_movement = current_hip[0] - previous_hip[0]
        if dx_movement > config.MOVEMENT_THRESHOLD:
            movement_direction = "RIGHT"  # Moving toward CONE3
        elif dx_movement < -config.MOVEMENT_THRESHOLD:
            movement_direction = "LEFT"   # Moving toward CONE1

    # Check if ball is aligned with player
    if abs(delta_x) < config.BALL_POSITION_THRESHOLD:
        return BallPositionResult(
            position="ALIGNED",
            movement_direction=movement_direction,
            ball_hip_delta_x=delta_x,
            color=config.BALL_POSITION_ALIGNED_COLOR
        )

    # If player is stationary, just report left/right position
    if movement_direction is None:
        return BallPositionResult(
            position="LEFT" if delta_x < 0 else "RIGHT",
            movement_direction=None,
            ball_hip_delta_x=delta_x,
            color=config.BALL_POSITION_NEUTRAL_COLOR
        )

    # Determine front/behind based on movement direction
    if movement_direction == "LEFT":
        # Moving left: FRONT = ball to left (negative delta)
        if delta_x < 0:
            position = "FRONT"
            color = config.BALL_POSITION_FRONT_COLOR
        else:
            position = "BEHIND"
            color = config.BALL_POSITION_BEHIND_COLOR
    else:  # RIGHT
        # Moving right: FRONT = ball to right (positive delta)
        if delta_x > 0:
            position = "FRONT"
            color = config.BALL_POSITION_FRONT_COLOR
        else:
            position = "BEHIND"
            color = config.BALL_POSITION_BEHIND_COLOR

    return BallPositionResult(
        position=position,
        movement_direction=movement_direction,
        ball_hip_delta_x=delta_x,
        color=color
    )


def determine_torso_facing(
    persons: dict,
    config: TripleConeAnnotationConfig
) -> Optional[str]:
    """
    Determine if player torso is facing LEFT or RIGHT.

    Uses nose position relative to hip:
    - Facing RIGHT: nose.x > hip.x (head is to the right of body center)
    - Facing LEFT: nose.x < hip.x (head is to the left of body center)

    This captures turn intention since head turns before body.

    Returns 'RIGHT', 'LEFT', or None if data unreliable.
    """
    if not persons:
        return None

    # Get first person's keypoints (tuple format: x, y, confidence)
    first_person_id = min(persons.keys())
    keypoints = persons[first_person_id]

    nose = keypoints.get('nose')
    hip = keypoints.get('hip')

    # Fallback: if central hip not available, average left/right hip
    if not hip or hip[2] < config.MIN_KEYPOINT_CONFIDENCE:
        left_hip = keypoints.get('left_hip')
        right_hip = keypoints.get('right_hip')
        if left_hip and right_hip:
            if left_hip[2] >= config.MIN_KEYPOINT_CONFIDENCE and right_hip[2] >= config.MIN_KEYPOINT_CONFIDENCE:
                hip = ((left_hip[0] + right_hip[0]) / 2,
                       (left_hip[1] + right_hip[1]) / 2,
                       (left_hip[2] + right_hip[2]) / 2)

    if not nose or not hip:
        return None

    # Check confidence
    if nose[2] < config.MIN_KEYPOINT_CONFIDENCE:
        return None
    if isinstance(hip, tuple) and len(hip) >= 3:
        if hip[2] < config.MIN_KEYPOINT_CONFIDENCE:
            return None

    # Compare X positions: nose relative to hip
    diff = nose[0] - hip[0]

    if diff > config.NOSE_HIP_FACING_THRESHOLD:
        return "RIGHT"
    elif diff < -config.NOSE_HIP_FACING_THRESHOLD:
        return "LEFT"
    else:
        return None  # Aligned/neutral - don't display


def determine_ball_position_vs_intention(
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    facing_direction: Optional[str],
    config: TripleConeAnnotationConfig
) -> IntentionPositionResult:
    """
    Determine if ball is in front of or behind player's FACING direction (intention).

    Unlike momentum-based detection, this uses where the player is LOOKING
    (nose-hip orientation) rather than where they're MOVING.

    Args:
        ball_center: Ball (x, y) position
        hip_position: Player hip (x, y) position
        facing_direction: "LEFT", "RIGHT", or None from determine_torso_facing()
        config: Annotation config

    Returns:
        IntentionPositionResult with position relative to facing direction
    """
    if ball_center is None or hip_position is None:
        return IntentionPositionResult(
            position="UNKNOWN",
            facing_direction=facing_direction,
            ball_hip_delta_x=0.0,
            color=(180, 180, 180)  # Gray
        )

    ball_x = ball_center[0]
    hip_x = hip_position[0]
    delta_x = ball_x - hip_x

    # If no facing direction detected, return unknown
    if facing_direction is None:
        return IntentionPositionResult(
            position="UNKNOWN",
            facing_direction=None,
            ball_hip_delta_x=delta_x,
            color=(180, 180, 180)  # Gray
        )

    # Check if ball is aligned with player (same threshold as momentum-based)
    if abs(delta_x) < config.BALL_POSITION_THRESHOLD:
        return IntentionPositionResult(
            position="I-ALIGNED",
            facing_direction=facing_direction,
            ball_hip_delta_x=delta_x,
            color=config.INTENTION_ALIGNED_COLOR
        )

    # Determine front/behind based on facing direction
    if facing_direction == "LEFT":
        # Facing left: FRONT = ball to left (negative delta)
        if delta_x < 0:
            position = "I-FRONT"
            color = config.INTENTION_FRONT_COLOR
        else:
            position = "I-BEHIND"
            color = config.INTENTION_BEHIND_COLOR
    else:  # RIGHT
        # Facing right: FRONT = ball to right (positive delta)
        if delta_x > 0:
            position = "I-FRONT"
            color = config.INTENTION_FRONT_COLOR
        else:
            position = "I-BEHIND"
            color = config.INTENTION_BEHIND_COLOR

    return IntentionPositionResult(
        position=position,
        facing_direction=facing_direction,
        ball_hip_delta_x=delta_x,
        color=color
    )
