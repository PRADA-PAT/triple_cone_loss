"""
Indicator drawing functions for Triple Cone annotation.

Contains counters, position labels, and status indicators.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from ..annotation_config import TripleConeAnnotationConfig
    from ..annotation_data.structures import (
        BallPositionResult,
        IntentionPositionResult,
        BallTrackingState,
    )
    from .primitives import draw_dashed_line
except ImportError:
    from annotation_config import TripleConeAnnotationConfig
    from annotation_data.structures import (
        BallPositionResult,
        IntentionPositionResult,
        BallTrackingState,
    )
    from annotation_drawing.primitives import draw_dashed_line


def draw_ball_position_indicator(
    frame: np.ndarray,
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    result: BallPositionResult,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw visual indicators showing ball position relative to player."""
    if ball_center is None or hip_position is None:
        return

    if result.position == "UNKNOWN":
        return

    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [ball_center[0], ball_center[1], hip_position[0], hip_position[1]]):
        return

    hip_x = int(hip_position[0]) + x_offset
    hip_y = int(hip_position[1])
    ball_x = int(ball_center[0]) + x_offset
    ball_y = int(ball_center[1])

    # 1. Dashed vertical line through hip (use config dash/gap values)
    half_height = config.DIVIDER_LINE_HEIGHT // 2
    line_color = (100, 100, 100)

    dash_length = config.INTENTION_LINE_DASH_LENGTH  # Already scaled
    gap_length = config.INTENTION_LINE_GAP_LENGTH    # Already scaled
    y_start = hip_y - half_height
    y_end = hip_y + half_height
    y = y_start
    while y < y_end:
        y_next = min(y + dash_length, y_end)
        cv2.line(frame, (hip_x, y), (hip_x, y_next), line_color, 1, cv2.LINE_AA)
        y = y_next + gap_length

    # Hip marker (scale radius)
    hip_radius = max(2, int(5 * config.FONT_SCALE_FACTOR))
    cv2.circle(frame, (hip_x, hip_y), hip_radius, result.color, -1)
    cv2.circle(frame, (hip_x, hip_y), hip_radius, (255, 255, 255), 1)

    # 2. Connecting line from hip to ball
    cv2.line(frame, (hip_x, hip_y), (ball_x, ball_y),
             result.color, config.BALL_HIP_LINE_THICKNESS, cv2.LINE_AA)

    # 3. Position label above ball (scale font and positioning)
    label = result.position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * config.FONT_SCALE_FACTOR
    font_thickness = max(1, int(2 * config.FONT_SCALE_FACTOR))

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    label_x = ball_x - text_width // 2
    label_y = ball_y - int(30 * config.FONT_SCALE_FACTOR)

    padding = max(2, int(3 * config.FONT_SCALE_FACTOR))
    cv2.rectangle(
        frame,
        (label_x - padding, label_y - text_height - padding),
        (label_x + text_width + padding, label_y + padding),
        (0, 0, 0), -1
    )
    cv2.putText(
        frame, label,
        (label_x, label_y),
        font, font_scale, result.color, font_thickness, cv2.LINE_AA
    )


def draw_intention_position_indicator(
    frame: np.ndarray,
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    result: IntentionPositionResult,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw visual indicator for ball position vs intention (dashed line style).

    Shows where ball is relative to player's FACING direction (intention).
    Uses dashed line and label BELOW ball to distinguish from momentum indicator.
    """
    if ball_center is None or hip_position is None:
        return

    if result.position == "UNKNOWN":
        return

    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [ball_center[0], ball_center[1], hip_position[0], hip_position[1]]):
        return

    hip_x = int(hip_position[0]) + x_offset
    hip_y = int(hip_position[1])
    ball_x = int(ball_center[0]) + x_offset
    ball_y = int(ball_center[1])

    # 1. Dashed line from hip to ball (distinct from solid momentum line)
    draw_dashed_line(
        frame,
        (hip_x, hip_y),
        (ball_x, ball_y),
        result.color,
        config.BALL_HIP_LINE_THICKNESS,
        config.INTENTION_LINE_DASH_LENGTH,
        config.INTENTION_LINE_GAP_LENGTH
    )

    # 2. Position label BELOW ball (to distinguish from momentum label above)
    label = result.position  # "I-FRONT", "I-BEHIND", "I-ALIGNED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55  # Slightly smaller than momentum label
    font_thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    label_x = ball_x - text_width // 2
    label_y = ball_y + config.INTENTION_LABEL_OFFSET_Y  # Below ball (momentum is above)

    padding = 3
    cv2.rectangle(
        frame,
        (label_x - padding, label_y - text_height - padding),
        (label_x + text_width + padding, label_y + padding),
        (0, 0, 0), -1
    )
    cv2.putText(
        frame, label,
        (label_x, label_y),
        font, font_scale, result.color, font_thickness, cv2.LINE_AA
    )


def draw_behind_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw ball-behind duration counter."""
    if count <= 0:
        return

    text = f"BEHIND: {count}f"
    x = x_offset + config.BEHIND_COUNTER_POS_X
    y = config.BEHIND_COUNTER_POS_Y

    color = config.BEHIND_COUNTER_COLOR if is_active else config.BEHIND_COUNTER_PERSIST_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * config.FONT_SCALE_FACTOR))
    (tw, th), _ = cv2.getTextSize(text, font, config.BEHIND_COUNTER_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * config.FONT_SCALE_FACTOR))
    pad_y = max(4, int(10 * config.FONT_SCALE_FACTOR))
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font,
                config.BEHIND_COUNTER_FONT_SCALE, color, thickness, cv2.LINE_AA)


def draw_intention_behind_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw intention-based ball-behind duration counter."""
    if count <= 0:
        return

    text = f"BEHIND (I): {count}f"
    x = x_offset + config.BEHIND_COUNTER_POS_X
    y = config.INTENTION_BEHIND_COUNTER_POS_Y

    # Use intention colors (magenta active, lighter persist)
    color = config.INTENTION_BEHIND_COLOR if is_active else (255, 150, 255)  # Lighter magenta

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * config.FONT_SCALE_FACTOR))
    (tw, th), _ = cv2.getTextSize(text, font, config.BEHIND_COUNTER_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * config.FONT_SCALE_FACTOR))
    pad_y = max(4, int(10 * config.FONT_SCALE_FACTOR))
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font,
                config.BEHIND_COUNTER_FONT_SCALE, color, thickness, cv2.LINE_AA)


def draw_edge_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    edge_side: str,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw edge zone frame counter."""
    if count <= 0:
        return

    text = f"EDGE ({edge_side}): {count}f"
    x = x_offset + config.EDGE_COUNTER_POS_X
    y = config.EDGE_COUNTER_POS_Y

    color = config.EDGE_COUNTER_COLOR if is_active else config.EDGE_COUNTER_PERSIST_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * config.FONT_SCALE_FACTOR))
    (tw, th), _ = cv2.getTextSize(text, font, config.EDGE_COUNTER_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * config.FONT_SCALE_FACTOR))
    pad_y = max(4, int(10 * config.FONT_SCALE_FACTOR))
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font,
                config.EDGE_COUNTER_FONT_SCALE, color, thickness, cv2.LINE_AA)


def draw_off_screen_indicator(
    frame: np.ndarray,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw 'BALL OFF-SCREEN' indicator when ball is not detected.

    Args:
        frame: Video frame to draw on
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    text = config.OFF_SCREEN_TEXT
    x = x_offset + config.OFF_SCREEN_POS_X
    y = config.OFF_SCREEN_POS_Y

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    (tw, th), _ = cv2.getTextSize(text, font, config.OFF_SCREEN_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    pad_y = max(4, int(10 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))

    # Draw background box
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font,
                config.OFF_SCREEN_FONT_SCALE, config.OFF_SCREEN_COLOR, thickness, cv2.LINE_AA)


def draw_return_counter(
    frame: np.ndarray,
    frames_gone: int,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw counter showing how long ball was off-screen (after it returns).

    Shows "WAS GONE: Xf" to indicate how many frames the ball was not detected.

    Args:
        frame: Video frame to draw on
        frames_gone: Number of frames the ball was off-screen
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    text = f"WAS GONE: {frames_gone}f"
    x = x_offset + config.RETURN_COUNTER_POS_X
    y = config.RETURN_COUNTER_POS_Y

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(2 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    (tw, th), _ = cv2.getTextSize(text, font, config.RETURN_COUNTER_FONT_SCALE, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    pad_y = max(4, int(10 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))

    # Draw background box
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y), (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font,
                config.RETURN_COUNTER_FONT_SCALE, config.RETURN_COUNTER_COLOR, thickness, cv2.LINE_AA)


def draw_unified_tracking_indicator(
    frame: np.ndarray,
    state: BallTrackingState,
    counter: int,
    is_persisting: bool,
    persist_side: str,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw unified edge+off-screen tracking indicator.

    Displays different text/color based on state:
    - EDGE_LEFT/RIGHT: "EDGE LEFT: Xf" / "EDGE RIGHT: Xf" (yellow)
    - OFF_SCREEN_LEFT/RIGHT: "LEFT EDGE + OFF: Xf" / "RIGHT EDGE + OFF: Xf" (red)
    - Persisting after return: "WAS LEFT: Xf" / "WAS RIGHT: Xf" (orange)
    - DISAPPEARED_MID: "DETECTION LOST" (gray) - no counter

    Args:
        frame: Video frame to draw on
        state: Current BallTrackingState
        counter: Number of frames in current state sequence
        is_persisting: True when showing persist display after returning to normal
        persist_side: "LEFT" or "RIGHT" - which side the ball exited from (for persist display)
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    # Determine text and color based on state
    text = ""
    color = config.UNIFIED_EDGE_COLOR
    pos_x = x_offset + config.UNIFIED_COUNTER_POS_X
    pos_y = config.UNIFIED_COUNTER_POS_Y

    if state == BallTrackingState.EDGE_LEFT:
        text = f"EDGE LEFT: {counter}f"
        color = config.UNIFIED_EDGE_COLOR
    elif state == BallTrackingState.EDGE_RIGHT:
        text = f"EDGE RIGHT: {counter}f"
        color = config.UNIFIED_EDGE_COLOR
    elif state == BallTrackingState.OFF_SCREEN_LEFT:
        text = f"LEFT EDGE + OFF: {counter}f"
        color = config.UNIFIED_OFF_SCREEN_COLOR
    elif state == BallTrackingState.OFF_SCREEN_RIGHT:
        text = f"RIGHT EDGE + OFF: {counter}f"
        color = config.UNIFIED_OFF_SCREEN_COLOR
    elif state == BallTrackingState.DISAPPEARED_MID:
        # Mid-field disappearance - show different indicator
        text = config.MID_FIELD_DISAPPEAR_TEXT
        color = config.MID_FIELD_DISAPPEAR_COLOR
        pos_x = x_offset + config.MID_FIELD_DISAPPEAR_POS_X
        pos_y = config.MID_FIELD_DISAPPEAR_POS_Y
    elif is_persisting and counter > 0:
        # Persist display after returning to normal
        text = f"WAS {persist_side}: {counter}f"
        color = config.UNIFIED_PERSIST_COLOR
    else:
        return  # Nothing to draw (NORMAL state with no persist)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.UNIFIED_COUNTER_FONT_SCALE
    thickness = max(1, int(2 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    pad_y = max(4, int(10 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))

    # Draw background box
    cv2.rectangle(frame, (pos_x - pad_x, pos_y - th - pad_y),
                  (pos_x + tw + pad_x * 2, pos_y + pad_y), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (pos_x, pos_y), font,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_vertical_deviation_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    direction: Optional[str],
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw vertical deviation counter showing ball is moving UP or DOWN.

    In a triple cone drill, the ball should mostly move horizontally.
    This counter indicates when the ball momentum is deviating vertically.

    Args:
        frame: Video frame to draw on
        count: Number of consecutive frames with vertical deviation
        is_active: True if currently deviating, False if showing persist display
        direction: "UP" or "DOWN" (direction of vertical deviation)
        config: Annotation configuration
        x_offset: Horizontal offset (for sidebar)
    """
    if count <= 0:
        return

    dir_str = direction if direction else "?"
    text = f"VERT DEV ({dir_str}): {count}f"

    x = x_offset + config.VERTICAL_DEVIATION_COUNTER_POS_X
    y = config.VERTICAL_DEVIATION_COUNTER_POS_Y

    # Choose color based on direction and active state
    if is_active:
        if direction == "UP":
            color = config.VERTICAL_DEVIATION_UP_COLOR
        else:  # DOWN
            color = config.VERTICAL_DEVIATION_DOWN_COLOR
    else:
        color = config.VERTICAL_DEVIATION_PERSIST_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.VERTICAL_DEVIATION_COUNTER_FONT_SCALE
    thickness = max(1, int(2 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Scale padding proportionally
    pad_x = max(2, int(5 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    pad_y = max(4, int(10 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))

    # Draw background box
    cv2.rectangle(frame, (x - pad_x, y - th - pad_y),
                  (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font,
                font_scale, color, thickness, cv2.LINE_AA)
