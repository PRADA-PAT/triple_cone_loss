"""
Primitive drawing functions for Triple Cone annotation.

Contains basic drawing operations: bounding boxes, skeleton, cone markers, arrows.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from ..annotation_config import (
        TripleConeAnnotationConfig,
        SKELETON_CONNECTIONS,
        KEYPOINT_BODY_PART,
        KEYPOINT_COLORS,
        CONE_COLOR_PALETTE,
    )
    from ..annotation_data.structures import ConeData
except ImportError:
    from annotation_config import (
        TripleConeAnnotationConfig,
        SKELETON_CONNECTIONS,
        KEYPOINT_BODY_PART,
        KEYPOINT_COLORS,
        CONE_COLOR_PALETTE,
    )
    from annotation_data.structures import ConeData

# Import DetectedCone for generic cone drawing
try:
    from detection.data_structures import DetectedCone
except ImportError:
    DetectedCone = None  # Will fail gracefully if not available


def draw_bbox(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
              color: Tuple[int, int, int], label: str, config: TripleConeAnnotationConfig,
              x_offset: int = 0) -> None:
    """Draw a bounding box with label."""
    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [x1, y1, x2, y2]):
        return
    x1, y1, x2, y2 = int(x1) + x_offset, int(y1), int(x2) + x_offset, int(y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
    )
    cv2.rectangle(frame, (x1, y1 - text_height - 8),
                  (x1 + text_width + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.TEXT_BG_COLOR, config.FONT_THICKNESS)


def draw_triple_cone_markers(
    frame: np.ndarray,
    cone1: ConeData,
    cone2: ConeData,
    cone3: ConeData,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw 3 cone markers with actual bounding boxes from detection."""
    cones = [
        ("CONE1 (HOME)", cone1, config.CONE1_COLOR),
        ("CONE2", cone2, config.CONE2_COLOR),
        ("CONE3", cone3, config.CONE3_COLOR),
    ]

    for label, cone, color in cones:
        # Draw actual bounding box from detection
        half_w = cone.width / 2
        half_h = cone.height / 2
        x1 = int(cone.center_x - half_w) + x_offset
        y1 = int(cone.center_y - half_h)
        x2 = int(cone.center_x + half_w) + x_offset
        y2 = int(cone.center_y + half_h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label above
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_x = x1
        label_y = y1 - 8

        # Background for label
        cv2.rectangle(frame,
                      (label_x - 2, label_y - text_height - 2),
                      (label_x + text_width + 2, label_y + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_cone_markers(
    frame: np.ndarray,
    detected_cones: List,  # List[DetectedCone]
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw N cone markers using color palette.

    Used by annotate_video.py for generic multi-drill support.
    Each cone gets a color from CONE_COLOR_PALETTE based on its index.

    Args:
        frame: Canvas to draw on
        detected_cones: List of DetectedCone objects with position and definition
        config: Annotation config
        x_offset: X offset for sidebar
    """
    for i, cone in enumerate(detected_cones):
        color = CONE_COLOR_PALETTE[i % len(CONE_COLOR_PALETTE)]
        x, y = cone.position
        label = cone.definition.label

        # Draw cone marker (filled circle with border)
        center = (int(x) + x_offset, int(y))
        cv2.circle(frame, center, 10, color, -1)
        cv2.circle(frame, center, 10, (0, 0, 0), 2)

        # Draw label above the cone
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_x = center[0] - text_width // 2
        label_y = center[1] - 18

        # Background for label
        cv2.rectangle(frame,
                      (label_x - 2, label_y - text_height - 2),
                      (label_x + text_width + 2, label_y + 2),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, keypoints: Dict[str, Tuple[float, float, float]],
                  config: TripleConeAnnotationConfig, x_offset: int = 0) -> None:
    """Draw pose skeleton with keypoints."""
    # Draw connections first
    for kp1_name, kp2_name in SKELETON_CONNECTIONS:
        if kp1_name in keypoints and kp2_name in keypoints:
            x1, y1, conf1 = keypoints[kp1_name]
            x2, y2, conf2 = keypoints[kp2_name]

            if conf1 >= config.MIN_KEYPOINT_CONFIDENCE and conf2 >= config.MIN_KEYPOINT_CONFIDENCE:
                pt1 = (int(x1) + x_offset, int(y1))
                pt2 = (int(x2) + x_offset, int(y2))
                cv2.line(frame, pt1, pt2, config.POSE_SKELETON_COLOR, config.SKELETON_THICKNESS)

    # Draw keypoints
    for kp_name, (x, y, conf) in keypoints.items():
        if conf >= config.MIN_KEYPOINT_CONFIDENCE:
            pt = (int(x) + x_offset, int(y))
            body_part = KEYPOINT_BODY_PART.get(kp_name, 'torso')
            color = KEYPOINT_COLORS.get(body_part, config.POSE_KEYPOINT_COLOR)
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, color, -1)
            cv2.circle(frame, pt, config.KEYPOINT_RADIUS, (0, 0, 0), 1)


def get_momentum_color(magnitude: float, config: TripleConeAnnotationConfig) -> Tuple[int, int, int]:
    """Calculate momentum arrow color based on speed magnitude."""
    if magnitude <= config.MOMENTUM_SPEED_LOW:
        t = 0.0
    elif magnitude >= config.MOMENTUM_SPEED_HIGH:
        t = 1.0
    else:
        t = (magnitude - config.MOMENTUM_SPEED_LOW) / (config.MOMENTUM_SPEED_HIGH - config.MOMENTUM_SPEED_LOW)

    if t <= 0.5:
        ratio = t * 2
        b = int(config.MOMENTUM_COLOR_LOW[0] + ratio * (config.MOMENTUM_COLOR_MID[0] - config.MOMENTUM_COLOR_LOW[0]))
        g = int(config.MOMENTUM_COLOR_LOW[1] + ratio * (config.MOMENTUM_COLOR_MID[1] - config.MOMENTUM_COLOR_LOW[1]))
        r = int(config.MOMENTUM_COLOR_LOW[2] + ratio * (config.MOMENTUM_COLOR_MID[2] - config.MOMENTUM_COLOR_LOW[2]))
    else:
        ratio = (t - 0.5) * 2
        b = int(config.MOMENTUM_COLOR_MID[0] + ratio * (config.MOMENTUM_COLOR_HIGH[0] - config.MOMENTUM_COLOR_MID[0]))
        g = int(config.MOMENTUM_COLOR_MID[1] + ratio * (config.MOMENTUM_COLOR_HIGH[1] - config.MOMENTUM_COLOR_MID[1]))
        r = int(config.MOMENTUM_COLOR_MID[2] + ratio * (config.MOMENTUM_COLOR_HIGH[2] - config.MOMENTUM_COLOR_MID[2]))

    return (b, g, r)


def draw_momentum_arrow(
    frame: np.ndarray,
    current_hip: Tuple[float, float],
    previous_hip: Tuple[float, float],
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw thick horizontal momentum arrow with color gradient."""
    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [current_hip[0], current_hip[1], previous_hip[0], previous_hip[1]]):
        return

    dx = current_hip[0] - previous_hip[0]
    horizontal_magnitude = abs(dx)

    if horizontal_magnitude < config.MOMENTUM_MIN_LENGTH:
        return

    color = get_momentum_color(horizontal_magnitude, config)
    scaled_length = min(horizontal_magnitude * config.MOMENTUM_SCALE, config.MOMENTUM_MAX_LENGTH)

    direction = 1 if dx > 0 else -1
    arrow_dx = direction * scaled_length

    start_x = int(current_hip[0]) + x_offset
    start_y = int(current_hip[1])
    end_x = int(start_x + arrow_dx)
    end_y = start_y

    cv2.arrowedLine(
        frame,
        (start_x, start_y),
        (end_x, end_y),
        color,
        config.MOMENTUM_THICKNESS,
        tipLength=0.25,
        line_type=cv2.LINE_AA
    )


def get_ball_momentum_color(magnitude: float, config: TripleConeAnnotationConfig) -> Tuple[int, int, int]:
    """Calculate ball momentum arrow color based on speed magnitude (orange gradient)."""
    if magnitude <= config.BALL_MOMENTUM_SPEED_LOW:
        t = 0.0
    elif magnitude >= config.BALL_MOMENTUM_SPEED_HIGH:
        t = 1.0
    else:
        t = (magnitude - config.BALL_MOMENTUM_SPEED_LOW) / (config.BALL_MOMENTUM_SPEED_HIGH - config.BALL_MOMENTUM_SPEED_LOW)

    if t <= 0.5:
        ratio = t * 2
        b = int(config.BALL_MOMENTUM_COLOR_LOW[0] + ratio * (config.BALL_MOMENTUM_COLOR_MID[0] - config.BALL_MOMENTUM_COLOR_LOW[0]))
        g = int(config.BALL_MOMENTUM_COLOR_LOW[1] + ratio * (config.BALL_MOMENTUM_COLOR_MID[1] - config.BALL_MOMENTUM_COLOR_LOW[1]))
        r = int(config.BALL_MOMENTUM_COLOR_LOW[2] + ratio * (config.BALL_MOMENTUM_COLOR_MID[2] - config.BALL_MOMENTUM_COLOR_LOW[2]))
    else:
        ratio = (t - 0.5) * 2
        b = int(config.BALL_MOMENTUM_COLOR_MID[0] + ratio * (config.BALL_MOMENTUM_COLOR_HIGH[0] - config.BALL_MOMENTUM_COLOR_MID[0]))
        g = int(config.BALL_MOMENTUM_COLOR_MID[1] + ratio * (config.BALL_MOMENTUM_COLOR_HIGH[1] - config.BALL_MOMENTUM_COLOR_MID[1]))
        r = int(config.BALL_MOMENTUM_COLOR_MID[2] + ratio * (config.BALL_MOMENTUM_COLOR_HIGH[2] - config.BALL_MOMENTUM_COLOR_MID[2]))

    return (b, g, r)


def draw_ball_momentum_arrow(
    frame: np.ndarray,
    current_ball: Tuple[float, float],
    previous_ball: Tuple[float, float],
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw momentum arrow for ball showing movement direction and speed (orange gradient)."""
    # Skip if any coordinate is NaN
    if any(pd.isna(v) for v in [current_ball[0], current_ball[1], previous_ball[0], previous_ball[1]]):
        return

    dx = current_ball[0] - previous_ball[0]
    dy = current_ball[1] - previous_ball[1]
    magnitude = np.sqrt(dx * dx + dy * dy)

    if magnitude < config.BALL_MOMENTUM_MIN_LENGTH:
        return

    color = get_ball_momentum_color(magnitude, config)
    scaled_length = min(magnitude * config.BALL_MOMENTUM_SCALE, config.BALL_MOMENTUM_MAX_LENGTH)

    # Normalize direction
    norm_dx = dx / magnitude
    norm_dy = dy / magnitude

    start_x = int(current_ball[0]) + x_offset
    start_y = int(current_ball[1])
    end_x = int(start_x + norm_dx * scaled_length)
    end_y = int(start_y + norm_dy * scaled_length)

    cv2.arrowedLine(
        frame,
        (start_x, start_y),
        (end_x, end_y),
        color,
        config.BALL_MOMENTUM_THICKNESS,
        tipLength=0.25,
        line_type=cv2.LINE_AA
    )


def calculate_ball_vertical_deviation(
    dx: float,
    dy: float,
    magnitude: float,
    threshold_degrees: float = 60.0
) -> Tuple[bool, Optional[str], float]:
    """
    Calculate if ball momentum is deviating vertically from horizontal path.

    In a triple cone drill, the ball should mostly move left-right (horizontal).
    This function detects when the ball is moving significantly up or down.

    Args:
        dx: Horizontal velocity component (positive = right)
        dy: Vertical velocity component (positive = down, in video coords)
        magnitude: Total velocity magnitude
        threshold_degrees: Angle from horizontal to consider "vertical" (default 60°)

    Returns:
        (is_deviating, direction, angle_from_horizontal)
        - is_deviating: True if angle > threshold
        - direction: "UP" (dy < 0) or "DOWN" (dy > 0) or None
        - angle_from_horizontal: 0-90 degrees (0 = pure horizontal, 90 = pure vertical)
    """
    import math

    if magnitude < 0.001:
        return False, None, 0.0

    # Calculate angle from horizontal (0-90 degrees)
    # atan2(|dy|, |dx|) gives angle from X-axis
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_degrees = math.degrees(angle_rad)

    is_deviating = angle_degrees > threshold_degrees

    if is_deviating:
        # In video coordinates, dy < 0 means moving UP (toward top of screen)
        direction = "UP" if dy < 0 else "DOWN"
    else:
        direction = None

    return is_deviating, direction, angle_degrees


def draw_intention_arrow(
    frame: np.ndarray,
    persons: dict,
    facing_direction: str,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw a horizontal arrow above the player's head showing intention (torso facing).

    The arrow is parallel to the video frame, pointing LEFT or RIGHT based on
    nose-hip orientation. This captures turn intention since head turns before body.

    Args:
        frame: Canvas to draw on
        persons: Dict of person_id -> keypoints dict
        facing_direction: "LEFT" or "RIGHT"
        config: Annotation config
        x_offset: X offset for sidebar
    """
    if not persons or facing_direction not in ("LEFT", "RIGHT"):
        return

    # Get head position (prefer nose, fallback to head keypoint)
    first_person_id = min(persons.keys())
    keypoints = persons[first_person_id]

    # Try nose first, then head keypoint
    head_pos = None
    for kp_name in ['nose', 'head']:
        kp = keypoints.get(kp_name)
        if kp and kp[2] >= config.MIN_KEYPOINT_CONFIDENCE:
            head_pos = (kp[0], kp[1])
            break

    if head_pos is None:
        return

    # Arrow center point (above head)
    arrow_center_x = int(head_pos[0]) + x_offset
    arrow_center_y = int(head_pos[1]) - config.INTENTION_ARROW_OFFSET_Y

    # Arrow direction and endpoints
    half_length = config.INTENTION_ARROW_LENGTH // 2

    if facing_direction == "RIGHT":
        start_x = arrow_center_x - half_length
        end_x = arrow_center_x + half_length
        color = config.TORSO_FACING_COLOR_RIGHT
    else:  # LEFT
        start_x = arrow_center_x + half_length
        end_x = arrow_center_x - half_length
        color = config.TORSO_FACING_COLOR_LEFT

    # Draw the arrow (horizontal, parallel to video)
    cv2.arrowedLine(
        frame,
        (start_x, arrow_center_y),
        (end_x, arrow_center_y),
        color,
        config.INTENTION_ARROW_THICKNESS,
        tipLength=config.INTENTION_ARROW_TIP_LENGTH,
        line_type=cv2.LINE_AA
    )


def draw_dashed_line(
    frame: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: int = 8,
    gap_length: int = 6
) -> None:
    """Draw a dashed line from pt1 to pt2."""
    x1, y1 = pt1
    x2, y2 = pt2

    # Calculate total distance and direction
    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx * dx + dy * dy)

    if distance < 1:
        return

    # Normalize direction
    dx /= distance
    dy /= distance

    # Draw dashes
    segment_length = dash_length + gap_length
    current_pos = 0.0

    while current_pos < distance:
        # Start of dash
        start_x = int(x1 + dx * current_pos)
        start_y = int(y1 + dy * current_pos)

        # End of dash (limited by total distance)
        end_pos = min(current_pos + dash_length, distance)
        end_x = int(x1 + dx * end_pos)
        end_y = int(y1 + dy * end_pos)

        cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)

        current_pos += segment_length


def draw_debug_axes(
    frame: np.ndarray,
    ball_center: Optional[Tuple[float, float]],
    config: TripleConeAnnotationConfig,
    video_width: int,
    video_height: int,
    x_offset: int = 0
) -> None:
    """Draw debug axes through ball position (always on)."""
    if ball_center is None:
        return
    # Skip if coordinates are NaN
    if pd.isna(ball_center[0]) or pd.isna(ball_center[1]):
        return

    ball_x = int(ball_center[0]) + x_offset
    ball_y = int(ball_center[1])

    # Vertical line through ball X
    cv2.line(frame, (ball_x, 0), (ball_x, video_height),
             config.DEBUG_AXES_COLOR, config.DEBUG_AXES_THICKNESS)

    # Horizontal line through ball Y
    cv2.line(frame, (x_offset, ball_y), (x_offset + video_width, ball_y),
             config.DEBUG_AXES_COLOR, config.DEBUG_AXES_THICKNESS)


def draw_edge_zones(
    frame: np.ndarray,
    video_width: int,
    video_height: int,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> np.ndarray:
    """Draw semi-transparent edge zone overlays."""
    overlay = frame.copy()

    # Left edge zone
    cv2.rectangle(
        overlay,
        (x_offset, 0),
        (x_offset + config.EDGE_MARGIN, video_height),
        config.EDGE_ZONE_COLOR, -1
    )
    # Right edge zone
    cv2.rectangle(
        overlay,
        (x_offset + video_width - config.EDGE_MARGIN, 0),
        (x_offset + video_width, video_height),
        config.EDGE_ZONE_COLOR, -1
    )

    return cv2.addWeighted(overlay, config.EDGE_ZONE_ALPHA, frame,
                           1 - config.EDGE_ZONE_ALPHA, 0)
