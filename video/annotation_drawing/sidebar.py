"""
Sidebar drawing functions for Triple Cone annotation.

Draws the debug sidebar with coordinates, zone status, and legends.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from ..annotation_config import (
        TripleConeAnnotationConfig,
        KEYPOINT_BODY_PART,
        KEYPOINT_COLORS,
        TRACKED_KEYPOINTS,
    )
    from ..annotation_data.structures import (
        BallPositionResult,
        IntentionPositionResult,
        TurnEvent,
        ConeData,
    )
except ImportError:
    from annotation_config import (
        TripleConeAnnotationConfig,
        KEYPOINT_BODY_PART,
        KEYPOINT_COLORS,
        TRACKED_KEYPOINTS,
    )
    from annotation_data.structures import (
        BallPositionResult,
        IntentionPositionResult,
        TurnEvent,
        ConeData,
    )


def draw_sidebar_section_header(frame: np.ndarray, y: int, text: str,
                                color: Tuple[int, int, int], config: TripleConeAnnotationConfig) -> int:
    """Draw a section header in the sidebar. Returns next y position."""
    cv2.line(frame, (config.SIDEBAR_PADDING, y),
             (config.SIDEBAR_WIDTH - config.SIDEBAR_PADDING, y),
             config.SIDEBAR_HEADER_COLOR, 1)

    y += 18
    cv2.putText(frame, text, (config.SIDEBAR_PADDING, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                color, 1, cv2.LINE_AA)

    return y + 8


def draw_sidebar_row(frame: np.ndarray, y: int, label: str,
                     x_coord: Optional[float], y_coord: Optional[float],
                     color: Tuple[int, int, int], config: TripleConeAnnotationConfig) -> int:
    """Draw a coordinate row in the sidebar. Returns next y position."""
    # Check for both None and NaN values
    x_valid = x_coord is not None and pd.notna(x_coord)
    y_valid = y_coord is not None and pd.notna(y_coord)
    if x_valid and y_valid:
        coord_str = f"({int(x_coord):4d}, {int(y_coord):4d})"
    else:
        coord_str = "   --"

    label_x = config.SIDEBAR_PADDING + max(2, int(5 * config.FONT_SCALE_FACTOR))
    cv2.putText(frame, f"{label}:", (label_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.05 * config.FONT_SCALE_FACTOR,
                color, 1, cv2.LINE_AA)

    # Coordinate column offset proportional to sidebar width (130/300 = 43% from right edge)
    coord_offset = int(130 * config.RESOLUTION_SCALE)  # Keep linear for position
    coord_x = config.SIDEBAR_WIDTH - coord_offset
    cv2.putText(frame, coord_str, (coord_x, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.05 * config.FONT_SCALE_FACTOR,
                (200, 200, 200), 1, cv2.LINE_AA)

    return y + config.SIDEBAR_LINE_HEIGHT


def draw_sidebar(frame: np.ndarray, frame_id: int,
                 cone1: ConeData,
                 cone2: ConeData,
                 cone3: ConeData,
                 ball_center: Optional[Tuple[float, float]],
                 pose_keypoints: Dict[str, Tuple[float, float, float]],
                 config: TripleConeAnnotationConfig,
                 active_zone: Optional[str] = None,
                 ball_position_result: Optional[BallPositionResult] = None,
                 intention_result: Optional[IntentionPositionResult] = None,
                 turn_events: Optional[List[TurnEvent]] = None) -> None:
    """Draw the sidebar with all object coordinates for Triple Cone drill."""
    # Fill sidebar background
    frame[:, :config.SIDEBAR_WIDTH] = config.SIDEBAR_BG_COLOR

    y = 25

    # Frame number header
    cv2.putText(frame, f"FRAME: {frame_id}", (config.SIDEBAR_PADDING, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y += 35

    # ===== CONES SECTION (3 cones) =====
    y = draw_sidebar_section_header(frame, y, "CONES (static)",
                                    config.CONE1_COLOR, config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    y = draw_sidebar_row(frame, y, "CONE1", cone1.center_x, cone1.center_y,
                         config.CONE1_COLOR, config)
    y = draw_sidebar_row(frame, y, "CONE2", cone2.center_x, cone2.center_y,
                         config.CONE2_COLOR, config)
    y = draw_sidebar_row(frame, y, "CONE3", cone3.center_x, cone3.center_y,
                         config.CONE3_COLOR, config)

    y += max(4, int(10 * config.FONT_SCALE_FACTOR))

    # ===== BALL SECTION =====
    y = draw_sidebar_section_header(frame, y, "BALL (dynamic)",
                                    config.BALL_COLOR, config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    ball_x = ball_center[0] if ball_center else None
    ball_y = ball_center[1] if ball_center else None
    y = draw_sidebar_row(frame, y, "BALL", ball_x, ball_y,
                         config.BALL_COLOR, config)

    y += max(4, int(10 * config.FONT_SCALE_FACTOR))

    # ===== POSE SECTION =====
    y = draw_sidebar_section_header(frame, y, "POSE (dynamic)",
                                    config.POSE_SKELETON_COLOR, config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    for kp_name, display_name in TRACKED_KEYPOINTS:
        if kp_name in pose_keypoints:
            x, ycoord, conf = pose_keypoints[kp_name]
            if conf >= config.MIN_KEYPOINT_CONFIDENCE:
                body_part = KEYPOINT_BODY_PART.get(kp_name, 'torso')
                color = KEYPOINT_COLORS.get(body_part, config.POSE_KEYPOINT_COLOR)
                y = draw_sidebar_row(frame, y, display_name, x, ycoord, color, config)
            else:
                y = draw_sidebar_row(frame, y, display_name, None, None,
                                     (100, 100, 100), config)
        else:
            y = draw_sidebar_row(frame, y, display_name, None, None,
                                 (100, 100, 100), config)

    y += max(6, int(15 * config.FONT_SCALE_FACTOR))

    # ===== TURNING ZONE SECTION (3 zones) =====
    y = draw_sidebar_section_header(frame, y, "TURNING ZONE",
                                    (150, 150, 150), config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    if active_zone == "CONE1":
        zone_text = "Ball in: CONE1 (HOME)"
        zone_color = config.CONE1_ZONE_COLOR
    elif active_zone == "CONE2":
        zone_text = "Ball in: CONE2"
        zone_color = config.CONE2_ZONE_COLOR
    elif active_zone == "CONE3":
        zone_text = "Ball in: CONE3"
        zone_color = config.CONE3_ZONE_COLOR
    else:
        zone_text = "Ball in: --"
        zone_color = (100, 100, 100)

    cv2.putText(frame, zone_text,
                (config.SIDEBAR_PADDING + max(2, int(5 * config.FONT_SCALE_FACTOR)), y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                zone_color, 1, cv2.LINE_AA)

    y += config.SIDEBAR_LINE_HEIGHT + max(4, int(10 * config.FONT_SCALE_FACTOR))

    # ===== BALL POSITION SECTION =====
    y = draw_sidebar_section_header(frame, y, "BALL POSITION",
                                    (150, 150, 150), config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    text_indent = config.SIDEBAR_PADDING + max(2, int(5 * config.FONT_SCALE_FACTOR))

    # Momentum-based position (existing)
    if ball_position_result is not None:
        momentum_text = f"Momentum: {ball_position_result.position}"
        cv2.putText(frame, momentum_text,
                    (text_indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    ball_position_result.color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Momentum: --",
                    (text_indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    (100, 100, 100), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT

    # Intention-based position (new)
    if intention_result is not None:
        intention_text = f"Intention: {intention_result.position}"
        cv2.putText(frame, intention_text,
                    (text_indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    intention_result.color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Intention: --",
                    (text_indent, y),
                    cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE,
                    (100, 100, 100), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT

    # Delta X (shared)
    delta_x = ball_position_result.ball_hip_delta_x if ball_position_result else 0.0
    delta_text = f"Delta X: {delta_x:+.0f}px"
    cv2.putText(frame, delta_text,
                (text_indent, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1 * config.FONT_SCALE_FACTOR,
                (180, 180, 180), 1, cv2.LINE_AA)

    y += config.SIDEBAR_LINE_HEIGHT + max(4, int(10 * config.FONT_SCALE_FACTOR))

    # ===== TURN EVENTS SECTION =====
    if config.DRAW_EVENT_LOG and turn_events:
        y = draw_sidebar_section_header(frame, y, "TURN EVENTS",
                                        (150, 150, 150), config)
        y += max(2, int(5 * config.FONT_SCALE_FACTOR))

        recent = turn_events[-config.EVENT_LOG_MAX_EVENTS:]
        for event in recent:
            # Format: "12.5s CONE2: L->R"
            event_text = f"{event.timestamp:5.1f}s {event.zone}: {event.from_direction[0]}->{event.to_direction[0]}"

            # Color by zone
            if event.zone == "CONE1":
                event_color = config.CONE1_ZONE_COLOR
            elif event.zone == "CONE2":
                event_color = config.CONE2_ZONE_COLOR
            elif event.zone == "CONE3":
                event_color = config.CONE3_ZONE_COLOR
            else:
                event_color = (150, 150, 150)

            cv2.putText(frame, event_text,
                        (config.SIDEBAR_PADDING + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1,
                        event_color, 1, cv2.LINE_AA)
            y += config.SIDEBAR_LINE_HEIGHT - 4

    # ===== ARROW LEGEND SECTION =====
    y += max(6, int(15 * config.FONT_SCALE_FACTOR))
    y = draw_sidebar_section_header(frame, y, "ARROW LEGEND",
                                    (150, 150, 150), config)
    y += max(2, int(5 * config.FONT_SCALE_FACTOR))

    # Scale legend layout values (with minimum values for readability)
    scale = config.FONT_SCALE_FACTOR  # Use font scale for legend
    legend_font_scale = max(0.35, 0.45 * scale)  # Minimum 0.35 for readability
    legend_height = max(8, int(12 * scale))
    swatch_width = max(12, int(18 * scale))
    text_offset = max(16, int(22 * scale))
    col2_offset = max(45, int(60 * scale))
    col3_offset = max(90, int(115 * scale))

    # Momentum arrow legend (at hip)
    cv2.putText(frame, "MOMENTUM (at hip):",
                (config.SIDEBAR_PADDING + max(2, int(5 * scale)), y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1 * scale,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT - max(1, int(2 * scale))

    # Draw small color samples for momentum
    sample_x = config.SIDEBAR_PADDING + max(4, int(10 * scale))
    cv2.rectangle(frame, (sample_x, y - legend_height), (sample_x + swatch_width, y), config.MOMENTUM_COLOR_LOW, -1)
    cv2.putText(frame, "Slow", (sample_x + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.MOMENTUM_COLOR_LOW, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + col2_offset, y - legend_height), (sample_x + col2_offset + swatch_width, y), config.MOMENTUM_COLOR_MID, -1)
    cv2.putText(frame, "Med", (sample_x + col2_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.MOMENTUM_COLOR_MID, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + col3_offset, y - legend_height), (sample_x + col3_offset + swatch_width, y), config.MOMENTUM_COLOR_HIGH, -1)
    cv2.putText(frame, "Fast", (sample_x + col3_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.MOMENTUM_COLOR_HIGH, 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT + max(2, int(5 * scale))

    # Intention arrow legend (above head)
    cv2.putText(frame, "INTENTION (above head):",
                (config.SIDEBAR_PADDING + max(2, int(5 * scale)), y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1 * scale,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT - max(1, int(2 * scale))

    # Draw small color samples for intention (2 columns only)
    intent_col2_offset = max(50, int(70 * scale))
    cv2.rectangle(frame, (sample_x, y - legend_height), (sample_x + swatch_width, y), config.TORSO_FACING_COLOR_RIGHT, -1)
    cv2.putText(frame, "Right", (sample_x + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.TORSO_FACING_COLOR_RIGHT, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + intent_col2_offset, y - legend_height), (sample_x + intent_col2_offset + swatch_width, y), config.TORSO_FACING_COLOR_LEFT, -1)
    cv2.putText(frame, "Left", (sample_x + intent_col2_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.TORSO_FACING_COLOR_LEFT, 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT + max(2, int(5 * scale))

    # Ball momentum arrow legend (at ball)
    cv2.putText(frame, "BALL MOMENTUM:",
                (config.SIDEBAR_PADDING + max(2, int(5 * scale)), y),
                cv2.FONT_HERSHEY_SIMPLEX, config.SIDEBAR_FONT_SCALE - 0.1 * scale,
                (200, 200, 200), 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT - max(1, int(2 * scale))

    # Draw small color samples for ball momentum (orange gradient)
    cv2.rectangle(frame, (sample_x, y - legend_height), (sample_x + swatch_width, y), config.BALL_MOMENTUM_COLOR_LOW, -1)
    cv2.putText(frame, "Slow", (sample_x + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.BALL_MOMENTUM_COLOR_LOW, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + col2_offset, y - legend_height), (sample_x + col2_offset + swatch_width, y), config.BALL_MOMENTUM_COLOR_MID, -1)
    cv2.putText(frame, "Med", (sample_x + col2_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.BALL_MOMENTUM_COLOR_MID, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (sample_x + col3_offset, y - legend_height), (sample_x + col3_offset + swatch_width, y), config.BALL_MOMENTUM_COLOR_HIGH, -1)
    cv2.putText(frame, "Fast", (sample_x + col3_offset + text_offset, y),
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, config.BALL_MOMENTUM_COLOR_HIGH, 1, cv2.LINE_AA)
    y += config.SIDEBAR_LINE_HEIGHT

    # Vertical separator line
    cv2.line(frame, (config.SIDEBAR_WIDTH - 1, 0),
             (config.SIDEBAR_WIDTH - 1, frame.shape[0]),
             (60, 60, 60), 2)
