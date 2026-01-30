#!/usr/bin/env python3
"""
Generic Video Annotation for Multi-Drill Analysis.

Creates annotated videos for any drill type (3, 5, 7+ cones) using:
- STATIC cone positions from parquet detection (mean positions per player)
- DYNAMIC ball positions from parquet detection
- DYNAMIC pose skeleton from parquet detection
- LEFT SIDEBAR showing all object coordinates in real-time
- Turn detection and ball-behind tracking

Supports auto-detection of drill type from path patterns.

Usage:
    python annotate_video.py path/to/player_data_folder/
    python annotate_video.py path/to/data/ --drill-type seven_cone_weave
    python annotate_video.py --list
"""

import argparse
import sys
from pathlib import Path
from collections import deque
from typing import List, Optional

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Import from local modules - handle both package and direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .annotation_config import TripleConeAnnotationConfig, scale_config_for_resolution, CONE_COLOR_PALETTE
    from .annotation_utils import convert_to_h264, get_available_videos, get_drills_structure
    from .annotation_data import (
        BallTrackingState,
        ConeData,
        load_all_cone_positions,
        load_ball_data,
        load_pose_data,
        prepare_pose_lookup,
    )
    from .annotation_drawing import (
        draw_sidebar,
        draw_bbox,
        draw_cone_markers,
        draw_skeleton,
        draw_momentum_arrow,
        draw_ball_momentum_arrow,
        calculate_ball_vertical_deviation,
        draw_intention_arrow,
        draw_debug_axes,
        draw_edge_zones,
        draw_area_zone,
        draw_ball_position_indicator,
        draw_intention_position_indicator,
        draw_behind_counter,
        draw_intention_behind_counter,
        draw_edge_counter,
        draw_off_screen_indicator,
        draw_return_counter,
        draw_unified_tracking_indicator,
        draw_vertical_deviation_counter,
    )
    from .annotation_analysis import (
        determine_ball_position_relative_to_player,
        determine_torso_facing,
        determine_ball_position_vs_intention,
        check_edge_zone_status,
        update_ball_tracking_state,
    )
    from .turn_tracker import TripleConeTurnTracker
except ImportError:
    from annotation_config import TripleConeAnnotationConfig, scale_config_for_resolution, CONE_COLOR_PALETTE
    from annotation_utils import convert_to_h264, get_available_videos, get_drills_structure
    from annotation_data import (
        BallTrackingState,
        ConeData,
        load_all_cone_positions,
        load_ball_data,
        load_pose_data,
        prepare_pose_lookup,
    )
    from annotation_drawing import (
        draw_sidebar,
        draw_bbox,
        draw_cone_markers,
        draw_skeleton,
        draw_momentum_arrow,
        draw_ball_momentum_arrow,
        calculate_ball_vertical_deviation,
        draw_intention_arrow,
        draw_debug_axes,
        draw_edge_zones,
        draw_area_zone,
        draw_ball_position_indicator,
        draw_intention_position_indicator,
        draw_behind_counter,
        draw_intention_behind_counter,
        draw_edge_counter,
        draw_off_screen_indicator,
        draw_return_counter,
        draw_unified_tracking_indicator,
        draw_vertical_deviation_counter,
    )
    from annotation_analysis import (
        determine_ball_position_relative_to_player,
        determine_torso_facing,
        determine_ball_position_vs_intention,
        check_edge_zone_status,
        update_ball_tracking_state,
    )
    from turn_tracker import TripleConeTurnTracker

# Detection module imports
from detection.drill_config_loader import DrillConfigLoader, assign_cones_to_config
from detection.data_structures import ConeType, DetectedCone, DrillTypeConfig
from detection.turning_zones import TurningZone, TripleConeZoneConfig


def create_zones_for_turn_cones(
    detected_cones: List[DetectedCone],
    cone_data_list: List[ConeData],
    zone_config: TripleConeZoneConfig,
    image_height: int
) -> List[tuple]:
    """
    Create elliptical turning zones only for turn-type cones.

    Uses Y-position based perspective model: cones higher in frame (farther from
    camera) get more horizontal stretch, cones lower in frame (closer) get less.
    This simulates how a circle on the ground plane appears as an ellipse due to
    camera tilt.

    Args:
        detected_cones: List of all detected cones with definitions
        cone_data_list: List of ConeData with bbox dimensions (same order as detected_cones)
        zone_config: Zone configuration with base radius
        image_height: Height of the video frame in pixels

    Returns:
        List of (label, TurningZone) tuples for turn-type cones only
    """
    # Perspective squeeze constants - tune these visually
    CONE_MULTIPLIER = 10.0  # Zone horizontal diameter = N times cone width
    MIN_SQUEEZE = 4.0      # Compression at bottom of frame (closest to camera, rounder)
    MAX_SQUEEZE = 8.0      # Compression at top of frame (farthest from camera, flatter)

    zones = []
    for cone, cone_data in zip(detected_cones, cone_data_list):
        if cone.definition.type == ConeType.TURN:
            x, y = cone.position

            # Zone center at cone's BASE (bottom of bbox), not bbox center
            center_y = y + (cone_data.height / 2)

            # Horizontal diameter = CONE_MULTIPLIER * cone_width
            # semi_major = half of horizontal diameter
            semi_major = (CONE_MULTIPLIER / 2) * cone_data.width

            # Y-position based perspective model
            # normalized_y: 0.0 at top (far), 1.0 at bottom (near)
            normalized_y = center_y / image_height

            # Linear interpolation: more squeeze at top, less at bottom
            squeeze = MAX_SQUEEZE - (normalized_y * (MAX_SQUEEZE - MIN_SQUEEZE))

            # Vertical axis compressed by perspective (squeeze > 1 means more compression)
            semi_minor = semi_major / squeeze

            # Create ellipse
            zone = TurningZone(
                name=cone.definition.label,
                center_px=x,
                center_py=center_y,
                semi_major=semi_major,   # Horizontal (5x cone width diameter)
                semi_minor=semi_minor,   # Vertical (compressed by perspective)
            )
            zones.append((cone.definition.label, zone))

            # Debug output
            print(f"    {cone.definition.label}: cone_w={cone_data.width:.0f}, cone_h={cone_data.height:.0f}, "
                  f"bbox_center_y={y:.0f}, zone_center_y={center_y:.0f} (bottom of bbox), "
                  f"squeeze={squeeze:.2f}, zone={semi_major*2:.0f}x{semi_minor*2:.0f} (diameter)")

    return zones


def draw_turning_zones(
    frame: np.ndarray,
    turn_zones: List[tuple],
    ball_center: Optional[tuple],
    x_offset: int = 0,
    alpha: float = 0.25,
) -> str:
    """
    Draw turning zones and return which zone the ball is in (if any).

    Args:
        frame: Canvas to draw on
        turn_zones: List of (label, TurningZone) tuples
        ball_center: Ball position or None
        x_offset: X offset for sidebar
        alpha: Zone transparency

    Returns:
        Label of active zone if ball is inside, None otherwise
    """
    overlay = frame.copy()
    active_zone = None

    for i, (label, zone) in enumerate(turn_zones):
        color = CONE_COLOR_PALETTE[i % len(CONE_COLOR_PALETTE)]

        # Check if ball is in this zone
        is_active = False
        if ball_center and not any(pd.isna(v) for v in ball_center):
            if zone.contains_point(ball_center[0], ball_center[1]):
                is_active = True
                active_zone = label

        # Use highlight color if ball is inside
        draw_color = (0, 255, 255) if is_active else color

        # Draw ellipse (TurningZone uses center_px, center_py, semi_major, semi_minor)
        center = (int(zone.center_px) + x_offset, int(zone.center_py))
        axes = (int(zone.semi_major), int(zone.semi_minor))
        cv2.ellipse(overlay, center, axes, 0, 0, 360, draw_color, -1)

    # Blend overlay with original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return active_zone


def annotate_video(
    video_path: Path,
    parquet_dir: Path,
    output_path: Path,
    drill_config: Optional[DrillTypeConfig] = None,
    config: TripleConeAnnotationConfig = None
) -> bool:
    """
    Annotate video with cone positions, ball/pose tracking, and debug visualization.

    Args:
        video_path: Path to source video
        parquet_dir: Directory containing parquet files
        output_path: Path for output video
        drill_config: Optional drill configuration (auto-detected if None)
        config: Annotation config

    Returns:
        True if successful, False otherwise
    """
    if config is None:
        config = TripleConeAnnotationConfig()

    # Load cone positions from parquet
    cone_parquet = list(parquet_dir.glob("*_cone.parquet"))
    if not cone_parquet:
        print(f"  Error: No cone parquet found in {parquet_dir}")
        return False

    print(f"  Loading cone positions from parquet...")
    try:
        cone_data_list = load_all_cone_positions(cone_parquet[0])  # Returns List[ConeData]
    except Exception as e:
        print(f"  Error loading cones: {e}")
        return False

    print(f"    Found {len(cone_data_list)} cones")

    # Match cones to drill config - filter if too many detected
    if drill_config:
        if len(cone_data_list) > drill_config.cone_count:
            # More cones detected than expected - keep only the most reliable ones
            print(f"    Detected {len(cone_data_list)} cones, config expects {drill_config.cone_count}")
            print(f"    Filtering to top {drill_config.cone_count} by frame count...")
            cone_data_list = load_all_cone_positions(cone_parquet[0], max_cones=drill_config.cone_count)
            print(f"    Kept {len(cone_data_list)} most frequently detected cones")
        elif len(cone_data_list) < drill_config.cone_count:
            # Fewer cones than expected - fall back to generic mode
            print(f"  Warning: Expected {drill_config.cone_count} cones, found {len(cone_data_list)}")
            print(f"  Falling back to generic mode...")
            drill_config = None

    # Extract positions as tuples for assign_cones_to_config
    cone_positions = [cone_data.center for cone_data in cone_data_list]

    if drill_config:
        detected_cones = assign_cones_to_config(cone_positions, drill_config)
        print(f"    Matched to drill type: {drill_config.name}")
    else:
        # Create generic cone definitions
        from detection.data_structures import ConeDefinition
        detected_cones = [
            DetectedCone(
                position=cone_data.center,
                definition=ConeDefinition(
                    position=i,
                    type=ConeType.TURN,  # Treat all as turn cones for generic case
                    label=f"cone_{i+1}"
                )
            )
            for i, cone_data in enumerate(cone_data_list)
        ]
        print(f"    Using generic cone labels (no drill config matched)")

    # Print cone positions
    for cone in detected_cones:
        x, y = cone.position
        cone_type = cone.definition.type.value
        print(f"    {cone.definition.label} ({cone_type}): ({x:.0f}, {y:.0f})")

    # Load parquet data
    ball_parquets = list(parquet_dir.glob("*_football.parquet"))
    pose_parquets = list(parquet_dir.glob("*_pose.parquet"))

    if not ball_parquets or not pose_parquets:
        print(f"  Error: Missing parquet files in {parquet_dir}")
        return False

    print(f"  Loading parquet data...")
    ball_df = load_ball_data(ball_parquets[0], use_postprocessed=config.USE_POSTPROCESSED_BALL)
    pose_df = load_pose_data(pose_parquets[0])
    actually_used_pp = ball_df.attrs.get('using_postprocessed', False)
    if config.USE_POSTPROCESSED_BALL and not actually_used_pp:
        print(f"    Ball data: raw columns (fallback - no _pp columns available)")
    else:
        print(f"    Ball data: {'post-processed (_pp)' if actually_used_pp else 'raw'} columns")

    # Create lookup structures
    ball_cols = ['x1', 'y1', 'x2', 'y2', 'confidence']
    if 'interpolated' in ball_df.columns:
        ball_cols.append('interpolated')
    ball_lookup = ball_df.groupby('frame_id').apply(
        lambda g: g[ball_cols].to_dict('records')
    ).to_dict()
    pose_lookup = prepare_pose_lookup(pose_df)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Error: Cannot open video: {video_path}")
        return False

    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Scale config for resolution
    scale_config_for_resolution(config, orig_width)

    # Create turning zones for turn-type cones only
    # Uses Y-position based perspective model (farther cones = flatter ellipses)
    print(f"  Creating turning zones (Y-position perspective model)...")
    zone_config = TripleConeZoneConfig.default()
    turn_zones = create_zones_for_turn_cones(detected_cones, cone_data_list, zone_config, orig_height)
    turn_cone_count = len(turn_zones)
    print(f"    Created {turn_cone_count} turning zones")

    canvas_width = config.SIDEBAR_WIDTH + orig_width
    canvas_height = orig_height

    print(f"  Video: {orig_width}x{orig_height} @ {fps:.1f}fps, {total_frames} frames")
    print(f"  Output canvas: {canvas_width}x{canvas_height}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_width, canvas_height))

    if not out.isOpened():
        print(f"  Error: Cannot create output video: {output_path}")
        cap.release()
        return False

    print(f"  Processing frames...")

    # Initialize trackers
    hip_history: deque = deque(maxlen=config.MOMENTUM_LOOKBACK_FRAMES + 1)
    ball_history: deque = deque(maxlen=config.MOMENTUM_LOOKBACK_FRAMES + 1)
    turn_tracker = TripleConeTurnTracker()

    # Behind counters
    behind_counter: int = 0
    behind_display_value: int = 0
    behind_display_timer: int = 0
    behind_persist_frames = int(config.BEHIND_COUNTER_PERSIST_SECONDS * fps)

    intention_behind_counter: int = 0
    intention_behind_display_value: int = 0
    intention_behind_display_timer: int = 0

    # Unified tracking state machine
    unified_tracking_state: BallTrackingState = BallTrackingState.NORMAL
    unified_counter: int = 0
    unified_persist_value: int = 0
    unified_persist_timer: int = 0
    unified_persist_side: str = "NONE"
    unified_persist_frames = int(config.UNIFIED_COUNTER_PERSIST_SECONDS * fps)

    # Vertical deviation tracking
    vertical_deviation_counter: int = 0
    vertical_deviation_display_value: int = 0
    vertical_deviation_display_timer: int = 0
    vertical_deviation_direction: str = None
    vertical_deviation_persist_frames = int(config.VERTICAL_DEVIATION_PERSIST_SECONDS * fps)

    for frame_id in tqdm(range(total_frames), desc="  Annotating", unit="frame"):
        ret, video_frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps

        # Create canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:, config.SIDEBAR_WIDTH:] = video_frame

        # Get ball data
        balls = ball_lookup.get(frame_id, [])
        ball_center = None
        ball_is_off_screen = True
        for ball in balls:
            if ball['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                is_interpolated = ball.get('interpolated', False)
                if not is_interpolated:
                    ball_is_off_screen = False

                center_x = (ball['x1'] + ball['x2']) / 2
                center_y = (ball['y1'] + ball['y2']) / 2
                ball_center = (center_x, center_y)
                break

        # Get pose keypoints
        persons = pose_lookup.get(frame_id, {})
        pose_keypoints = {}
        if persons:
            first_person_id = min(persons.keys())
            pose_keypoints = persons[first_person_id]

        # Get hip position
        current_hip = None
        for _, keypoints in persons.items():
            hip_data = keypoints.get('hip')
            if hip_data and hip_data[2] >= config.MIN_KEYPOINT_CONFIDENCE:
                current_hip = (hip_data[0], hip_data[1])
                break

        # Update histories
        if current_hip:
            hip_history.append(current_hip)
        previous_hip = hip_history[0] if len(hip_history) >= 2 else None

        if ball_center and not any(pd.isna(v) for v in ball_center):
            ball_history.append(ball_center)
        previous_ball = ball_history[0] if len(ball_history) >= 2 else None

        # Get movement direction
        movement_direction = None
        if current_hip and previous_hip:
            dx = current_hip[0] - previous_hip[0]
            if dx > config.MOVEMENT_THRESHOLD:
                movement_direction = "RIGHT"
            elif dx < -config.MOVEMENT_THRESHOLD:
                movement_direction = "LEFT"

        # Check turning zone
        active_zone = None
        if ball_center and not any(pd.isna(v) for v in ball_center):
            for label, zone in turn_zones:
                if zone.contains_point(ball_center[0], ball_center[1]):
                    active_zone = label
                    break

        # Update turn tracker
        turn_tracker.update(frame_id, timestamp, active_zone, movement_direction)

        # Ball position detection
        ball_position_result = None
        if config.DRAW_BALL_POSITION:
            ball_position_result = determine_ball_position_relative_to_player(
                ball_center, current_hip, previous_hip, config
            )

        # Torso facing detection
        torso_facing = None
        if config.DRAW_TORSO_FACING:
            torso_facing = determine_torso_facing(persons, config)

        # Intention-based ball position detection
        intention_result = None
        if config.DRAW_BALL_POSITION_INTENTION and torso_facing:
            intention_result = determine_ball_position_vs_intention(
                ball_center, current_hip, torso_facing, config
            )

        # Update momentum behind counter
        if ball_position_result and ball_position_result.position == "BEHIND":
            behind_counter += 1
            behind_display_value = behind_counter
            behind_display_timer = behind_persist_frames
        else:
            if behind_counter > 0:
                behind_display_value = behind_counter
                behind_display_timer = behind_persist_frames
            behind_counter = 0

        if behind_display_timer > 0:
            behind_display_timer -= 1

        # Update intention behind counter
        if intention_result and intention_result.position == "I-BEHIND":
            intention_behind_counter += 1
            intention_behind_display_value = intention_behind_counter
            intention_behind_display_timer = behind_persist_frames
        else:
            if intention_behind_counter > 0:
                intention_behind_display_value = intention_behind_counter
                intention_behind_display_timer = behind_persist_frames
            intention_behind_counter = 0

        if intention_behind_display_timer > 0:
            intention_behind_display_timer -= 1

        # Edge zone detection
        ball_x = ball_center[0] if ball_center else None
        edge_status = check_edge_zone_status(ball_x, orig_width, config)

        # Unified state machine
        if config.UNIFIED_TRACKING_ENABLED:
            prev_state = unified_tracking_state
            ball_visible = not ball_is_off_screen
            new_state, should_reset = update_ball_tracking_state(
                unified_tracking_state, ball_visible, edge_status
            )

            if should_reset:
                if new_state == BallTrackingState.NORMAL and prev_state != BallTrackingState.NORMAL:
                    unified_persist_value = unified_counter
                    unified_persist_timer = unified_persist_frames
                    if prev_state in (BallTrackingState.EDGE_LEFT, BallTrackingState.OFF_SCREEN_LEFT):
                        unified_persist_side = "LEFT"
                    elif prev_state in (BallTrackingState.EDGE_RIGHT, BallTrackingState.OFF_SCREEN_RIGHT):
                        unified_persist_side = "RIGHT"
                unified_counter = 0

            if new_state in (BallTrackingState.EDGE_LEFT, BallTrackingState.EDGE_RIGHT,
                            BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT):
                unified_counter += 1

            unified_tracking_state = new_state

            if unified_tracking_state == BallTrackingState.NORMAL and unified_persist_timer > 0:
                unified_persist_timer -= 1

        # Vertical deviation detection
        if config.DETECT_BALL_VERTICAL_DEVIATION and ball_center and previous_ball:
            dx = ball_center[0] - previous_ball[0]
            dy = ball_center[1] - previous_ball[1]
            magnitude = np.sqrt(dx * dx + dy * dy)

            if magnitude >= config.VERTICAL_DEVIATION_MIN_SPEED:
                is_deviating, direction, angle = calculate_ball_vertical_deviation(
                    dx, dy, magnitude, config.VERTICAL_DEVIATION_THRESHOLD
                )

                if is_deviating:
                    vertical_deviation_counter += 1
                    vertical_deviation_display_value = vertical_deviation_counter
                    vertical_deviation_display_timer = vertical_deviation_persist_frames
                    vertical_deviation_direction = direction
                else:
                    if vertical_deviation_counter > 0:
                        vertical_deviation_display_value = vertical_deviation_counter
                        vertical_deviation_display_timer = vertical_deviation_persist_frames
                    vertical_deviation_counter = 0
            else:
                if vertical_deviation_counter > 0:
                    vertical_deviation_display_value = vertical_deviation_counter
                    vertical_deviation_display_timer = vertical_deviation_persist_frames
                vertical_deviation_counter = 0

        if vertical_deviation_display_timer > 0:
            vertical_deviation_display_timer -= 1

        # === DRAW EVERYTHING ===

        # 1. Turning zones (only for turn-type cones)
        draw_turning_zones(
            canvas, turn_zones, ball_center,
            x_offset=config.SIDEBAR_WIDTH,
            alpha=config.ZONE_ALPHA,
        )

        # 2. Edge zones
        canvas = draw_edge_zones(canvas, orig_width, orig_height, config, x_offset=config.SIDEBAR_WIDTH)

        # 3. Sidebar (using dynamic cone list)
        draw_sidebar_generic(
            canvas, frame_id, detected_cones, ball_center, pose_keypoints,
            config, active_zone, ball_position_result,
            intention_result=intention_result,
            turn_events=turn_tracker.get_recent_events(config.EVENT_LOG_MAX_EVENTS)
        )

        # 4. Area zone (filled polygon connecting area-type cones)
        canvas = draw_area_zone(canvas, detected_cones, config, x_offset=config.SIDEBAR_WIDTH)

        # 5. Cone markers (N cones with palette colors and bounding boxes)
        draw_cone_markers(canvas, detected_cones, config, x_offset=config.SIDEBAR_WIDTH,
                         cone_data_list=cone_data_list, draw_bbox=True)

        # 6. Ball bbox
        for ball in balls:
            if ball['confidence'] >= config.MIN_BBOX_CONFIDENCE:
                label = f"Ball {ball['confidence']:.2f}"
                draw_bbox(canvas, ball['x1'], ball['y1'], ball['x2'], ball['y2'],
                         config.BALL_COLOR, label, config, x_offset=config.SIDEBAR_WIDTH)

        # 5.5. Ball momentum arrow
        if config.DRAW_BALL_MOMENTUM_ARROW and ball_center and previous_ball:
            draw_ball_momentum_arrow(canvas, ball_center, previous_ball, config, x_offset=config.SIDEBAR_WIDTH)

        # 6. Debug axes
        if config.DRAW_DEBUG_AXES:
            draw_debug_axes(canvas, ball_center, config, orig_width, orig_height, x_offset=config.SIDEBAR_WIDTH)

        # 7. Pose skeleton
        for _, keypoints in persons.items():
            draw_skeleton(canvas, keypoints, config, x_offset=config.SIDEBAR_WIDTH)

        # 8. Ball position indicator
        if config.DRAW_BALL_POSITION and ball_position_result:
            draw_ball_position_indicator(canvas, ball_center, current_hip, ball_position_result,
                                        config, x_offset=config.SIDEBAR_WIDTH)

        # 8.5. Intention arrow
        if config.DRAW_TORSO_FACING and torso_facing:
            draw_intention_arrow(canvas, persons, torso_facing, config, x_offset=config.SIDEBAR_WIDTH)

        # 8.6. Intention position indicator
        if config.DRAW_BALL_POSITION_INTENTION and intention_result:
            draw_intention_position_indicator(canvas, ball_center, current_hip, intention_result,
                                             config, x_offset=config.SIDEBAR_WIDTH)

        # 9. Momentum arrow
        if config.DRAW_MOMENTUM_ARROW and current_hip and previous_hip:
            draw_momentum_arrow(canvas, current_hip, previous_hip, config, x_offset=config.SIDEBAR_WIDTH)

        # 10. Behind counter
        if behind_display_timer > 0 or behind_counter > 0:
            draw_behind_counter(canvas, behind_display_value, is_active=(behind_counter > 0),
                               config=config, x_offset=config.SIDEBAR_WIDTH)

        # 11. Unified tracking indicator
        if config.UNIFIED_TRACKING_ENABLED:
            is_active_state = unified_tracking_state in (
                BallTrackingState.EDGE_LEFT, BallTrackingState.EDGE_RIGHT,
                BallTrackingState.OFF_SCREEN_LEFT, BallTrackingState.OFF_SCREEN_RIGHT,
                BallTrackingState.DISAPPEARED_MID
            )
            is_persisting = unified_tracking_state == BallTrackingState.NORMAL and unified_persist_timer > 0

            if is_active_state or is_persisting:
                draw_unified_tracking_indicator(
                    canvas,
                    state=unified_tracking_state,
                    counter=unified_counter if is_active_state else unified_persist_value,
                    is_persisting=is_persisting,
                    persist_side=unified_persist_side,
                    config=config,
                    x_offset=config.SIDEBAR_WIDTH
                )

        # 12. Intention behind counter
        if intention_behind_display_timer > 0 or intention_behind_counter > 0:
            draw_intention_behind_counter(canvas, intention_behind_display_value,
                                         is_active=(intention_behind_counter > 0),
                                         config=config, x_offset=config.SIDEBAR_WIDTH)

        # 13. Vertical deviation counter
        if config.DETECT_BALL_VERTICAL_DEVIATION:
            if vertical_deviation_display_timer > 0 or vertical_deviation_counter > 0:
                draw_vertical_deviation_counter(
                    canvas, vertical_deviation_display_value,
                    is_active=(vertical_deviation_counter > 0),
                    direction=vertical_deviation_direction,
                    config=config, x_offset=config.SIDEBAR_WIDTH
                )

        out.write(canvas)

    cap.release()
    out.release()

    print(f"  Saved (mp4v): {output_path}")

    # Convert to H.264
    h264_path = convert_to_h264(output_path)
    if h264_path:
        return True
    else:
        print(f"  Warning: H.264 conversion failed, keeping mp4v version")
        return True


def draw_sidebar_generic(
    canvas: np.ndarray,
    frame_id: int,
    detected_cones: List[DetectedCone],
    ball_center: Optional[tuple],
    pose_keypoints: dict,
    config: TripleConeAnnotationConfig,
    active_zone: Optional[str],
    ball_position_result,
    intention_result=None,
    turn_events=None
) -> None:
    """
    Draw sidebar with dynamic cone list.

    Adapts the standard sidebar to show N cones instead of hardcoded 3.
    """
    # Draw sidebar background
    cv2.rectangle(canvas, (0, 0), (config.SIDEBAR_WIDTH, canvas.shape[0]),
                  config.SIDEBAR_BG_COLOR, -1)

    y = config.SIDEBAR_PADDING
    line_height = config.SIDEBAR_LINE_HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.SIDEBAR_FONT_SCALE

    # Frame info header
    cv2.rectangle(canvas, (0, y - 5), (config.SIDEBAR_WIDTH, y + line_height + 5),
                  config.SIDEBAR_HEADER_COLOR, -1)
    cv2.putText(canvas, f"Frame: {frame_id}", (config.SIDEBAR_PADDING, y + line_height - 5),
                font, font_scale, config.TEXT_COLOR, 1)
    y += line_height + 15

    # Cones section
    cv2.putText(canvas, f"Cones ({len(detected_cones)}):", (config.SIDEBAR_PADDING, y + line_height - 5),
                font, font_scale, config.TEXT_COLOR, 1)
    y += line_height + 5

    # Show cones (limit to 5, with ellipsis if more)
    max_display = 5
    cones_to_show = detected_cones[:max_display] if len(detected_cones) > max_display else detected_cones

    for i, cone in enumerate(cones_to_show):
        x, cy = cone.position
        color = CONE_COLOR_PALETTE[i % len(CONE_COLOR_PALETTE)]
        label = cone.definition.label
        # Truncate label if too long
        if len(label) > 12:
            label = label[:10] + ".."
        text = f"  {i}: {label} ({x:.0f})"
        cv2.putText(canvas, text, (config.SIDEBAR_PADDING, y + line_height - 5),
                    font, font_scale * 0.9, color, 1)
        y += line_height

    if len(detected_cones) > max_display:
        remaining = len(detected_cones) - max_display
        cv2.putText(canvas, f"  ... ({remaining} more)", (config.SIDEBAR_PADDING, y + line_height - 5),
                    font, font_scale * 0.9, (150, 150, 150), 1)
        y += line_height

    y += 10

    # Ball section
    cv2.putText(canvas, "Ball:", (config.SIDEBAR_PADDING, y + line_height - 5),
                font, font_scale, config.TEXT_COLOR, 1)
    y += line_height

    if ball_center and not any(pd.isna(v) for v in ball_center):
        text = f"  ({ball_center[0]:.0f}, {ball_center[1]:.0f})"
        cv2.putText(canvas, text, (config.SIDEBAR_PADDING, y + line_height - 5),
                    font, font_scale * 0.9, config.BALL_COLOR, 1)
    else:
        cv2.putText(canvas, "  (not detected)", (config.SIDEBAR_PADDING, y + line_height - 5),
                    font, font_scale * 0.9, (100, 100, 100), 1)
    y += line_height + 5

    # Active zone
    if active_zone:
        cv2.putText(canvas, f"Zone: {active_zone}", (config.SIDEBAR_PADDING, y + line_height - 5),
                    font, font_scale, (0, 255, 255), 1)
        y += line_height + 10

    # Turn events
    if turn_events and config.DRAW_EVENT_LOG:
        cv2.putText(canvas, "Events:", (config.SIDEBAR_PADDING, y + line_height - 5),
                    font, font_scale, config.TEXT_COLOR, 1)
        y += line_height + 5

        for event in turn_events[-5:]:  # Show last 5 events
            text = f"  {event}"
            if len(text) > 30:
                text = text[:28] + ".."
            cv2.putText(canvas, text, (config.SIDEBAR_PADDING, y + line_height - 5),
                        font, font_scale * 0.8, (180, 180, 180), 1)
            y += line_height


def main():
    parser = argparse.ArgumentParser(
        description="Annotate drill videos with debug visualization (supports N cones)"
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        help="Path to parquet data directory or video name"
    )
    parser.add_argument(
        "--drill-type", "-d",
        help="Explicit drill type (e.g., triple_cone, seven_cone_weave)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available videos"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Process ALL available videos"
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(__file__).parent.parent / "videos",
        help="Directory containing source videos"
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=Path(__file__).parent.parent / "video_detection_pose_ball_cones",
        help="Directory containing parquet data folders"
    )
    parser.add_argument(
        "--resolution", "-r",
        choices=["original", "720p"],
        default="720p",
        help="Which resolution to process (720p uses downscaled videos/parquets)"
    )
    parser.add_argument(
        "--drills-dir",
        type=Path,
        help="Directory containing drill type folders (e.g., drills/)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process videos even if output already exists"
    )

    args = parser.parse_args()

    # Initialize drill config loader
    loader = DrillConfigLoader()

    # Override directories for 720p resolution
    if args.resolution == "720p":
        base_dir = Path(__file__).parent.parent
        args.videos_dir = base_dir / "videos_720p"
        args.parquet_dir = base_dir / "video_detection_pose_ball_cones_720p"
        print(f"\n[720p MODE] Using downscaled videos and parquets\n")

    if args.list:
        if args.drills_dir:
            # New drills/ folder mode
            drill_folders = get_drills_structure(args.drills_dir, loader)
            print(f"\nAvailable drills in {args.drills_dir}/:\n")
            total_players = 0
            for drill in drill_folders:
                print(f"{drill.drill_path.name}/ ({drill.drill_name})")
                for player in drill.players:
                    status = "[done]" if player.has_output else "[    ]"
                    print(f"  {status} {player.name}")
                    total_players += 1
                print()
            print(f"Total: {total_players} players across {len(drill_folders)} drill types")
        else:
            # Legacy mode
            available = get_available_videos(args.videos_dir, args.parquet_dir)
            print("\n Available videos:\n")
            for name, video_path, parquet_path in available:
                drill_id = loader.detect_drill_type_from_path(str(parquet_path))
                drill_label = f"[{drill_id}]" if drill_id else "[unknown]"
                output_path = parquet_path / f"{name}_annotated.mp4"
                status = "[done]" if output_path.exists() else "[    ]"
                print(f"  {status} {name} {drill_label}")
            print(f"\nTotal: {len(available)} videos")
        return 0

    if not args.data_path and not args.all and not args.drills_dir:
        print("Error: Please specify a data path, use --list, use --all, or use --drills-dir")
        return 1

    if args.all:
        if args.drills_dir:
            # New drills/ folder mode - process all drills, all players
            drill_folders = get_drills_structure(args.drills_dir, loader)
            total_players = sum(len(d.players) for d in drill_folders)
            print(f"\n Processing ALL {total_players} players across {len(drill_folders)} drills...\n")

            success_count = 0
            fail_count = 0
            skip_count = 0
            player_num = 0

            for drill in drill_folders:
                drill_config = None
                if drill.drill_type != "unknown":
                    try:
                        drill_config = loader.get_drill_type(drill.drill_type)
                    except ValueError:
                        pass

                for player in drill.players:
                    player_num += 1

                    if player.has_output and not args.force:
                        print(f"[{player_num}/{total_players}] Skipping {player.name} (already annotated)")
                        skip_count += 1
                        continue

                    print(f"\n{'='*60}")
                    print(f"[{player_num}/{total_players}] {drill.drill_path.name}/{player.name}")
                    print(f"{'='*60}")

                    if drill_config:
                        print(f"  Drill type: {drill_config.name}")

                    output_path = player.parquet_dir / f"{player.name}_annotated.mp4"
                    config = TripleConeAnnotationConfig()
                    success = annotate_video(player.video_path, player.parquet_dir, output_path, drill_config, config)

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1

            print(f"\n{'='*60}")
            print(f" BATCH COMPLETE")
            print(f"{'='*60}")
            print(f"  Processed: {success_count}")
            print(f"  Skipped:   {skip_count}")
            print(f"  Failed:    {fail_count}")

            return 0 if fail_count == 0 else 1
        else:
            # Legacy mode
            available = get_available_videos(args.videos_dir, args.parquet_dir)
            print(f"\n Processing ALL {len(available)} videos...\n")

            success_count = 0
            fail_count = 0

            for i, (name, video_path, parquet_path) in enumerate(available, 1):
                print(f"\n{'='*60}")
                print(f"[{i}/{len(available)}] Processing: {name}")
                print(f"{'='*60}")

                drill_id = args.drill_type or loader.detect_drill_type_from_path(str(parquet_path))
                drill_config = loader.get_drill_type(drill_id) if drill_id else None

                if drill_config:
                    print(f"  Drill type: {drill_config.name}")

                output_path = parquet_path / f"{name}_annotated.mp4"
                config = TripleConeAnnotationConfig()
                success = annotate_video(video_path, parquet_path, output_path, drill_config, config)

                if success:
                    success_count += 1
                else:
                    fail_count += 1

            print(f"\n{'='*60}")
            print(f" BATCH COMPLETE")
            print(f"{'='*60}")
            print(f"  Processed: {success_count}")
            print(f"  Failed:    {fail_count}")

            return 0 if fail_count == 0 else 1

    # Single video/folder processing
    if args.data_path:
        data_path = args.data_path

        # If data_path is a directory, use it directly
        if data_path.is_dir():
            parquet_path = data_path

            # Find video file in the directory
            video_files = list(parquet_path.glob("*.mp4")) + list(parquet_path.glob("*.MOV"))
            source_videos = [v for v in video_files if "_annotated" not in v.name.lower()]

            if not source_videos:
                # Try parent directory for video (legacy mode)
                parent = parquet_path.parent
                name = parquet_path.name.replace("_tc", "").replace("_triple", "")
                source_videos = list(parent.glob(f"*{name}*.mp4"))

            if not source_videos:
                # Try videos directory (legacy mode)
                source_videos = list(args.videos_dir.glob(f"*{parquet_path.name}*.mp4"))

            if not source_videos:
                print(f"Error: Cannot find video for {parquet_path}")
                return 1

            video_path = source_videos[0]
        else:
            # Search by name in legacy mode
            available = get_available_videos(args.videos_dir, args.parquet_dir)
            search_term = str(data_path).lower()
            match = None
            for name, video_path, parquet_path in available:
                if search_term in name.lower():
                    match = (name, video_path, parquet_path)
                    break

            if not match:
                print(f"Error: Video not found matching: {data_path}")
                print("Use --list to see available videos")
                return 1

            name, video_path, parquet_path = match

        # Auto-detect or use explicit drill type
        drill_id = args.drill_type or loader.detect_drill_type_from_path(str(parquet_path))
        drill_config = None
        if drill_id:
            try:
                drill_config = loader.get_drill_type(drill_id)
                print(f"\n Drill type: {drill_config.name}")
            except ValueError:
                print(f"\n Warning: Unknown drill type '{drill_id}', using auto-detection")

        print(f"\n Annotating {video_path.name}...\n")

        output_path = parquet_path / f"{parquet_path.name}_annotated.mp4"
        config = TripleConeAnnotationConfig()
        success = annotate_video(video_path, parquet_path, output_path, drill_config, config)

        if success:
            print(f"\n Done! Output: {output_path}")
            return 0
        else:
            print("\n Annotation failed!")
            return 1

    # No data_path but drills_dir specified without --all - show help
    if args.drills_dir and not args.all:
        print("Error: Use --all with --drills-dir to process all, or specify a player folder path")
        print("Example: python video/annotate_video.py drills/7_cone_weave/player_x/")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
