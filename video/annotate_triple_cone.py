#!/usr/bin/env python3
"""
Video Annotation for Triple Cone Drill Analysis.

Creates annotated videos using:
- STATIC cone positions from parquet detection (mean positions per player)
- DYNAMIC ball positions from parquet detection
- DYNAMIC pose skeleton from parquet detection
- LEFT SIDEBAR showing all object coordinates in real-time
- Turn detection and ball-behind tracking

Triple Cone Drill Pattern:
    CONE1 (HOME) -> CONE2 (turn) -> CONE1 (turn) -> CONE3 (turn) -> CONE1 (turn) -> repeat

Usage:
    python annotate_triple_cone.py "Alex mochar"
    python annotate_triple_cone.py --list
    python annotate_triple_cone.py --all
"""

import argparse
import sys
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

# Import from local modules - handle both package and direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

try:
    # When imported as part of a package (e.g., from video.annotate_triple_cone import ...)
    from .annotation_config import TripleConeAnnotationConfig, scale_config_for_resolution
    from .annotation_utils import convert_to_h264, get_available_videos
    from .annotation_data import (
        BallTrackingState,
        load_cone_positions_from_parquet,
        load_ball_data,
        load_pose_data,
        prepare_pose_lookup,
    )
    from .annotation_drawing import (
        draw_sidebar,
        draw_bbox,
        draw_triple_cone_markers,
        draw_skeleton,
        draw_momentum_arrow,
        draw_ball_momentum_arrow,
        calculate_ball_vertical_deviation,
        draw_intention_arrow,
        draw_debug_axes,
        draw_edge_zones,
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
    # When run directly as a script (python video/annotate_triple_cone.py)
    from annotation_config import TripleConeAnnotationConfig, scale_config_for_resolution
    from annotation_utils import convert_to_h264, get_available_videos
    from annotation_data import (
        BallTrackingState,
        load_cone_positions_from_parquet,
        load_ball_data,
        load_pose_data,
        prepare_pose_lookup,
    )
    from annotation_drawing import (
        draw_sidebar,
        draw_bbox,
        draw_triple_cone_markers,
        draw_skeleton,
        draw_momentum_arrow,
        draw_ball_momentum_arrow,
        calculate_ball_vertical_deviation,
        draw_intention_arrow,
        draw_debug_axes,
        draw_edge_zones,
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

# Turning zones from detection module
from detection.turning_zones import (
    TripleConeZoneConfig, create_triple_cone_zones, draw_triple_cone_zones,
)


def annotate_triple_cone_video(video_path: Path, parquet_dir: Path, output_path: Path,
                               config: TripleConeAnnotationConfig = None) -> bool:
    """
    Annotate Triple Cone drill video with cone positions, ball/pose tracking,
    and debug visualization.
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
        cone1, cone2, cone3 = load_cone_positions_from_parquet(cone_parquet[0])
    except Exception as e:
        print(f"  Error loading cones: {e}")
        return False

    print(f"    CONE1 (HOME): ({cone1.center_x:.0f}, {cone1.center_y:.0f}) [{cone1.width:.0f}x{cone1.height:.0f}]")
    print(f"    CONE2 (CENTER): ({cone2.center_x:.0f}, {cone2.center_y:.0f}) [{cone2.width:.0f}x{cone2.height:.0f}]")
    print(f"    CONE3 (RIGHT): ({cone3.center_x:.0f}, {cone3.center_y:.0f}) [{cone3.width:.0f}x{cone3.height:.0f}]")

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

    # Create turning zones
    print(f"  Creating turning zones...")
    zone_config = TripleConeZoneConfig.default()
    turning_zones = create_triple_cone_zones(cone1.center, cone2.center, cone3.center, zone_config)
    print(f"    Zone radius: {zone_config.cone1_zone_radius:.0f}px, stretch_y: {zone_config.stretch_y}")

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

    # Legacy edge counter
    edge_counter: int = 0
    edge_display_value: int = 0
    edge_display_timer: int = 0
    edge_last_side: str = "NONE"
    edge_persist_frames = int(config.EDGE_COUNTER_PERSIST_SECONDS * fps)

    # Legacy off-screen tracking
    off_screen_counter: int = 0
    return_display_value: int = 0
    return_display_timer: int = 0
    return_persist_frames = int(config.RETURN_COUNTER_PERSIST_SECONDS * fps)

    # Unified tracking state machine
    unified_tracking_state: BallTrackingState = BallTrackingState.NORMAL
    unified_counter: int = 0
    unified_persist_value: int = 0
    unified_persist_timer: int = 0
    unified_persist_side: str = "NONE"
    unified_persist_frames = int(config.UNIFIED_COUNTER_PERSIST_SECONDS * fps)

    # Vertical deviation tracking (ball going UP/DOWN instead of LEFT/RIGHT)
    vertical_deviation_counter: int = 0
    vertical_deviation_display_value: int = 0
    vertical_deviation_display_timer: int = 0
    vertical_deviation_direction: str = None  # "UP" or "DOWN"
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
            active_zone = turning_zones.get_zone_at_point(ball_center[0], ball_center[1])

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

        if config.UNIFIED_TRACKING_ENABLED:
            # Unified state machine
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
        else:
            # Legacy separate counters
            if edge_status.in_edge_zone:
                if edge_last_side != "NONE" and edge_last_side != edge_status.edge_side:
                    edge_counter = 1
                else:
                    edge_counter += 1
                edge_display_value = edge_counter
                edge_last_side = edge_status.edge_side
                edge_display_timer = edge_persist_frames
            else:
                if edge_counter > 0:
                    edge_display_timer = edge_persist_frames
                edge_counter = 0
                if edge_display_timer > 0:
                    edge_display_timer -= 1

            if ball_is_off_screen:
                off_screen_counter += 1
            else:
                if off_screen_counter > 0:
                    return_display_value = off_screen_counter
                    return_display_timer = return_persist_frames
                off_screen_counter = 0

            if return_display_timer > 0 and not ball_is_off_screen:
                return_display_timer -= 1

        # Vertical deviation detection (ball going UP/DOWN instead of LEFT/RIGHT)
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
                # Ball moving too slowly to detect deviation
                if vertical_deviation_counter > 0:
                    vertical_deviation_display_value = vertical_deviation_counter
                    vertical_deviation_display_timer = vertical_deviation_persist_frames
                vertical_deviation_counter = 0

        if vertical_deviation_display_timer > 0:
            vertical_deviation_display_timer -= 1

        # === DRAW EVERYTHING ===

        # 1. Turning zones
        draw_triple_cone_zones(
            canvas, turning_zones, ball_center,
            x_offset=config.SIDEBAR_WIDTH,
            cone1_color=config.CONE1_ZONE_COLOR,
            cone2_color=config.CONE2_ZONE_COLOR,
            cone3_color=config.CONE3_ZONE_COLOR,
            highlight_color=config.ZONE_HIGHLIGHT_COLOR,
            alpha=config.ZONE_ALPHA,
        )

        # 2. Edge zones
        canvas = draw_edge_zones(canvas, orig_width, orig_height, config, x_offset=config.SIDEBAR_WIDTH)

        # 3. Sidebar
        draw_sidebar(
            canvas, frame_id, cone1, cone2, cone3, ball_center, pose_keypoints,
            config, active_zone, ball_position_result,
            intention_result=intention_result,
            turn_events=turn_tracker.get_recent_events(config.EVENT_LOG_MAX_EVENTS)
        )

        # 4. Cone markers
        draw_triple_cone_markers(canvas, cone1, cone2, cone3, config, x_offset=config.SIDEBAR_WIDTH)

        # 5. Ball bbox
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

        # 11. Edge/Unified tracking indicator
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
        else:
            if edge_display_timer > 0 or edge_counter > 0:
                draw_edge_counter(canvas, edge_display_value, is_active=(edge_counter > 0),
                                 edge_side=edge_last_side, config=config, x_offset=config.SIDEBAR_WIDTH)

        # 12. Intention behind counter
        if intention_behind_display_timer > 0 or intention_behind_counter > 0:
            draw_intention_behind_counter(canvas, intention_behind_display_value,
                                         is_active=(intention_behind_counter > 0),
                                         config=config, x_offset=config.SIDEBAR_WIDTH)

        # 13. Off-screen indicator (legacy)
        if not config.UNIFIED_TRACKING_ENABLED:
            if config.DRAW_OFF_SCREEN_INDICATOR and ball_is_off_screen:
                draw_off_screen_indicator(canvas, config, x_offset=config.SIDEBAR_WIDTH)

        # 14. Return counter (legacy)
        if not config.UNIFIED_TRACKING_ENABLED:
            if return_display_timer > 0 and not ball_is_off_screen:
                draw_return_counter(canvas, return_display_value, config, x_offset=config.SIDEBAR_WIDTH)

        # 15. Vertical deviation counter
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


def main():
    parser = argparse.ArgumentParser(
        description="Annotate Triple Cone drill videos with debug visualization"
    )
    parser.add_argument(
        "video_name",
        nargs="?",
        help="Name of video to process (partial match supported)"
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
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have annotated output"
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

    args = parser.parse_args()

    # Override directories for 720p resolution
    if args.resolution == "720p":
        base_dir = Path(__file__).parent.parent
        args.videos_dir = base_dir / "videos_720p"
        args.parquet_dir = base_dir / "video_detection_pose_ball_cones_720p"
        print(f"\n[720p MODE] Using downscaled videos and parquets\n")

    available = get_available_videos(args.videos_dir, args.parquet_dir)

    if args.list:
        print("\n Available Triple Cone videos:\n")
        for name, video_path, parquet_path in available:
            output_path = parquet_path / f"{name}_triple_cone_test.mp4"
            status = "[done]" if output_path.exists() else "[    ]"
            print(f"  {status} {name}")
        print(f"\nTotal: {len(available)} videos")
        return 0

    if args.all:
        print(f"\n Processing ALL {len(available)} Triple Cone videos...\n")

        success_count = 0
        skip_count = 0
        fail_count = 0

        for i, (name, video_path, parquet_path) in enumerate(available, 1):
            output_path = parquet_path / f"{name}_triple_cone_test.mp4"

            if args.skip_existing and output_path.exists():
                print(f"[{i}/{len(available)}] Skipping {name} (already exists)")
                skip_count += 1
                continue

            print(f"\n{'='*60}")
            print(f"[{i}/{len(available)}] Processing: {name}")
            print(f"{'='*60}")

            config = TripleConeAnnotationConfig()
            success = annotate_triple_cone_video(video_path, parquet_path, output_path, config)

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

    if not args.video_name:
        print("Error: Please specify a video name, use --list, or use --all")
        return 1

    # Find matching video
    to_process = None
    search_term = args.video_name.lower()
    for name, video_path, parquet_path in available:
        name_lower = name.lower()
        if search_term in name_lower or search_term.replace(" ", "_") in name_lower or search_term.replace("_", " ") in name_lower:
            to_process = (name, video_path, parquet_path)
            break

    if not to_process:
        print(f"Error: Video not found matching: {args.video_name}")
        print("Use --list to see available videos")
        return 1

    name, video_path, parquet_path = to_process

    print(f"\n Annotating {name}...\n")

    output_path = parquet_path / f"{name}_triple_cone_test.mp4"
    config = TripleConeAnnotationConfig()
    success = annotate_triple_cone_video(video_path, parquet_path, output_path, config)

    if success:
        print(f"\n Done! Output: {output_path}")
        return 0
    else:
        print("\n Annotation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
