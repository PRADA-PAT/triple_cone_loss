"""
Triple Cone Drill Ball Control Detection - Starter Script

Run for a single player:
    python run_detection.py alex_mochar

Run for all players:
    python run_detection.py --all

List available players:
    python run_detection.py --list

Run validation tests (compare detection vs ground truth):
    python run_detection.py --test
    python run_detection.py --test --frame-tolerance 30  # Custom tolerance (default: 45 frames)
    OR set TEST_MODE = True in the config section

Note: Frame tolerance is FPS-independent. Default 45 frames ≈ 1.5s at 30fps.
"""
import sys
from pathlib import Path

# Use local detection module (self-contained package)
from detection import (
    detect_ball_control,
    load_parquet_data,
    CSVExporter,
    AppConfig,
    TripleConeConeDetector,
    get_video_fps,
)

# Optional visualization (requires OpenCV)
try:
    from annotation.drill_visualizer import DrillVisualizer
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

# ============================================================
# CONFIGURATION - Edit these paths as needed
# ============================================================

# Base directories (using local folders - self-contained)
# Using 720p version by default for faster processing
VIDEO_DIR = Path(__file__).parent / "videos_720p"
OUTPUT_BASE = Path(__file__).parent / "output"

# Parquet data directory (local to this project)
# Structure: PARQUET_BASE / {folder_name} / {folder_name}_football.parquet, etc.
# Using 720p version by default for faster processing
PARQUET_BASE = Path(__file__).parent / "video_detection_pose_ball_cones_720p"

# All available players with their video files
# Triple Cone drill naming: "Drill_1_Triple cone Turn _dubaiacademy_{player_name}.MOV"
PLAYERS = {
    "alex_mochar": "Drill_1_Triple cone Turn _dubaiacademy_Alex mochar.MOV",
    "alfie_bates": "Drill_1_Triple cone Turn _dubaiacademy_Alfie Bates.MOV",
    "alonso_carrasco": "Drill_1_Triple cone Turn _dubaiacademy_Alonso Carrasco.MOV",
    "antonio_rodini_attoh": "Drill_1_Triple cone Turn _dubaiacademy_Antonio Rodini Attoh.MOV",
    "essa_ahmed": "Drill_1_Triple cone Turn _dubaiacademy_Essa Ahmed.MOV",
    "freddie_keilering": "Drill_1_Triple cone Turn _dubaiacademy_Freddie Keilering.MOV",
    "george_maclachlan": "Drill_1_Triple cone Turn _dubaiacademy_George Maclachlan.MOV",
    "isaac_barlow": "Drill_1_Triple cone Turn _dubaiacademy_Isaac Barlow.MOV",
    "jacobo_fernandez": "Drill_1_Triple cone Turn _dubaiacademy_Jacobo Fernandez.MOV",
    "jonty_robbins": "Drill_1_Triple cone Turn _dubaiacademy_Jonty Robbins.MOV",
    "kingston_robison": "Drill_1_Triple cone Turn _dubaiacademy_Kingston robison.MOV",
    "krill_prokuronov": "Drill_1_Triple cone Turn _dubaiacademy_Krill Prokuronov.MOV",
    "leon_nelson": "Drill_1_Triple cone Turn _dubaiacademy_Leon Nelson.MOV",
    "louis_hopps": "Drill_1_Triple cone Turn _dubaiacademy_Louis Hopps.MOV",
    "luca_focaccia": "Drill_1_Triple cone Turn _dubaiacademy_Luca Focaccia.MOV",
    "oliver_rourke": "Drill_1_Triple cone Turn _dubaiacademy_Oliver Rourke.MOV",
    "quin_rowe": "Drill_1_Triple cone Turn _dubaiacademy_Quin Rowe.MOV",
    "rachid_dittrich": "Drill_1_Triple cone Turn _dubaiacademy_Rachid Dittrich.MOV",
    "rayan_khadim": "Drill_1_Triple cone Turn _dubaiacademy_Rayan Khadim.MOV",
    "robin_staubach": "Drill_1_Triple cone Turn _dubaiacademy_Robin Staubach.MOV",
    "sanad_alkhresheh": "Drill_1_Triple cone Turn _dubaiacademy_Sanad Alkhresheh.MOV",
    "tiernan_cotter": "Drill_1_Triple cone Turn _dubaiacademy_Tiernan Cotter.MOV",
    "timofej_jeckl": "Drill_1_Triple cone Turn _dubaiacademy_Timofej Jeckl.MOV",
    "tise_ogundele": "Drill_1_Triple cone Turn _dubaiacademy_Tise Ogundele.MOV",
    "travis_muturi": "Drill_1_Triple cone Turn _dubaiacademy_Travis Muturi.MOV",
    "vincent_brady": "Drill_1_Triple cone Turn _dubaiacademy_Vincent Brady.MOV",
    "yassin_abdullah": "Drill_1_Triple cone Turn _dubaiacademy_Yassin Abdullah.MOV",
    # Loss videos (with ground truth annotations)
    "amal_mastan": "Amal Mastan_triple.mp4",
    "anaya_ahmed": "anaya ahmed_triple.mp4",
    "arsen_nurtayev": "Arsen Nurtayev_triple.mp4",
    "benjamin_orhan": "benjamin orhan_triple.mp4",
    "cadha_boyle": "Cadha Boyle_triple.mp4",
    "dylan_white": "Dylan White_triple.mp4",
    "fredrick_mccormack": "fredrick mccormack_triple.mp4",
    "hind_alassar": "hind Alassar _triple.mp4",
    "jabr_el_manssouri": "Jabr El Manssouri_triple.mp4",
    "jaeden_lindley": "jaeden lindley_triple.mp4",
    "julia_assy": "Julia Assy_triple.mp4",
    "lui_padraig_conway": "Lui Padraig Conway_triple.mp4",
    "olivier_fee": "olivier fee_triple.mp4",
    "omar_farag": "Omar Farag lw_triple.mp4",
    "raphael_samuel": "Raphael samuel_triple.mp4",
    "sarah_rashid": "sarah rashid_triple.mp4",
    "sebastian_bough": "sebastian bough_triple.mp4",
}

# Options
CREATE_VIDEO = False  # Set to True to create annotated video
DEFAULT_FPS = 30.0  # Fallback FPS if video cannot be read (actual FPS is auto-detected)
DETECTION_MODE = "standard"  # "standard", "strict", or "lenient"

# ============================================================
# TEST MODE CONFIGURATION
# ============================================================
TEST_MODE = False  # Set to True to run validation against ground truth
GROUND_TRUTH_CSV = Path(__file__).parent / "ground_truth.csv"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_results"
# Frame tolerance for matching detected events to ground truth.
# Default 45 frames (≈1.5s at 30fps) accounts for human annotation imprecision.
# Can be overridden with --frame-tolerance CLI argument.
TEST_FRAME_TOLERANCE = 45  # +/- frames for matching events

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_parquet_paths(player_name: str) -> dict:
    """Get parquet file paths for a player.

    Triple Cone naming convention:
    - Video: "Drill_1_Triple cone Turn _dubaiacademy_{Player Name}.MOV"
    - Folder: "Drill_1_Triple cone Turn _dubaiacademy_{Player Name}/"
    - Files: "{folder_name}_football.parquet", "_pose.parquet", "_cone.parquet"
    """
    # Get video filename from PLAYERS dict
    video_file = PLAYERS.get(player_name)
    if not video_file:
        # Fallback: construct from player_name
        player_dir = PARQUET_BASE / player_name
        return {
            "ball": player_dir / f"{player_name}_football.parquet",
            "pose": player_dir / f"{player_name}_pose.parquet",
            "cone": player_dir / f"{player_name}_cone.parquet",
            "dir": player_dir,
        }

    # Folder name = video filename without .MOV extension
    folder_name = video_file.replace(".MOV", "")
    player_dir = PARQUET_BASE / folder_name

    return {
        "ball": player_dir / f"{folder_name}_football.parquet",
        "pose": player_dir / f"{folder_name}_pose.parquet",
        "cone": player_dir / f"{folder_name}_cone.parquet",
        "dir": player_dir,
    }


def process_player(player_name: str) -> int:
    """Process a single player's Triple Cone drill."""
    print("\n" + "=" * 60)
    print(f"TRIPLE CONE DRILL: {player_name.upper()}")
    print("=" * 60)

    # Get paths
    video_file = PLAYERS.get(player_name)
    if not video_file:
        print(f"ERROR: Unknown player '{player_name}'")
        print("Use --list to see available players")
        return 1

    video_path = VIDEO_DIR / video_file
    parquet_paths = get_parquet_paths(player_name)
    output_dir = OUTPUT_BASE / player_name

    # Check files
    print("\nChecking files...")
    print(f"  Video: {video_path}")
    print(f"    Exists: {video_path.exists()}")

    missing_parquet = []
    for name, path in parquet_paths.items():
        exists = path.exists()
        print(f"  {name.capitalize()}: {path}")
        print(f"    Exists: {exists}")
        if not exists:
            missing_parquet.append(name)

    if missing_parquet:
        print(f"\nERROR: Missing parquet files: {missing_parquet}")
        print(f"Expected location: {PARQUET_BASE / player_name}/")
        print("\nRun object detection first to generate parquet files.")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("\n[1/5] Loading data...")
    ball_df = load_parquet_data(str(parquet_paths["ball"]))
    pose_df = load_parquet_data(str(parquet_paths["pose"]))
    cone_df = load_parquet_data(str(parquet_paths["cone"]))

    print(f"  Ball records: {len(ball_df)}")
    print(f"  Pose records: {len(pose_df)}")
    print(f"  Cone records: {len(cone_df)}")

    # Get actual FPS from video file (important: some videos are 25fps, others 30fps)
    if video_path.exists():
        fps = get_video_fps(str(video_path), default_fps=DEFAULT_FPS)
        print(f"  Video FPS: {fps:.2f} (from video file)")
    else:
        fps = DEFAULT_FPS
        print(f"  Video FPS: {fps:.2f} (default - video not found)")

    # 2. Create config
    print(f"\n[2/5] Setting up Triple Cone detection (mode: {DETECTION_MODE})...")
    if DETECTION_MODE == "strict":
        config = AppConfig.with_strict_detection()
    elif DETECTION_MODE == "lenient":
        config = AppConfig.with_lenient_detection()
    else:
        config = AppConfig.for_triple_cone()

    config.fps = fps

    # 3. Run detection (pass parquet dir for manual cone annotations)
    print("\n[3/5] Running detection...")
    result = detect_ball_control(
        ball_df, pose_df, cone_df,
        config=config, fps=fps,
        parquet_dir=str(parquet_paths["dir"]),
        video_path=str(video_path) if video_path.exists() else None
    )

    if not result.success:
        print(f"  ERROR: {result.error}")
        return 1

    print(f"  Frames processed: {result.total_frames}")
    print(f"  Laps completed: {result.total_laps}")

    # 4. Export CSVs
    print("\n[4/5] Exporting results...")
    exporter = CSVExporter()

    # Main exports
    events_path = output_dir / "loss_events.csv"
    frames_path = output_dir / "frame_analysis.csv"
    exporter.export_events(result, str(events_path))
    exporter.export_frame_analysis(result, str(frames_path))
    print(f"  Events: {events_path}")
    print(f"  Frames: {frames_path}")

    # 5. Create video (optional)
    if CREATE_VIDEO:
        if not HAS_VISUALIZER:
            print("\n[5/5] Skipping video (OpenCV not available)")
        elif not video_path.exists():
            print(f"\n[5/5] Skipping video (file not found: {video_path})")
        else:
            print("\n[5/5] Creating annotated video...")
            visualizer = DrillVisualizer(config.visualization)
            video_output = output_dir / "annotated_video.mp4"

            success = visualizer.create_annotated_video(
                str(video_path),
                str(video_output),
                result,
                cone_df,
                ball_df,
                pose_df
            )

            if success:
                print(f"  Video: {video_output}")
            else:
                print("  Video creation failed")
    else:
        print("\n[5/5] Skipping video (set CREATE_VIDEO=True to enable)")

    # Summary
    print("\n" + "-" * 60)
    print("TRIPLE CONE DRILL SUMMARY")
    print("-" * 60)
    print(f"Player: {player_name}")
    print(f"Detection mode: {DETECTION_MODE}")
    print()
    print("DRILL METRICS:")
    print(f"  Total frames: {result.total_frames}")
    print(f"  Laps completed: {result.total_laps}")
    print()
    print("BALL CONTROL:")
    print(f"  Loss events: {result.total_loss_events}")
    print(f"  Loss duration: {result.total_loss_duration_frames} frames")
    print(f"  Control %: {result.control_percentage:.1f}%")
    print()

    # Detailed loss events with timestamps and event types
    if result.events:
        # Event type names for display
        type_names = {
            "boundary": "OUT_OF_BOUNDS",
            "ball_behind": "BALL_BEHIND",
            "loss_distance": "DISTANCE",
            "loss_velocity": "VELOCITY",
        }

        print("LOSS EVENT DETAILS:")
        print("-" * 60)
        type_counts = {}
        for event in result.events:
            start_sec = event.start_timestamp
            end_sec = event.end_timestamp if event.end_timestamp else start_sec
            duration = event.duration_seconds
            gate_info = f" near {event.gate_context}" if event.gate_context else ""

            # Get event type name
            event_type_val = event.event_type.value if event.event_type else "unknown"
            event_type_name = type_names.get(event_type_val, event_type_val.upper())
            type_counts[event_type_name] = type_counts.get(event_type_name, 0) + 1

            print(f"  Event #{event.event_id}: {start_sec:.2f}s - {end_sec:.2f}s")
            print(f"    Type: {event_type_name}")
            print(f"    Duration: {duration:.2f}s | Severity: {event.severity}{gate_info}")

        # Event type summary
        if type_counts:
            print()
            print("EVENT TYPE SUMMARY:")
            for type_name, count in sorted(type_counts.items()):
                print(f"    {type_name}: {count}")
        print()
    else:
        print("LOSS EVENT DETAILS:")
        print("  No loss events detected - excellent ball control!")
        print()

    print(f"Output: {output_dir}/")
    print("-" * 60)

    return 0


def list_players():
    """List all available players."""
    print("\nAvailable players:")
    print("-" * 40)
    for name, video in sorted(PLAYERS.items()):
        video_exists = (VIDEO_DIR / video).exists()
        parquet_exists = all(p.exists() for p in get_parquet_paths(name).values())
        status = []
        if video_exists:
            status.append("video")
        if parquet_exists:
            status.append("parquet")
        status_str = f" [{', '.join(status)}]" if status else " [no data]"
        print(f"  {name}{status_str}")
    print()
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Parquet directory: {PARQUET_BASE}")


def run_test_mode(frame_tolerance: int = TEST_FRAME_TOLERANCE) -> int:
    """
    Run batch testing on all players with ground truth validation.

    Args:
        frame_tolerance: Frame tolerance for matching events (default 45 frames ≈ 1.5s at 30fps)

    Returns:
        0 on success, 1 on error
    """
    from testing import (
        load_ground_truth,
        create_test_result,
        generate_report,
        save_report,
        save_csv_results,
        TestResult,
        OverallTestSummary,
        DEFAULT_FRAME_TOLERANCE
    )
    from typing import Dict

    print("\n" + "=" * 60)
    print("TRIPLE CONE DETECTION - TEST MODE")
    print(f"Frame Tolerance: +/- {frame_tolerance} frames (~{frame_tolerance/30.0:.1f}s at 30fps)")
    print("=" * 60)

    # Load ground truth
    if not GROUND_TRUTH_CSV.exists():
        print(f"ERROR: Ground truth file not found: {GROUND_TRUTH_CSV}")
        print("Create ground_truth.csv with manual annotations first.")
        return 1

    ground_truth = load_ground_truth(str(GROUND_TRUTH_CSV))
    print(f"Loaded ground truth for {len(ground_truth)} players")

    # Track results
    results: Dict[str, TestResult] = {}
    videos_processed = 0
    videos_skipped = 0
    skipped_players = []

    # Process each player with ground truth
    players_to_test = [p for p in PLAYERS if p in ground_truth]
    players_without_gt = [p for p in PLAYERS if p not in ground_truth]

    if players_without_gt:
        print(f"\nNote: {len(players_without_gt)} players have no ground truth")
        print(f"  Missing: {', '.join(players_without_gt[:5])}{'...' if len(players_without_gt) > 5 else ''}")

    print(f"\nProcessing {len(players_to_test)} players with ground truth...")
    print("-" * 60)

    for player_name in players_to_test:
        print(f"\n[{videos_processed + videos_skipped + 1}/{len(players_to_test)}] {player_name}...")

        # Check if parquet data exists
        parquet_paths = get_parquet_paths(player_name)
        missing = [name for name, path in parquet_paths.items() if not path.exists()]

        if missing:
            print(f"  SKIPPED: Missing parquet files: {missing}")
            videos_skipped += 1
            skipped_players.append(player_name)
            continue

        # Load data
        try:
            ball_df = load_parquet_data(str(parquet_paths["ball"]))
            pose_df = load_parquet_data(str(parquet_paths["pose"]))
            cone_df = load_parquet_data(str(parquet_paths["cone"]))
        except Exception as e:
            print(f"  SKIPPED: Error loading data: {e}")
            videos_skipped += 1
            skipped_players.append(player_name)
            continue

        # Run detection
        if DETECTION_MODE == "strict":
            config = AppConfig.with_strict_detection()
        elif DETECTION_MODE == "lenient":
            config = AppConfig.with_lenient_detection()
        else:
            config = AppConfig.for_triple_cone()

        # Get video path for this player
        video_file = PLAYERS.get(player_name)
        video_path = VIDEO_DIR / video_file if video_file else None

        # Get actual FPS from video file
        if video_path and video_path.exists():
            fps = get_video_fps(str(video_path), default_fps=DEFAULT_FPS)
        else:
            fps = DEFAULT_FPS
        config.fps = fps

        result = detect_ball_control(
            ball_df, pose_df, cone_df,
            config=config, fps=fps,
            parquet_dir=str(parquet_paths["dir"]),
            video_path=str(video_path) if video_path and video_path.exists() else None
        )

        if not result.success:
            print(f"  SKIPPED: Detection failed: {result.error}")
            videos_skipped += 1
            skipped_players.append(player_name)
            continue

        # Compare with ground truth
        gt_events = ground_truth.get(player_name, [])
        test_result = create_test_result(
            player_name=player_name,
            detected_events=result.events,
            ground_truth_events=gt_events,
            frame_tolerance=frame_tolerance
        )

        results[player_name] = test_result
        videos_processed += 1

        # Brief output
        print(f"  GT: {test_result.ground_truth_count} | "
              f"Detected: {test_result.detected_count} | "
              f"TP: {len(test_result.true_positives)} | "
              f"FP: {len(test_result.false_positives)} | "
              f"FN: {len(test_result.false_negatives)}")

    # Calculate overall summary
    total_tp = sum(len(r.true_positives) for r in results.values())
    total_fp = sum(len(r.false_positives) for r in results.values())
    total_fn = sum(len(r.false_negatives) for r in results.values())
    total_gt = sum(r.ground_truth_count for r in results.values())
    total_det = sum(r.detected_count for r in results.values())

    # Overall metrics
    if total_tp + total_fp > 0:
        overall_precision = total_tp / (total_tp + total_fp)
    else:
        overall_precision = 1.0 if total_fn == 0 else 0.0

    if total_tp + total_fn > 0:
        overall_recall = total_tp / (total_tp + total_fn)
    else:
        overall_recall = 1.0 if total_fp == 0 else 0.0

    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    else:
        overall_f1 = 0.0

    summary = OverallTestSummary(
        total_videos=len(PLAYERS),
        videos_with_ground_truth=len(players_to_test),
        videos_processed=videos_processed,
        videos_skipped=videos_skipped,
        total_ground_truth_events=total_gt,
        total_detected_events=total_det,
        total_true_positives=total_tp,
        total_false_positives=total_fp,
        total_false_negatives=total_fn,
        overall_precision=overall_precision,
        overall_recall=overall_recall,
        overall_f1=overall_f1,
        per_player_results=results
    )

    # Save CSV results
    summary_csv, events_csv = save_csv_results(results, summary, TEST_OUTPUT_DIR)
    print(f"\n✓ Test results saved to CSV:")
    print(f"  Summary: {summary_csv}")
    print(f"  Events:  {events_csv}")

    # Generate and save verbose text report (includes detailed FP/FN info)
    report = generate_report(results, summary, frame_tolerance)
    report_path = save_report(report, TEST_OUTPUT_DIR, summary)
    print(f"\n✓ Detailed text report saved:")
    print(f"  Report:  {report_path}")
    print(f"  Latest:  {TEST_OUTPUT_DIR / 'LATEST_report.txt'}")

    # Generate and display brief summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Videos processed: {videos_processed}/{len(players_to_test)}")
    print(f"Ground truth events: {total_gt}")
    print(f"Detected events: {total_det}")
    print()
    print(f"True Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print()
    print(f"Precision: {overall_precision*100:.1f}%")
    print(f"Recall:    {overall_recall*100:.1f}%")
    print(f"F1 Score:  {overall_f1*100:.1f}%")
    print("=" * 60)
    print(f"\nFor detailed FP/FN breakdown, see: {TEST_OUTPUT_DIR / 'LATEST_report.txt'}")

    if skipped_players:
        print(f"\nSkipped players: {', '.join(skipped_players)}")

    return 0


def parse_frame_tolerance_arg() -> int:
    """Parse --frame-tolerance argument from command line."""
    frame_tolerance = TEST_FRAME_TOLERANCE
    for i, arg in enumerate(sys.argv):
        if arg == "--frame-tolerance" and i + 1 < len(sys.argv):
            try:
                frame_tolerance = int(sys.argv[i + 1])
                print(f"Using custom frame tolerance: {frame_tolerance} frames (~{frame_tolerance/30.0:.1f}s at 30fps)")
            except ValueError:
                print(f"Warning: Invalid frame tolerance value '{sys.argv[i + 1]}', using default {TEST_FRAME_TOLERANCE} frames")
    return frame_tolerance


def main():
    """Main entry point."""
    # Check for test mode first
    if TEST_MODE:
        frame_tolerance = parse_frame_tolerance_arg()
        return run_test_mode(frame_tolerance=frame_tolerance)

    # Check for --test command line argument
    if len(sys.argv) >= 2 and sys.argv[1] == "--test":
        frame_tolerance = parse_frame_tolerance_arg()
        return run_test_mode(frame_tolerance=frame_tolerance)

    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    arg = sys.argv[1]

    if arg == "--list":
        list_players()
        return 0

    if arg == "--all":
        print("Processing all players...")
        results = {}
        all_player_events = {}  # Store loss events for summary

        total_players = len(PLAYERS)
        for idx, player in enumerate(PLAYERS, 1):
            print(f"\n[{idx}/{total_players}] Processing {player}...", end=" ")
            try:
                # Get paths and check if data exists
                parquet_paths = get_parquet_paths(player)
                if not all(p.exists() for p in [parquet_paths["ball"], parquet_paths["pose"], parquet_paths["cone"]]):
                    print("SKIPPED (missing data)")
                    results[player] = "SKIPPED (missing data)"
                    continue

                # Load and process
                ball_df = load_parquet_data(str(parquet_paths["ball"]))
                pose_df = load_parquet_data(str(parquet_paths["pose"]))
                cone_df = load_parquet_data(str(parquet_paths["cone"]))

                if DETECTION_MODE == "strict":
                    config = AppConfig.with_strict_detection()
                elif DETECTION_MODE == "lenient":
                    config = AppConfig.with_lenient_detection()
                else:
                    config = AppConfig.for_triple_cone()

                # Get video path for this player
                video_file = PLAYERS.get(player)
                video_path = VIDEO_DIR / video_file if video_file else None

                # Get actual FPS from video file
                if video_path and video_path.exists():
                    fps = get_video_fps(str(video_path), default_fps=DEFAULT_FPS)
                else:
                    fps = DEFAULT_FPS
                config.fps = fps

                result = detect_ball_control(
                    ball_df, pose_df, cone_df,
                    config=config, fps=fps,
                    parquet_dir=str(parquet_paths["dir"]),
                    video_path=str(video_path) if video_path and video_path.exists() else None
                )

                if result.success:
                    num_events = len(result.events)
                    print(f"OK ({num_events} loss event{'s' if num_events != 1 else ''})")
                    results[player] = "SUCCESS"
                    all_player_events[player] = result.events

                    # Export CSVs
                    output_dir = OUTPUT_BASE / player
                    output_dir.mkdir(parents=True, exist_ok=True)
                    exporter = CSVExporter()
                    exporter.export_events(result, str(output_dir / "loss_events.csv"))
                    exporter.export_frame_analysis(result, str(output_dir / "frame_analysis.csv"))
                else:
                    print(f"FAILED: {result.error}")
                    results[player] = f"FAILED: {result.error}"
            except Exception as e:
                print(f"ERROR: {e}")
                results[player] = f"ERROR: {e}"

        # Print summary
        print("\n" + "=" * 70)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 70)

        # Status summary
        print("\nPROCESSING STATUS:")
        print("-" * 50)
        success_count = sum(1 for s in results.values() if s == "SUCCESS")
        failed_count = len(results) - success_count
        for player, status in results.items():
            status_icon = "✓" if status == "SUCCESS" else "✗"
            print(f"  {status_icon} {player}: {status}")
        print(f"\nTotal: {success_count} succeeded, {failed_count} failed/skipped")

        # Loss events summary with event type breakdown
        print("\n" + "=" * 70)
        print("LOSS OF CONTROL DETECTION SUMMARY")
        print("=" * 70)

        # Event type names for display
        type_names = {
            "boundary": "OUT_OF_BOUNDS",
            "ball_behind": "BALL_BEHIND",
            "loss_distance": "DISTANCE",
            "loss_velocity": "VELOCITY",
        }

        total_events = 0
        type_counts = {}

        for player, events in sorted(all_player_events.items()):
            total_events += len(events)
            print(f"\n{player.upper()}: {len(events)} loss event(s)")
            if events:
                print("-" * 60)
                for event in events:
                    start_sec = event.start_timestamp
                    end_sec = event.end_timestamp if event.end_timestamp else start_sec
                    duration = event.duration_seconds
                    gate_info = f" near {event.gate_context}" if event.gate_context else ""

                    # Get event type name
                    event_type_val = event.event_type.value if event.event_type else "unknown"
                    event_type_name = type_names.get(event_type_val, event_type_val.upper())

                    # Track type counts
                    type_counts[event_type_name] = type_counts.get(event_type_name, 0) + 1

                    print(f"  #{event.event_id}: {start_sec:.2f}s - {end_sec:.2f}s "
                          f"[{event_type_name}] (duration: {duration:.2f}s{gate_info})")
            else:
                print("  No loss events - excellent ball control!")

        print("\n" + "=" * 70)
        print("EVENT TYPE BREAKDOWN")
        print("=" * 70)
        if type_counts:
            for type_name, count in sorted(type_counts.items()):
                pct = (count / total_events * 100) if total_events > 0 else 0
                bar = "█" * int(pct / 5)  # Scale bar to 20 chars max
                print(f"  {type_name:15s}: {count:3d} ({pct:5.1f}%) {bar}")
        else:
            print("  No events detected")

        print("\n" + "-" * 70)
        print(f"TOTAL: {total_events} loss events across {len(all_player_events)} players")
        print("-" * 70)

        print(f"\n📁 Results saved to: {OUTPUT_BASE.resolve()}/")
        print("   Each player has: loss_events.csv, frame_analysis.csv")

        return 0

    # Single player
    return process_player(arg)


if __name__ == "__main__":
    sys.exit(main())
