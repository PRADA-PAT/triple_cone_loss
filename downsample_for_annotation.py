#!/usr/bin/env python3
"""
Downsample Triple Cone videos and parquets from 2816x1584 to 720p (1280x720).

Creates separate output folders to preserve original data:
- videos_720p/
- video_detection_pose_ball_cones_720p/

Usage:
    python downsample_for_annotation.py           # Process all 27 players
    python downsample_for_annotation.py --player "Alex mochar"  # Single player
    python downsample_for_annotation.py --list    # List available players
    python downsample_for_annotation.py --dry-run # Preview without processing
    python downsample_for_annotation.py --skip-existing  # Skip already processed
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Source resolution (from ffprobe of original videos)
SOURCE_WIDTH = 2816
SOURCE_HEIGHT = 1584

# Target resolution (720p)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Calculate exact scale factor
SCALE_FACTOR = TARGET_WIDTH / SOURCE_WIDTH  # 0.454545...

# Verify aspect ratio match (both should equal SCALE_FACTOR)
assert abs(TARGET_HEIGHT / SOURCE_HEIGHT - SCALE_FACTOR) < 0.001, \
    f"Aspect ratio mismatch: width scale {SCALE_FACTOR:.6f} != height scale {TARGET_HEIGHT / SOURCE_HEIGHT:.6f}"

# File naming pattern
FILE_PREFIX = "Drill_1_Triple cone Turn _dubaiacademy_"

# Columns to scale in each parquet type (PIXEL coordinates only)
PIXEL_COLUMNS = {
    'football': ['x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'width', 'height'],
    'cone': ['x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 'width', 'height'],
    'pose': ['x', 'y'],
}

# Columns to preserve unchanged (world coordinates in meters)
# These are NOT scaled because they represent real-world measurements
WORLD_COLUMNS = [
    'field_center_x', 'field_center_y', 'field_z',
    'field_x', 'field_y', 'depth_m'
]


# =============================================================================
# VIDEO DOWNSCALING
# =============================================================================

def downscale_video(
    input_path: Path,
    output_path: Path,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
) -> Tuple[bool, str]:
    """
    Downscale video using ffmpeg with optimized settings.

    FFmpeg command breakdown:
    - -hwaccel videotoolbox: Use macOS hardware acceleration for decoding
    - -i input: Input file
    - -vf scale=W:H: Video filter for scaling
    - -c:v libx264: H.264 codec for wide compatibility
    - -preset fast: Good speed/compression balance
    - -crf 23: Quality setting (18-28 range, lower=better)
    - -pix_fmt yuv420p: Standard pixel format for compatibility
    - -movflags +faststart: Move moov atom for web streaming
    - -an: No audio (drill videos typically have no useful audio)

    Returns:
        (success: bool, message: str)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'ffmpeg', '-y',
        '-hide_banner', '-loglevel', 'warning',
        '-hwaccel', 'videotoolbox',  # macOS GPU acceleration
        '-i', str(input_path),
        '-vf', f'scale={target_width}:{target_height}',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-an',  # No audio
        str(output_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout per video
        )

        if result.returncode == 0:
            # Get output file size
            size_mb = output_path.stat().st_size / (1024 * 1024)
            return True, f"OK ({size_mb:.1f} MB)"
        else:
            return False, f"FFmpeg error: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        return False, "Timeout (>5 min)"
    except FileNotFoundError:
        return False, "FFmpeg not found - install with: brew install ffmpeg"


# =============================================================================
# PARQUET SCALING
# =============================================================================

def scale_parquet(
    input_path: Path,
    output_path: Path,
    parquet_type: str,  # 'football', 'cone', or 'pose'
    scale_factor: float = SCALE_FACTOR,
) -> Tuple[bool, str]:
    """
    Scale pixel coordinate columns in parquet file.

    Preserves:
    - All world/field coordinates (in meters)
    - All non-coordinate columns (confidence, class_id, etc.)
    - Original dtypes

    Returns:
        (success: bool, message: str)
    """
    try:
        df = pd.read_parquet(input_path)
        original_rows = len(df)

        # Get columns to scale for this parquet type
        columns_to_scale = PIXEL_COLUMNS.get(parquet_type, [])

        # Scale each pixel coordinate column
        scaled_count = 0
        for col in columns_to_scale:
            if col in df.columns:
                df[col] = df[col] * scale_factor
                scaled_count += 1

        # Save to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        return True, f"OK ({original_rows} rows, {scaled_count} cols scaled)"

    except Exception as e:
        return False, f"Error: {str(e)[:100]}"


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def get_player_list(base_dir: Path) -> List[str]:
    """Get list of player names from parquet directory."""
    parquet_base = base_dir / "video_detection_pose_ball_cones"

    players = []
    for folder in sorted(parquet_base.iterdir()):
        if folder.is_dir() and folder.name.startswith(FILE_PREFIX):
            player_name = folder.name[len(FILE_PREFIX):]
            players.append(player_name)

    return players


def process_player(
    base_dir: Path,
    player_name: str,
    output_videos_dir: Path,
    output_parquets_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> Tuple[int, int]:
    """
    Process a single player's video and parquets.

    Returns:
        (successes, failures)
    """
    full_name = f"{FILE_PREFIX}{player_name}"

    successes = 0
    failures = 0

    # === Video ===
    video_input = base_dir / "videos" / f"{full_name}.MOV"
    video_output = output_videos_dir / f"{full_name}.mp4"  # Convert to .mp4

    if video_input.exists():
        if skip_existing and video_output.exists():
            tqdm.write(f"    [SKIP] Video exists: {video_output.name}")
            successes += 1
        elif dry_run:
            tqdm.write(f"    [DRY-RUN] Video: {video_input.name} -> {video_output.name}")
            successes += 1
        else:
            tqdm.write(f"    Video: {video_input.name}...")
            success, msg = downscale_video(video_input, video_output)
            if success:
                tqdm.write(f"      {msg}")
                successes += 1
            else:
                tqdm.write(f"      FAILED: {msg}")
                failures += 1
    else:
        tqdm.write(f"    [WARN] Video not found: {video_input.name}")
        failures += 1

    # === Parquets ===
    parquet_input_dir = base_dir / "video_detection_pose_ball_cones" / full_name
    parquet_output_dir = output_parquets_dir / full_name

    if parquet_input_dir.exists():
        for parquet_file in sorted(parquet_input_dir.glob("*.parquet")):
            # Determine parquet type from filename
            if "_football.parquet" in parquet_file.name:
                ptype = "football"
            elif "_cone.parquet" in parquet_file.name:
                ptype = "cone"
            elif "_pose.parquet" in parquet_file.name:
                ptype = "pose"
            else:
                tqdm.write(f"    [WARN] Unknown parquet type: {parquet_file.name}")
                continue

            output_path = parquet_output_dir / parquet_file.name

            if skip_existing and output_path.exists():
                tqdm.write(f"    [SKIP] Parquet exists: {parquet_file.name}")
                successes += 1
            elif dry_run:
                tqdm.write(f"    [DRY-RUN] Parquet ({ptype}): {parquet_file.name}")
                successes += 1
            else:
                success, msg = scale_parquet(parquet_file, output_path, ptype)
                if success:
                    tqdm.write(f"    Parquet ({ptype}): {msg}")
                    successes += 1
                else:
                    tqdm.write(f"    Parquet FAILED ({parquet_file.name}): {msg}")
                    failures += 1
    else:
        tqdm.write(f"    [WARN] Parquet dir not found: {parquet_input_dir}")
        failures += 1

    return successes, failures


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_player(
    base_dir: Path,
    player_name: str,
) -> bool:
    """
    Verify that downscaling was performed correctly for a player.

    Checks:
    1. Video exists and has correct resolution
    2. Parquet coordinates are scaled correctly
    """
    full_name = f"{FILE_PREFIX}{player_name}"

    # Check video
    video_720p = base_dir / "videos_720p" / f"{full_name}.mp4"
    if not video_720p.exists():
        print(f"  [FAIL] Video not found: {video_720p}")
        return False

    # Check video resolution via ffprobe
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        str(video_720p)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        dims = result.stdout.strip().split(',')
        if len(dims) == 2:
            w, h = int(dims[0]), int(dims[1])
            if w != TARGET_WIDTH or h != TARGET_HEIGHT:
                print(f"  [FAIL] Wrong resolution: {w}x{h} (expected {TARGET_WIDTH}x{TARGET_HEIGHT})")
                return False
    except Exception as e:
        print(f"  [WARN] Could not verify video resolution: {e}")

    # Check parquet scaling
    orig_dir = base_dir / "video_detection_pose_ball_cones" / full_name
    scaled_dir = base_dir / "video_detection_pose_ball_cones_720p" / full_name

    for ptype in ['football', 'cone', 'pose']:
        orig_file = list(orig_dir.glob(f"*_{ptype}.parquet"))
        scaled_file = list(scaled_dir.glob(f"*_{ptype}.parquet"))

        if not orig_file or not scaled_file:
            continue

        orig_df = pd.read_parquet(orig_file[0])
        scaled_df = pd.read_parquet(scaled_file[0])

        # Check a pixel coordinate column
        check_col = 'center_x' if ptype in ['football', 'cone'] else 'x'
        if check_col in orig_df.columns and check_col in scaled_df.columns:
            orig_val = orig_df[check_col].iloc[0]
            scaled_val = scaled_df[check_col].iloc[0]
            expected = orig_val * SCALE_FACTOR

            if abs(scaled_val - expected) > 0.01:
                print(f"  [FAIL] {ptype} scaling wrong: {scaled_val:.2f} != {expected:.2f}")
                return False

    print(f"  [OK] {player_name}")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Downsample Triple Cone videos and parquets to 720p",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python downsample_for_annotation.py --list           # List available players
  python downsample_for_annotation.py --dry-run        # Preview what will be processed
  python downsample_for_annotation.py -p "Alex mochar" # Process single player
  python downsample_for_annotation.py                  # Process all players
  python downsample_for_annotation.py --verify         # Verify all processed files
        """
    )
    parser.add_argument(
        "--player", "-p",
        help="Process single player (partial name match)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available players"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview without processing"
    )
    parser.add_argument(
        "--skip-existing", "-s",
        action="store_true",
        help="Skip files that already exist in output"
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify output files (resolution, scaling)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Base directory for triple_cone_loss"
    )

    args = parser.parse_args()
    base_dir = args.base_dir.resolve()

    # Output directories
    output_videos_dir = base_dir / "videos_720p"
    output_parquets_dir = base_dir / "video_detection_pose_ball_cones_720p"

    # Get player list
    players = get_player_list(base_dir)

    if not players:
        print(f"Error: No players found in {base_dir / 'video_detection_pose_ball_cones'}")
        return 1

    if args.list:
        print(f"\nAvailable players ({len(players)}):\n")
        for i, p in enumerate(players, 1):
            print(f"  {i:2d}. {p}")
        return 0

    # Filter to single player if specified
    if args.player:
        search = args.player.lower()
        players = [p for p in players if search in p.lower()]
        if not players:
            print(f"No player found matching: {args.player}")
            return 1

    # Verification mode
    if args.verify:
        print(f"\n{'='*60}")
        print(f"Verifying 720p output files")
        print(f"{'='*60}\n")

        all_ok = True
        for player in players:
            if not verify_player(base_dir, player):
                all_ok = False

        return 0 if all_ok else 1

    # Processing mode
    print(f"\n{'='*60}")
    print(f"Triple Cone Downscaling: {SOURCE_WIDTH}x{SOURCE_HEIGHT} -> {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print(f"Scale Factor: {SCALE_FACTOR:.6f}")
    print(f"{'='*60}")
    print(f"Base directory: {base_dir}")
    print(f"Players to process: {len(players)}")
    print(f"Output videos: {output_videos_dir}")
    print(f"Output parquets: {output_parquets_dir}")
    if args.dry_run:
        print("\n[DRY-RUN MODE - no files will be created]\n")
    if args.skip_existing:
        print("[SKIP-EXISTING MODE - will skip already processed files]\n")
    print()

    total_success = 0
    total_fail = 0

    for i, player in enumerate(tqdm(players, desc="Processing players", unit="player"), 1):
        tqdm.write(f"\n[{i}/{len(players)}] {player}")
        s, f = process_player(
            base_dir, player,
            output_videos_dir, output_parquets_dir,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
        total_success += s
        total_fail += f

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Successes: {total_success}")
    print(f"Failures:  {total_fail}")

    if not args.dry_run and total_fail == 0:
        print(f"\nTo verify output, run:")
        print(f"  python {Path(__file__).name} --verify")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
