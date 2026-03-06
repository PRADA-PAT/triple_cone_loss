"""
Utility functions for video annotation.

Contains video discovery and codec conversion utilities.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from detection.drill_config_loader import DrillConfigLoader


@dataclass
class PlayerFolder:
    """Represents a player's data folder within a drill."""
    name: str
    video_path: Path
    parquet_dir: Path
    has_output: bool


@dataclass
class DrillFolder:
    """Represents a drill type folder containing player folders."""
    drill_type: str      # from config or "unknown"
    drill_name: str      # human readable name
    drill_path: Path
    players: List[PlayerFolder]


def convert_to_h264(input_path: Path) -> Optional[Path]:
    """
    Convert video to H.264 codec using ffmpeg.

    Args:
        input_path: Path to mp4v video file

    Returns:
        Path to converted H.264 file, or None if conversion failed
    """
    temp_path = input_path.parent / f"{input_path.stem}_h264_temp.mp4"

    print(f"  Converting to H.264 for compatibility...")

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(temp_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0 and temp_path.exists():
            backup_path = input_path.parent / f"{input_path.stem}_mp4v.mp4"
            input_path.rename(backup_path)
            temp_path.rename(input_path)
            print(f"  Converted to H.264: {input_path}")
            if backup_path.exists():
                backup_path.unlink()
            return input_path
        else:
            print(f"  FFmpeg error: {result.stderr}")
            if temp_path.exists():
                temp_path.unlink()
            return None

    except FileNotFoundError:
        print(f"  FFmpeg not found - keeping mp4v format")
        return None
    except subprocess.TimeoutExpired:
        print(f"  FFmpeg timeout (>5min) - keeping mp4v format")
        if temp_path.exists():
            temp_path.unlink()
        return None
    except Exception as e:
        print(f"  Conversion error: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return None


def get_available_videos(videos_dir: Path, parquet_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    Get list of videos with matching parquet data.

    Args:
        videos_dir: Directory containing source video files
        parquet_dir: Directory containing parquet data folders

    Returns:
        List of tuples: (name, video_path, parquet_path)
    """
    available = []

    for parquet_path in sorted(parquet_dir.iterdir()):
        if not parquet_path.is_dir():
            continue

        base_name = parquet_path.name

        # Check for required parquet files
        cone_files = list(parquet_path.glob("*_cone.parquet"))
        ball_files = list(parquet_path.glob("*_football.parquet"))
        pose_files = list(parquet_path.glob("*_pose.parquet"))

        if not (cone_files and ball_files and pose_files):
            continue

        # Look for matching video
        video_path = videos_dir / f"{base_name}.MOV"
        if not video_path.exists():
            video_path = videos_dir / f"{base_name}.mp4"
        if video_path.exists():
            available.append((base_name, video_path, parquet_path))

    return available


def resolve_parquet_dir(player_path: Path) -> Optional[Path]:
    """
    Find the directory containing parquet files for a player folder.

    Checks multiple locations:
    1. player_path/*_cone.parquet (legacy pattern, e.g. name_tc_cone.parquet)
    2. player_path/cone.parquet (short name in player dir)
    3. player_path/pipeline/cone.parquet (drill_data standard)

    Returns:
        Path to the directory containing parquets, or None if not found
    """
    # Legacy pattern: player_path/*_cone.parquet, *_football.parquet, *_pose.parquet
    if (list(player_path.glob("*_cone.parquet")) and
            list(player_path.glob("*_football.parquet")) and
            list(player_path.glob("*_pose.parquet"))):
        return player_path

    # Short name in player dir: cone.parquet, ball.parquet, pose.parquet
    if ((player_path / "cone.parquet").exists() and
            (player_path / "ball.parquet").exists() and
            (player_path / "pose.parquet").exists()):
        return player_path

    # Pipeline subfolder: pipeline/cone.parquet, pipeline/ball.parquet, pipeline/pose.parquet
    pipeline_dir = player_path / "pipeline"
    if (pipeline_dir.is_dir() and
            (pipeline_dir / "cone.parquet").exists() and
            (pipeline_dir / "ball.parquet").exists() and
            (pipeline_dir / "pose.parquet").exists()):
        return pipeline_dir

    return None


def resolve_output_path(player_path: Path, parquet_dir: Path) -> Path:
    """
    Determine the output path for the annotated debug video.

    If parquets are in a pipeline/ subfolder (drill_data convention),
    output to features/debug_video.mp4. Otherwise use legacy naming.

    Returns:
        Path for the output video file
    """
    if parquet_dir.name == "pipeline" and parquet_dir.parent == player_path:
        # drill_data convention: output to features/debug_video.mp4
        return player_path / "features" / "debug_video.mp4"
    else:
        # Legacy convention: output alongside parquets
        return parquet_dir / f"{player_path.name}_annotated.mp4"


def check_has_output(player_path: Path) -> bool:
    """Check if annotated output already exists (legacy or features/ convention)."""
    # Legacy: *_annotated.mp4 in player dir
    if list(player_path.glob("*_annotated.mp4")):
        return True
    # drill_data convention: features/debug_video.mp4
    if (player_path / "features" / "debug_video.mp4").exists():
        return True
    return False


def get_drills_structure(drills_dir: Path, loader: 'DrillConfigLoader') -> List[DrillFolder]:
    """
    Scan drills folder and return structure of drill types and players.

    Expected structure (supports both layouts):
        drills_dir/
          {drill_type_folder}/
            {player_folder}/
              *.mp4 or *.MOV (source video)
              *_cone.parquet OR pipeline/cone.parquet
              *_football.parquet OR pipeline/ball.parquet
              *_pose.parquet OR pipeline/pose.parquet

    Args:
        drills_dir: Root directory containing drill type folders
        loader: DrillConfigLoader for detecting drill types

    Returns:
        List of DrillFolder objects, each containing list of PlayerFolder objects
    """
    results = []

    if not drills_dir.exists():
        return results

    for drill_path in sorted(drills_dir.iterdir()):
        if not drill_path.is_dir() or drill_path.name.startswith('.'):
            continue

        # Detect drill type from folder name
        drill_id = loader.detect_drill_type_from_path(str(drill_path))
        if drill_id:
            try:
                config = loader.get_drill_type(drill_id)
                drill_name = config.name
            except ValueError:
                drill_id = "unknown"
                drill_name = f"{drill_path.name} (unknown)"
        else:
            drill_id = "unknown"
            drill_name = f"{drill_path.name} (unknown)"

        players = []
        for player_path in sorted(drill_path.iterdir()):
            if not player_path.is_dir() or player_path.name.startswith('.'):
                continue

            # Find source video (exclude annotated outputs and pipeline/)
            videos = list(player_path.glob("*.mp4")) + list(player_path.glob("*.MOV")) + list(player_path.glob("*.mov"))
            source_videos = [v for v in videos if "_annotated" not in v.name.lower()]
            if not source_videos:
                continue

            # Find parquet directory (supports legacy and pipeline/ layouts)
            parquet_dir = resolve_parquet_dir(player_path)
            if not parquet_dir:
                continue

            # Check for existing output
            has_output = check_has_output(player_path)

            players.append(PlayerFolder(
                name=player_path.name,
                video_path=source_videos[0],
                parquet_dir=parquet_dir,
                has_output=has_output
            ))

        if players:
            results.append(DrillFolder(
                drill_type=drill_id,
                drill_name=drill_name,
                drill_path=drill_path,
                players=players
            ))

    return results
