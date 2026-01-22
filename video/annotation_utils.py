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


def get_drills_structure(drills_dir: Path, loader: 'DrillConfigLoader') -> List[DrillFolder]:
    """
    Scan drills folder and return structure of drill types and players.

    Expected structure:
        drills_dir/
          {drill_type_folder}/
            {player_folder}/
              *.mp4 (source video)
              *_cone.parquet
              *_football.parquet
              *_pose.parquet

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

            # Find source video (exclude annotated outputs)
            videos = list(player_path.glob("*.mp4")) + list(player_path.glob("*.MOV"))
            source_videos = [v for v in videos if "_annotated" not in v.name.lower()]
            if not source_videos:
                continue

            # Check for required parquet files
            has_cone = bool(list(player_path.glob("*_cone.parquet")))
            has_ball = bool(list(player_path.glob("*_football.parquet")))
            has_pose = bool(list(player_path.glob("*_pose.parquet")))

            if not (has_cone and has_ball and has_pose):
                continue

            # Check for existing output
            has_output = bool(list(player_path.glob("*_annotated.mp4")))

            players.append(PlayerFolder(
                name=player_path.name,
                video_path=source_videos[0],
                parquet_dir=player_path,
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
