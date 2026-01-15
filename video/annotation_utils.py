"""
Utility functions for video annotation.

Contains video discovery and codec conversion utilities.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple


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
