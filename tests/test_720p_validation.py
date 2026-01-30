"""Test suite to validate videos and parquets are 720p and in sync for overlay."""
import pytest
import subprocess
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq


# 720p dimensions
EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 720

# Allow small tolerance for off-screen keypoints (pose can have negative values)
COORD_TOLERANCE = 50

# Frame count sync tolerance (allow small mismatch)
FRAME_COUNT_TOLERANCE = 10

# Drills directory
DRILLS_DIR = Path(__file__).parent.parent / "drills"


def discover_drill_folders() -> list[Path]:
    """Discover all drill folders containing video and parquet files."""
    drill_folders = []

    if not DRILLS_DIR.exists():
        return drill_folders

    # Recursively find folders with both .mp4 and .parquet files
    for folder in DRILLS_DIR.rglob("*"):
        if not folder.is_dir():
            continue

        mp4_files = list(folder.glob("*.mp4"))
        parquet_files = list(folder.glob("*.parquet"))

        # Must have at least one video and one parquet (exclude annotated videos)
        source_videos = [f for f in mp4_files if "_annotated" not in f.name]
        if source_videos and parquet_files:
            drill_folders.append(folder)

    return sorted(drill_folders)


def get_video_info(video_path: Path) -> dict:
    """Get video resolution and frame count using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,codec_name",
        "-of", "json",
        str(video_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"error": result.stderr}

        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]

        # nb_frames might be missing for some codecs, try alternative
        nb_frames = stream.get("nb_frames")
        if nb_frames:
            nb_frames = int(nb_frames)
        else:
            # Fallback: count frames
            cmd_count = [
                "ffprobe", "-v", "error",
                "-count_frames",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "json",
                str(video_path)
            ]
            result_count = subprocess.run(cmd_count, capture_output=True, text=True, timeout=120)
            if result_count.returncode == 0:
                data_count = json.loads(result_count.stdout)
                nb_frames = int(data_count.get("streams", [{}])[0].get("nb_read_frames", 0))

        return {
            "width": stream.get("width"),
            "height": stream.get("height"),
            "frame_count": nb_frames,
            "codec": stream.get("codec_name"),
        }
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        return {"error": str(e)}


def get_parquet_coord_ranges(parquet_path: Path) -> dict:
    """Get coordinate ranges from a parquet file."""
    try:
        df = pq.read_table(parquet_path).to_pandas()
    except Exception as e:
        return {"error": str(e)}

    result = {"frame_count": 0, "columns": list(df.columns)}

    # Determine frame column
    if "frame_id" in df.columns:
        result["frame_count"] = int(df["frame_id"].max()) + 1
    elif "frame_idx" in df.columns:
        result["frame_count"] = int(df["frame_idx"].max()) + 1

    # Check X coordinate columns
    x_cols = [c for c in df.columns if c in ["x", "x1", "x2", "center_x"]]
    for col in x_cols:
        result[f"{col}_min"] = float(df[col].min())
        result[f"{col}_max"] = float(df[col].max())

    # Check Y coordinate columns
    y_cols = [c for c in df.columns if c in ["y", "y1", "y2", "center_y"]]
    for col in y_cols:
        result[f"{col}_min"] = float(df[col].min())
        result[f"{col}_max"] = float(df[col].max())

    return result


class DrillValidationResult:
    """Container for drill validation results."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.video_path: Optional[Path] = None
        self.video_info: dict = {}
        self.parquet_files: dict[str, dict] = {}  # filename -> coord ranges
        self.errors: list[str] = []

    def add_error(self, msg: str):
        self.errors.append(msg)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_drill_folder(folder: Path) -> DrillValidationResult:
    """Validate a single drill folder for 720p compliance."""
    result = DrillValidationResult(folder)

    # Find source video (exclude annotated)
    videos = [f for f in folder.glob("*.mp4") if "_annotated" not in f.name]
    if not videos:
        result.add_error("No source video found (only annotated videos present)")
        return result

    # Use first source video
    result.video_path = videos[0]
    result.video_info = get_video_info(result.video_path)

    if "error" in result.video_info:
        result.add_error(f"Video read error: {result.video_info['error']}")
        return result

    # Check video resolution
    width = result.video_info.get("width")
    height = result.video_info.get("height")

    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        result.add_error(
            f"Video resolution {width}x{height} != expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
        )

    video_frame_count = result.video_info.get("frame_count")

    # Validate each parquet file
    for parquet_path in folder.glob("*.parquet"):
        pq_name = parquet_path.name
        coord_info = get_parquet_coord_ranges(parquet_path)
        result.parquet_files[pq_name] = coord_info

        if "error" in coord_info:
            result.add_error(f"{pq_name}: read error - {coord_info['error']}")
            continue

        # Check X coordinate bounds
        for key, value in coord_info.items():
            if key.endswith("_min") and "x" in key.replace("_min", ""):
                col_name = key.replace("_min", "")
                min_val = value
                max_val = coord_info.get(f"{col_name}_max", 0)

                if min_val < -COORD_TOLERANCE:
                    result.add_error(
                        f"{pq_name}: {col_name} min={min_val:.1f} < -{COORD_TOLERANCE} (off-screen left)"
                    )
                if max_val > EXPECTED_WIDTH + COORD_TOLERANCE:
                    result.add_error(
                        f"{pq_name}: {col_name} max={max_val:.1f} > {EXPECTED_WIDTH + COORD_TOLERANCE} (off-screen right)"
                    )

            elif key.endswith("_min") and "y" in key.replace("_min", ""):
                col_name = key.replace("_min", "")
                min_val = value
                max_val = coord_info.get(f"{col_name}_max", 0)

                if min_val < -COORD_TOLERANCE:
                    result.add_error(
                        f"{pq_name}: {col_name} min={min_val:.1f} < -{COORD_TOLERANCE} (off-screen top)"
                    )
                if max_val > EXPECTED_HEIGHT + COORD_TOLERANCE:
                    result.add_error(
                        f"{pq_name}: {col_name} max={max_val:.1f} > {EXPECTED_HEIGHT + COORD_TOLERANCE} (off-screen bottom)"
                    )

        # Check frame count sync
        pq_frame_count = coord_info.get("frame_count", 0)
        if video_frame_count and pq_frame_count:
            diff = abs(video_frame_count - pq_frame_count)
            if diff > FRAME_COUNT_TOLERANCE:
                result.add_error(
                    f"{pq_name}: frame count {pq_frame_count} differs from video {video_frame_count} by {diff} frames"
                )

    return result


# Discover drill folders for parameterization
DRILL_FOLDERS = discover_drill_folders()


@pytest.mark.skipif(not DRILLS_DIR.exists(), reason="Drills directory not found")
@pytest.mark.skipif(len(DRILL_FOLDERS) == 0, reason="No drill folders found")
@pytest.mark.parametrize(
    "drill_folder",
    DRILL_FOLDERS,
    ids=[f.relative_to(DRILLS_DIR).as_posix() for f in DRILL_FOLDERS]
)
def test_drill_720p_compliance(drill_folder: Path):
    """Test that a drill folder's video and parquets are 720p compliant."""
    result = validate_drill_folder(drill_folder)

    if result.errors:
        error_msg = f"\n{drill_folder.name} validation failed:\n"
        for err in result.errors:
            error_msg += f"  - {err}\n"
        pytest.fail(error_msg)


@pytest.mark.skipif(not DRILLS_DIR.exists(), reason="Drills directory not found")
def test_drills_discovered():
    """Sanity check that drill folders were discovered."""
    folders = discover_drill_folders()
    assert len(folders) > 0, f"No drill folders found in {DRILLS_DIR}"
    print(f"\nDiscovered {len(folders)} drill folders:")
    for f in folders:
        print(f"  - {f.relative_to(DRILLS_DIR)}")


class TestVideoResolution:
    """Focused tests for video resolution validation."""

    @pytest.mark.skipif(len(DRILL_FOLDERS) == 0, reason="No drill folders found")
    def test_all_videos_are_720p(self):
        """Check all source videos are exactly 720p."""
        failures = []

        for folder in DRILL_FOLDERS:
            videos = [f for f in folder.glob("*.mp4") if "_annotated" not in f.name]
            for video in videos:
                info = get_video_info(video)
                if "error" in info:
                    failures.append(f"{video.name}: {info['error']}")
                elif info.get("width") != EXPECTED_WIDTH or info.get("height") != EXPECTED_HEIGHT:
                    failures.append(
                        f"{video.name}: {info.get('width')}x{info.get('height')}"
                    )

        if failures:
            pytest.fail(f"Videos not 720p:\n" + "\n".join(f"  - {f}" for f in failures))


class TestParquetCoordinates:
    """Focused tests for parquet coordinate validation."""

    @pytest.mark.skipif(len(DRILL_FOLDERS) == 0, reason="No drill folders found")
    def test_all_parquet_coords_within_720p_bounds(self):
        """Check all parquet coordinates are within 720p bounds."""
        failures = []

        for folder in DRILL_FOLDERS:
            for pq_path in folder.glob("*.parquet"):
                info = get_parquet_coord_ranges(pq_path)
                if "error" in info:
                    failures.append(f"{pq_path.name}: {info['error']}")
                    continue

                # Check each coordinate column
                for key, value in info.items():
                    if "_max" in key:
                        col = key.replace("_max", "")
                        if "x" in col and value > EXPECTED_WIDTH + COORD_TOLERANCE:
                            failures.append(f"{pq_path.name}: {col} max={value:.1f} > {EXPECTED_WIDTH}")
                        elif "y" in col and value > EXPECTED_HEIGHT + COORD_TOLERANCE:
                            failures.append(f"{pq_path.name}: {col} max={value:.1f} > {EXPECTED_HEIGHT}")

        if failures:
            pytest.fail(f"Parquet coords out of bounds:\n" + "\n".join(f"  - {f}" for f in failures))


if __name__ == "__main__":
    # Run discovery and show results
    print(f"Drills directory: {DRILLS_DIR}")
    print(f"Exists: {DRILLS_DIR.exists()}")

    folders = discover_drill_folders()
    print(f"\nDiscovered {len(folders)} drill folders:")

    for folder in folders:
        result = validate_drill_folder(folder)
        status = "PASS" if result.is_valid else "FAIL"
        print(f"\n[{status}] {folder.relative_to(DRILLS_DIR)}")

        if result.video_info:
            print(f"  Video: {result.video_info.get('width')}x{result.video_info.get('height')}, "
                  f"{result.video_info.get('frame_count')} frames")

        for pq_name, info in result.parquet_files.items():
            if "error" not in info:
                print(f"  {pq_name}: {info.get('frame_count', '?')} frames")

        for err in result.errors:
            print(f"  ERROR: {err}")
