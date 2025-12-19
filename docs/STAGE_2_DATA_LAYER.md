# STAGE 2: Data Layer

**Duration**: ~20 minutes
**Prerequisites**: Stage 1 complete (config.py, data_structures.py exist)
**Outcome**: Can load parquet files and extract ankle positions

---

## Project Context

**Project Path**: `/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/`

**Data Path**:
```
/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/
├── Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet
├── Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet
└── Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet
```

**Key Constants**:
```python
ANKLE_KEYPOINTS = ['left_ankle', 'right_ankle']
FPS = 30.0
```

---

## Prerequisites Check

Before starting Stage 2, verify Stage 1 is complete:

```bash
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"
python -c "from f8_loss import AppConfig, FrameData; print('Stage 1 OK')"
```

---

## Files to Create

```
f8_loss/
├── data_loader.py           # ← CREATE THIS
└── tests/
    └── test_data_loader.py  # ← CREATE THIS
```

---

## Parquet Schema Reference

### Cone Parquet (`_cone.parquet`)
| Column | Type | Description |
|--------|------|-------------|
| `frame_id` | int32 | Frame number (0-1508) |
| `object_id` | Int32 | Cone ID (1-7) |
| `center_x`, `center_y` | float32 | Bbox center (pixel) |
| `field_center_x`, `field_center_y` | float32 | Field coordinates |

### Football Parquet (`_football.parquet`)
| Column | Type | Description |
|--------|------|-------------|
| `frame_id` | int32 | Frame number |
| `center_x`, `center_y` | float32 | Ball center (pixel) |
| `field_center_x`, `field_center_y` | float32 | Field coordinates |

### Pose Parquet (`_pose.parquet`)
| Column | Type | Description |
|--------|------|-------------|
| `frame_idx` | int32 | Frame number |
| `timestamp` | float32 | Time in seconds |
| `keypoint_name` | category | e.g., "left_ankle", "right_ankle" |
| `x`, `y` | float32 | Pixel coordinates |
| `field_x`, `field_y` | float32 | Field coordinates |
| `confidence` | float32 | Keypoint confidence |

---

## 1. data_loader.py - Data Loading Module

```python
"""
Data loading module for Ball Control Detection System.

Handles loading parquet files and preprocessing for analysis.
Only uses ankle keypoints (left_ankle, right_ankle) for stability.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Only use ankles for ball-foot distance (more stable than toes)
ANKLE_KEYPOINTS = ['left_ankle', 'right_ankle']


def load_parquet_data(path: str) -> pd.DataFrame:
    """
    Load a parquet file with validation.

    Args:
        path: Path to parquet file

    Returns:
        DataFrame with loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Parquet file is empty: {path}")

    logger.info(f"Loaded {len(df)} records from {path.name}")
    return df


def load_all_data(
    cone_path: str,
    ball_path: str,
    pose_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three parquet files.

    Args:
        cone_path: Path to cone parquet
        ball_path: Path to football parquet
        pose_path: Path to pose parquet

    Returns:
        Tuple of (cone_df, ball_df, pose_df)
    """
    logger.info("Loading all parquet files...")

    cone_df = load_parquet_data(cone_path)
    ball_df = load_parquet_data(ball_path)
    pose_df = load_parquet_data(pose_path)

    logger.info(f"Loaded: {len(cone_df)} cones, {len(ball_df)} balls, {len(pose_df)} poses")

    return cone_df, ball_df, pose_df


def extract_ankle_positions(pose_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only ankle keypoints from pose data.

    Args:
        pose_df: Full pose DataFrame with all keypoints

    Returns:
        DataFrame with only ankle keypoints (left_ankle, right_ankle)
    """
    ankle_df = pose_df[pose_df['keypoint_name'].isin(ANKLE_KEYPOINTS)].copy()

    if ankle_df.empty:
        raise ValueError(
            f"No ankle keypoints found. Available keypoints: "
            f"{pose_df['keypoint_name'].unique().tolist()}"
        )

    logger.info(f"Extracted {len(ankle_df)} ankle records from {len(pose_df)} pose records")
    return ankle_df


def get_closest_ankle_per_frame(
    ankle_df: pd.DataFrame,
    ball_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Find the closest ankle to ball for each frame.

    For each frame, computes distance from ball to both ankles
    and returns the closest one.

    Args:
        ankle_df: DataFrame with ankle positions (from extract_ankle_positions)
        ball_df: DataFrame with ball positions

    Returns:
        DataFrame with one row per frame containing:
        - frame_id
        - ankle_x, ankle_y (pixel coordinates)
        - ankle_field_x, ankle_field_y (field coordinates)
        - closest_ankle (keypoint name)
        - ball_ankle_distance (field units)
    """
    results = []
    skipped_frames = 0

    for frame_id in ball_df['frame_id'].unique():
        # Get ball position for this frame
        ball_rows = ball_df[ball_df['frame_id'] == frame_id]
        if ball_rows.empty:
            skipped_frames += 1
            continue

        ball_row = ball_rows.iloc[0]
        ball_x = ball_row['field_center_x']
        ball_y = ball_row['field_center_y']

        # Get ankles for this frame
        # Note: pose uses 'frame_idx', ball uses 'frame_id'
        frame_ankles = ankle_df[ankle_df['frame_idx'] == frame_id]

        if frame_ankles.empty:
            skipped_frames += 1
            continue

        # Calculate distance to each ankle
        frame_ankles = frame_ankles.copy()
        frame_ankles['distance'] = np.sqrt(
            (frame_ankles['field_x'] - ball_x)**2 +
            (frame_ankles['field_y'] - ball_y)**2
        )

        # Get closest ankle
        closest_idx = frame_ankles['distance'].idxmin()
        closest = frame_ankles.loc[closest_idx]

        results.append({
            'frame_id': frame_id,
            'ankle_x': closest['x'],
            'ankle_y': closest['y'],
            'ankle_field_x': closest['field_x'],
            'ankle_field_y': closest['field_y'],
            'closest_ankle': closest['keypoint_name'],
            'ball_ankle_distance': closest['distance'],
        })

    if skipped_frames > 0:
        logger.warning(f"Skipped {skipped_frames} frames due to missing data")

    result_df = pd.DataFrame(results)
    logger.info(f"Computed closest ankle for {len(result_df)} frames")

    return result_df


def validate_data_alignment(
    cone_df: pd.DataFrame,
    ball_df: pd.DataFrame,
    pose_df: pd.DataFrame
) -> dict:
    """
    Validate that all data files are properly aligned.

    Args:
        cone_df, ball_df, pose_df: Loaded DataFrames

    Returns:
        Dictionary with validation statistics
    """
    stats = {}

    # Frame ranges
    cone_frames = set(cone_df['frame_id'].unique())
    ball_frames = set(ball_df['frame_id'].unique())
    pose_frames = set(pose_df['frame_idx'].unique())

    stats['cone_frame_range'] = (min(cone_frames), max(cone_frames))
    stats['ball_frame_range'] = (min(ball_frames), max(ball_frames))
    stats['pose_frame_range'] = (min(pose_frames), max(pose_frames))

    # Common frames
    common_frames = cone_frames & ball_frames & pose_frames
    stats['common_frames'] = len(common_frames)
    stats['total_unique_frames'] = len(cone_frames | ball_frames | pose_frames)

    # Coverage
    stats['coverage_pct'] = (
        len(common_frames) / stats['total_unique_frames'] * 100
        if stats['total_unique_frames'] > 0 else 0
    )

    # Record counts
    stats['cone_records'] = len(cone_df)
    stats['ball_records'] = len(ball_df)
    stats['pose_records'] = len(pose_df)

    logger.info(f"Data alignment: {stats['coverage_pct']:.1f}% coverage "
                f"({stats['common_frames']} common frames)")

    return stats


def get_frame_data(
    frame_id: int,
    ball_df: pd.DataFrame,
    ankle_df: pd.DataFrame,
    cone_df: pd.DataFrame
) -> Optional[dict]:
    """
    Get all data for a single frame.

    Args:
        frame_id: Frame number
        ball_df: Ball DataFrame
        ankle_df: Ankle DataFrame (filtered)
        cone_df: Cone DataFrame

    Returns:
        Dictionary with frame data or None if missing
    """
    # Ball
    ball_row = ball_df[ball_df['frame_id'] == frame_id]
    if ball_row.empty:
        return None
    ball_row = ball_row.iloc[0]

    # Ankles
    frame_ankles = ankle_df[ankle_df['frame_idx'] == frame_id]
    if frame_ankles.empty:
        return None

    # Calculate closest ankle
    ball_x = ball_row['field_center_x']
    ball_y = ball_row['field_center_y']

    frame_ankles = frame_ankles.copy()
    frame_ankles['distance'] = np.sqrt(
        (frame_ankles['field_x'] - ball_x)**2 +
        (frame_ankles['field_y'] - ball_y)**2
    )

    closest = frame_ankles.loc[frame_ankles['distance'].idxmin()]

    # Cones
    frame_cones = cone_df[cone_df['frame_id'] == frame_id]

    return {
        'frame_id': frame_id,
        'ball': {
            'x': ball_row['center_x'],
            'y': ball_row['center_y'],
            'field_x': ball_x,
            'field_y': ball_y,
        },
        'ankle': {
            'x': closest['x'],
            'y': closest['y'],
            'field_x': closest['field_x'],
            'field_y': closest['field_y'],
            'name': closest['keypoint_name'],
            'distance': closest['distance'],
        },
        'cones': frame_cones[['object_id', 'center_x', 'center_y',
                              'field_center_x', 'field_center_y']].to_dict('records'),
    }
```

---

## 2. tests/test_data_loader.py - Data Loader Tests

```python
"""Stage 2 Tests: Data Layer - Parquet loading and ankle extraction."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from f8_loss.data_loader import (
    load_parquet_data,
    load_all_data,
    extract_ankle_positions,
    get_closest_ankle_per_frame,
    validate_data_alignment,
    ANKLE_KEYPOINTS,
)

# Real data paths for integration tests
DATA_DIR = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar")
CONE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet"
BALL_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet"
POSE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet"


class TestLoadParquetData:
    """Tests for load_parquet_data function."""

    def test_load_missing_file_raises(self):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_parquet_data("/nonexistent/path.parquet")

    def test_load_valid_parquet(self, tmp_path):
        """Test loading a valid parquet file."""
        df = pd.DataFrame({'frame_id': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        result = load_parquet_data(str(path))
        assert len(result) == 3
        assert 'frame_id' in result.columns

    @pytest.mark.skipif(not CONE_PATH.exists(), reason="Real data not available")
    def test_load_real_cone_data(self):
        """Test loading real cone parquet."""
        df = load_parquet_data(str(CONE_PATH))
        assert len(df) > 0
        assert 'frame_id' in df.columns
        assert 'object_id' in df.columns
        assert 'field_center_x' in df.columns


class TestExtractAnklePositions:
    """Tests for extract_ankle_positions function."""

    def test_extract_ankles_only(self):
        """Test that only ankle keypoints are extracted."""
        pose_df = pd.DataFrame({
            'frame_idx': [0, 0, 0, 0, 1, 1, 1, 1],
            'keypoint_name': [
                'left_ankle', 'right_ankle', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle', 'left_knee', 'right_knee',
            ],
            'x': [100, 110, 90, 95, 105, 115, 92, 97],
            'y': [200, 210, 180, 185, 205, 215, 182, 187],
            'field_x': [50, 55, 45, 47, 52, 57, 46, 48],
            'field_y': [100, 105, 90, 92, 102, 107, 91, 93],
        })

        result = extract_ankle_positions(pose_df)

        assert len(result) == 4  # 2 ankles per frame x 2 frames
        assert set(result['keypoint_name'].unique()) == {'left_ankle', 'right_ankle'}

    def test_extract_raises_if_no_ankles(self):
        """Test that ValueError is raised if no ankles found."""
        pose_df = pd.DataFrame({
            'frame_idx': [0, 0],
            'keypoint_name': ['left_knee', 'right_knee'],
            'x': [100, 110],
            'y': [200, 210],
            'field_x': [50, 55],
            'field_y': [100, 105],
        })

        with pytest.raises(ValueError, match="No ankle keypoints found"):
            extract_ankle_positions(pose_df)

    @pytest.mark.skipif(not POSE_PATH.exists(), reason="Real data not available")
    def test_extract_ankles_from_real_data(self):
        """Test extracting ankles from real pose data."""
        pose_df = load_parquet_data(str(POSE_PATH))
        ankle_df = extract_ankle_positions(pose_df)

        assert len(ankle_df) > 0
        assert all(kp in ANKLE_KEYPOINTS for kp in ankle_df['keypoint_name'].unique())


class TestGetClosestAnklePerFrame:
    """Tests for get_closest_ankle_per_frame function."""

    def test_finds_closest_ankle(self):
        """Test that closest ankle is correctly identified."""
        ankle_df = pd.DataFrame({
            'frame_idx': [0, 0, 1, 1],
            'keypoint_name': ['left_ankle', 'right_ankle', 'left_ankle', 'right_ankle'],
            'x': [100, 200, 100, 200],
            'y': [100, 100, 100, 100],
            'field_x': [10.0, 20.0, 10.0, 20.0],
            'field_y': [10.0, 10.0, 10.0, 10.0],
        })

        ball_df = pd.DataFrame({
            'frame_id': [0, 1],
            'field_center_x': [12.0, 18.0],  # Frame 0: closer to left, Frame 1: closer to right
            'field_center_y': [10.0, 10.0],
        })

        result = get_closest_ankle_per_frame(ankle_df, ball_df)

        assert len(result) == 2
        assert result[result['frame_id'] == 0]['closest_ankle'].values[0] == 'left_ankle'
        assert result[result['frame_id'] == 1]['closest_ankle'].values[0] == 'right_ankle'

    def test_returns_correct_columns(self):
        """Test that result has expected columns."""
        ankle_df = pd.DataFrame({
            'frame_idx': [0, 0],
            'keypoint_name': ['left_ankle', 'right_ankle'],
            'x': [100, 200],
            'y': [100, 100],
            'field_x': [10.0, 20.0],
            'field_y': [10.0, 10.0],
        })

        ball_df = pd.DataFrame({
            'frame_id': [0],
            'field_center_x': [15.0],
            'field_center_y': [10.0],
        })

        result = get_closest_ankle_per_frame(ankle_df, ball_df)

        expected_cols = ['frame_id', 'ankle_x', 'ankle_y', 'ankle_field_x',
                        'ankle_field_y', 'closest_ankle', 'ball_ankle_distance']
        for col in expected_cols:
            assert col in result.columns

    @pytest.mark.skipif(not all(p.exists() for p in [BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_closest_ankle_with_real_data(self):
        """Test closest ankle computation with real data."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        ankle_df = extract_ankle_positions(pose_df)

        result = get_closest_ankle_per_frame(ankle_df, ball_df)

        assert len(result) > 0
        assert result['ball_ankle_distance'].min() >= 0
        assert all(result['closest_ankle'].isin(ANKLE_KEYPOINTS))


class TestValidateDataAlignment:
    """Tests for validate_data_alignment function."""

    def test_alignment_statistics(self):
        """Test alignment validation returns expected stats."""
        cone_df = pd.DataFrame({'frame_id': [0, 1, 2], 'object_id': [1, 1, 1]})
        ball_df = pd.DataFrame({'frame_id': [0, 1, 2]})
        pose_df = pd.DataFrame({'frame_idx': [0, 1, 2]})

        stats = validate_data_alignment(cone_df, ball_df, pose_df)

        assert stats['common_frames'] == 3
        assert stats['coverage_pct'] == 100.0

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_alignment_with_real_data(self):
        """Test alignment validation with real data."""
        cone_df, ball_df, pose_df = load_all_data(
            str(CONE_PATH), str(BALL_PATH), str(POSE_PATH)
        )

        stats = validate_data_alignment(cone_df, ball_df, pose_df)

        assert stats['coverage_pct'] > 90  # Should have high coverage
        assert stats['common_frames'] > 1000


class TestIntegration:
    """Integration tests with real data."""

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_full_data_loading_pipeline(self):
        """Test complete data loading pipeline."""
        # Load all data
        cone_df, ball_df, pose_df = load_all_data(
            str(CONE_PATH), str(BALL_PATH), str(POSE_PATH)
        )

        # Validate data
        assert len(cone_df) > 10000  # Expected ~10,563
        assert len(ball_df) > 1500   # Expected ~1,522
        assert len(pose_df) > 30000  # Expected ~39,156

        # Extract ankles
        ankle_df = extract_ankle_positions(pose_df)
        assert len(ankle_df) == len(pose_df) / 13  # 2 ankles out of 26 keypoints

        # Get closest ankle per frame
        closest_df = get_closest_ankle_per_frame(ankle_df, ball_df)
        assert len(closest_df) > 1400  # Most ball frames should have ankle data

        # Verify distances are reasonable
        assert closest_df['ball_ankle_distance'].median() < 500  # Field units


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 3. Update __init__.py (Add data_loader exports)

Add to existing `__init__.py`:

```python
# Add after existing imports
from .data_loader import (
    load_parquet_data,
    load_all_data,
    extract_ankle_positions,
    get_closest_ankle_per_frame,
    validate_data_alignment,
    ANKLE_KEYPOINTS,
)

# Add to __all__
__all__ = [
    # ... existing exports ...
    # Data loading
    'load_parquet_data',
    'load_all_data',
    'extract_ankle_positions',
    'get_closest_ankle_per_frame',
    'validate_data_alignment',
    'ANKLE_KEYPOINTS',
]
```

---

## Validation Commands

```bash
# Navigate to project
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"

# Run Stage 2 tests
pytest tests/test_data_loader.py -v

# Quick validation with real data
python -c "
from f8_loss.data_loader import load_all_data, extract_ankle_positions, get_closest_ankle_per_frame

DATA_DIR = '/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar'

cone_df, ball_df, pose_df = load_all_data(
    f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet',
    f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet',
    f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet',
)

print(f'Cones: {len(cone_df)} records')
print(f'Balls: {len(ball_df)} records')
print(f'Poses: {len(pose_df)} records')

ankle_df = extract_ankle_positions(pose_df)
print(f'Ankles: {len(ankle_df)} records')

closest_df = get_closest_ankle_per_frame(ankle_df, ball_df)
print(f'Closest ankle per frame: {len(closest_df)} frames')
print(f'Median ball-ankle distance: {closest_df[\"ball_ankle_distance\"].median():.1f}')

print('Stage 2 COMPLETE')
"
```

---

## Ready for Stage 3 Checklist

- [ ] `data_loader.py` created with all functions
- [ ] `tests/test_data_loader.py` passes all tests
- [ ] Can load real parquet files without errors
- [ ] `extract_ankle_positions()` returns only ankle keypoints
- [ ] `get_closest_ankle_per_frame()` finds correct ankle
- [ ] Median ball-ankle distance is reasonable (<500 field units)

---

## Next Stage Preview

**Stage 3: Core Detection** will implement:
- `ball_control_detector.py` - Main detector class with placeholder logic
- Detection pipeline: merge data → process frames → track state

**Context needed for Stage 3**:
- FrameData, LossEvent, DetectionResult from data_structures.py
- AppConfig, DetectionConfig from config.py
- Data loader functions
