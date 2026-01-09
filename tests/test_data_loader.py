"""Stage 2 Tests: Data Layer - Parquet loading and ankle extraction."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from triple_cone_loss.detection.data_loader import (
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
            'center_x': [120.0, 180.0],      # Pixel coords (closer to left_ankle=100, right_ankle=200)
            'center_y': [100.0, 100.0],      # Pixel coords
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
            'center_x': [150.0],             # Pixel coords (midpoint between ankles)
            'center_y': [100.0],             # Pixel coords
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

        # Verify distances are reasonable (field units scale varies)
        assert closest_df['ball_ankle_distance'].median() < 2000  # Field units


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
