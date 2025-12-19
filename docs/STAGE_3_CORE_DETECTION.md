# STAGE 3: Core Detection

**Duration**: ~30 minutes
**Prerequisites**: Stage 1 (types) + Stage 2 (data loader) complete
**Outcome**: Detector runs end-to-end with placeholder logic

---

## Project Context

**Project Path**: `/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/`

**Data Path**:
```
/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar/
```

**Key Constants**:
```python
FPS = 30.0
TOTAL_FRAMES = 1509
ANKLE_KEYPOINTS = ['left_ankle', 'right_ankle']
```

---

## Prerequisites Check

```bash
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"

# Verify Stage 1 + 2 are complete
python -c "
from f8_loss import AppConfig, FrameData, DetectionResult
from f8_loss.data_loader import load_parquet_data, extract_ankle_positions
print('Prerequisites OK')
"
```

---

## Files to Create

```
f8_loss/
├── ball_control_detector.py  # ← CREATE THIS
└── tests/
    └── test_detector.py      # ← CREATE THIS
```

---

## Important: Placeholder Methods

The detector has 4 placeholder methods that need actual detection logic later:

1. `_analyze_frame()` - Currently returns None
2. `_calculate_control_score()` - Currently returns (0.0, {})
3. `_detect_loss_condition()` - Currently returns None
4. `_check_state_transition()` - Currently no-op

**The infrastructure is fully implemented** - data flows correctly through the pipeline.
Detection algorithms will be added in a future iteration.

---

## 1. ball_control_detector.py - Core Detector

```python
"""
Ball Control Detector - Main detection class.

PLACEHOLDER METHODS (implement detection logic later):
- _analyze_frame(): Currently returns None
- _calculate_control_score(): Currently returns (0.0, {})
- _detect_loss_condition(): Currently returns None
- _check_state_transition(): Currently no-op

The infrastructure and data flow are fully implemented.
"""
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .config import AppConfig, DetectionConfig
from .data_structures import (
    ControlState, EventType, FrameData,
    LossEvent, DetectionResult
)
from .data_loader import (
    load_parquet_data, extract_ankle_positions,
    get_closest_ankle_per_frame
)

logger = logging.getLogger(__name__)


class BallControlDetector:
    """
    Main class for detecting ball control loss events.

    Usage:
        detector = BallControlDetector()
        result = detector.detect(ball_df, pose_df, cone_df)

    Or with config:
        config = AppConfig()
        config.detection.control_radius = 150.0
        detector = BallControlDetector(config)
        result = detector.detect(ball_df, pose_df, cone_df)
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize detector with optional config."""
        self.config = config or AppConfig()
        self._detection_config = self.config.detection
        self._drill_config = self.config.drill

        # State tracking
        self._current_state = ControlState.UNKNOWN
        self._events: List[LossEvent] = []
        self._frame_data: List[FrameData] = []
        self._event_counter = 0

        logger.info("BallControlDetector initialized")

    def detect(
        self,
        ball_df: pd.DataFrame,
        pose_df: pd.DataFrame,
        cone_df: pd.DataFrame,
        fps: float = 30.0
    ) -> DetectionResult:
        """
        Run ball control detection.

        Args:
            ball_df: Ball detection DataFrame
            pose_df: Pose keypoint DataFrame
            cone_df: Cone detection DataFrame
            fps: Video FPS for timestamps

        Returns:
            DetectionResult with events and frame data
        """
        try:
            logger.info("Starting detection...")
            logger.info(f"  Ball frames: {len(ball_df)}")
            logger.info(f"  Pose records: {len(pose_df)}")
            logger.info(f"  Cone records: {len(cone_df)}")

            # Reset state
            self._reset_state()

            # Extract ankles and find closest per frame
            ankle_df = extract_ankle_positions(pose_df)
            merged_df = get_closest_ankle_per_frame(ankle_df, ball_df)

            if merged_df.empty:
                return DetectionResult(
                    success=False,
                    total_frames=0,
                    events=[],
                    frame_data=[],
                    error="No valid frames after merging"
                )

            # Merge with ball data
            ball_cols = ['frame_id', 'center_x', 'center_y',
                        'field_center_x', 'field_center_y']
            available_cols = [c for c in ball_cols if c in ball_df.columns]
            merged_df = merged_df.merge(ball_df[available_cols], on='frame_id')

            merged_df.rename(columns={
                'center_x': 'ball_x',
                'center_y': 'ball_y',
                'field_center_x': 'ball_field_x',
                'field_center_y': 'ball_field_y',
            }, inplace=True)

            # Calculate ball velocity
            merged_df = merged_df.sort_values('frame_id')
            merged_df['ball_velocity'] = np.sqrt(
                merged_df['ball_field_x'].diff()**2 +
                merged_df['ball_field_y'].diff()**2
            ).fillna(0)

            # Process each frame
            total_frames = len(merged_df)

            for _, row in merged_df.iterrows():
                frame_id = int(row['frame_id'])
                timestamp = frame_id / fps

                # Analyze frame
                frame_result = self._analyze_frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    row=row,
                    cone_df=cone_df
                )

                if frame_result:
                    self._frame_data.append(frame_result)
                    self._check_state_transition(frame_result)

            # Finalize events
            self._finalize_events()

            result = DetectionResult(
                success=True,
                total_frames=total_frames,
                events=self._events,
                frame_data=self._frame_data
            )

            logger.info(f"Detection complete: {result.total_loss_events} events")
            logger.info(f"  Processed frames: {total_frames}")
            logger.info(f"  Frame data generated: {len(self._frame_data)}")

            return result

        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            return DetectionResult(
                success=False,
                total_frames=0,
                events=[],
                frame_data=[],
                error=str(e)
            )

    def _reset_state(self):
        """Reset internal state for new detection run."""
        self._current_state = ControlState.UNKNOWN
        self._events = []
        self._frame_data = []
        self._event_counter = 0

    def _analyze_frame(
        self,
        frame_id: int,
        timestamp: float,
        row: pd.Series,
        cone_df: pd.DataFrame
    ) -> Optional[FrameData]:
        """
        Analyze a single frame.

        PLACEHOLDER: Implement actual detection logic.

        Should:
        1. Get ball and ankle positions from row
        2. Calculate control score using _calculate_control_score()
        3. Find nearest cone
        4. Determine current gate
        5. Return FrameData

        Currently returns a basic FrameData with placeholder values.
        """
        # ============================================
        # PLACEHOLDER - IMPLEMENT DETECTION LOGIC HERE
        # ============================================

        # Get positions from row
        ball_pos = (row['ball_field_x'], row['ball_field_y'])
        ankle_pos = (row['ankle_field_x'], row['ankle_field_y'])

        # Calculate control score (placeholder)
        control_score, _ = self._calculate_control_score(
            ball_pos, ankle_pos, row['ball_velocity'], self._frame_data
        )

        # Get nearest cone
        nearest_cone_id, nearest_cone_dist = self._get_nearest_cone(
            ball_pos, cone_df, frame_id
        )

        # Determine gate (placeholder)
        current_gate = self._determine_gate(ball_pos, cone_df, frame_id)

        # Determine control state (placeholder - always UNKNOWN for now)
        control_state = ControlState.UNKNOWN

        return FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            ball_x=row['ball_x'],
            ball_y=row['ball_y'],
            ball_field_x=row['ball_field_x'],
            ball_field_y=row['ball_field_y'],
            ball_velocity=row['ball_velocity'],
            ankle_x=row['ankle_x'],
            ankle_y=row['ankle_y'],
            ankle_field_x=row['ankle_field_x'],
            ankle_field_y=row['ankle_field_y'],
            closest_ankle=row['closest_ankle'],
            nearest_cone_id=nearest_cone_id,
            nearest_cone_distance=nearest_cone_dist,
            current_gate=current_gate,
            ball_ankle_distance=row['ball_ankle_distance'],
            control_score=control_score,
            control_state=control_state,
        )

    def _calculate_control_score(
        self,
        ball_pos: Tuple[float, float],
        ankle_pos: Tuple[float, float],
        ball_velocity: float,
        history: List[FrameData]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate control score.

        PLACEHOLDER: Implement scoring logic.

        Should consider:
        - Proximity (ball-ankle distance)
        - Velocity (lower = more control)
        - Stability (position variance)
        - History (recent control state)

        Returns:
            (overall_score 0-1, component_scores dict)
        """
        # ============================================
        # PLACEHOLDER - IMPLEMENT SCORING LOGIC HERE
        # ============================================

        # Basic distance-based score for now
        distance = np.sqrt(
            (ball_pos[0] - ankle_pos[0])**2 +
            (ball_pos[1] - ankle_pos[1])**2
        )

        # Simple score: 1.0 at distance 0, 0.0 at control_radius
        control_radius = self._detection_config.control_radius
        score = max(0.0, 1.0 - (distance / control_radius))

        return score, {'distance_score': score}

    def _detect_loss_condition(
        self,
        frame_data: FrameData,
        history: List[FrameData]
    ) -> Optional[EventType]:
        """
        Detect if current frame indicates loss of control.

        PLACEHOLDER: Implement detection logic.

        Should check:
        - Distance exceeding threshold
        - Velocity spike
        - Direction change away from player

        Returns:
            EventType if loss detected, None otherwise
        """
        # ============================================
        # PLACEHOLDER - IMPLEMENT DETECTION LOGIC HERE
        # ============================================
        return None

    def _check_state_transition(self, frame_result: FrameData):
        """
        Check for state transitions and create events.

        PLACEHOLDER: Implement state machine.

        Should:
        1. Compare current state with frame_result.control_state
        2. On transition to LOST: create LossEvent
        3. On transition from LOST: close LossEvent
        """
        # ============================================
        # PLACEHOLDER - IMPLEMENT STATE MACHINE HERE
        # ============================================
        pass

    def _finalize_events(self):
        """Close any open events at end of detection."""
        for event in self._events:
            if event.end_frame is None:
                event.notes += " [Unclosed]"

    def _get_nearest_cone(
        self,
        ball_pos: Tuple[float, float],
        cone_df: pd.DataFrame,
        frame_id: int
    ) -> Tuple[int, float]:
        """Get nearest cone to ball position."""
        frame_cones = cone_df[cone_df['frame_id'] == frame_id]

        if frame_cones.empty:
            return -1, float('inf')

        distances = np.sqrt(
            (frame_cones['field_center_x'] - ball_pos[0])**2 +
            (frame_cones['field_center_y'] - ball_pos[1])**2
        )

        idx = distances.idxmin()
        return int(frame_cones.loc[idx, 'object_id']), float(distances.min())

    def _determine_gate(
        self,
        ball_pos: Tuple[float, float],
        cone_df: pd.DataFrame,
        frame_id: int
    ) -> Optional[str]:
        """
        Determine which gate the ball is currently in/near.

        Returns gate name (e.g., "G3") or None if not in a gate.
        """
        # Get cones for this frame
        frame_cones = cone_df[cone_df['frame_id'] == frame_id]
        if frame_cones.empty:
            return None

        # Find two closest cones
        distances = np.sqrt(
            (frame_cones['field_center_x'] - ball_pos[0])**2 +
            (frame_cones['field_center_y'] - ball_pos[1])**2
        )

        sorted_cones = frame_cones.loc[distances.nsmallest(2).index]
        if len(sorted_cones) < 2:
            return None

        cone_ids = sorted(sorted_cones['object_id'].tolist())

        # Check against gate definitions
        for gate_name, (c1, c2) in self._drill_config.gate_definitions.items():
            if sorted(cone_ids) == sorted([c1, c2]):
                return gate_name

        return None


# Convenience function
def detect_ball_control(
    ball_df: pd.DataFrame,
    pose_df: pd.DataFrame,
    cone_df: pd.DataFrame,
    config: Optional[AppConfig] = None,
    fps: float = 30.0
) -> DetectionResult:
    """
    Convenience function for detection.

    Args:
        ball_df: Ball detection DataFrame
        pose_df: Pose keypoint DataFrame
        cone_df: Cone detection DataFrame
        config: Optional AppConfig
        fps: Video FPS

    Returns:
        DetectionResult
    """
    detector = BallControlDetector(config)
    return detector.detect(ball_df, pose_df, cone_df, fps)
```

---

## 2. tests/test_detector.py - Detector Tests

```python
"""Stage 3 Tests: Core Detection - BallControlDetector."""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from f8_loss.ball_control_detector import (
    BallControlDetector,
    detect_ball_control,
)
from f8_loss.config import AppConfig, DetectionConfig
from f8_loss.data_structures import DetectionResult, ControlState
from f8_loss.data_loader import load_parquet_data, extract_ankle_positions

# Real data paths
DATA_DIR = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar")
CONE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet"
BALL_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet"
POSE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet"


@pytest.fixture
def sample_data():
    """Create minimal sample test data."""
    ball_df = pd.DataFrame({
        'frame_id': [0, 1, 2, 3, 4],
        'center_x': [100, 110, 120, 130, 140],
        'center_y': [200, 210, 220, 230, 240],
        'field_center_x': [50.0, 55.0, 60.0, 65.0, 70.0],
        'field_center_y': [100.0, 105.0, 110.0, 115.0, 120.0],
    })

    pose_df = pd.DataFrame({
        'frame_idx': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        'keypoint_name': ['left_ankle', 'right_ankle'] * 5,
        'x': [90, 95, 100, 105, 110, 115, 120, 125, 130, 135],
        'y': [190, 195, 200, 205, 210, 215, 220, 225, 230, 235],
        'field_x': [45.0, 48.0, 50.0, 53.0, 55.0, 58.0, 60.0, 63.0, 65.0, 68.0],
        'field_y': [95.0, 98.0, 100.0, 103.0, 105.0, 108.0, 110.0, 113.0, 115.0, 118.0],
        'confidence': [0.9] * 10,
    })

    cone_df = pd.DataFrame({
        'frame_id': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        'object_id': [3, 4] * 5,
        'center_x': [50, 150] * 5,
        'center_y': [300, 300] * 5,
        'field_center_x': [25.0, 75.0] * 5,
        'field_center_y': [150.0, 150.0] * 5,
    })

    return ball_df, pose_df, cone_df


class TestBallControlDetector:
    """Tests for BallControlDetector class."""

    def test_initialization_default_config(self):
        """Test detector initializes with default config."""
        detector = BallControlDetector()
        assert detector is not None
        assert detector.config.fps == 30.0
        assert detector._detection_config.control_radius == 120.0

    def test_initialization_custom_config(self):
        """Test detector initializes with custom config."""
        config = AppConfig()
        config.detection.control_radius = 200.0
        config.fps = 60.0

        detector = BallControlDetector(config)
        assert detector.config.fps == 60.0
        assert detector._detection_config.control_radius == 200.0

    def test_detect_returns_result(self, sample_data):
        """Test detect() returns DetectionResult."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        result = detector.detect(ball_df, pose_df, cone_df)

        assert isinstance(result, DetectionResult)
        assert result.success is True

    def test_detect_processes_all_frames(self, sample_data):
        """Test detect() processes all available frames."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        result = detector.detect(ball_df, pose_df, cone_df)

        assert result.total_frames == 5

    def test_detect_generates_frame_data(self, sample_data):
        """Test detect() generates frame data for each frame."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        result = detector.detect(ball_df, pose_df, cone_df)

        # With placeholder logic, should generate frame data
        assert len(result.frame_data) == 5

    def test_detect_empty_ball_df(self, sample_data):
        """Test detect() handles empty ball DataFrame."""
        _, pose_df, cone_df = sample_data
        empty_ball = pd.DataFrame(columns=['frame_id', 'center_x', 'center_y',
                                           'field_center_x', 'field_center_y'])

        detector = BallControlDetector()
        result = detector.detect(empty_ball, pose_df, cone_df)

        assert result.success is False
        assert "No valid frames" in result.error

    def test_detect_calculates_ball_velocity(self, sample_data):
        """Test detect() calculates ball velocity."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        result = detector.detect(ball_df, pose_df, cone_df)

        # Check that velocity is calculated in frame data
        if result.frame_data:
            # First frame should have velocity 0 or small
            assert result.frame_data[0].ball_velocity >= 0
            # Subsequent frames should have calculated velocity
            assert result.frame_data[1].ball_velocity >= 0


class TestConvenienceFunction:
    """Tests for detect_ball_control convenience function."""

    def test_convenience_function(self, sample_data):
        """Test detect_ball_control works."""
        ball_df, pose_df, cone_df = sample_data

        result = detect_ball_control(ball_df, pose_df, cone_df)

        assert isinstance(result, DetectionResult)
        assert result.success is True

    def test_convenience_function_with_config(self, sample_data):
        """Test detect_ball_control with custom config."""
        ball_df, pose_df, cone_df = sample_data
        config = AppConfig()
        config.detection.control_radius = 50.0

        result = detect_ball_control(ball_df, pose_df, cone_df, config=config)

        assert result.success is True


class TestNearestCone:
    """Tests for nearest cone detection."""

    def test_nearest_cone_found(self, sample_data):
        """Test nearest cone is correctly identified."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        # Ball at (50, 100), cones at (25, 150) and (75, 150)
        # Distance to cone 3: sqrt((50-25)^2 + (100-150)^2) = sqrt(625+2500) = ~55.9
        # Distance to cone 4: sqrt((50-75)^2 + (100-150)^2) = sqrt(625+2500) = ~55.9
        cone_id, distance = detector._get_nearest_cone((50.0, 100.0), cone_df, 0)

        assert cone_id in [3, 4]
        assert distance < 60


class TestGateDetection:
    """Tests for gate detection."""

    def test_gate_detection_basic(self, sample_data):
        """Test gate detection finds correct gate."""
        ball_df, pose_df, cone_df = sample_data
        detector = BallControlDetector()

        # Ball position near cones 3 and 4
        gate = detector._determine_gate((50.0, 150.0), cone_df, 0)

        # Should be G4 (cones 4, 3) based on drill config
        assert gate == "G4"


class TestIntegration:
    """Integration tests with real data."""

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_detect_with_real_data(self):
        """Test detection with real parquet files."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        detector = BallControlDetector()
        result = detector.detect(ball_df, pose_df, cone_df)

        assert result.success is True
        assert result.total_frames > 1400  # Most frames should be processed
        assert len(result.frame_data) > 1400

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_frame_data_has_valid_values(self):
        """Test that frame data has reasonable values."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        detector = BallControlDetector()
        result = detector.detect(ball_df, pose_df, cone_df)

        # Check frame data quality
        for fd in result.frame_data[:10]:  # Check first 10
            assert fd.frame_id >= 0
            assert fd.timestamp >= 0
            assert fd.ball_ankle_distance >= 0
            assert fd.nearest_cone_id in [1, 2, 3, 4, 5, 6, 7, -1]
            assert 0 <= fd.control_score <= 1

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_detection_result_summary(self):
        """Test detection result summary statistics."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        result = detect_ball_control(ball_df, pose_df, cone_df)

        # With placeholder logic, should have 0 events but valid summary
        assert result.total_loss_events >= 0
        assert 0 <= result.control_percentage <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## 3. Update __init__.py (Add detector exports)

Add to existing `__init__.py`:

```python
# Add after existing imports
from .ball_control_detector import (
    BallControlDetector,
    detect_ball_control,
)

# Add to __all__
__all__ = [
    # ... existing exports ...
    # Detector
    'BallControlDetector',
    'detect_ball_control',
]
```

---

## Validation Commands

```bash
# Navigate to project
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"

# Run Stage 3 tests
pytest tests/test_detector.py -v

# Quick validation with real data
python -c "
from f8_loss import detect_ball_control, load_parquet_data

DATA_DIR = '/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar'

ball_df = load_parquet_data(f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet')
pose_df = load_parquet_data(f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet')
cone_df = load_parquet_data(f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet')

result = detect_ball_control(ball_df, pose_df, cone_df)

print(f'Success: {result.success}')
print(f'Total frames: {result.total_frames}')
print(f'Frame data generated: {len(result.frame_data)}')
print(f'Loss events: {result.total_loss_events}')
print(f'Control percentage: {result.control_percentage:.1f}%')

if result.frame_data:
    fd = result.frame_data[0]
    print(f'Sample frame 0: ball-ankle dist={fd.ball_ankle_distance:.1f}, score={fd.control_score:.2f}')

print('Stage 3 COMPLETE')
"
```

---

## Ready for Stage 4 Checklist

- [ ] `ball_control_detector.py` created
- [ ] `tests/test_detector.py` passes all tests
- [ ] `detect_ball_control()` runs on real data without errors
- [ ] DetectionResult has valid `success=True`
- [ ] Frame data is generated for all frames
- [ ] Nearest cone detection works (returns valid cone IDs)
- [ ] Gate detection works (returns gate names like "G4")

---

## Next Stage Preview

**Stage 4: Export & Public API** will implement:
- `csv_exporter.py` - Export results to CSV
- Complete `__init__.py` with all public exports

**Context needed for Stage 4**:
- DetectionResult structure
- CSV schema for loss_events.csv and frame_analysis.csv
