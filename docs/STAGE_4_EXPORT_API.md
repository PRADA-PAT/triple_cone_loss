# STAGE 4: Export & Public API

**Duration**: ~20 minutes
**Prerequisites**: Stages 1-3 complete (types, data loader, detector)
**Outcome**: Full pipeline working - load → detect → export CSV

---

## Project Context

**Project Path**: `/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/`

**Output Directory**: `./output/` (created automatically)

---

## Prerequisites Check

```bash
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"

# Verify Stage 1-3 are complete
python -c "
from f8_loss import detect_ball_control, load_parquet_data
from f8_loss.data_structures import DetectionResult
print('Prerequisites OK')
"
```

---

## Files to Create/Update

```
f8_loss/
├── csv_exporter.py    # ← CREATE THIS
├── __init__.py        # ← UPDATE (complete version)
└── tests/
    └── test_exporter.py  # ← CREATE THIS
```

---

## CSV Output Schemas

### loss_events.csv
```csv
# Ball Control Loss Events
# Total frames: 1522
# Total events: 3
# Control percentage: 97.5%
#
event_id,event_type,start_frame,end_frame,start_timestamp,end_timestamp,duration_frames,duration_seconds,ball_x,ball_y,player_x,player_y,distance_at_loss,velocity_at_loss,nearest_cone_id,gate_context,recovered,recovery_frame,severity,notes
1,loss_distance,145,168,4.833,5.600,23,0.767,325.4,218.7,412.3,245.1,156.8,42.3,4,G2,True,168,medium,Recovered near gate
```

### frame_analysis.csv
```csv
frame_id,timestamp,ball_x,ball_y,ball_field_x,ball_field_y,ball_velocity,ankle_x,ankle_y,ankle_field_x,ankle_field_y,closest_ankle,nearest_cone_id,nearest_cone_distance,current_gate,ball_ankle_distance,control_score,control_state
0,0.000,808.0,353.0,477.988,155.361,0.0,792.5,368.2,461.2,170.5,right_ankle,4,85.2,G2,24.3,0.82,unknown
```

---

## 1. csv_exporter.py - CSV Export Module

```python
"""
CSV export for Ball Control Detection results.

Exports:
- Loss events CSV with metadata header
- Frame-by-frame analysis CSV
"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from .data_structures import DetectionResult

logger = logging.getLogger(__name__)


class CSVExporter:
    """Export detection results to CSV files."""

    def __init__(self, decimal_precision: int = 3):
        """
        Initialize exporter.

        Args:
            decimal_precision: Number of decimal places for float columns
        """
        self.decimal_precision = decimal_precision

    def export_events(
        self,
        result: DetectionResult,
        output_path: str
    ) -> bool:
        """
        Export loss events to CSV.

        Creates a CSV with metadata header containing:
        - Total frames processed
        - Total events detected
        - Control percentage

        Args:
            result: Detection result
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            if not result.events:
                # Create empty CSV with headers
                columns = [
                    'event_id', 'event_type', 'start_frame', 'end_frame',
                    'start_timestamp', 'end_timestamp', 'duration_frames',
                    'duration_seconds', 'ball_x', 'ball_y', 'player_x', 'player_y',
                    'distance_at_loss', 'velocity_at_loss', 'nearest_cone_id',
                    'gate_context', 'recovered', 'recovery_frame', 'severity', 'notes'
                ]
                df = pd.DataFrame(columns=columns)
            else:
                df = pd.DataFrame([e.to_dict() for e in result.events])

            # Round numeric columns
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].round(self.decimal_precision)

            # Write with metadata header
            with open(output_path, 'w') as f:
                f.write(f"# Ball Control Loss Events\n")
                f.write(f"# Total frames: {result.total_frames}\n")
                f.write(f"# Total events: {result.total_loss_events}\n")
                f.write(f"# Control percentage: {result.control_percentage:.1f}%\n")
                f.write("#\n")
                df.to_csv(f, index=False)

            logger.info(f"Exported {len(result.events)} events to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def export_frame_analysis(
        self,
        result: DetectionResult,
        output_path: str
    ) -> bool:
        """
        Export frame-by-frame analysis to CSV.

        Args:
            result: Detection result
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            if not result.frame_data:
                logger.warning("No frame data to export")
                return False

            # Create output directory if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame([f.to_dict() for f in result.frame_data])

            # Round numeric columns
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].round(self.decimal_precision)

            df.to_csv(output_path, index=False)

            logger.info(f"Exported {len(df)} frames to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def export_all(
        self,
        result: DetectionResult,
        output_dir: str,
        events_filename: str = "loss_events.csv",
        frames_filename: str = "frame_analysis.csv"
    ) -> dict:
        """
        Export both events and frame analysis CSVs.

        Args:
            result: Detection result
            output_dir: Output directory
            events_filename: Events CSV filename
            frames_filename: Frames CSV filename

        Returns:
            Dict with export status for each file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        events_path = output_dir / events_filename
        frames_path = output_dir / frames_filename

        return {
            'events': {
                'success': self.export_events(result, str(events_path)),
                'path': str(events_path),
            },
            'frames': {
                'success': self.export_frame_analysis(result, str(frames_path)),
                'path': str(frames_path),
            }
        }


def export_to_csv(
    result: DetectionResult,
    path: str,
    include_frames: bool = False
) -> bool:
    """
    Convenience function to export detection results to CSV.

    Args:
        result: Detection result
        path: Output path for events CSV
        include_frames: If True, also export frame analysis to *_frames.csv

    Returns:
        True if successful
    """
    exporter = CSVExporter()
    success = exporter.export_events(result, path)

    if include_frames and success:
        frames_path = path.replace('.csv', '_frames.csv')
        exporter.export_frame_analysis(result, frames_path)

    return success
```

---

## 2. __init__.py - Complete Public API

Replace the existing `__init__.py` with this complete version:

```python
"""
Ball Control Detection System for Figure-8 Cone Drills.

A modular system for detecting when a player loses control of the ball
during cone drill exercises.

Quick Start:
    from f8_loss import detect_ball_control, load_parquet_data, export_to_csv

    # Load data
    ball_df = load_parquet_data("ball.parquet")
    pose_df = load_parquet_data("pose.parquet")
    cone_df = load_parquet_data("cone.parquet")

    # Detect
    result = detect_ball_control(ball_df, pose_df, cone_df)

    # Export
    export_to_csv(result, "output.csv")

Classes:
    - AppConfig: Main configuration container
    - BallControlDetector: Core detection class
    - CSVExporter: CSV export functionality
    - DrillVisualizer: Debug video visualization (Stage 5)

Data Structures:
    - FrameData: Per-frame analysis data
    - LossEvent: A detected loss-of-control event
    - DetectionResult: Complete detection output
    - ControlState: Ball control state enum
    - EventType: Loss event type enum
"""

# Configuration
from .config import (
    AppConfig,
    DrillConfig,
    DetectionConfig,
    PathConfig,
    VisualizationConfig,
    DetectionMode,
)

# Data structures
from .data_structures import (
    ControlState,
    EventType,
    FrameData,
    LossEvent,
    DetectionResult,
)

# Data loading
from .data_loader import (
    load_parquet_data,
    load_all_data,
    extract_ankle_positions,
    get_closest_ankle_per_frame,
    validate_data_alignment,
    ANKLE_KEYPOINTS,
)

# Detection
from .ball_control_detector import (
    BallControlDetector,
    detect_ball_control,
)

# Export
from .csv_exporter import (
    CSVExporter,
    export_to_csv,
)


__version__ = "0.1.0"

__all__ = [
    # Version
    '__version__',
    # Config
    'AppConfig',
    'DrillConfig',
    'DetectionConfig',
    'PathConfig',
    'VisualizationConfig',
    'DetectionMode',
    # Data structures
    'ControlState',
    'EventType',
    'FrameData',
    'LossEvent',
    'DetectionResult',
    # Data loading
    'load_parquet_data',
    'load_all_data',
    'extract_ankle_positions',
    'get_closest_ankle_per_frame',
    'validate_data_alignment',
    'ANKLE_KEYPOINTS',
    # Detection
    'BallControlDetector',
    'detect_ball_control',
    # Export
    'CSVExporter',
    'export_to_csv',
]
```

---

## 3. tests/test_exporter.py - Exporter Tests

```python
"""Stage 4 Tests: Export & Public API - CSV export functionality."""
import pytest
import pandas as pd
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from f8_loss.csv_exporter import CSVExporter, export_to_csv
from f8_loss.data_structures import (
    DetectionResult, LossEvent, FrameData,
    EventType, ControlState
)
from f8_loss import (
    detect_ball_control, load_parquet_data,
    AppConfig, BallControlDetector
)

# Real data paths
DATA_DIR = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar")
CONE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet"
BALL_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet"
POSE_PATH = DATA_DIR / "Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet"


@pytest.fixture
def sample_result():
    """Create sample detection result."""
    events = [
        LossEvent(
            event_id=1,
            event_type=EventType.LOSS_DISTANCE,
            start_frame=100,
            end_frame=120,
            start_timestamp=3.33,
            end_timestamp=4.0,
            ball_position=(200.0, 150.0),
            player_position=(180.0, 160.0),
            distance_at_loss=25.0,
            velocity_at_loss=45.0,
            nearest_cone_id=3,
            gate_context="G2",
            recovered=True,
            recovery_frame=120,
        )
    ]

    frame_data = [
        FrameData(
            frame_id=i,
            timestamp=i / 30.0,
            ball_x=100.0 + i,
            ball_y=200.0 + i,
            ball_field_x=50.0 + i * 0.5,
            ball_field_y=100.0 + i * 0.5,
            ball_velocity=10.0,
            ankle_x=90.0 + i,
            ankle_y=190.0 + i,
            ankle_field_x=45.0 + i * 0.5,
            ankle_field_y=95.0 + i * 0.5,
            closest_ankle="right_ankle",
            nearest_cone_id=4,
            nearest_cone_distance=85.0,
            current_gate="G3",
            ball_ankle_distance=15.0,
            control_score=0.85,
            control_state=ControlState.CONTROLLED,
        )
        for i in range(10)
    ]

    return DetectionResult(
        success=True,
        total_frames=1000,
        events=events,
        frame_data=frame_data,
    )


@pytest.fixture
def empty_result():
    """Create empty detection result."""
    return DetectionResult(
        success=True,
        total_frames=100,
        events=[],
        frame_data=[],
    )


class TestCSVExporter:
    """Tests for CSVExporter class."""

    def test_export_events_creates_file(self, sample_result, tmp_path):
        """Test export_events creates CSV file."""
        exporter = CSVExporter()
        output_path = tmp_path / "events.csv"

        success = exporter.export_events(sample_result, str(output_path))

        assert success
        assert output_path.exists()

    def test_export_events_has_metadata_header(self, sample_result, tmp_path):
        """Test export_events includes metadata header."""
        exporter = CSVExporter()
        output_path = tmp_path / "events.csv"
        exporter.export_events(sample_result, str(output_path))

        content = output_path.read_text()

        assert "# Ball Control Loss Events" in content
        assert "# Total frames:" in content
        assert "# Total events:" in content
        assert "# Control percentage:" in content

    def test_export_events_correct_columns(self, sample_result, tmp_path):
        """Test export_events has correct columns."""
        exporter = CSVExporter()
        output_path = tmp_path / "events.csv"
        exporter.export_events(sample_result, str(output_path))

        # Read CSV skipping comment lines
        df = pd.read_csv(output_path, comment='#')

        expected_cols = ['event_id', 'event_type', 'start_frame', 'end_frame']
        for col in expected_cols:
            assert col in df.columns

    def test_export_empty_events(self, empty_result, tmp_path):
        """Test export_events handles empty event list."""
        exporter = CSVExporter()
        output_path = tmp_path / "events.csv"

        success = exporter.export_events(empty_result, str(output_path))

        assert success
        assert output_path.exists()

        # Should have headers but no data rows
        df = pd.read_csv(output_path, comment='#')
        assert len(df) == 0
        assert 'event_id' in df.columns

    def test_export_frame_analysis(self, sample_result, tmp_path):
        """Test export_frame_analysis creates CSV."""
        exporter = CSVExporter()
        output_path = tmp_path / "frames.csv"

        success = exporter.export_frame_analysis(sample_result, str(output_path))

        assert success
        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert len(df) == 10  # 10 frame_data entries
        assert 'frame_id' in df.columns
        assert 'control_score' in df.columns

    def test_export_frame_analysis_empty(self, empty_result, tmp_path):
        """Test export_frame_analysis handles empty frame data."""
        exporter = CSVExporter()
        output_path = tmp_path / "frames.csv"

        success = exporter.export_frame_analysis(empty_result, str(output_path))

        assert success is False  # Should return False for empty

    def test_export_all(self, sample_result, tmp_path):
        """Test export_all exports both files."""
        exporter = CSVExporter()

        results = exporter.export_all(sample_result, str(tmp_path))

        assert results['events']['success']
        assert results['frames']['success']
        assert Path(results['events']['path']).exists()
        assert Path(results['frames']['path']).exists()

    def test_decimal_precision(self, sample_result, tmp_path):
        """Test decimal precision is applied."""
        exporter = CSVExporter(decimal_precision=2)
        output_path = tmp_path / "frames.csv"
        exporter.export_frame_analysis(sample_result, str(output_path))

        df = pd.read_csv(output_path)
        # Check that floats are rounded
        # (this is approximate due to float representation)
        assert df['control_score'].iloc[0] == 0.85


class TestExportToCSV:
    """Tests for export_to_csv convenience function."""

    def test_convenience_function(self, sample_result, tmp_path):
        """Test export_to_csv convenience function."""
        output_path = tmp_path / "output.csv"

        success = export_to_csv(sample_result, str(output_path))

        assert success
        assert output_path.exists()

    def test_convenience_function_with_frames(self, sample_result, tmp_path):
        """Test export_to_csv with include_frames=True."""
        output_path = tmp_path / "output.csv"
        frames_path = tmp_path / "output_frames.csv"

        success = export_to_csv(sample_result, str(output_path), include_frames=True)

        assert success
        assert output_path.exists()
        assert frames_path.exists()


class TestIntegration:
    """Integration tests with real data."""

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_full_pipeline_to_csv(self, tmp_path):
        """Test complete pipeline: load → detect → export."""
        # Load
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        # Detect
        result = detect_ball_control(ball_df, pose_df, cone_df)

        # Export
        exporter = CSVExporter()
        export_results = exporter.export_all(result, str(tmp_path))

        assert export_results['events']['success']
        assert export_results['frames']['success']

        # Verify CSVs are readable
        events_df = pd.read_csv(export_results['events']['path'], comment='#')
        frames_df = pd.read_csv(export_results['frames']['path'])

        assert 'event_id' in events_df.columns
        assert len(frames_df) > 1000

    @pytest.mark.skipif(not all(p.exists() for p in [CONE_PATH, BALL_PATH, POSE_PATH]),
                        reason="Real data not available")
    def test_csv_content_quality(self, tmp_path):
        """Test that CSV content has reasonable values."""
        ball_df = load_parquet_data(str(BALL_PATH))
        pose_df = load_parquet_data(str(POSE_PATH))
        cone_df = load_parquet_data(str(CONE_PATH))

        result = detect_ball_control(ball_df, pose_df, cone_df)

        exporter = CSVExporter()
        exporter.export_all(result, str(tmp_path))

        frames_df = pd.read_csv(tmp_path / "frame_analysis.csv")

        # Check data quality
        assert frames_df['frame_id'].min() >= 0
        assert frames_df['control_score'].min() >= 0
        assert frames_df['control_score'].max() <= 1
        assert frames_df['ball_ankle_distance'].min() >= 0


class TestPublicAPI:
    """Test the public API (__init__.py exports)."""

    def test_import_all_exports(self):
        """Test all expected exports are available."""
        import f8_loss

        # Config
        assert hasattr(f8_loss, 'AppConfig')
        assert hasattr(f8_loss, 'DrillConfig')
        assert hasattr(f8_loss, 'DetectionConfig')

        # Data structures
        assert hasattr(f8_loss, 'FrameData')
        assert hasattr(f8_loss, 'LossEvent')
        assert hasattr(f8_loss, 'DetectionResult')
        assert hasattr(f8_loss, 'ControlState')

        # Functions
        assert hasattr(f8_loss, 'load_parquet_data')
        assert hasattr(f8_loss, 'detect_ball_control')
        assert hasattr(f8_loss, 'export_to_csv')

        # Classes
        assert hasattr(f8_loss, 'BallControlDetector')
        assert hasattr(f8_loss, 'CSVExporter')

    def test_version(self):
        """Test version is defined."""
        import f8_loss
        assert hasattr(f8_loss, '__version__')
        assert f8_loss.__version__ == "0.1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Validation Commands

```bash
# Navigate to project
cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss"

# Run Stage 4 tests
pytest tests/test_exporter.py -v

# Full pipeline validation
python -c "
from f8_loss import detect_ball_control, load_parquet_data, export_to_csv
from f8_loss.csv_exporter import CSVExporter
from pathlib import Path

DATA_DIR = '/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone_output/Drill_1_7 Cone_dubaiacademy_Alex Mochar'
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Load
ball_df = load_parquet_data(f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_football.parquet')
pose_df = load_parquet_data(f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_pose.parquet')
cone_df = load_parquet_data(f'{DATA_DIR}/Drill_1_7 Cone_dubaiacademy_Alex Mochar_cone.parquet')

# Detect
result = detect_ball_control(ball_df, pose_df, cone_df)

# Export
exporter = CSVExporter()
results = exporter.export_all(result, str(OUTPUT_DIR))

print(f'Events CSV: {results[\"events\"][\"success\"]} - {results[\"events\"][\"path\"]}')
print(f'Frames CSV: {results[\"frames\"][\"success\"]} - {results[\"frames\"][\"path\"]}')

# Verify
import pandas as pd
events_df = pd.read_csv(results['events']['path'], comment='#')
frames_df = pd.read_csv(results['frames']['path'])

print(f'Events: {len(events_df)} rows')
print(f'Frames: {len(frames_df)} rows')

print('Stage 4 COMPLETE')
"

# Verify output files
ls -la output/
head -20 output/loss_events.csv
head -5 output/frame_analysis.csv
```

---

## Ready for Stage 5 Checklist

- [ ] `csv_exporter.py` created
- [ ] `__init__.py` updated with all exports
- [ ] `tests/test_exporter.py` passes all tests
- [ ] Full pipeline works: load → detect → export
- [ ] `output/loss_events.csv` created with metadata header
- [ ] `output/frame_analysis.csv` created with all frame data
- [ ] CSV files are readable with pandas

---

## Next Stage Preview

**Stage 5: CLI & Visualization** will implement:
- `main.py` - Command-line interface
- `drill_visualizer.py` - Debug video with annotations

**Context needed for Stage 5**:
- All previous modules
- Video path: `/Users/pradyumn/Desktop/FOOTBALL data /AIM/7 Cone/Drill_1_7 Cone_dubaiacademy_Alex Mochar.MOV`
- OpenCV video writing
