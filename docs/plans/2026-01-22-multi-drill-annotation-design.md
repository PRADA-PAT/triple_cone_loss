# Multi-Drill Annotation System Design

**Date**: 2026-01-22
**Status**: Approved
**Scope**: Annotation-only adaptability (detection logic unchanged)

## Overview

Make the video annotation system work with any drill type (3, 5, 7+ cones) by:
1. Reading drill type from config (with path-based auto-detection)
2. Reading cone positions from data (parquet or merged CSV)
3. Drawing turning zones only around turn-type cones
4. Using a color palette for N cones

## File Structure

### New/Modified Files

```
video/
├── annotate_video.py              # NEW: Generic entry point
├── annotation_config.py           # MODIFY: Add cone color palette
├── annotation_data/
│   └── loaders.py                 # MODIFY: Add merged CSV loader, dual-format support
└── drawing/
    └── cone_drawing.py            # MODIFY: Generic N-cone drawing

detection/
├── drill_types_config.yaml        # MODIFY: Add path_patterns for auto-detection
└── drill_config_loader.py         # MODIFY: Add detect_drill_type_from_path()
```

### Unchanged

- `annotate_triple_cone.py` - existing workflow untouched
- All detection logic - unchanged
- Ball-behind, momentum, edge detection - work as-is (no cone dependency)

## Drill Type Auto-Detection

### Config with Path Patterns

```yaml
# detection/drill_types_config.yaml
drill_types:
  triple_cone:
    name: "Triple Cone Drill"
    cone_count: 3
    path_patterns:
      - "_tc/"
      - "_tc_"
      - "triple_cone"
      - "_triple/"
      - "_triple_"
    cones:
      - position: 0
        type: turn
        label: turn_cone_1
      - position: 1
        type: turn
        label: turn_cone_2
      - position: 2
        type: turn
        label: turn_cone_3

  seven_cone_weave:
    name: "7 Cone Weave"
    cone_count: 7
    path_patterns:
      - "7_cone_weave"
      - "seven_cone"
      - "passing_receiving_turning"
    cones:
      - position: 0
        type: turn
        label: turn_cone_1
      - position: 1
        type: weave
        label: cone_2
      - position: 2
        type: weave
        label: cone_3
      - position: 3
        type: weave
        label: cone_4
      - position: 4
        type: weave
        label: cone_5
      - position: 5
        type: weave
        label: cone_6
      - position: 6
        type: turn
        label: turn_cone_2
```

### Detection Logic

```python
# detection/drill_config_loader.py

def detect_drill_type_from_path(self, path: str) -> Optional[str]:
    """Match path against configured patterns, return drill_id or None."""
    path_lower = path.lower()
    for drill_id, config in self._drill_types.items():
        for pattern in config.path_patterns:
            if pattern.lower() in path_lower:
                return drill_id
    return None  # Unknown drill type
```

## Data Loading

### Dual Format Support

The system handles two data formats:

| Format | Path Pattern | Ball Position | Cone Position |
|--------|--------------|---------------|---------------|
| **Parquet (3 files)** | `*_cone.parquet`, `*_football.parquet`, `*_pose.parquet` | `center_x_pp`, `center_y_pp` | `center_x_pp` per `object_id` |
| **Merged CSV** | `*_merged.csv` | `x_center`, `y_center` | `cone_center_x` per `cone_id` |

### Column Mappings

| Data | Parquet | Merged CSV |
|------|---------|------------|
| Ball X | `center_x_pp` | `x_center` |
| Ball Y | `center_y_pp` | `y_center` |
| Interpolated | `interpolated` | `interpolated` |
| Frame ID | ball: `frame_id`, pose: `frame_idx` | `frame` |
| Cone X | `center_x_pp` (mean per `object_id`) | `cone_center_x` (mean per `cone_id`) |

### Pose Keypoint Mapping

| Annotation Needs | Parquet `keypoint_name` | Merged CSV Column |
|------------------|-------------------------|-------------------|
| Hip center | `hip` | `MidHip_x`, `MidHip_y`, `MidHip_confidence` |
| Nose | `nose` | `Nose_x`, `Nose_y`, `Nose_confidence` |
| Left ankle | `left_ankle` | `LAnkle_x`, `LAnkle_y`, `LAnkle_confidence` |
| Right ankle | `right_ankle` | `RAnkle_x`, `RAnkle_y`, `RAnkle_confidence` |

### DrillData Class

```python
@dataclass
class DrillData:
    """Unified data container for both formats."""
    cone_positions: List[Tuple[float, float]]

    # Format-specific storage
    ball_df: Optional[pd.DataFrame] = None      # Parquet format
    pose_df: Optional[pd.DataFrame] = None      # Parquet format
    merged_df: Optional[pd.DataFrame] = None    # CSV format

    # Column mappings (set by loader)
    ball_x_col: str = 'center_x_pp'
    ball_y_col: str = 'center_y_pp'
    interpolated_col: str = 'interpolated'

    @property
    def is_parquet_format(self) -> bool:
        return self.ball_df is not None
```

### Unified Loader

```python
def load_drill_data(data_path: Path) -> DrillData:
    """Auto-detect format and load data."""
    if data_path.suffix == '.csv':
        return load_from_merged_csv(data_path)
    elif data_path.suffix == '.parquet' or data_path.is_dir():
        return load_from_parquet_files(data_path)
    else:
        raise ValueError(f"Unknown data format: {data_path}")
```

### Parquet Loader

```python
def load_from_parquet_files(base_path: Path) -> DrillData:
    """Load from separate parquet files."""
    # Find files
    if base_path.is_dir():
        cone_file = next(base_path.glob("*_cone.parquet"))
        ball_file = next(base_path.glob("*_football.parquet"))
        pose_file = next(base_path.glob("*_pose.parquet"))
    else:
        stem = str(base_path).rsplit('_', 1)[0]  # Remove suffix like _cone
        cone_file = Path(f"{stem}_cone.parquet")
        ball_file = Path(f"{stem}_football.parquet")
        pose_file = Path(f"{stem}_pose.parquet")

    cone_df = read_parquet_safe(cone_file)
    ball_df = read_parquet_safe(ball_file)
    pose_df = read_parquet_safe(pose_file)

    # Extract cone positions (post-processed, mean per object_id)
    cone_positions = []
    for obj_id in sorted(cone_df['object_id'].unique()):
        obj_data = cone_df[cone_df['object_id'] == obj_id]
        x = obj_data['center_x_pp'].mean()
        y = obj_data['center_y_pp'].mean()
        cone_positions.append((x, y))
    cone_positions.sort(key=lambda c: c[0])  # Left-to-right

    return DrillData(
        ball_df=ball_df,
        pose_df=pose_df,
        cone_positions=cone_positions,
        ball_x_col='center_x_pp',
        ball_y_col='center_y_pp',
        interpolated_col='interpolated',
    )
```

### Merged CSV Loader

```python
def load_from_merged_csv(csv_path: Path) -> DrillData:
    """Load from single merged CSV."""
    df = pd.read_csv(csv_path)

    # Extract cone positions (mean per cone_id)
    cone_positions = []
    for cone_id in sorted(df['cone_id'].dropna().unique()):
        cone_data = df[df['cone_id'] == cone_id]
        x = cone_data['cone_center_x'].mean()
        y = cone_data['cone_center_y'].mean()
        cone_positions.append((x, y))
    cone_positions.sort(key=lambda c: c[0])

    return DrillData(
        merged_df=df,
        cone_positions=cone_positions,
        ball_x_col='x_center',
        ball_y_col='y_center',
        interpolated_col='interpolated',
    )
```

### Pose Extraction Helper

```python
def get_pose_for_frame(frame_id: int, data: DrillData) -> dict:
    """Extract pose keypoints for a frame, handling both formats."""

    if data.is_parquet_format:
        # Long format: filter by frame
        frame_pose = data.pose_df[data.pose_df['frame_idx'] == frame_id]

        def get_keypoint(name):
            row = frame_pose[frame_pose['keypoint_name'] == name]
            if row.empty:
                return None, None, 0.0
            r = row.iloc[0]
            return r['x'], r['y'], r['confidence']

        return {
            'hip': get_keypoint('hip'),
            'nose': get_keypoint('nose'),
            'left_ankle': get_keypoint('left_ankle'),
            'right_ankle': get_keypoint('right_ankle'),
        }

    else:
        # Wide format: direct column access
        row = data.merged_df[data.merged_df['frame'] == frame_id].iloc[0]

        return {
            'hip': (row['MidHip_x'], row['MidHip_y'], row['MidHip_confidence']),
            'nose': (row['Nose_x'], row['Nose_y'], row['Nose_confidence']),
            'left_ankle': (row['LAnkle_x'], row['LAnkle_y'], row['LAnkle_confidence']),
            'right_ankle': (row['RAnkle_x'], row['RAnkle_y'], row['RAnkle_confidence']),
        }
```

## Cone Colors & Drawing

### Color Palette

```python
# annotation_config.py

# Distinct colors for up to 10 cones (BGR format)
CONE_COLOR_PALETTE = [
    (200, 200, 0),    # Teal (index 0)
    (200, 100, 200),  # Purple (index 1)
    (100, 200, 200),  # Orange (index 2)
    (0, 200, 0),      # Green (index 3)
    (200, 0, 200),    # Magenta (index 4)
    (0, 200, 200),    # Yellow (index 5)
    (200, 100, 100),  # Light blue (index 6)
    (100, 100, 200),  # Salmon (index 7)
    (150, 200, 100),  # Lime (index 8)
    (100, 150, 200),  # Peach (index 9)
]

TURN_CONE_BRIGHTNESS_BOOST = 1.2
```

### Drawing Logic

```python
def draw_cones(canvas, detected_cones: List[DetectedCone], config):
    """Draw all cones with type-appropriate visualization."""
    for i, cone in enumerate(detected_cones):
        color = CONE_COLOR_PALETTE[i % len(CONE_COLOR_PALETTE)]

        # Boost brightness for turn cones
        if cone.definition.type == ConeType.TURN:
            color = brighten(color, TURN_CONE_BRIGHTNESS_BOOST)

        # Draw cone marker
        draw_cone_marker(canvas, cone.position, color, cone.definition.label)

        # Only draw turning zone for turn-type cones
        if cone.definition.type == ConeType.TURN:
            draw_turning_zone(canvas, cone.position, color, alpha=0.25)
```

## Entry Point

### annotate_video.py

```python
def annotate_video(data_path: Path, video_path: Path, output_path: Path):
    """Generic video annotation for any drill type."""

    # 1. Auto-detect drill type from path
    loader = DrillConfigLoader()
    drill_id = loader.detect_drill_type_from_path(str(data_path))

    if drill_id:
        drill_config = loader.get_drill_type(drill_id)
        print(f"Auto-detected drill type: {drill_config.name}")
    else:
        drill_config = None
        print("Unknown drill type - using generic annotation")

    # 2. Load data (auto-detects format)
    data = load_drill_data(data_path)

    # 3. Validate cone count if config exists
    if drill_config:
        if len(data.cone_positions) != drill_config.cone_count:
            print(f"Warning: Expected {drill_config.cone_count} cones, "
                  f"found {len(data.cone_positions)}")

    # 4. Assign cones to config definitions
    if drill_config:
        detected_cones = assign_cones_to_config(data.cone_positions, drill_config)
    else:
        detected_cones = make_generic_cones(data.cone_positions)

    # 5. Run annotation loop
    annotate_frames(video_path, output_path, data, detected_cones, config)
```

### CLI Usage

```bash
# Auto-detects from path
python video/annotate_video.py drills/7_cone_weave/tfa_17688437681328196_merged.csv

# Parquet format (directory or single file)
python video/annotate_video.py video_detection_pose_ball_cones_720p/Amal\ Mastan_triple/

# Explicit override
python video/annotate_video.py data.csv --drill-type triple_cone
```

## Validation

At runtime:
1. Auto-detect drill type from path → e.g., `seven_cone_weave`
2. Load config → expects 7 cones
3. Read data → find N unique cone positions
4. Validate: warn if N != expected (but continue - data is truth)

## What Stays the Same

- `annotate_triple_cone.py` - untouched for backwards compatibility
- Ball-behind detection logic - no cone dependency
- Momentum arrows - no cone dependency
- Edge/boundary tracking - no cone dependency
- All detection logic in `detection/` - unchanged

## Implementation Order

1. Add `path_patterns` to `drill_types_config.yaml`
2. Add `detect_drill_type_from_path()` to `drill_config_loader.py`
3. Add `DrillData` class and loaders to `annotation_data/loaders.py`
4. Add `CONE_COLOR_PALETTE` to `annotation_config.py`
5. Create `video/annotate_video.py` entry point
6. Add generic cone drawing functions
7. Test with both data formats
