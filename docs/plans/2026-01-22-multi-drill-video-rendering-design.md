# Multi-Drill Video Rendering Design

**Date**: 2026-01-22
**Status**: Approved
**Scope**: Add video rendering to `annotate_video.py` for N-cone drills

## Overview

Adapt the rendering logic from `annotate_triple_cone.py` into a new `annotate_video.py` that supports any number of cones (3, 5, 7+).

## Key Decisions

- **Data loading**: Reuse existing loader pattern (no new `DrillData` class for now)
- **Turning zones**: Only draw around cones with `type: turn`
- **Colors**: Use palette in order (cone index = color index)
- **Backwards compatible**: `annotate_triple_cone.py` stays unchanged

## File Structure

```
video/
├── annotate_triple_cone.py    # UNCHANGED - existing 3-cone workflow
├── annotate_video.py          # NEW - generic N-cone entry point
├── annotation_config.py       # MODIFY - add CONE_COLOR_PALETTE
└── annotation_drawing/
    ├── primitives.py          # MODIFY - add draw_cone_markers()
    └── sidebar.py             # MODIFY - dynamic cone list

detection/
├── drill_types_config.yaml    # MODIFY - add path_patterns
└── drill_config_loader.py     # MODIFY - add detect_drill_type_from_path()
```

## CLI Usage

```bash
# Auto-detects drill type from path pattern
python video/annotate_video.py path/to/player_data_folder/

# Explicit drill type override
python video/annotate_video.py path/to/data/ --drill-type seven_cone_weave

# List available videos
python video/annotate_video.py --list
```

## Rendering Loop Changes

### 1. Cone Loading (before loop)

```python
# Load N cone positions
cone_positions = load_all_cone_positions(cone_parquet)  # List of (x, y)

# Match to drill config
detected_cones = assign_cones_to_config(cone_positions, drill_config)
# Result: [DetectedCone(position, definition), ...]
```

### 2. Turning Zones (before loop)

```python
# Create zones ONLY for turn-type cones
turn_cones = [c for c in detected_cones if c.definition.type == ConeType.TURN]
turning_zones = create_zones_for_turn_cones(turn_cones, zone_config)
```

### 3. Cone Drawing (in loop)

```python
# Draw all cones with palette colors
draw_cone_markers(canvas, detected_cones, config, x_offset=SIDEBAR_WIDTH)
# Internally uses CONE_COLOR_PALETTE[i] for cone i
```

### 4. Unchanged Components

These work without modification (no cone dependency):
- Ball bounding box drawing
- Pose skeleton drawing
- Momentum arrows (player + ball)
- Ball-behind detection (intention + momentum based)
- Edge zone tracking
- All counters (behind, edge, vertical deviation)

## New/Modified Functions

### 1. `annotation_config.py` - Add Color Palette

```python
# 10 distinct colors (BGR) - enough for most drills
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
```

### 2. `drill_config_loader.py` - Path Detection

```python
def detect_drill_type_from_path(self, path: str) -> Optional[str]:
    """Match path against configured patterns, return drill_id or None."""
    path_lower = path.lower()
    for drill_id, config in self._drill_types.items():
        for pattern in config.path_patterns:
            if pattern.lower() in path_lower:
                return drill_id
    return None
```

Requires adding `path_patterns` to YAML:
```yaml
triple_cone:
  path_patterns:
    - "_tc/"
    - "_tc_"
    - "triple_cone"
```

### 3. `annotation_data/loaders.py` - N-Cone Loader

```python
def load_all_cone_positions(parquet_path: Path) -> List[Tuple[float, float]]:
    """Load all cone positions from parquet, sorted left-to-right by X."""
    cone_df = read_parquet_safe(parquet_path)
    cone_df = cone_df[cone_df['object_id'].notna()]

    positions = []
    for obj_id in sorted(cone_df['object_id'].unique()):
        obj_data = cone_df[cone_df['object_id'] == obj_id]
        x = obj_data['center_x'].mean()
        y = obj_data['center_y'].mean()
        if pd.notna(x) and pd.notna(y):
            positions.append((x, y))

    positions.sort(key=lambda p: p[0])  # Left-to-right
    return positions
```

### 4. `annotation_drawing/primitives.py` - Generic Cone Drawing

```python
def draw_cone_markers(
    frame: np.ndarray,
    detected_cones: List[DetectedCone],
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw N cone markers using color palette."""
    for i, cone in enumerate(detected_cones):
        color = CONE_COLOR_PALETTE[i % len(CONE_COLOR_PALETTE)]
        x, y = cone.position
        label = cone.definition.label

        # Draw marker (circle + label)
        center = (int(x) + x_offset, int(y))
        cv2.circle(frame, center, 8, color, -1)
        cv2.circle(frame, center, 8, (0, 0, 0), 2)

        # Draw label above
        cv2.putText(frame, label, (center[0] - 20, center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
```

### 5. Zone Creation Helper

```python
def create_zones_for_turn_cones(
    turn_cones: List[DetectedCone],
    zone_config: TripleConeZoneConfig
) -> List[TurningZone]:
    """Create elliptical turning zones only for turn-type cones."""
    zones = []
    for cone in turn_cones:
        x, y = cone.position
        zone = TurningZone(
            center_x=x,
            center_y=y,
            radius_x=zone_config.cone1_zone_radius,
            radius_y=zone_config.cone1_zone_radius * zone_config.stretch_y,
            label=cone.definition.label
        )
        zones.append(zone)
    return zones
```

## Sidebar Changes

### Dynamic Cone List

Old (hardcoded 3 cones):
```
CONE1 (HOME): (467, 780)
CONE2: (1393, 790)
CONE3: (2316, 801)
```

New (N cones):
```
Cones (7):
  0: turn_cone_1 (52)
  1: cone_2 (180)
  2: cone_3 (308)
  ...
```

### Handling Many Cones

If more than 5 cones, show abbreviated list:
```
Cones (7):
  0: turn_cone_1 (52)
  1: cone_2 (180)
  ... (3 more)
  6: turn_cone_2 (820)
```

## Implementation Order

1. Add `CONE_COLOR_PALETTE` to `annotation_config.py`
2. Add `path_patterns` to `drill_types_config.yaml`
3. Add `detect_drill_type_from_path()` to `drill_config_loader.py`
4. Add `load_all_cone_positions()` to `annotation_data/loaders.py`
5. Add `draw_cone_markers()` to `annotation_drawing/primitives.py`
6. Create `video/annotate_video.py` entry point
7. Update sidebar in `annotation_drawing/sidebar.py`
8. Test with triple_cone data (should match existing output)
9. Test with 7-cone data

## Testing

### Validation Approach

1. Run on existing triple-cone video
2. Compare output visually to `annotate_triple_cone.py` output
3. Verify: same cone positions, same colors, same zones

### Expected Differences

- Cone labels will show config labels (e.g., "turn_cone_1") instead of "CONE1 (HOME)"
- Sidebar format slightly different

## Out of Scope

- New `DrillData` class (can add later if needed)
- Merged CSV format support (parquet only for now)
- Detection logic changes (annotation only)
