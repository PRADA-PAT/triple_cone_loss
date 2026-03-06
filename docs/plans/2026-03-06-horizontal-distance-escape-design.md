# Horizontal Distance Escape Detection — Design

**Date:** 2026-03-06
**Stage:** Experimental (visualization-only)
**Status:** Approved

## Problem

The current ball-behind detection uses horizontal delta_x only for direction (sign), not magnitude. A player who lets the ball drift far ahead along the drill line (X-axis) isn't detected as losing control unless the ball goes off-screen or stays "behind" for sustained frames.

## Feature Summary

Calculate `|ball_pixel_x - hip_pixel_x|` per frame and visualize it on the annotated video. When the horizontal distance exceeds a threshold (~120px at 720p), flag it as "escaped." Track consecutive escaped frames with a counter.

This is a **new, supplementary** loss event type (`BALL_ESCAPED_HORIZONTAL`) — separate from existing ball-behind and boundary detection. It will be visualization-only initially, promoted to detection after validation.

## Design

### Calculation

```
horizontal_distance = abs(ball_pixel_x - hip_pixel_x)
```

Pixel coordinates, same source as existing ball-behind detection.

### Threshold Zones (720p)

| Zone | Range | Color | Meaning |
|------|-------|-------|---------|
| Safe | 0–80px | Green | Ball close to player |
| Warning | 80–120px | Yellow | Ball drifting away |
| Escaped | >120px | Red | Ball has escaped horizontally |

For reference: cone-to-cone spacing is ~920px, player body width ~50-80px.

### No Turning Zone Suppression

Unlike momentum-based ball-behind detection, horizontal escape is NOT suppressed in turning zones. If the ball is 120px away during a turn, that's still a loss of control.

### Visualization Components

1. **Sidebar value:** `H-Dist: 134px` — color-coded green/yellow/red
2. **Field line:** Horizontal line from hip to ball (at hip Y), colored by zone
3. **Escaped counter:** `H-Esc: 5f` — consecutive frames in red zone, resets when distance drops below threshold. Red when counting, grey when 0.

### Configuration

Added to annotation config:

```python
DRAW_HORIZONTAL_DISTANCE: bool = True
H_DIST_WARNING_THRESHOLD: float = 80.0
H_DIST_ESCAPED_THRESHOLD: float = 120.0
```

## Files Modified

| File | Change |
|------|--------|
| `video/annotation_config.py` | Add 3 config values |
| `video/annotation_analysis/ball_position.py` | Add `calculate_horizontal_distance()` function |
| `video/annotate_video.py` | Main loop: calculate, draw sidebar, draw field line, track counter |
| `video/annotation_drawing.py` | Drawing helpers if needed |

## Files NOT Modified

- `detection/ball_control_detector.py` — no detection changes (viz-only stage)
- `video/annotate_triple_cone.py` — legacy, not updated

## Out of Scope

- Vertical (Y-axis) distance
- Speed-relative threshold scaling
- Directional distinction (ahead vs behind)
- Detection/loss event generation (future promotion)
