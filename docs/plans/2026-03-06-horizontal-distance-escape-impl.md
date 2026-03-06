# Horizontal Distance Escape Detection — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add horizontal distance visualization (sidebar value + field line + escaped counter) to the generic video annotator.

**Architecture:** Add config fields, a pure calculation function, a drawing function, and wire them into the main annotation loop. Follows the same pattern as existing behind-counter and vertical-deviation features.

**Tech Stack:** Python, OpenCV, numpy, pytest

---

### Task 1: Add config fields to annotation_config.py

**Files:**
- Modify: `video/annotation_config.py:82-96` (after vertical deviation block)
- Modify: `video/annotation_config.py:283-298` (scale_config_for_resolution)

**Step 1: Add config fields**

Add after the `VERTICAL_DEVIATION_PERSIST_SECONDS` block (line 96), before the ball position section (line 98):

```python
    # Horizontal distance escape detection
    DRAW_HORIZONTAL_DISTANCE: bool = True
    H_DIST_WARNING_THRESHOLD: float = 80.0   # Yellow zone start (pixels, auto-scaled)
    H_DIST_ESCAPED_THRESHOLD: float = 120.0  # Red zone start (pixels, auto-scaled)
    H_DIST_SAFE_COLOR: Tuple[int, int, int] = (0, 255, 0)       # Green (BGR)
    H_DIST_WARNING_COLOR: Tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
    H_DIST_ESCAPED_COLOR: Tuple[int, int, int] = (0, 0, 255)    # Red (BGR)
    H_DIST_PERSIST_COLOR: Tuple[int, int, int] = (0, 165, 255)  # Orange (BGR)
    H_DIST_LINE_THICKNESS: int = 2
    H_DIST_COUNTER_POS_X: int = 50
    H_DIST_COUNTER_POS_Y: int = 400  # Below vertical deviation counter
    H_DIST_COUNTER_FONT_SCALE: float = 1.2
    H_DIST_PERSIST_SECONDS: float = 3.0
```

**Step 2: Add resolution scaling**

In `scale_config_for_resolution()`, add after the vertical deviation scaling lines (around line 298):

```python
    # Scale horizontal distance thresholds
    config.H_DIST_WARNING_THRESHOLD *= resolution_scale
    config.H_DIST_ESCAPED_THRESHOLD *= resolution_scale
    config.H_DIST_COUNTER_POS_X = int(config.H_DIST_COUNTER_POS_X * resolution_scale)
    config.H_DIST_COUNTER_POS_Y = int(config.H_DIST_COUNTER_POS_Y * resolution_scale)
    config.H_DIST_COUNTER_FONT_SCALE *= font_scale
    config.H_DIST_LINE_THICKNESS = max(1, int(config.H_DIST_LINE_THICKNESS * font_scale))
```

**Step 3: Commit**

```bash
git add video/annotation_config.py
git commit -m "feat: add horizontal distance escape config fields"
```

---

### Task 2: Add HorizontalDistanceResult dataclass

**Files:**
- Modify: `video/annotation_data/structures.py` (add after IntentionPositionResult)

**Step 1: Add dataclass**

Add after `IntentionPositionResult` (line 27):

```python
@dataclass
class HorizontalDistanceResult:
    """Result of horizontal distance calculation between ball and player hip."""
    distance: float           # abs(ball_x - hip_x) in pixels
    zone: str                 # "SAFE", "WARNING", or "ESCAPED"
    color: Tuple[int, int, int]  # BGR color for this zone
```

**Step 2: Commit**

```bash
git add video/annotation_data/structures.py
git commit -m "feat: add HorizontalDistanceResult dataclass"
```

---

### Task 3: Add calculate_horizontal_distance function

**Files:**
- Modify: `video/annotation_analysis/ball_position.py` (add function at end)
- Modify: `video/annotation_analysis/__init__.py` (add export)
- Create: `tests/test_horizontal_distance.py`

**Step 1: Write tests**

Create `tests/test_horizontal_distance.py`:

```python
"""Tests for horizontal distance escape detection."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video.annotation_config import TripleConeAnnotationConfig
from video.annotation_analysis.ball_position import calculate_horizontal_distance


@pytest.fixture
def config():
    return TripleConeAnnotationConfig()


def test_none_ball_returns_none(config):
    result = calculate_horizontal_distance(None, (500, 400), config)
    assert result is None


def test_none_hip_returns_none(config):
    result = calculate_horizontal_distance((500, 400), None, config)
    assert result is None


def test_safe_zone(config):
    # Ball 30px from hip (< 80px warning threshold)
    result = calculate_horizontal_distance((530, 400), (500, 400), config)
    assert result.zone == "SAFE"
    assert result.distance == 30.0
    assert result.color == config.H_DIST_SAFE_COLOR


def test_warning_zone(config):
    # Ball 100px from hip (80-120px)
    result = calculate_horizontal_distance((600, 400), (500, 400), config)
    assert result.zone == "WARNING"
    assert result.distance == 100.0
    assert result.color == config.H_DIST_WARNING_COLOR


def test_escaped_zone(config):
    # Ball 150px from hip (> 120px)
    result = calculate_horizontal_distance((650, 400), (500, 400), config)
    assert result.zone == "ESCAPED"
    assert result.distance == 150.0
    assert result.color == config.H_DIST_ESCAPED_COLOR


def test_distance_is_absolute(config):
    # Ball to LEFT of hip — distance should still be positive
    result = calculate_horizontal_distance((400, 400), (500, 400), config)
    assert result.distance == 100.0
    assert result.zone == "WARNING"


def test_exact_warning_boundary(config):
    # Exactly at warning threshold — should be WARNING
    result = calculate_horizontal_distance((580, 400), (500, 400), config)
    assert result.zone == "WARNING"
    assert result.distance == 80.0


def test_exact_escaped_boundary(config):
    # Exactly at escaped threshold — should be ESCAPED
    result = calculate_horizontal_distance((620, 400), (500, 400), config)
    assert result.zone == "ESCAPED"
    assert result.distance == 120.0
```

**Step 2: Run tests to verify they fail**

Run: `PYTHONPATH="." pytest tests/test_horizontal_distance.py -v`
Expected: FAIL with "cannot import name 'calculate_horizontal_distance'"

**Step 3: Implement function**

Add to end of `video/annotation_analysis/ball_position.py`:

```python
def calculate_horizontal_distance(
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    config: TripleConeAnnotationConfig
) -> Optional['HorizontalDistanceResult']:
    """
    Calculate horizontal distance between ball and player hip.

    Returns None if either position is unavailable.
    """
    if ball_center is None or hip_position is None:
        return None

    from ..annotation_data.structures import HorizontalDistanceResult

    distance = abs(ball_center[0] - hip_position[0])

    if distance >= config.H_DIST_ESCAPED_THRESHOLD:
        zone = "ESCAPED"
        color = config.H_DIST_ESCAPED_COLOR
    elif distance >= config.H_DIST_WARNING_THRESHOLD:
        zone = "WARNING"
        color = config.H_DIST_WARNING_COLOR
    else:
        zone = "SAFE"
        color = config.H_DIST_SAFE_COLOR

    return HorizontalDistanceResult(distance=distance, zone=zone, color=color)
```

Also add the import at the top of ball_position.py — no new import needed since we use a lazy import inside the function to avoid circular imports.

**Step 4: Add export to `__init__.py`**

In `video/annotation_analysis/__init__.py`, add to imports and `__all__`:

```python
from .ball_position import (
    determine_ball_position_relative_to_player,
    determine_torso_facing,
    determine_ball_position_vs_intention,
    calculate_horizontal_distance,
)

__all__ = [
    # Ball position
    'determine_ball_position_relative_to_player',
    'determine_torso_facing',
    'determine_ball_position_vs_intention',
    'calculate_horizontal_distance',
    # Tracking state
    'check_edge_zone_status',
    'update_ball_tracking_state',
]
```

**Step 5: Run tests to verify they pass**

Run: `PYTHONPATH="." pytest tests/test_horizontal_distance.py -v`
Expected: All 8 tests PASS

**Step 6: Commit**

```bash
git add video/annotation_analysis/ball_position.py video/annotation_analysis/__init__.py video/annotation_data/structures.py tests/test_horizontal_distance.py
git commit -m "feat: add calculate_horizontal_distance function with tests"
```

---

### Task 4: Add drawing function for horizontal distance counter

**Files:**
- Modify: `video/annotation_drawing/indicators.py` (add function after draw_vertical_deviation_counter)
- Modify: `video/annotation_drawing/__init__.py` (add export)

**Step 1: Add draw function**

Add to end of `video/annotation_drawing/indicators.py`:

```python
def draw_horizontal_distance_counter(
    frame: np.ndarray,
    count: int,
    is_active: bool,
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """Draw horizontal distance escape counter."""
    if count <= 0:
        return

    text = f"H-ESC: {count}f"
    x = x_offset + config.H_DIST_COUNTER_POS_X
    y = config.H_DIST_COUNTER_POS_Y

    color = config.H_DIST_ESCAPED_COLOR if is_active else config.H_DIST_PERSIST_COLOR

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.H_DIST_COUNTER_FONT_SCALE
    thickness = max(1, int(2 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x = max(2, int(5 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))
    pad_y = max(4, int(10 * getattr(config, 'FONT_SCALE_FACTOR', 1.0)))

    cv2.rectangle(frame, (x - pad_x, y - th - pad_y),
                  (x + tw + pad_x * 2, y + pad_y), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font,
                font_scale, color, thickness, cv2.LINE_AA)
```

**Step 2: Add export to `__init__.py`**

In `video/annotation_drawing/__init__.py`, add `draw_horizontal_distance_counter` to the imports from `.indicators` and to `__all__`.

**Step 3: Commit**

```bash
git add video/annotation_drawing/indicators.py video/annotation_drawing/__init__.py
git commit -m "feat: add draw_horizontal_distance_counter indicator"
```

---

### Task 5: Add field line drawing function

**Files:**
- Modify: `video/annotation_drawing/indicators.py` (add function)
- Modify: `video/annotation_drawing/__init__.py` (add export)

**Step 1: Add draw function**

Add to `video/annotation_drawing/indicators.py`:

```python
def draw_horizontal_distance_line(
    frame: np.ndarray,
    ball_center: Optional[Tuple[float, float]],
    hip_position: Optional[Tuple[float, float]],
    color: Tuple[int, int, int],
    config: TripleConeAnnotationConfig,
    x_offset: int = 0
) -> None:
    """
    Draw horizontal line from hip to ball at hip's Y position.

    Shows the horizontal separation between player and ball.
    Line is purely horizontal (at hip Y), colored by distance zone.
    """
    if ball_center is None or hip_position is None:
        return

    if any(pd.isna(v) for v in [ball_center[0], ball_center[1], hip_position[0], hip_position[1]]):
        return

    hip_x = int(hip_position[0]) + x_offset
    hip_y = int(hip_position[1])
    ball_x = int(ball_center[0]) + x_offset

    # Draw horizontal line at hip Y
    cv2.line(frame, (hip_x, hip_y), (ball_x, hip_y),
             color, config.H_DIST_LINE_THICKNESS, cv2.LINE_AA)

    # Draw small vertical tick marks at endpoints
    tick_height = max(4, int(8 * config.FONT_SCALE_FACTOR))
    cv2.line(frame, (hip_x, hip_y - tick_height), (hip_x, hip_y + tick_height),
             color, config.H_DIST_LINE_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (ball_x, hip_y - tick_height), (ball_x, hip_y + tick_height),
             color, config.H_DIST_LINE_THICKNESS, cv2.LINE_AA)
```

**Step 2: Add export**

Add `draw_horizontal_distance_line` to `video/annotation_drawing/__init__.py` imports and `__all__`.

**Step 3: Commit**

```bash
git add video/annotation_drawing/indicators.py video/annotation_drawing/__init__.py
git commit -m "feat: add draw_horizontal_distance_line indicator"
```

---

### Task 6: Wire everything into annotate_video.py main loop

**Files:**
- Modify: `video/annotate_video.py`

This task adds imports, state variables, per-frame logic, and drawing calls.

**Step 1: Add imports**

At the top of `annotate_video.py`, in the import block (around line 58-68), add `calculate_horizontal_distance` to the analysis imports and `draw_horizontal_distance_counter`, `draw_horizontal_distance_line` to the drawing imports.

In the `try` block for annotation_analysis imports:
```python
    from .annotation_analysis import (
        determine_ball_position_relative_to_player,
        determine_torso_facing,
        determine_ball_position_vs_intention,
        check_edge_zone_status,
        update_ball_tracking_state,
        calculate_horizontal_distance,
    )
```

In the `try` block for annotation_drawing imports:
```python
    from .annotation_drawing import (
        ...existing imports...
        draw_horizontal_distance_counter,
        draw_horizontal_distance_line,
    )
```

Also add the same to the `except ImportError` fallback blocks.

**Step 2: Add state variables**

After the vertical deviation tracking variables (around line 429), add:

```python
    # Horizontal distance escape tracking
    h_dist_escaped_counter: int = 0
    h_dist_display_value: int = 0
    h_dist_display_timer: int = 0
    h_dist_persist_frames = int(config.H_DIST_PERSIST_SECONDS * fps)
```

**Step 3: Add per-frame calculation**

After the vertical deviation detection block (around line 607), add:

```python
        # Horizontal distance escape detection
        h_dist_result = None
        if config.DRAW_HORIZONTAL_DISTANCE:
            h_dist_result = calculate_horizontal_distance(ball_center, current_hip, config)

            if h_dist_result and h_dist_result.zone == "ESCAPED":
                h_dist_escaped_counter += 1
                h_dist_display_value = h_dist_escaped_counter
                h_dist_display_timer = h_dist_persist_frames
            else:
                if h_dist_escaped_counter > 0:
                    h_dist_display_value = h_dist_escaped_counter
                    h_dist_display_timer = h_dist_persist_frames
                h_dist_escaped_counter = 0

            if h_dist_display_timer > 0:
                h_dist_display_timer -= 1
```

**Step 4: Add drawing calls**

After the vertical deviation counter drawing (around line 712), add:

```python
        # 14. Horizontal distance escape
        if config.DRAW_HORIZONTAL_DISTANCE:
            # Field line (hip to ball horizontal)
            if h_dist_result:
                draw_horizontal_distance_line(
                    canvas, ball_center, current_hip, h_dist_result.color,
                    config, x_offset=config.SIDEBAR_WIDTH
                )

            # Escaped counter
            if h_dist_display_timer > 0 or h_dist_escaped_counter > 0:
                draw_horizontal_distance_counter(
                    canvas, h_dist_display_value,
                    is_active=(h_dist_escaped_counter > 0),
                    config=config, x_offset=config.SIDEBAR_WIDTH
                )
```

**Step 5: Add H-Dist to the generic sidebar**

In `draw_sidebar_generic()` (around line 804, after the ball position text), add:

```python
    # Horizontal distance
    if hasattr(config, 'DRAW_HORIZONTAL_DISTANCE') and config.DRAW_HORIZONTAL_DISTANCE:
        # This will be populated by the caller - for now show placeholder
        # The actual value is drawn via the counter/line, not sidebar
        pass
```

Actually, looking at the existing sidebar more carefully — the sidebar already shows `Delta X` from `ball_position_result.ball_hip_delta_x`. Since `H-Dist` is just `abs(delta_x)`, the sidebar already provides the raw signed value. The new visualization adds the **field line** (visual on video area) and the **counter** (overlay on video area), which are both drawn directly on the canvas rather than in the sidebar. No sidebar changes needed.

**Step 6: Commit**

```bash
git add video/annotate_video.py
git commit -m "feat: wire horizontal distance escape into annotation loop"
```

---

### Task 7: Verify end-to-end with a test video

**Step 1: Run the annotator on a test player**

Pick any player with existing data. Run:

```bash
PYTHONPATH="." python video/annotate_video.py /path/to/drill_data/some_drill/some_player/
```

Or if using legacy layout:

```bash
PYTHONPATH="." python video/annotate_video.py --drills-dir /path/to/drill_data/ --list
```

Pick one player and annotate them.

**Step 2: Watch the output video**

Open the generated `debug_video.mp4` (or `_annotated.mp4`) and verify:
- Green/yellow/red horizontal line appears between hip and ball
- Line is horizontal (at hip Y level)
- `H-ESC: Xf` counter appears when ball is far away
- Counter persists briefly after ball returns to safe zone
- Colors match zones: green close, yellow moderate, red far

**Step 3: Run all tests**

```bash
PYTHONPATH="." pytest tests/ -v
```

Expected: All tests pass including the new `test_horizontal_distance.py`.

**Step 4: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "fix: horizontal distance escape fixups from visual verification"
```
