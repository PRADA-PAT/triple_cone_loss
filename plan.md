# Ball Control Loss Detection - Implementation Plan

## Objective

Implement multi-signal detection logic in the `detect_loss()` method to identify the **moment of mis-hit** during Figure-8 soccer drills, with **high precision** (minimize false alarms) and track loss until **stable control recovery**.

---

## User Requirements

| Requirement | Value |
|-------------|-------|
| Loss Onset | Moment of mis-hit (when bad kick contact occurs) |
| Loss Duration | Until player regains stable control |
| Sensitivity | High precision - only flag definite losses, minimize false alarms |

---

## Target File

**File to Modify**: `detection/ball_control_detector.py`

**Method to Modify**: `detect_loss()` (lines 328-386)

**Specific Lines to Replace**: Lines 380-386 (the current simple distance threshold logic)

**Current Implementation** (to be replaced):
```python
loss_threshold = self._detection_config.loss_distance_threshold  # 200.0

if distance > loss_threshold:
    return True, EventType.LOSS_DISTANCE

return False, None
```

---

## Available Imports and Data

**Imports Already Available** (line 17):
- `import numpy as np`

**FrameData Fields Available in `history` Parameter**:
```python
@dataclass
class FrameData:
    frame_id: int
    timestamp: float
    ball_x: float           # Ball pixel X position
    ball_y: float           # Ball pixel Y position
    ball_field_x: float     # Ball field X position
    ball_field_y: float     # Ball field Y position
    ball_velocity: float    # Ball velocity magnitude
    ankle_x: float          # Closest ankle pixel X
    ankle_y: float          # Closest ankle pixel Y
    ball_ankle_distance: float  # Pre-calculated distance
    drill_phase: Optional[DrillPhase]
    drill_direction: Optional[DrillDirection]
```

**EventType Enum Values Available**:
- `EventType.LOSS_DISTANCE` - Loss due to distance threshold
- `EventType.LOSS_VELOCITY` - Loss due to velocity anomaly
- `EventType.LOSS_DIRECTION` - Loss due to direction anomaly
- `EventType.BOUNDARY` - Ball out of frame

---

## Detection Algorithm Design

### The Mis-Hit Pattern (Observed from abdullah_nasib video)

The loss event occurs in a distinct sequence:

1. **Pre-hit hesitation**: Velocity drops as player prepares to turn (~14→7 px/frame)
2. **Bad contact (MIS-HIT)**: Velocity suddenly spikes as ball shoots away (~7→22 px/frame)
3. **Ball escaping**: Distance increases rapidly, ball moves in unexpected direction
4. **Extended loss**: Ball goes out of frame or stays far from player
5. **Recovery**: Player brings ball back under foot (ends loss event)

### Detection Signals and Weights

| Signal | Weight | Threshold | Description |
|--------|--------|-----------|-------------|
| Ball Missing (NaN) | Immediate | NaN check | Ball out of frame = definite loss |
| Velocity Spike Pattern | 0.35 | vmax/vmin > 2.5 | Hesitation followed by spike = mis-hit signature |
| Distance Acceleration | 0.35 | > 8 units/frame | Ball separating rapidly from player |
| Absolute Distance | 0.20 | > 250 units | Ball far from player (conservative) |
| Horizontal Escape | 0.10 | > 25 px/frame | Ball moving horizontally fast |

### Decision Rule

**Loss is detected when**: Combined weighted score > 0.55

This requires multiple signals to fire, ensuring high precision (few false positives).

---

## Complete Implementation Code

Replace lines 380-386 in `detect_loss()` method with the following code:

```python
        # ============================================================
        # MULTI-SIGNAL LOSS DETECTION (HIGH PRECISION)
        # Detects moment of mis-hit using velocity patterns and distance
        # ============================================================

        # Phase 1: Critical check - ball missing (out of frame)
        # If ball position is None or NaN, it's out of frame = definite loss
        if ball_pos is None:
            return True, EventType.BOUNDARY
        try:
            if np.isnan(ball_pos[0]) or np.isnan(ball_pos[1]):
                return True, EventType.BOUNDARY
        except (TypeError, IndexError):
            pass  # ball_pos might not be indexable as expected

        # Phase 2: Insufficient history - use conservative distance-only fallback
        # Need at least 15 frames of history for temporal pattern detection
        if len(history) < 15:
            # Very conservative threshold during cold start
            if distance > 300:
                return True, EventType.LOSS_DISTANCE
            return False, None

        # Phase 3: Extract temporal features from history
        recent = history[-15:]
        velocities = [h.ball_velocity for h in recent if h.ball_velocity is not None]
        distances = [h.ball_ankle_distance for h in recent if h.ball_ankle_distance is not None]
        ball_x_positions = [h.ball_x for h in recent if h.ball_x is not None]

        # Safety checks for empty lists
        if len(velocities) < 5 or len(distances) < 5 or len(ball_x_positions) < 5:
            # Fallback to simple distance check
            if distance > 250:
                return True, EventType.LOSS_DISTANCE
            return False, None

        # Feature A: Velocity spike ratio (mis-hit signature)
        # Pattern: hesitation (low velocity) → bad contact → ball shoots away (high velocity)
        # Looking at last 10 frames for minimum, last 5 for maximum (recent spike)
        min_vel_recent = min(velocities[-10:]) if len(velocities) >= 10 else min(velocities)
        max_vel_recent = max(velocities[-5:]) if len(velocities) >= 5 else max(velocities)
        velocity_spike_ratio = max_vel_recent / (min_vel_recent + 0.1)  # +0.1 to avoid div by zero

        # Feature B: Distance acceleration (ball separating from player)
        # Calculate average frame-to-frame distance change over last 10 frames
        distance_deltas = np.diff(distances[-10:]) if len(distances) >= 10 else np.diff(distances)
        distance_accel = float(np.mean(distance_deltas)) if len(distance_deltas) > 0 else 0.0

        # Feature C: Horizontal velocity magnitude (ball escaping sideways)
        # Looking at last 5 frames for immediate horizontal movement
        x_deltas = np.diff(ball_x_positions[-5:]) if len(ball_x_positions) >= 5 else np.diff(ball_x_positions)
        horizontal_vel = float(np.mean(np.abs(x_deltas))) if len(x_deltas) > 0 else 0.0

        # Phase 4: Compute weighted loss score (HIGH PRECISION thresholds)
        loss_score = 0.0
        event_type = None

        # Signal A: Velocity spike pattern (35% weight) - MIS-HIT SIGNATURE
        # A spike ratio > 2.5 means velocity more than doubled after hesitation
        # This is the primary indicator of a mis-hit
        if velocity_spike_ratio > 2.5:
            loss_score += 0.35
            event_type = EventType.LOSS_VELOCITY

        # Signal B: Distance acceleration (35% weight) - BALL SEPARATING
        # If distance is growing > 8 units/frame on average, ball is escaping
        if distance_accel > 8.0:
            # Scale contribution based on severity (cap at 1.0)
            contribution = 0.35 * min(distance_accel / 12.0, 1.0)
            loss_score += contribution
            event_type = event_type or EventType.LOSS_DISTANCE

        # Signal C: Absolute distance (20% weight) - BALL FAR FROM PLAYER
        # Conservative threshold of 250 (higher than default 200) for high precision
        if distance > 250:
            # Scale contribution based on how far over threshold
            contribution = 0.20 * min((distance - 250) / 100, 1.0)
            loss_score += contribution
            event_type = event_type or EventType.LOSS_DISTANCE

        # Signal D: Horizontal escape velocity (10% weight) - BALL SHOOTING SIDEWAYS
        # High horizontal velocity indicates ball escaping to the side
        if horizontal_vel > 25:
            loss_score += 0.10
            event_type = event_type or EventType.LOSS_DIRECTION

        # Phase 5: Decision with HIGH PRECISION threshold
        # Require score > 0.55 to flag as loss (need multiple signals to fire)
        # This ensures we only catch definite losses
        threshold = 0.55

        if loss_score > threshold:
            return True, event_type

        return False, None
```

---

## Validation and Testing

### Test Single Player First

```bash
cd /Users/pradyumn/Desktop/FOOTBALL\ data\ /AIM/f8_loss
python run_detection.py abdullah_nasib
```

**Expected Output**:
- Loss detected starting at approximately 23.5 seconds (tolerance ±0.5s)
- Loss should continue until ball recovered (~28.5s)
- Event type should be `LOSS_VELOCITY` or `LOSS_DISTANCE`

### Run Full Test Suite

```bash
python run_detection.py --test
```

**Success Criteria**:
- F1 Score > 0.70 across all 29 players
- Low false positive rate (high precision)
- Most real loss events detected (reasonable recall)

### Ground Truth Reference

The `ground_truth.csv` file contains:
- 29 players total
- 27 players with 1 loss event each
- 2 players with perfect control (no losses)
- Matching tolerance: ±0.5 seconds

---

## Threshold Tuning Guide

If the detection results need adjustment:

| Problem | Solution |
|---------|----------|
| Too many false positives (detecting losses that didn't happen) | Increase decision threshold from 0.55 → 0.60 or 0.65 |
| Missing real losses (false negatives) | Decrease decision threshold to 0.50 or lower individual signal thresholds |
| Detecting loss too early | Increase velocity spike ratio threshold from 2.5 → 3.0 |
| Detecting loss too late | Decrease velocity spike ratio threshold to 2.0 |
| Missing ball-out-of-frame events | Check NaN handling logic |
| Wrong event type being assigned | Adjust signal weights to prioritize correct detection type |

### Signal-Specific Tuning

| Signal | Current | More Sensitive | More Conservative |
|--------|---------|----------------|-------------------|
| Velocity Spike Ratio | > 2.5 | > 2.0 | > 3.0 |
| Distance Acceleration | > 8.0 | > 5.0 | > 12.0 |
| Absolute Distance | > 250 | > 200 | > 300 |
| Horizontal Velocity | > 25 | > 15 | > 35 |
| Decision Threshold | > 0.55 | > 0.45 | > 0.65 |

---

## Recovery Detection

Recovery is handled automatically by `_handle_state_change()` method (no changes needed).

Recovery occurs when:
- Ball-ankle distance drops below `control_radius` (120 units)
- Distance stays low for several consecutive frames
- This ends the loss event and marks `recovered = True`

---

## Files Summary

| File | Action |
|------|--------|
| `detection/ball_control_detector.py` | **MODIFY** - Replace lines 380-386 in `detect_loss()` |
| `ground_truth.csv` | READ ONLY - Reference for validation |
| `run_detection.py` | USE - Run detection and tests |

---

## Implementation Checklist

- [ ] Open `detection/ball_control_detector.py`
- [ ] Navigate to `detect_loss()` method (line 328)
- [ ] Find the current implementation (lines 380-386)
- [ ] Replace with the multi-signal detection code provided above
- [ ] Save the file
- [ ] Run `python run_detection.py abdullah_nasib` to test
- [ ] Verify loss detected at ~23.5s
- [ ] Run `python run_detection.py --test` for full validation
- [ ] Tune thresholds if needed using the guide above
