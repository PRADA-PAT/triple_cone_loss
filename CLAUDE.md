# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ball Control Detection System for Triple Cone Drills - analyzes video-derived parquet data to detect when a player loses control of the ball during Triple Cone soccer training drills.

**NOTE**: This is a self-contained package. All imports use local modules (detection/, annotation/, video/). Data is stored locally in videos/ and video_detection_pose_ball_cones/.

**Cone Arrangement (bird's eye view, 3 cones on horizontal line):**
```
Camera View (looking down from above):

     CONE1 (HOME)         CONE2 (CENTER)        CONE3 (RIGHT)
         o -------------------- o -------------------- o
      (467)                 (1393)                 (2316)

Y-coords: ~778-801px (all roughly same height - straight line)

Drill pattern (one repetition):
CONE1 → CONE2(turn) → CONE1(turn) → CONE3(turn) → CONE1(turn) → repeat

<==== BACKWARD (player runs left toward CONE1) ============>
<==================== FORWARD (player runs right) =========>
```

**Important**: Cone positions come from **parquet data** (mean positions across frames), sorted by X position left-to-right as CONE1, CONE2, CONE3.

## Package Structure

The codebase is organized into **3 clean folders**:

```
triple_cone_loss/
├── detection/                    # FOLDER 1: Loss of control calculation logic
│   ├── ball_control_detector.py  # Core detection engine with detect_loss()
│   ├── triple_cone_detector.py   # 3-cone phase tracking & turn detection
│   ├── data_structures.py        # Data models, enums, classes
│   ├── data_loader.py            # Parquet loading
│   ├── config.py                 # Configuration classes
│   ├── csv_exporter.py           # CSV export functionality
│   └── turning_zones.py          # Elliptical turning zones (CONE1, CONE2, CONE3)
│
├── annotation/                   # FOLDER 2: Visualization tools
│   └── drill_visualizer.py       # Debug visualization (optional)
│
├── video/                        # FOLDER 3: Video generation with loss events
│   ├── annotate_triple_cone.py      # PRIMARY: Debug visualization (experimental features)
│   ├── annotate_videos.py           # Basic overlay (parquet cones, no sidebar)
│   └── drill_event_tracker.py       # Cone crossing & turn event tracking
│
├── __init__.py                   # Root exports (backwards compatible)
├── run_detection.py              # Main starter script
├── main.py                       # CLI entry point
├── testing.py                    # Validation framework
└── tests/                        # Test suite
```

## Common Commands

```bash
# Run detection for a single player
python run_detection.py abdullah_nasib

# Run detection for all players
python run_detection.py --all

# List available players and their data status
python run_detection.py --list

# Run validation tests against ground truth
python run_detection.py --test

# Run tests (from within triple_cone_loss directory)
PYTHONPATH="." pytest tests/ -v

# Run a specific test file
PYTHONPATH="." pytest tests/test_detector.py -v

# Run tests with coverage
PYTHONPATH="." pytest tests/ --cov=detection

# Run validation with custom frame tolerance (default: 45 frames = 1.5s)
python run_detection.py --test --frame-tolerance 90  # 90 frames = 3.0s tolerance
```

## Testing & Validation

### Ground Truth Validation

The `--test` flag runs detection against `ground_truth.csv` and calculates precision/recall/F1.

```bash
python run_detection.py --test                      # Default 45 frame tolerance
python run_detection.py --test --frame-tolerance 90 # 3.0s tolerance (recommended)
```

### Frame Tolerance

Frame tolerance defines how close a detected event must be to ground truth to count as a True Positive.

| Tolerance | Frames | Seconds | Use Case |
|-----------|--------|---------|----------|
| Strict | 45 | 1.5s | Default, may miss valid detections |
| **Recommended** | **90** | **3.0s** | Accounts for annotation imprecision |
| Lenient | 120 | 4.0s | For exploratory analysis |

### Current Performance (as of 2026-01-13)

| Metric | 45 frames (1.5s) | 90 frames (3.0s) |
|--------|------------------|------------------|
| F1 Score | 20.4% | **32.7%** |
| Precision | 17.2% | 27.6% |
| Recall | 25.0% | 40.0% |
| True Positives | 5 | 8 |
| False Positives | 24 | 21 |
| False Negatives | 15 | 12 |

### Detection Gaps (Not Yet Detected)

These ground truth event types are **not currently detected** by the system:

| Pattern | Count | Examples | Notes |
|---------|-------|----------|-------|
| Overshoot/miss turn | 9 | Ball pushed too far, player misses cone | Needs new detector |
| Ball behind during turn | 2 | Ball gets behind player mid-turn | Turning zone suppression may hide these |
| Ball launched OOB | 2 | Player kicks ball out of frame | Some detected, timing mismatch |
| Cone crash | 1 | Player collides with cone | Needs cone proximity detector |
| Large radius turn | 1 | Takes wide path around cone | Needs path analysis |

### Test Output Files

After running `--test`, results are saved to `test_results/`:
- `test_summary.csv` - Per-player TP/FP/FN counts
- `test_events.csv` - All detected events with match status
- `LATEST_report.txt` - Detailed human-readable report
- `test_report_F1-XX.X_YYYYMMDD_HH-MM-SS.txt` - Timestamped reports

## Architecture

### Core Detection Pipeline

```
cone.parquet ──> load_triple_cone_layout_from_parquet() ──> TripleConeLayout
                                                                   |
ball.parquet ────────────────────────────────────────> BallControlDetector
                                                                   |
pose.parquet ──> extract_ankle_positions() ────────────────────────|
                                                                   |
                         ┌─────────────────────────────────────────┘
                         v
              TripleConeDetector (phase tracking, turn detection)
                         |
                         v
                  DetectionResult
                         |
                         v
                    CSVExporter
```

### Key Module Locations

**Detection entry points:**
- `detection/ball_control_detector.py` - `detect_ball_control()` convenience function
- `detection/ball_control_detector.py` - `BallControlDetector.detect()` main orchestrator

**Detection logic location:** The `detect_loss()` method in `detection/ball_control_detector.py` contains all loss detection logic. Modify ONLY this method to implement detection algorithms.

**BallControlDetector** delegates to:
- `detection/triple_cone_detector.py` - 3-cone phase tracking + turn detection
- `detection/data_loader.py` - Data preprocessing

### Imports

```python
# Direct from local detection module (recommended)
from detection import BallControlDetector, ControlState, AppConfig
from detection.data_loader import load_triple_cone_layout_from_parquet

# Visualization tools
from annotation import DrillVisualizer

# Turning zones (3-cone)
from detection.turning_zones import create_triple_cone_zones, TripleConeZoneSet, TripleConeZoneConfig

# Triple Cone detector
from detection.triple_cone_detector import TripleConeDetector, TurnEvent, DrillState
```

### Data Structures (`detection/data_structures.py`)

**Triple Cone Structures (3-cone):**
- `TripleConeLayout`: 3-cone positions (cone1/HOME, cone2/CENTER, cone3/RIGHT)
- `TripleConeDrillPhase`: Current phase in drill (AT_CONE1, GOING_TO_CONE2, etc.)

**Detection Structures:**
- `FrameData`: Per-frame analysis (positions, velocities, control scores, drill phase)
- `LossEvent`: Loss-of-control event with start/end frames, severity
- `DetectionResult`: Complete output container

**State Enums:**
- `ControlState`: CONTROLLED, TRANSITION, LOST, RECOVERING, UNKNOWN
- `DrillDirection`: FORWARD, BACKWARD, STATIONARY

**Turning Zone Structures (`detection/turning_zones.py`):**
- `TurningZone`: Elliptical zone with point-in-zone detection via ellipse equation
- `TripleConeZoneConfig`: Configuration for 3 zone sizes and camera perspective stretch factors
- `TripleConeZoneSet`: Container for CONE1, CONE2, CONE3 zones with convenience methods

### Configuration (`detection/config.py`)

Factory methods: `AppConfig.for_triple_cone()`, `.with_strict_detection()`, `.with_lenient_detection()`

Key detection thresholds in `DetectionConfig`:
- `control_radius`: 120.0 (normal control distance)
- `loss_distance_threshold`: 200.0 (distance indicating loss)
- `loss_velocity_spike`: 100.0 (velocity indicating sudden loss)
- `loss_duration_frames`: 5 (frames to confirm sustained loss)

## Data Flow

**Input Files:**

| File | Format | Purpose | Coordinates |
|------|--------|---------|-------------|
| `*_cone.parquet` | Parquet | Cone positions (3-cone layout from mean positions) | Pixel + Field |
| `*_football.parquet` | Parquet | Ball positions per frame | Pixel + Field |
| `*_pose.parquet` | Parquet | 26 keypoints/person/frame (only ankles used) | Pixel + Field |

**Output CSVs:**
- `loss_events.csv`: Detected loss events with timestamps and context
- `frame_analysis.csv`: Per-frame metrics and states

## Key Design Decisions

- **3-Cone Architecture**: Uses TripleConeLayout with CONE1 (HOME), CONE2 (CENTER), CONE3 (RIGHT)
- **Parquet-based Cone Detection**: Primary source is `*_cone.parquet` with mean positions sorted by X coordinate
- **Cone Proximity Zones**: Uses elliptical zones around each cone for phase detection (not gate line intersection)
- **Pixel Space for Zones**: Turning zone detection uses pixel coordinates
- **Field Space for Control**: Ball-ankle distance scoring uses field coordinates
- **Ankles only**: Uses ankle keypoints for ball-foot distance (`left_ankle`, `right_ankle`)
- **3-Folder Structure**: Code organized into detection/, annotation/, video/
- **Backwards Compatible**: Root `__init__.py` re-exports key classes
- **Control scoring**: Weighted combination: 60% distance, 25% velocity, 15% stability
- **Turn detection**: Direction reversal detected within elliptical turning zones
- **Visualization optional**: OpenCV import wrapped in try/except; detection works without it

## Development Workflow: Visualization-First Approach

### Philosophy
New detection features are developed using a **visualization-first** approach:
1. **Visualize**: Implement the feature in `video/annotate_triple_cone.py` first
2. **Validate**: Watch annotated videos to verify the logic "looks right"
3. **Promote**: If satisfied, rewrite the logic in `detection/ball_control_detector.py`
4. **Sync**: Maintain identical thresholds/logic in both files

### Why This Works
- Visual feedback is immediate and intuitive
- Easier to debug edge cases by watching video
- Detection logic can be ported cleanly once validated
- Dual maintenance is acceptable for portability

## Feature Development Stages

| Stage | Location | Description |
|-------|----------|-------------|
| **Experimental** | `video/annotate_triple_cone.py` only | Testing visually, not yet validated |
| **Validated** | `video/annotate_triple_cone.py` only | Looks correct, ready to promote |
| **Promoted** | Both viz + detection | In production, must stay in sync |

### Current Feature Status

| Feature | Stage | Viz File | Detection File |
|---------|-------|----------|----------------|
| Ball behind player | Promoted | `annotate_triple_cone.py` | `ball_control_detector.py` |
| Edge zone detection | Promoted | `annotate_triple_cone.py` | `ball_control_detector.py` |
| Momentum arrow | Viz-only | `annotate_triple_cone.py` | N/A (display only) |
| Behind counter | Viz-only | `annotate_triple_cone.py` | N/A (display only) |
| Torso facing direction | Promoted | `annotate_triple_cone.py` | N/A (viz helper) |
| Intention-based ball position | Promoted | `annotate_triple_cone.py` | N/A (viz helper) |
| Intention arrow | Viz-only | `annotate_triple_cone.py` | N/A (display only) |
| Intention behind counter | Viz-only | `annotate_triple_cone.py` | N/A (display only) |

## Adding Experimental Visualization Features

### Step-by-Step Guide

1. **Add configuration flags** in `TripleConeAnnotationConfig`:
   ```python
   # Feature toggle
   DRAW_NEW_FEATURE: bool = True

   # Feature thresholds
   NEW_FEATURE_THRESHOLD: float = 30.0
   ```

2. **Add state tracking** in main loop (if needed):
   ```python
   feature_state = None
   feature_history = deque(maxlen=10)
   ```

3. **Implement detection logic** as a helper function:
   ```python
   def determine_new_feature(persons: dict, config) -> Optional[str]:
       # Extract keypoints, calculate, return result
   ```

4. **Draw visualization** following layer order:
   - Add to appropriate layer (after pose skeleton, before counters)
   - Use consistent color coding
   - Add to sidebar if relevant

5. **Test visually** with multiple videos

6. **If validated**, port logic to `ball_control_detector.py`:
   - Add same thresholds as instance variables
   - Implement equivalent helper method
   - Update FrameData with new fields if needed

## Drill Phase Tracking

The `TripleConeDetector` tracks drill phases using proximity zones:

| Phase | Description |
|-------|-------------|
| `AT_CONE1` | Player at home cone (left) |
| `GOING_TO_CONE2` | Moving toward center cone |
| `AT_CONE2` | At center cone, turning |
| `RETURNING_FROM_CONE2` | Returning to home from center |
| `GOING_TO_CONE3` | Moving toward right cone |
| `AT_CONE3` | At right cone, turning |
| `RETURNING_FROM_CONE3` | Returning to home from right |
| `COMPLETED` | Drill finished |
| `UNKNOWN` | Phase not determined |

## CRITICAL: Visualization and Detection Logic Consistency

**The visualization logic in `video/annotate_triple_cone.py` MUST use the same thresholds and logic as the detection in `detection/ball_control_detector.py`.**

The video annotation serves as a **debug tool** - what you see in the annotated video (FRONT/BEHIND labels, colors) should match exactly what the detection algorithm calculates. If they differ, debugging becomes impossible.

**Synchronized thresholds:**

| Parameter | Detection (`ball_control_detector.py`) | Visualization (`annotate_triple_cone.py`) |
|-----------|----------------------------------------|------------------------------------------|
| Ball-behind threshold | `_behind_threshold = 20.0` | `BALL_POSITION_THRESHOLD = 20.0` |
| Movement threshold | `_movement_threshold = 3.0` | `MOVEMENT_THRESHOLD = 3.0` |
| Edge margin | `EDGE_MARGIN = 50` (in detect_loss) | `EDGE_MARGIN = 50` |
| Hip confidence | `>= 0.3` | `MIN_KEYPOINT_CONFIDENCE = 0.3` |

**When modifying detection logic:**
1. Update the threshold/logic in `detection/ball_control_detector.py`
2. Update the SAME threshold/logic in `video/annotate_triple_cone.py`
3. Regenerate annotated video to verify visually

## Video Generation Requirements

**IMPORTANT**: When generating annotated videos, always use H.264 codec for compatibility with VS Code and modern video players.

**OpenCV creates incompatible mp4v format** - must convert using ffmpeg:

```python
# OpenCV writes mp4v (incompatible with many players)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

# After writing, convert to H.264 with ffmpeg:
ffmpeg -y -i input.mp4 -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -movflags +faststart output.mp4
```

**Required ffmpeg parameters:**
- `-c:v libx264`: H.264 video codec (universal compatibility)
- `-preset medium`: Balance between speed and compression
- `-crf 23`: Good quality (18-28 range, lower = better)
- `-pix_fmt yuv420p`: Standard pixel format for compatibility
- `-movflags +faststart`: Move moov atom for web streaming

**Video annotation scripts:**
- `video/annotate_triple_cone.py`: **PRIMARY** - Full debug visualization with sidebar, turning zones, momentum arrows, counters. Use this for experimental features.
- `video/annotate_videos.py`: Basic overlay using parquet cone detection (per-frame positions, no sidebar)

## Generating Annotated Debug Videos

### Primary visualization (for experimental features):
```bash
# Generate annotated video with full debug overlay
python video/annotate_triple_cone.py <player_name>

# Example:
python video/annotate_triple_cone.py abdullah_nasib
```

### Output:
- Creates annotated video with sidebar, turning zones, ball position, momentum arrows
- Output saved to same directory as source video with `_annotated` suffix
- Auto-converts to H.264 for VS Code/browser compatibility

## Available Pose Keypoints

All 26 keypoints available in pose parquet for experimental features:

**Head/Face:** nose, head, left_eye, right_eye, left_ear, right_ear
**Torso:** neck, hip, left_hip, right_hip, left_shoulder, right_shoulder
**Arms:** left_elbow, right_elbow, left_wrist, right_wrist
**Legs:** left_knee, right_knee, left_ankle, right_ankle
**Feet:** left_heel, right_heel, left_big_toe, right_big_toe, left_small_toe, right_small_toe

**Currently used:** ankles (ball-foot distance), hip (movement direction, ball-behind)

## File Naming Convention

Player data follows pattern:
```
video_detection_pose_ball_cones/
  {player_name}_tc/
    {player_name}_tc_football.parquet
    {player_name}_tc_pose.parquet
    {player_name}_tc_cone.parquet
```

Some players use `{player_name}` without `_tc` suffix - `run_detection.py` handles both conventions.

## Detection Logic Deep Dive (For Porting to Other Applications)

This section provides detailed documentation of the loss-of-control detection algorithms, designed to be self-contained enough for porting to other applications.

### Overview of Detection Types

The system detects three types of loss events:

| Event Type | Enum Value | Description |
|------------|------------|-------------|
| `BOUNDARY_VIOLATION` | `boundary` | Ball exits video frame |
| `BALL_BEHIND_INTENTION` | `ball_behind_intention` | Ball behind player's facing direction (PRIMARY) |
| `BALL_BEHIND_PLAYER` | `ball_behind` | Ball behind player's movement direction (FALLBACK) |

---

### 1. Ball Behind Detection (Two Subtypes)

The system uses two methods to detect when the ball gets "behind" the player. **Intention-based** is checked first (PRIMARY), then **momentum-based** (FALLBACK).

#### 1.1 Intention-Based Detection (`BALL_BEHIND_INTENTION`) - PRIMARY

**Concept**: Detects loss when ball is behind where the player is **LOOKING** (body orientation), not where they're moving.

**Why it's better**: Captures turn intention since head/torso turns before body moves. Detects loss earlier and more accurately during direction changes.

**Algorithm**:

```
INPUTS:
  - ball_pixel_pos: (x, y) position of ball in pixels
  - hip_pixel_pos: (x, y) position of player's hip center
  - nose_pixel_pos: (x, y) position of player's nose

STEP 1: Determine Facing Direction (from nose-hip vector)
  diff = nose_x - hip_x
  if diff > NOSE_HIP_FACING_THRESHOLD (5.0px):
      facing_direction = "RIGHT"  # Head is to right of body = facing right
  elif diff < -NOSE_HIP_FACING_THRESHOLD:
      facing_direction = "LEFT"   # Head is to left of body = facing left
  else:
      facing_direction = None     # Neutral/ambiguous

STEP 2: Determine Ball Position Relative to Facing
  delta_x = ball_x - hip_x

  if abs(delta_x) < BEHIND_THRESHOLD (9.0px):
      position = "I-ALIGNED"   # Ball directly at player
      is_behind = False
  elif facing_direction == "LEFT":
      if delta_x < 0:
          position = "I-FRONT"  # Ball to left, facing left = FRONT
          is_behind = False
      else:
          position = "I-BEHIND" # Ball to right, facing left = BEHIND
          is_behind = True
  elif facing_direction == "RIGHT":
      if delta_x > 0:
          position = "I-FRONT"  # Ball to right, facing right = FRONT
          is_behind = False
      else:
          position = "I-BEHIND" # Ball to left, facing right = BEHIND
          is_behind = True

STEP 3: Check for Sustained Pattern (temporal filtering)
  - Look at last N frames (INTENTION_SUSTAINED_FRAMES = 12, ~0.4s at 30fps)
  - Count consecutive frames where ball_behind_intention == True
  - Also verify facing direction was consistent (70% threshold)
  - If sustained → trigger BALL_BEHIND_INTENTION event
```

**Key Constants (720p)**:
```python
NOSE_HIP_FACING_THRESHOLD = 5.0   # Min nose-hip offset to determine facing
BEHIND_THRESHOLD = 9.0            # Min ball-hip offset to be "behind"
INTENTION_SUSTAINED_FRAMES = 12   # ~0.4s at 30fps to confirm
MIN_KEYPOINT_CONFIDENCE = 0.3     # Pose keypoint confidence threshold
```

**Pose Keypoints Required**:
- `nose`: (x, y, confidence) - for facing direction
- `hip`: (x, y, confidence) - body center reference
- Fallback: `left_hip` + `right_hip` averaged if `hip` not available

---

#### 1.2 Momentum-Based Detection (`BALL_BEHIND_PLAYER`) - FALLBACK

**Concept**: Detects loss when ball is behind where the player is **MOVING** (velocity direction).

**When used**: Only checked if intention-based detection didn't trigger (nose data unavailable or inconsistent).

**Algorithm**:

```
INPUTS:
  - ball_pixel_pos: (x, y) position of ball
  - hip_pixel_pos: (x, y) position of hip
  - hip_history: deque of last 15 hip positions (~0.5s at 30fps)
  - in_turning_zone: "CONE1"/"CONE2"/"CONE3" or None

STEP 1: Calculate Movement Direction from Hip History
  if len(hip_history) >= 2:
      prev_hip = hip_history[0]  # Oldest
      curr_hip = hip_history[-1] # Current
      dx = curr_hip.x - prev_hip.x

      if dx > MOVEMENT_THRESHOLD (1.4px):
          player_direction = "RIGHT"  # Moving rightward
      elif dx < -MOVEMENT_THRESHOLD:
          player_direction = "LEFT"   # Moving leftward
      else:
          player_direction = None     # Stationary (use last known direction)

STEP 2: Check if Ball is Behind Movement Direction
  delta_x = ball_x - hip_x

  if player_direction == "LEFT":
      # Moving left: ball to RIGHT of hip = BEHIND
      is_behind = delta_x > BEHIND_THRESHOLD (9.0px)
  elif player_direction == "RIGHT":
      # Moving right: ball to LEFT of hip = BEHIND
      is_behind = delta_x < -BEHIND_THRESHOLD

STEP 3: Skip if in Turning Zone
  # Ball being "behind" is expected during turns
  if in_turning_zone is not None:
      return False, None  # No loss detected

STEP 4: Check for Sustained Pattern
  - Look at last N frames (BEHIND_SUSTAINED_FRAMES = 10, ~0.33s at 30fps)
  - Count consecutive frames where ball_behind_player == True
  - Also verify movement direction was consistent (70% threshold)
  - If sustained → trigger BALL_BEHIND_PLAYER event
```

**Key Constants (720p)**:
```python
BEHIND_THRESHOLD = 9.0           # Min ball-hip offset (pixels)
MOVEMENT_THRESHOLD = 1.4         # Min hip movement to determine direction (pixels)
BEHIND_SUSTAINED_FRAMES = 10     # ~0.33s at 30fps to confirm
HIP_HISTORY_SIZE = 15            # ~0.5s history for direction calculation
```

**Important**: Momentum-based detection is **suppressed in turning zones** because the ball naturally falls "behind" during turns.

---

### 2. Boundary/Edge Detection (`BOUNDARY_VIOLATION`)

**Concept**: Detects when the ball exits the video frame by tracking ball visibility near screen edges.

**Key Insight**: Uses the `interpolated` flag from ball detection. When `interpolated=True`, the ball wasn't actually detected (position was filled in by the tracker). This signals the ball went off-screen.

#### State Machine (`BallTrackingState`)

```
States:
  NORMAL           - Ball visible, not near edge
  EDGE_LEFT        - Ball in left edge zone (warning)
  EDGE_RIGHT       - Ball in right edge zone (warning)
  OFF_SCREEN_LEFT  - Ball disappeared via left edge (loss!)
  OFF_SCREEN_RIGHT - Ball disappeared via right edge (loss!)
  DISAPPEARED_MID  - Ball disappeared mid-field (detection failure, ignore)

Transitions:
  NORMAL → EDGE_LEFT/RIGHT    (ball enters edge zone)
  EDGE_LEFT → OFF_SCREEN_LEFT (ball disappears while in left edge)
  EDGE_RIGHT → OFF_SCREEN_RIGHT (ball disappears while in right edge)
  OFF_SCREEN_* → NORMAL       (ball reappears)
  NORMAL → DISAPPEARED_MID    (ball disappears mid-field - not a loss)
```

**Algorithm**:

```
INPUTS:
  - ball_pixel_pos: (x, y) current ball position
  - ball_interpolated: True if ball wasn't actually detected this frame
  - video_width: Width of video in pixels

CONSTANTS:
  EDGE_MARGIN = 50            # Pixels from edge to define "edge zone"
  BOUNDARY_SUSTAINED_FRAMES = 15  # ~0.5s at 30fps to confirm

STEP 1: Determine Ball Visibility
  ball_visible = NOT ball_interpolated

STEP 2: Check Edge Zone Status
  left_distance = ball_x
  right_distance = video_width - ball_x

  if right_distance < EDGE_MARGIN:
      in_edge_zone = True, edge_side = "RIGHT"
  elif left_distance < EDGE_MARGIN:
      in_edge_zone = True, edge_side = "LEFT"
  else:
      in_edge_zone = False, edge_side = "NONE"

STEP 3: Update State Machine
  prev_state = current_state

  if NOT ball_visible:
      # Ball disappeared - check if was in edge zone
      if prev_state == EDGE_LEFT:
          current_state = OFF_SCREEN_LEFT
      elif prev_state == EDGE_RIGHT:
          current_state = OFF_SCREEN_RIGHT
      elif prev_state in (OFF_SCREEN_LEFT, OFF_SCREEN_RIGHT):
          # Stay in off-screen state
          pass
      else:
          # Disappeared mid-field - detection failure, not a loss
          current_state = DISAPPEARED_MID
          reset_counter()
  else:
      # Ball is visible
      if in_edge_zone:
          current_state = EDGE_LEFT or EDGE_RIGHT (based on edge_side)
      else:
          current_state = NORMAL
          reset_counter()

STEP 4: Increment Counter and Check Threshold
  if current_state in (OFF_SCREEN_LEFT, OFF_SCREEN_RIGHT):
      boundary_counter += 1
      if boundary_counter >= BOUNDARY_SUSTAINED_FRAMES:
          trigger BOUNDARY_VIOLATION event
  elif current_state in (EDGE_LEFT, EDGE_RIGHT):
      boundary_counter += 1  # Preparing for potential exit
```

**Key Constants**:
```python
EDGE_MARGIN = 50                 # Pixels from screen edge
BOUNDARY_SUSTAINED_FRAMES = 15   # ~0.5s at 30fps
MIN_TIMESTAMP = 3.0              # Skip first 3 seconds (setup)
```

**Critical Requirement**: The ball tracker must provide an `interpolated` flag (or equivalent) indicating whether the ball was actually detected or its position was estimated/filled-in.

---

### 3. Turning Zone Suppression

The system uses elliptical "turning zones" around each cone to suppress false positives during turns.

**Why needed**: During a turn, the ball naturally gets "behind" the player briefly. Without suppression, every turn would trigger `BALL_BEHIND_PLAYER`.

**Implementation**:
```
# Elliptical zone check
def is_point_in_turning_zone(point_x, point_y, zone):
    dx = point_x - zone.center_x
    dy = point_y - zone.center_y
    # Ellipse equation: (dx/rx)² + (dy/ry)² <= 1
    return (dx/zone.radius_x)² + (dy/zone.radius_y)² <= 1.0
```

**Note**: Intention-based detection does NOT use turning zone suppression because facing direction should be tracked even during turns.

---

### 4. Porting Checklist

To port this detection logic to another application:

**Required Inputs**:
1. Ball position per frame (x, y in pixels)
2. Ball visibility flag (`interpolated` or equivalent)
3. Player hip position per frame (or left_hip + right_hip)
4. Player nose position per frame (for intention-based detection)
5. Pose keypoint confidence values
6. Video dimensions (width, height)

**Optional Inputs** (for turning zone suppression):
1. Cone/marker positions (for defining turning zones)
2. Turning zone radii

**Key Functions to Implement**:
1. `determine_facing_direction(nose_pos, hip_pos) → "LEFT"/"RIGHT"/None`
2. `is_ball_behind_intention(ball_pos, hip_pos, facing) → (bool, position_str)`
3. `is_ball_behind_momentum(ball_pos, hip_pos, movement_dir) → bool`
4. `update_ball_tracking_state(visible, in_edge, edge_side) → new_state`

**Coordinate System Assumptions**:
- X increases left-to-right (0 = left edge, video_width = right edge)
- Y increases top-to-bottom (typical video coordinates)
- All positions in pixel space (not normalized)

---

## Migration Notes (5-cone → 3-cone)

The codebase was migrated from a 5-cone gate-based architecture to a 3-cone proximity-based architecture:

| Before (5-cone) | After (3-cone) |
|-----------------|----------------|
| `DrillPhase` | `TripleConeDrillPhase` |
| `DrillLayout` | `TripleConeLayout` |
| `TurningZoneSet` | `TripleConeZoneSet` |
| `create_turning_zones()` | `create_triple_cone_zones()` |
| `load_cone_annotations()` | `load_triple_cone_layout_from_parquet()` |
| Gate line intersection | Cone proximity zones |
| `GatePassage` structure | Removed (no gates) |
| `ConeRole`, `ConeAnnotation` | Removed (simplified) |
| `gate_passages.csv` output | Removed |
| `cone_roles.csv` output | Removed |
