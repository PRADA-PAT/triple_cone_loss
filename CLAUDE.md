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

**Important**: Cone positions come from **parquet data** (mean positions across frames), sorted by X position left-to-right as CONE1, CONE2, CONE3. JSON annotation (`cone_annotations.json`) is an alternative source with pre-labeled roles.

## Package Structure

The codebase is organized into **3 clean folders**:

```
triple_cone_loss/
├── detection/                    # FOLDER 1: Loss of control calculation logic
│   ├── ball_control_detector.py  # Core detection engine with detect_loss()
│   ├── triple_cone_detector.py   # 3-cone phase tracking & turn detection
│   ├── data_structures.py        # Data models, enums, classes
│   ├── data_loader.py            # Parquet/JSON loading
│   ├── config.py                 # Configuration classes
│   ├── csv_exporter.py           # CSV export functionality
│   └── turning_zones.py          # Elliptical turning zones (CONE1, CONE2, CONE3)
│
├── annotation/                   # FOLDER 2: Cone annotation & visualization
│   ├── cone_annotator.py         # Interactive GUI annotation tool
│   ├── drill_visualizer.py       # Debug visualization (optional)
│   └── annotate_cones.py         # Annotation utilities
│
├── video/                        # FOLDER 3: Video generation with loss events
│   ├── annotate_triple_cone.py      # PRIMARY: Debug visualization (experimental features)
│   ├── annotate_with_json_cones.py  # Alternative: JSON-based cone positions
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
```

## Architecture

### Core Detection Pipeline

```
cone.parquet ──> load_triple_cone_layout_from_parquet() ──> TripleConeLayout
                              OR                                   |
cone_annotations.json ──> load_triple_cone_annotations()          |
                                                                   v
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
from detection.data_loader import load_triple_cone_layout_from_parquet, load_triple_cone_annotations

# Annotation tools
from annotation import ConeAnnotator, DrillVisualizer

# Video generation
from video.annotate_with_json_cones import annotate_video_with_json_cones

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
| `*_cone.parquet` | Parquet | Cone positions (primary source for 3-cone layout) | Pixel + Field |
| `cone_annotations.json` | JSON | Alternative: pre-labeled cone positions | Pixel (px, py) |
| `*_football.parquet` | Parquet | Ball positions per frame | Pixel + Field |
| `*_pose.parquet` | Parquet | 26 keypoints/person/frame (only ankles used) | Pixel + Field |

**JSON Cone Annotation Format (3-cone):**
```json
{
  "video": "player_name_tc.MOV",
  "cones": {
    "cone1": {"px": 467, "py": 801},
    "cone2": {"px": 1393, "py": 791},
    "cone3": {"px": 2316, "py": 778}
  }
}
```

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
- `video/annotate_with_json_cones.py`: Alternative using JSON cone annotations (static positions)
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
    cone_annotations.json
```

Some players use `{player_name}` without `_tc` suffix - `run_detection.py` handles both conventions.

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
