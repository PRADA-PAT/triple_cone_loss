# Multi-Drill Annotation System Design

**Date:** 2025-01-23
**Status:** Approved

## Overview

Extend the annotation system to support multiple drill types from a unified `drills/` folder structure, with a single command to annotate all players across all drills.

## Folder Structure

```
drills/
  triple_cone_drill/
    abdullah_nasib/
      abdullah_nasib.mp4                 # Source video
      abdullah_nasib_cone.parquet        # Cone detections
      abdullah_nasib_football.parquet    # Ball tracking
      abdullah_nasib_pose.parquet        # Pose keypoints
      abdullah_nasib_annotated.mp4       # Output (generated)
    player_2/
      ...

  7_cone_weave/
    player_x/
      ...

  figure_of_8_drill/
    ...

  chest_control_dribble/
    ...
```

### Conventions

- **Drill type detection:** Folder name maps to `drill_types_config.yaml` via `path_patterns`
- **Parquet naming:** Files must end with `_cone.parquet`, `_football.parquet`, `_pose.parquet`
- **Video:** Any `.mp4` in the player folder (first one found)
- **Output:** Same folder as source, named `{name}_annotated.mp4`
- **Max videos per folder:** ~2 (small scale)

## CLI Interface

### Commands

```bash
# Annotate ALL drills, ALL players
python video/annotate_video.py --all --drills-dir drills/

# List what would be processed
python video/annotate_video.py --list --drills-dir drills/

# Single player in drills folder
python video/annotate_video.py drills/7_cone_weave/player_x/

# Explicit drill type override (if auto-detect fails)
python video/annotate_video.py drills/some_folder/ --drill-type seven_cone_weave
```

### List Output Example

```
Available drills in drills/:

triple_cone_drill/ (3 cones, turn/turn/turn)
  [done] abdullah_nasib
  [    ] player_2

7_cone_weave/ (7 cones, turn/weave/weave/weave/weave/weave/turn)
  [    ] player_x

figure_of_8_drill/ (unknown - not in config)
  [    ] player_y

Total: 4 players across 3 drill types
```

### Behavior

- `--all` without `--drills-dir`: Uses existing `video_detection_pose_ball_cones/` (backwards compatible)
- `--all --drills-dir drills/`: Uses new structure
- Skips folders with existing `*_annotated.mp4` (use `--force` to re-process)

## Drill Configuration

Central config at `detection/drill_types_config.yaml`. Expand as new drills are added.

### Cone Types

| Type | Behavior | Visual |
|------|----------|--------|
| `turn` | Elliptical turning zone drawn | Colored ellipse |
| `weave` | No zone, just marker | Cone marker only |
| `area` | No zone, just marker | Cone marker only |

### Adding New Drills

```yaml
drill_types:
  # Add new drill:
  figure_of_8:
    name: "Figure of 8 Drill"
    cone_count: 2
    path_patterns:
      - "figure_of_8"
      - "fig8"
    cones:
      - position: 0
        type: turn
        label: cone_left
      - position: 1
        type: turn
        label: cone_right
```

### Fallback Behavior

If drill folder name not in config: All detected cones treated as `turn` type with generic labels (`cone_1`, `cone_2`, etc.)

## Initial Drills Supported

| Drill | Folder Name | Cones | Config Status |
|-------|-------------|-------|---------------|
| Triple Cone | `triple_cone_drill/` | 3 (all turn) | Already in config |
| 7 Cone Weave | `7_cone_weave/` | 7 (2 turn, 5 weave) | Already in config |
| Chest Control | `chest_control_dribble/` | 5 (1 turn, 4 area) | Already in config |
| Figure of 8 | `figure_of_8_drill/` | TBD | Add later |

## Code Changes

### Files to Modify

| File | Change |
|------|--------|
| `video/annotate_video.py` | Add `--drills-dir` flag, scan drill folders, find video+parquets |
| `video/annotation_utils.py` | Add `get_drills_structure()` helper to scan drills folder |

### No Changes Needed

- `detection/drill_config_loader.py` (already supports path pattern matching)
- Core annotation logic (already handles N cones)
- Drawing functions
- Detection modules

### New Helper Function

Add to `annotation_utils.py`:

```python
@dataclass
class PlayerFolder:
    name: str
    video_path: Path
    parquet_dir: Path
    has_output: bool

@dataclass
class DrillFolder:
    drill_type: str  # from config or "unknown"
    drill_name: str  # human readable
    drill_path: Path
    players: List[PlayerFolder]

def get_drills_structure(drills_dir: Path, loader: DrillConfigLoader) -> List[DrillFolder]:
    """
    Scan drills folder and return structure.

    Structure:
      drills_dir/
        {drill_type_folder}/
          {player_folder}/
            *.mp4
            *_cone.parquet
            *_football.parquet
            *_pose.parquet
    """
    results = []

    for drill_path in sorted(drills_dir.iterdir()):
        if not drill_path.is_dir() or drill_path.name.startswith('.'):
            continue

        # Detect drill type from folder name
        drill_id = loader.detect_drill_type_from_path(str(drill_path))
        if drill_id:
            config = loader.get_drill_type(drill_id)
            drill_name = config.name
        else:
            drill_id = "unknown"
            drill_name = f"{drill_path.name} (unknown)"

        players = []
        for player_path in sorted(drill_path.iterdir()):
            if not player_path.is_dir() or player_path.name.startswith('.'):
                continue

            # Find video
            videos = list(player_path.glob("*.mp4"))
            source_videos = [v for v in videos if "_annotated" not in v.name]
            if not source_videos:
                continue

            # Check for parquets
            has_cone = bool(list(player_path.glob("*_cone.parquet")))
            has_ball = bool(list(player_path.glob("*_football.parquet")))
            has_pose = bool(list(player_path.glob("*_pose.parquet")))

            if not (has_cone and has_ball and has_pose):
                continue

            # Check for existing output
            has_output = bool(list(player_path.glob("*_annotated.mp4")))

            players.append(PlayerFolder(
                name=player_path.name,
                video_path=source_videos[0],
                parquet_dir=player_path,
                has_output=has_output
            ))

        if players:
            results.append(DrillFolder(
                drill_type=drill_id,
                drill_name=drill_name,
                drill_path=drill_path,
                players=players
            ))

    return results
```

### Main Logic Update

In `annotate_video.py` main():

```python
if args.drills_dir:
    # New drills/ folder mode
    drill_folders = get_drills_structure(args.drills_dir, loader)

    if args.list:
        print(f"\nAvailable drills in {args.drills_dir}/:\n")
        total_players = 0
        for drill in drill_folders:
            print(f"{drill.drill_path.name}/ ({drill.drill_name})")
            for player in drill.players:
                status = "[done]" if player.has_output else "[    ]"
                print(f"  {status} {player.name}")
                total_players += 1
            print()
        print(f"Total: {total_players} players across {len(drill_folders)} drill types")
        return 0

    if args.all:
        for drill in drill_folders:
            drill_config = loader.get_drill_type(drill.drill_type) if drill.drill_type != "unknown" else None
            for player in drill.players:
                if player.has_output and not args.force:
                    print(f"Skipping {player.name} (already annotated)")
                    continue
                output_path = player.parquet_dir / f"{player.name}_annotated.mp4"
                annotate_video(player.video_path, player.parquet_dir, output_path, drill_config, config)
else:
    # Existing mode (backwards compatible)
    ...
```

## Migration Notes

- Existing `video_detection_pose_ball_cones/` system remains unchanged
- New `drills/` folder is independent
- Manual migration of existing triple cone data when ready
- Both systems can coexist

## Data Format

Each player folder must contain:
- One `.mp4` video file (source)
- `*_cone.parquet` - cone positions per frame
- `*_football.parquet` - ball tracking per frame
- `*_pose.parquet` - pose keypoints per frame

Same format as existing `video_detection_pose_ball_cones/` structure.
