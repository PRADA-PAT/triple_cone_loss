# Handover: Multi-Drill Annotation System

**Date:** 2025-01-23
**Status:** Implementation complete, ready for testing

---

## What Was Built

A dynamic multi-drill annotation system that processes videos from a unified `drills/` folder structure with a single command.

### Key Features
- `--drills-dir` flag to specify drill folder location
- `--all` to process all players across all drill types
- `--force` to re-process existing outputs
- `--list` to see available drills/players
- Smart cone filtering: if more cones detected than config expects, keeps only the most frequently detected ones
- Backwards compatible with existing `video_detection_pose_ball_cones/` system

---

## Folder Structure

```
drills/
  {drill_type}/
    {player_name}/
      {any_name}.mp4 or .MOV          # Source video
      {any_name}_cone.parquet          # Cone detections
      {any_name}_football.parquet      # Ball tracking
      {any_name}_pose.parquet          # Pose keypoints
      {player_name}_annotated.mp4      # OUTPUT (generated)
```

**Current data:**
```
drills/
  chest_control_dribble/
    Khaled Al Kaddah _ChestDribble/    # Ready to annotate
  figure_of_8_drill/
    abdullah_nasib_f8/                  # Ready to annotate
```

---

## CLI Commands

```bash
# List all drills and players
python video/annotate_video.py --list --drills-dir drills/

# Annotate ALL drills, ALL players
python video/annotate_video.py --all --drills-dir drills/

# Re-process even if output exists
python video/annotate_video.py --all --drills-dir drills/ --force

# Single player folder
python video/annotate_video.py drills/chest_control_dribble/Khaled\ Al\ Kaddah\ _ChestDribble/

# Legacy mode (unchanged)
python video/annotate_video.py --list
python video/annotate_video.py --all
```

---

## Drill Configuration

**File:** `detection/drill_types_config.yaml`

### Currently Configured Drills

| Drill | Cones | Turn | Weave | Area | Pattern Match |
|-------|-------|------|-------|------|---------------|
| triple_cone | 3 | 3 | 0 | 0 | `_tc/`, `triple_cone` |
| seven_cone_weave | 7 | 2 | 5 | 0 | `7_cone_weave`, `seven_cone` |
| chest_control | 12 | 2 | 6 | 4 | `chest_control` |
| figure_of_8 | 5 | 2 | 0 | 3 (gate) | `figure_of_8`, `_f8` |

### Cone Types
- `turn` - Elliptical turning zone drawn (yellow highlight when ball inside)
- `weave` - Marker only, no zone
- `area` - Marker only, no zone
- `gate` - Marker only, no zone (used in figure_of_8)

### Adding New Drills

```yaml
  new_drill_name:
    name: "Human Readable Name"
    cone_count: N
    path_patterns:
      - "folder_pattern"
    cones:
      - position: 0
        type: turn
        label: cone_label
      # ... repeat for each cone, left-to-right
```

---

## Smart Cone Filtering

**Problem:** Detection may find more cones than the drill actually has (spurious detections).

**Solution:** If detected cones > config expects:
1. Count how many frames each cone appears in
2. Keep only the top N cones (most frequent = most reliable)
3. Match to config

**Code:** `video/annotation_data/loaders.py` → `load_all_cone_positions(path, max_cones=N)`

---

## Files Modified

| File | Changes |
|------|---------|
| `video/annotation_utils.py` | Added `PlayerFolder`, `DrillFolder` dataclasses, `get_drills_structure()` |
| `video/annotate_video.py` | Added `--drills-dir`, `--force`, list/batch modes, cone filtering |
| `video/annotation_data/loaders.py` | Added `max_cones` param for filtering |
| `detection/drill_types_config.yaml` | Updated chest_control to 12 cones, added figure_of_8 |

---

## Current State

### Ready to Test
```bash
python video/annotate_video.py --all --drills-dir drills/
```

This will process:
1. **chest_control_dribble/Khaled Al Kaddah _ChestDribble** - 12 cones, 2 turning zones
2. **figure_of_8_drill/abdullah_nasib_f8** - 5 cones (filtered from 6), 2 turning zones

### Git Status
- All changes committed and pushed to `origin/main`
- Latest commit: `0f5e935 feat: smart cone filtering when count exceeds config`

---

## Pending / Future Work

1. **Test annotation output** - Run `--all --drills-dir drills/` and verify video quality
2. **Add more drills** - Populate other folders in `drills/` with parquet data
3. **Migrate legacy data** - Optionally move `video_detection_pose_ball_cones/` players to `drills/triple_cone_drill/`
4. **Figure of 8 config** - Verify the 5-cone layout is correct (currently 2 turn + 3 gate)

---

## Design Documents

- `docs/plans/2025-01-23-multi-drill-annotation-design.md` - Architecture decisions
- `docs/plans/2025-01-23-multi-drill-annotation-impl.md` - Implementation plan (8 tasks)

---

## Quick Reference

```bash
# See what's available
python video/annotate_video.py --list --drills-dir drills/

# Process everything
python video/annotate_video.py --all --drills-dir drills/

# Check config
cat detection/drill_types_config.yaml
```
