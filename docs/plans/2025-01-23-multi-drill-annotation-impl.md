# Multi-Drill Annotation System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `--drills-dir` flag to annotate_video.py to process all players across all drill types from a unified drills/ folder.

**Architecture:** Add data classes and scanner function to annotation_utils.py, then integrate into annotate_video.py's argument parser and main logic. Backwards compatible - existing mode unchanged.

**Tech Stack:** Python, pathlib, dataclasses, argparse

---

## Task 1: Add Data Classes to annotation_utils.py

**Files:**
- Modify: `video/annotation_utils.py:1-10` (add imports and dataclasses at top)

**Step 1: Add imports and dataclasses**

Add at the top of `video/annotation_utils.py` after the existing imports:

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from detection.drill_config_loader import DrillConfigLoader


@dataclass
class PlayerFolder:
    """Represents a player's data folder within a drill."""
    name: str
    video_path: Path
    parquet_dir: Path
    has_output: bool


@dataclass
class DrillFolder:
    """Represents a drill type folder containing player folders."""
    drill_type: str      # from config or "unknown"
    drill_name: str      # human readable name
    drill_path: Path
    players: List[PlayerFolder]
```

**Step 2: Verify file still imports**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python -c "from video.annotation_utils import PlayerFolder, DrillFolder; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add video/annotation_utils.py
git commit -m "feat: add PlayerFolder and DrillFolder dataclasses"
```

---

## Task 2: Add get_drills_structure() Function

**Files:**
- Modify: `video/annotation_utils.py` (add function at end)

**Step 1: Add the scanner function**

Add at the end of `video/annotation_utils.py`:

```python
def get_drills_structure(drills_dir: Path, loader: 'DrillConfigLoader') -> List[DrillFolder]:
    """
    Scan drills folder and return structure of drill types and players.

    Expected structure:
        drills_dir/
          {drill_type_folder}/
            {player_folder}/
              *.mp4 (source video)
              *_cone.parquet
              *_football.parquet
              *_pose.parquet

    Args:
        drills_dir: Root directory containing drill type folders
        loader: DrillConfigLoader for detecting drill types

    Returns:
        List of DrillFolder objects, each containing list of PlayerFolder objects
    """
    results = []

    if not drills_dir.exists():
        return results

    for drill_path in sorted(drills_dir.iterdir()):
        if not drill_path.is_dir() or drill_path.name.startswith('.'):
            continue

        # Detect drill type from folder name
        drill_id = loader.detect_drill_type_from_path(str(drill_path))
        if drill_id:
            try:
                config = loader.get_drill_type(drill_id)
                drill_name = config.name
            except ValueError:
                drill_id = "unknown"
                drill_name = f"{drill_path.name} (unknown)"
        else:
            drill_id = "unknown"
            drill_name = f"{drill_path.name} (unknown)"

        players = []
        for player_path in sorted(drill_path.iterdir()):
            if not player_path.is_dir() or player_path.name.startswith('.'):
                continue

            # Find source video (exclude annotated outputs)
            videos = list(player_path.glob("*.mp4")) + list(player_path.glob("*.MOV"))
            source_videos = [v for v in videos if "_annotated" not in v.name.lower()]
            if not source_videos:
                continue

            # Check for required parquet files
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

**Step 2: Verify function imports correctly**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python -c "from video.annotation_utils import get_drills_structure; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add video/annotation_utils.py
git commit -m "feat: add get_drills_structure() scanner function"
```

---

## Task 3: Add --drills-dir and --force Arguments

**Files:**
- Modify: `video/annotate_video.py:802-845` (argument parser section)

**Step 1: Add new arguments after --resolution**

Find the `--resolution` argument (around line 838-843) and add these two new arguments right after it:

```python
    parser.add_argument(
        "--drills-dir",
        type=Path,
        help="Directory containing drill type folders (e.g., drills/)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-process videos even if output already exists"
    )
```

**Step 2: Verify arguments parse correctly**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python video/annotate_video.py --help | grep -A2 "drills-dir"`

Expected output should show the --drills-dir argument

**Step 3: Commit**

```bash
git add video/annotate_video.py
git commit -m "feat: add --drills-dir and --force CLI arguments"
```

---

## Task 4: Add Import for get_drills_structure

**Files:**
- Modify: `video/annotate_video.py:36-46` (import section)

**Step 1: Update the import block**

Find the import block that imports from annotation_utils (around line 37-38 in the try block). Update it to also import `get_drills_structure`:

Change:
```python
    from .annotation_utils import convert_to_h264, get_available_videos
```

To:
```python
    from .annotation_utils import convert_to_h264, get_available_videos, get_drills_structure
```

And in the except ImportError block (around line 76-77), change:
```python
    from annotation_utils import convert_to_h264, get_available_videos
```

To:
```python
    from annotation_utils import convert_to_h264, get_available_videos, get_drills_structure
```

**Step 2: Verify imports work**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python -c "from video.annotate_video import get_drills_structure; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add video/annotate_video.py
git commit -m "feat: import get_drills_structure in annotate_video"
```

---

## Task 5: Add --drills-dir List Mode

**Files:**
- Modify: `video/annotate_video.py:857-868` (list mode section)

**Step 1: Add drills-dir list handling**

Find the `if args.list:` block (around line 857). Replace the entire block with:

```python
    if args.list:
        if args.drills_dir:
            # New drills/ folder mode
            drill_folders = get_drills_structure(args.drills_dir, loader)
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
        else:
            # Legacy mode
            available = get_available_videos(args.videos_dir, args.parquet_dir)
            print("\n Available videos:\n")
            for name, video_path, parquet_path in available:
                drill_id = loader.detect_drill_type_from_path(str(parquet_path))
                drill_label = f"[{drill_id}]" if drill_id else "[unknown]"
                output_path = parquet_path / f"{name}_annotated.mp4"
                status = "[done]" if output_path.exists() else "[    ]"
                print(f"  {status} {name} {drill_label}")
            print(f"\nTotal: {len(available)} videos")
        return 0
```

**Step 2: Verify list mode works**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python video/annotate_video.py --list --drills-dir drills/`

Expected: Should show drill folders (may be empty if no data yet)

**Step 3: Commit**

```bash
git add video/annotate_video.py
git commit -m "feat: add --list support for --drills-dir mode"
```

---

## Task 6: Add --drills-dir Batch Processing Mode

**Files:**
- Modify: `video/annotate_video.py:870-908` (--all processing section)

**Step 1: Update the validation check**

Find the line `if not args.data_path and not args.all:` (around line 870). Change it to:

```python
    if not args.data_path and not args.all and not args.drills_dir:
        print("Error: Please specify a data path, use --list, use --all, or use --drills-dir")
        return 1
```

**Step 2: Add drills-dir batch processing**

Find the `if args.all:` block (around line 874). Replace the entire block with:

```python
    if args.all:
        if args.drills_dir:
            # New drills/ folder mode - process all drills, all players
            drill_folders = get_drills_structure(args.drills_dir, loader)
            total_players = sum(len(d.players) for d in drill_folders)
            print(f"\n Processing ALL {total_players} players across {len(drill_folders)} drills...\n")

            success_count = 0
            fail_count = 0
            skip_count = 0
            player_num = 0

            for drill in drill_folders:
                drill_config = None
                if drill.drill_type != "unknown":
                    try:
                        drill_config = loader.get_drill_type(drill.drill_type)
                    except ValueError:
                        pass

                for player in drill.players:
                    player_num += 1

                    if player.has_output and not args.force:
                        print(f"[{player_num}/{total_players}] Skipping {player.name} (already annotated)")
                        skip_count += 1
                        continue

                    print(f"\n{'='*60}")
                    print(f"[{player_num}/{total_players}] {drill.drill_path.name}/{player.name}")
                    print(f"{'='*60}")

                    if drill_config:
                        print(f"  Drill type: {drill_config.name}")

                    output_path = player.parquet_dir / f"{player.name}_annotated.mp4"
                    config = TripleConeAnnotationConfig()
                    success = annotate_video(player.video_path, player.parquet_dir, output_path, drill_config, config)

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1

            print(f"\n{'='*60}")
            print(f" BATCH COMPLETE")
            print(f"{'='*60}")
            print(f"  Processed: {success_count}")
            print(f"  Skipped:   {skip_count}")
            print(f"  Failed:    {fail_count}")

            return 0 if fail_count == 0 else 1
        else:
            # Legacy mode
            available = get_available_videos(args.videos_dir, args.parquet_dir)
            print(f"\n Processing ALL {len(available)} videos...\n")

            success_count = 0
            fail_count = 0

            for i, (name, video_path, parquet_path) in enumerate(available, 1):
                print(f"\n{'='*60}")
                print(f"[{i}/{len(available)}] Processing: {name}")
                print(f"{'='*60}")

                drill_id = args.drill_type or loader.detect_drill_type_from_path(str(parquet_path))
                drill_config = loader.get_drill_type(drill_id) if drill_id else None

                if drill_config:
                    print(f"  Drill type: {drill_config.name}")

                output_path = parquet_path / f"{name}_annotated.mp4"
                config = TripleConeAnnotationConfig()
                success = annotate_video(video_path, parquet_path, output_path, drill_config, config)

                if success:
                    success_count += 1
                else:
                    fail_count += 1

            print(f"\n{'='*60}")
            print(f" BATCH COMPLETE")
            print(f"{'='*60}")
            print(f"  Processed: {success_count}")
            print(f"  Failed:    {fail_count}")

            return 0 if fail_count == 0 else 1
```

**Step 3: Verify batch mode starts correctly**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python video/annotate_video.py --all --drills-dir drills/`

Expected: Should show "Processing ALL 0 players across 0 drills" (or similar if no data)

**Step 4: Commit**

```bash
git add video/annotate_video.py
git commit -m "feat: add --all batch processing for --drills-dir mode"
```

---

## Task 7: Add Single Player Processing in drills/ Mode

**Files:**
- Modify: `video/annotate_video.py:910-972` (single video processing section)

**Step 1: Update single video processing**

Find the `# Single video processing` comment and the `data_path = args.data_path` line (around line 910-911). Replace everything from there to the end of main() with:

```python
    # Single video/folder processing
    if args.data_path:
        data_path = args.data_path

        # If data_path is a directory, use it directly
        if data_path.is_dir():
            parquet_path = data_path

            # Find video file in the directory
            video_files = list(parquet_path.glob("*.mp4")) + list(parquet_path.glob("*.MOV"))
            source_videos = [v for v in video_files if "_annotated" not in v.name.lower()]

            if not source_videos:
                # Try parent directory for video (legacy mode)
                parent = parquet_path.parent
                name = parquet_path.name.replace("_tc", "").replace("_triple", "")
                source_videos = list(parent.glob(f"*{name}*.mp4"))

            if not source_videos:
                # Try videos directory (legacy mode)
                source_videos = list(args.videos_dir.glob(f"*{parquet_path.name}*.mp4"))

            if not source_videos:
                print(f"Error: Cannot find video for {parquet_path}")
                return 1

            video_path = source_videos[0]
        else:
            # Search by name in legacy mode
            available = get_available_videos(args.videos_dir, args.parquet_dir)
            search_term = str(data_path).lower()
            match = None
            for name, video_path, parquet_path in available:
                if search_term in name.lower():
                    match = (name, video_path, parquet_path)
                    break

            if not match:
                print(f"Error: Video not found matching: {data_path}")
                print("Use --list to see available videos")
                return 1

            name, video_path, parquet_path = match

        # Auto-detect or use explicit drill type
        drill_id = args.drill_type or loader.detect_drill_type_from_path(str(parquet_path))
        drill_config = None
        if drill_id:
            try:
                drill_config = loader.get_drill_type(drill_id)
                print(f"\n Drill type: {drill_config.name}")
            except ValueError:
                print(f"\n Warning: Unknown drill type '{drill_id}', using auto-detection")

        print(f"\n Annotating {video_path.name}...\n")

        output_path = parquet_path / f"{parquet_path.name}_annotated.mp4"
        config = TripleConeAnnotationConfig()
        success = annotate_video(video_path, parquet_path, output_path, drill_config, config)

        if success:
            print(f"\n Done! Output: {output_path}")
            return 0
        else:
            print("\n Annotation failed!")
            return 1

    # No data_path but drills_dir specified without --all - show help
    if args.drills_dir and not args.all:
        print("Error: Use --all with --drills-dir to process all, or specify a player folder path")
        print("Example: python video/annotate_video.py drills/7_cone_weave/player_x/")
        return 1

    return 0
```

**Step 2: Verify single folder processing works**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python video/annotate_video.py --help`

Expected: Help text should show all options

**Step 3: Commit**

```bash
git add video/annotate_video.py
git commit -m "feat: update single folder processing for drills/ compatibility"
```

---

## Task 8: Final Integration Test

**Files:**
- No file changes, just verification

**Step 1: Verify --list with drills-dir**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python video/annotate_video.py --list --drills-dir drills/`

Expected: Lists drill folders (may show 0 players if no parquet data yet)

**Step 2: Verify --all with drills-dir (dry run)**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python video/annotate_video.py --all --drills-dir drills/`

Expected: Shows "Processing ALL X players across Y drills"

**Step 3: Verify legacy mode still works**

Run: `cd "/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss" && python video/annotate_video.py --list`

Expected: Shows videos from video_detection_pose_ball_cones_720p/ (legacy mode)

**Step 4: Commit final state**

```bash
git add -A
git commit -m "feat: complete multi-drill annotation system

Adds support for --drills-dir flag to process all players
across all drill types from unified drills/ folder.

- Add PlayerFolder and DrillFolder dataclasses
- Add get_drills_structure() scanner function
- Add --drills-dir and --force CLI arguments
- Support --list and --all modes with drills-dir
- Backwards compatible with existing mode

Closes: multi-drill-annotation design"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add dataclasses | annotation_utils.py |
| 2 | Add scanner function | annotation_utils.py |
| 3 | Add CLI arguments | annotate_video.py |
| 4 | Add import | annotate_video.py |
| 5 | Add list mode | annotate_video.py |
| 6 | Add batch mode | annotate_video.py |
| 7 | Update single processing | annotate_video.py |
| 8 | Integration test | (verification only) |

Total: 8 tasks, ~7 commits
