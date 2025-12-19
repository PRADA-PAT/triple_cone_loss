#!/usr/bin/env python3
"""
Manual Cone Annotation CLI for Figure-8 Drills.

Annotate single player:
    python annotate_cones.py abdullah_nasib

Annotate all unannotated videos:
    python annotate_cones.py --all

Show annotation status:
    python annotate_cones.py --status

Force re-annotation:
    python annotate_cones.py abdullah_nasib --force
"""
import sys
import argparse
from pathlib import Path

from cone_annotator import ConeAnnotator, get_annotation_status

# ============================================================
# CONFIGURATION - Same as run_detection.py
# ============================================================

VIDEO_DIR = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/videos")
PARQUET_BASE = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/f8_loss/video_detection_pose_ball_cones")

PLAYERS = {
    "abdullah_nasib": "abdullah_nasib_f8.MOV",
    "ali_buraq": "ali_buraq_f8.MOV",
    "archie_post": "archie_post_f8.MOV",
    "arjun_mital": "arjun_mital_f8.MOV",
    "arsen_said": "arsen_said_f8.MOV",
    "ava_peklar": "ava_peklar_f8.MOV",
    "cayden_kuforji": "cayden_kuforji_f8.MOV",
    "dameil_mendez": "dameil_mendez_f8.MOV",
    "dylan_white": "dylan_white_f8.MOV",
    "frederic_charbel": "frederic_charbel_f8.MOV",
    "haeley_anzaldo": "haeley_anzaldo_f8.MOV",
    "ismaail_ahmend": "ismaail_ahmend_f8.MOV",
    "lucas_correvon": "lucas_correvon_f8.MOV",
    "marwan_elazzouzi": "marwan_elazzouzi_f8.MOV",
    "maximillian_hall": "maximillian_hall.MOV",
    "maxwell_ross": "maxwell_ross_f8.MOV",
    "mike_basmadijan": "mike_basmadijan_f8.MOV",
    "miles_logon": "miles_logon_f8.MOV",
    "naomi_item": "naomi_item_f8.MOV",
    "noah_whyte": "noah_whyte_f8.MOV",
    "oliver_walsh": "oliver_walsh.MOV",
    "ollie_keefe": "ollie_keefe_f8.MOV",
    "omar_tariqu": "omar_tariqu_f8.MOV",
    "oscar_turner": "oscar_turner_f8.MOV",
    "poppy_henwoof": "poppy_henwoof.MOV",
    "riley_clemence": "riley_clemence.MOV",
    "shayne_saldanha": "shayne_saldanha_f8.MOV",
    "sonny_spicer": "sonny_spicer_f8.MOV",
}


def show_status():
    """Display annotation status for all players."""
    print("\n" + "=" * 60)
    print("CONE ANNOTATION STATUS")
    print("=" * 60)

    status = get_annotation_status(str(PARQUET_BASE), str(VIDEO_DIR), PLAYERS)

    annotated = []
    pending = []
    no_video = []

    for player, state in sorted(status.items()):
        if state == "annotated":
            annotated.append(player)
        elif state == "pending":
            pending.append(player)
        else:
            no_video.append(player)

    print(f"\nAnnotated ({len(annotated)}):")
    for p in annotated:
        print(f"  [OK] {p}")

    print(f"\nPending ({len(pending)}):")
    for p in pending:
        print(f"  [ ] {p}")

    if no_video:
        print(f"\nNo video ({len(no_video)}):")
        for p in no_video:
            print(f"  [X] {p}")

    print()
    print(f"Video directory: {VIDEO_DIR}")
    print(f"Parquet directory: {PARQUET_BASE}")
    print()


def get_parquet_dir(player_name: str) -> Path:
    """Get the parquet directory for a player (handles _f8 suffix)."""
    # Try with _f8 suffix first
    dir_f8 = PARQUET_BASE / f"{player_name}_f8"
    if dir_f8.exists():
        return dir_f8
    # Fall back to without suffix
    return PARQUET_BASE / player_name


def annotate_player(player_name: str, force: bool = False) -> bool:
    """
    Annotate a single player.

    Args:
        player_name: Player identifier
        force: If True, overwrite existing annotation

    Returns:
        True if annotation was saved
    """
    if player_name not in PLAYERS:
        print(f"Unknown player: {player_name}")
        print("Use --status to see available players")
        return False

    video_file = PLAYERS[player_name]
    video_path = VIDEO_DIR / video_file
    output_dir = get_parquet_dir(player_name)

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return False

    # Check existing annotation
    annotation_file = output_dir / "cone_annotations.json"
    if annotation_file.exists() and not force:
        print(f"Already annotated: {player_name}")
        print(f"  File: {annotation_file}")
        print("Use --force to re-annotate")
        return False

    # Run annotation
    print("\n" + "=" * 60)
    print(f"ANNOTATING: {player_name}")
    print("=" * 60)

    annotator = ConeAnnotator(str(video_path), str(output_dir))

    # If force, remove the existing check prompt
    if force and annotation_file.exists():
        annotation_file.unlink()

    return annotator.run()


def annotate_all():
    """Annotate all unannotated videos."""
    print("\n" + "=" * 60)
    print("BATCH CONE ANNOTATION")
    print("=" * 60)

    status = get_annotation_status(str(PARQUET_BASE), str(VIDEO_DIR), PLAYERS)

    pending = [p for p, s in status.items() if s == "pending"]

    if not pending:
        print("\nAll videos are already annotated!")
        show_status()
        return

    print(f"\nFound {len(pending)} videos to annotate")
    print("You can quit at any time with 'q'\n")

    completed = 0
    skipped = 0

    for i, player in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {player}")
        response = input("Annotate this video? [Y/n/q]: ").strip().lower()

        if response == 'q':
            print("\nStopping batch annotation")
            break
        elif response == 'n':
            skipped += 1
            continue

        if annotate_player(player):
            completed += 1
        else:
            skipped += 1

    print("\n" + "=" * 60)
    print("BATCH ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Remaining: {len(pending) - completed - skipped}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manual cone annotation for Figure-8 drills",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "player",
        nargs="?",
        help="Player name to annotate"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Annotate all unannotated videos"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show annotation status for all players"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force re-annotation even if already exists"
    )

    args = parser.parse_args()

    # Handle commands
    if args.status:
        show_status()
        return 0

    if args.all:
        annotate_all()
        return 0

    if args.player:
        success = annotate_player(args.player, args.force)
        return 0 if success else 1

    # No arguments - show help
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
