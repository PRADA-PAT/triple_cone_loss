#!/usr/bin/env python3
"""
Manual Cone Annotation CLI for Triple Cone Drills.

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

VIDEO_DIR = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss/videos")
PARQUET_BASE = Path("/Users/pradyumn/Desktop/FOOTBALL data /AIM/triple_cone_loss/video_detection_pose_ball_cones")

PLAYERS = {
    "abdullah_nasib": "abdullah_nasib_tc.MOV",
    "ali_buraq": "ali_buraq_tc.MOV",
    "archie_post": "archie_post_tc.MOV",
    "arjun_mital": "arjun_mital_tc.MOV",
    "arsen_said": "arsen_said_tc.MOV",
    "ava_peklar": "ava_peklar_tc.MOV",
    "cayden_kuforji": "cayden_kuforji_tc.MOV",
    "dameil_mendez": "dameil_mendez_tc.MOV",
    "dylan_white": "dylan_white_tc.MOV",
    "frederic_charbel": "frederic_charbel_tc.MOV",
    "haeley_anzaldo": "haeley_anzaldo_tc.MOV",
    "ismaail_ahmend": "ismaail_ahmend_tc.MOV",
    "lucas_correvon": "lucas_correvon_tc.MOV",
    "marwan_elazzouzi": "marwan_elazzouzi_tc.MOV",
    "maximillian_hall": "maximillian_hall.MOV",
    "maxwell_ross": "maxwell_ross_tc.MOV",
    "mike_basmadijan": "mike_basmadijan_tc.MOV",
    "miles_logon": "miles_logon_tc.MOV",
    "naomi_item": "naomi_item_tc.MOV",
    "noah_whyte": "noah_whyte_tc.MOV",
    "oliver_walsh": "oliver_walsh.MOV",
    "ollie_keefe": "ollie_keefe_tc.MOV",
    "omar_tariqu": "omar_tariqu_tc.MOV",
    "oscar_turner": "oscar_turner_tc.MOV",
    "poppy_henwoof": "poppy_henwoof.MOV",
    "riley_clemence": "riley_clemence.MOV",
    "shayne_saldanha": "shayne_saldanha_tc.MOV",
    "sonny_spicer": "sonny_spicer_tc.MOV",
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
    """Get the parquet directory for a player (handles _tc suffix)."""
    # Try with _tc suffix first
    dir_tc = PARQUET_BASE / f"{player_name}"
    if dir_tc.exists():
        return dir_tc
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
        description="Manual cone annotation for Triple Cone drills",
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
