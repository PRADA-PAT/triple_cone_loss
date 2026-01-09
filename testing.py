"""
Testing module for Triple Cone Ball Control Detection validation.

Compares detected loss events against manually annotated ground truth
with configurable time tolerance.

Provides detailed reporting including:
- Event type breakdown (BOUNDARY_VIOLATION vs BALL_BEHIND_PLAYER)
- Per-player detection analysis
- Overall precision/recall/F1 metrics

Usage:
    from testing import run_validation_suite, load_ground_truth

    ground_truth = load_ground_truth("ground_truth.csv")
    summary = run_validation_suite(ground_truth, tolerance=0.5)
    print(generate_report(summary))
"""
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter

try:
    from .detection.data_structures import LossEvent, EventType
except ImportError:
    from detection.data_structures import LossEvent, EventType

logger = logging.getLogger(__name__)

# Constants
# Default frame tolerance to account for human annotation imprecision.
# 45 frames ≈ 1.5s at 30fps. Manual video annotation is typically off by 1-2 seconds due to:
# - Subjective perception of when "loss" starts
# - Reaction time delay when marking timestamps
# - Frame-by-frame detection vs. human perception
DEFAULT_FRAME_TOLERANCE = 45  # frames (≈1.5s at 30fps)
DEFAULT_TOLERANCE = 1.5  # seconds (legacy, for backwards compatibility)


@dataclass
class MatchedEvent:
    """A detected event matched to ground truth."""
    detected_time: float
    ground_truth_time: float
    time_difference: float  # detected - ground_truth
    # Frame-based matching (primary for debugging)
    detected_frame: Optional[int] = None
    ground_truth_frame: Optional[int] = None
    frame_difference: Optional[int] = None  # detected - ground_truth
    event_type: Optional[EventType] = None  # Type of detection that matched
    loss_event: Optional[LossEvent] = None  # Full event object for detailed info

    @property
    def event_type_name(self) -> str:
        """Human-readable event type name."""
        if self.event_type is None:
            return "UNKNOWN"
        type_names = {
            EventType.BOUNDARY_VIOLATION: "OUT_OF_BOUNDS",
            EventType.BALL_BEHIND_PLAYER: "BALL_BEHIND",
            EventType.LOSS_DISTANCE: "DISTANCE",
            EventType.LOSS_VELOCITY: "VELOCITY",
            EventType.LOSS_DIRECTION: "DIRECTION",
        }
        return type_names.get(self.event_type, self.event_type.value.upper())


@dataclass
class DetectedEventInfo:
    """Information about a detected event (for false positives)."""
    timestamp: float
    event_type: EventType
    loss_event: LossEvent
    duration_seconds: float = 0.0
    # Frame-based info (primary for debugging)
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    duration_frames: int = 0

    @property
    def event_type_name(self) -> str:
        """Human-readable event type name."""
        type_names = {
            EventType.BOUNDARY_VIOLATION: "OUT_OF_BOUNDS",
            EventType.BALL_BEHIND_PLAYER: "BALL_BEHIND",
            EventType.LOSS_DISTANCE: "DISTANCE",
            EventType.LOSS_VELOCITY: "VELOCITY",
            EventType.LOSS_DIRECTION: "DIRECTION",
        }
        return type_names.get(self.event_type, self.event_type.value.upper())


@dataclass
class GroundTruthEvent:
    """A single ground truth event with frame and timestamp."""
    frame: Optional[int]
    timestamp: Optional[float]


@dataclass
class TestResult:
    """Results from comparing detection to ground truth for one player."""
    player_name: str

    # Ground truth info (with frames)
    ground_truth_events: List[GroundTruthEvent]  # List of GT events with frame+timestamp
    ground_truth_count: int

    # Detection info
    detected_events: List[float]  # List of start timestamps (legacy)
    detected_count: int

    # Full detected events with type info
    all_detected_events: List[LossEvent] = field(default_factory=list)

    # Matching results
    true_positives: List[MatchedEvent] = field(default_factory=list)
    false_positives: List[DetectedEventInfo] = field(default_factory=list)  # Detected but not in GT
    false_negatives: List[GroundTruthEvent] = field(default_factory=list)  # GT but not detected

    # Event type breakdown (computed in __post_init__)
    event_type_counts: Dict[str, int] = field(default_factory=dict)
    tp_by_type: Dict[str, int] = field(default_factory=dict)
    fp_by_type: Dict[str, int] = field(default_factory=dict)

    # Metrics (computed in __post_init__)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    def __post_init__(self):
        """Calculate metrics after initialization."""
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)

        # Precision: TP / (TP + FP)
        if tp + fp > 0:
            self.precision = tp / (tp + fp)
        else:
            self.precision = 1.0 if fn == 0 else 0.0

        # Recall: TP / (TP + FN)
        if tp + fn > 0:
            self.recall = tp / (tp + fn)
        else:
            self.recall = 1.0 if fp == 0 else 0.0

        # F1: 2 * (P * R) / (P + R)
        if self.precision + self.recall > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            self.f1_score = 0.0

        # Calculate event type breakdown
        self._calculate_type_breakdown()

    def _calculate_type_breakdown(self):
        """Calculate event counts by type."""
        # Count all detected events by type
        for event in self.all_detected_events:
            type_name = self._get_type_name(event.event_type)
            self.event_type_counts[type_name] = self.event_type_counts.get(type_name, 0) + 1

        # Count true positives by type
        for match in self.true_positives:
            if match.event_type:
                type_name = self._get_type_name(match.event_type)
                self.tp_by_type[type_name] = self.tp_by_type.get(type_name, 0) + 1

        # Count false positives by type
        for fp in self.false_positives:
            type_name = fp.event_type_name
            self.fp_by_type[type_name] = self.fp_by_type.get(type_name, 0) + 1

    def _get_type_name(self, event_type: EventType) -> str:
        """Get human-readable type name."""
        type_names = {
            EventType.BOUNDARY_VIOLATION: "OUT_OF_BOUNDS",
            EventType.BALL_BEHIND_PLAYER: "BALL_BEHIND",
            EventType.LOSS_DISTANCE: "DISTANCE",
            EventType.LOSS_VELOCITY: "VELOCITY",
            EventType.LOSS_DIRECTION: "DIRECTION",
        }
        return type_names.get(event_type, event_type.value.upper())


@dataclass
class OverallTestSummary:
    """Aggregated test results across all players."""
    total_videos: int
    videos_with_ground_truth: int
    videos_processed: int
    videos_skipped: int  # Missing parquet data

    total_ground_truth_events: int
    total_detected_events: int
    total_true_positives: int
    total_false_positives: int
    total_false_negatives: int

    overall_precision: float
    overall_recall: float
    overall_f1: float

    per_player_results: Dict[str, TestResult] = field(default_factory=dict)

    # Event type breakdown across all players
    total_by_type: Dict[str, int] = field(default_factory=dict)
    tp_by_type: Dict[str, int] = field(default_factory=dict)
    fp_by_type: Dict[str, int] = field(default_factory=dict)

    def calculate_type_breakdown(self):
        """Calculate aggregate event type breakdown from per-player results."""
        for result in self.per_player_results.values():
            # Total by type
            for type_name, count in result.event_type_counts.items():
                self.total_by_type[type_name] = self.total_by_type.get(type_name, 0) + count
            # TP by type
            for type_name, count in result.tp_by_type.items():
                self.tp_by_type[type_name] = self.tp_by_type.get(type_name, 0) + count
            # FP by type
            for type_name, count in result.fp_by_type.items():
                self.fp_by_type[type_name] = self.fp_by_type.get(type_name, 0) + count


def load_ground_truth(csv_path: str) -> Dict[str, List[GroundTruthEvent]]:
    """
    Load ground truth annotations from CSV file.

    Args:
        csv_path: Path to ground_truth.csv

    Returns:
        Dict mapping player_name -> list of GroundTruthEvent (sorted by frame)

    Example:
        {
            'abdullah_nasib': [GroundTruthEvent(375, 12.5), GroundTruthEvent(750, 25.0)],
            'ali_buraq': [GroundTruthEvent(249, 8.3)],
            'archie_post': [],  # No events
        }
    """
    ground_truth: Dict[str, List[GroundTruthEvent]] = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Strip whitespace from keys (CSV may have aligned columns)
            row = {k.strip(): v.strip() if isinstance(v, str) else v for k, v in row.items()}
            player_name = row['player_name']
            event_number = int(row['event_number'])

            # Initialize player if not seen
            if player_name not in ground_truth:
                ground_truth[player_name] = []

            # event_number=0 means explicitly no events
            if event_number == 0:
                continue

            # Parse frame number (primary identifier)
            start_frame: Optional[int] = None
            start_frame_str = row.get('loss_start_frame', '').strip()
            if start_frame_str:
                try:
                    start_frame = int(start_frame_str)
                except ValueError:
                    logger.warning(f"Invalid frame for {player_name}: {start_frame_str}")

            # Parse start time (secondary, for human readability)
            start_time: Optional[float] = None
            start_time_str = row.get('loss_start_time', '').strip()
            if start_time_str:
                try:
                    start_time = float(start_time_str)
                except ValueError:
                    logger.warning(f"Invalid time for {player_name}: {start_time_str}")

            # Only add if we have at least frame or time
            if start_frame is not None or start_time is not None:
                ground_truth[player_name].append(GroundTruthEvent(
                    frame=start_frame,
                    timestamp=start_time
                ))

    # Sort by frame (primary) or timestamp (fallback)
    for player in ground_truth:
        ground_truth[player].sort(key=lambda e: (e.frame if e.frame is not None else float('inf'),
                                                   e.timestamp if e.timestamp is not None else float('inf')))

    logger.info(f"Loaded ground truth for {len(ground_truth)} players")
    return ground_truth


def compare_events(
    detected: List[LossEvent],
    ground_truth: List[GroundTruthEvent],
    frame_tolerance: int = DEFAULT_FRAME_TOLERANCE
) -> Tuple[List[MatchedEvent], List[DetectedEventInfo], List[GroundTruthEvent]]:
    """
    Compare detected events against ground truth using frame-based matching.

    Uses greedy matching: for each ground truth event, find the closest
    unmatched detection within frame tolerance.

    Args:
        detected: List of detected LossEvent objects
        ground_truth: List of GroundTruthEvent objects (sorted by frame)
        frame_tolerance: Maximum frame difference for a match (default 45 frames)

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
        - true_positives: List of MatchedEvent with matched pairs, frames, and event types
        - false_positives: List of DetectedEventInfo not matching any GT
        - false_negatives: List of GroundTruthEvent not matched by any detection
    """
    # Sort detected events by frame
    sorted_detected = sorted(detected, key=lambda e: e.start_frame)
    gt_events = sorted(ground_truth, key=lambda e: (e.frame if e.frame is not None else float('inf')))

    # Track which detections have been matched
    matched_detections = set()
    matched_gt = set()

    true_positives: List[MatchedEvent] = []

    # For each ground truth event, find best matching detection
    for gt_idx, gt_event in enumerate(gt_events):
        best_match_idx = None
        best_match_diff = float('inf')

        for det_idx, event in enumerate(sorted_detected):
            if det_idx in matched_detections:
                continue

            # Use frame-based matching if GT has frame, otherwise fallback to time
            if gt_event.frame is not None:
                diff = abs(event.start_frame - gt_event.frame)
                if diff <= frame_tolerance and diff < best_match_diff:
                    best_match_idx = det_idx
                    best_match_diff = diff
            elif gt_event.timestamp is not None:
                # Fallback to time-based matching (convert tolerance: 45 frames ≈ 1.5s at 30fps)
                time_tolerance = frame_tolerance / 30.0
                diff = abs(event.start_timestamp - gt_event.timestamp)
                if diff <= time_tolerance and diff < best_match_diff:
                    best_match_idx = det_idx
                    best_match_diff = diff

        if best_match_idx is not None:
            # Found a match - include full event info with frames
            matched_event = sorted_detected[best_match_idx]
            matched_detections.add(best_match_idx)
            matched_gt.add(gt_idx)

            # Calculate frame difference
            frame_diff = None
            if gt_event.frame is not None:
                frame_diff = matched_event.start_frame - gt_event.frame

            # Calculate time difference
            time_diff = 0.0
            gt_time = gt_event.timestamp if gt_event.timestamp is not None else 0.0
            time_diff = matched_event.start_timestamp - gt_time

            true_positives.append(MatchedEvent(
                detected_time=matched_event.start_timestamp,
                ground_truth_time=gt_time,
                time_difference=time_diff,
                detected_frame=matched_event.start_frame,
                ground_truth_frame=gt_event.frame,
                frame_difference=frame_diff,
                event_type=matched_event.event_type,
                loss_event=matched_event
            ))

    # False positives: detections not matched to any GT (with full info)
    false_positives: List[DetectedEventInfo] = []
    for det_idx, event in enumerate(sorted_detected):
        if det_idx not in matched_detections:
            false_positives.append(DetectedEventInfo(
                timestamp=event.start_timestamp,
                event_type=event.event_type,
                loss_event=event,
                duration_seconds=event.duration_seconds,
                start_frame=event.start_frame,
                end_frame=event.end_frame,
                duration_frames=event.duration_frames
            ))

    # False negatives: GT events not matched by any detection
    false_negatives = [
        gt_events[i] for i in range(len(gt_events))
        if i not in matched_gt
    ]

    return true_positives, false_positives, false_negatives


def create_test_result(
    player_name: str,
    detected_events: List[LossEvent],
    ground_truth_events: List[GroundTruthEvent],
    frame_tolerance: int = DEFAULT_FRAME_TOLERANCE
) -> TestResult:
    """
    Create a TestResult for a single player.

    Args:
        player_name: Player identifier
        detected_events: List of detected LossEvent objects
        ground_truth_events: List of GroundTruthEvent objects (with frame+timestamp)
        frame_tolerance: Frame tolerance for matching (default 45 frames)

    Returns:
        TestResult with all metrics calculated including event type breakdown
    """
    detected_times = [e.start_timestamp for e in detected_events]

    true_positives, false_positives, false_negatives = compare_events(
        detected_events, ground_truth_events, frame_tolerance
    )

    return TestResult(
        player_name=player_name,
        ground_truth_events=ground_truth_events,
        ground_truth_count=len(ground_truth_events),
        detected_events=detected_times,
        detected_count=len(detected_times),
        all_detected_events=detected_events,  # Store full events for type breakdown
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )


def generate_report(
    results: Dict[str, TestResult],
    summary: OverallTestSummary,
    frame_tolerance: int = DEFAULT_FRAME_TOLERANCE
) -> str:
    """
    Generate formatted test report string with detailed event type breakdown.

    Args:
        results: Dict of player_name -> TestResult
        summary: Overall aggregated summary
        frame_tolerance: Frame tolerance used for matching

    Returns:
        Formatted report string ready for console/file output
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("TRIPLE CONE BALL CONTROL DETECTION - DETAILED TEST RESULTS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Frame Tolerance: +/- {frame_tolerance} frames (~{frame_tolerance/30.0:.1f}s at 30fps)")
    lines.append("=" * 80)
    lines.append("")

    # Detection Type Legend
    lines.append("DETECTION TYPES:")
    lines.append("  OUT_OF_BOUNDS  = Ball exits video frame (boundary violation)")
    lines.append("  BALL_BEHIND    = Ball stays behind player relative to movement")
    lines.append("-" * 80)
    lines.append("")

    # Per-player results
    for player_name in sorted(results.keys()):
        result = results[player_name]
        lines.append(f"{'=' * 60}")
        lines.append(f"PLAYER: {player_name.upper()}")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Ground Truth Events: {result.ground_truth_count}")
        lines.append(f"  Detected Events: {result.detected_count}")

        # Event type breakdown for this player
        if result.event_type_counts:
            type_summary = ", ".join(
                f"{t}: {c}" for t, c in sorted(result.event_type_counts.items())
            )
            lines.append(f"  Detection Types: {type_summary}")

        lines.append("")

        # True positives with frame details
        if result.true_positives:
            lines.append(f"  ✓ TRUE POSITIVES: {len(result.true_positives)}")
            lines.append("  " + "-" * 56)
            for m in result.true_positives:
                event = m.loss_event
                # Format GT frame/time
                gt_str = f"frame {m.ground_truth_frame}" if m.ground_truth_frame is not None else "frame ?"
                if m.ground_truth_time > 0:
                    gt_str += f" ({m.ground_truth_time:.1f}s)"
                # Format detected frame/time
                det_str = f"frame {m.detected_frame}" if m.detected_frame is not None else "frame ?"
                det_str += f" ({m.detected_time:.1f}s)"
                # Format duration
                dur_frames = event.duration_frames if event else 0
                dur_secs = event.duration_seconds if event else 0.0
                dur_str = f"{dur_frames} frames ({dur_secs:.1f}s)"
                # Format difference
                diff_str = ""
                if m.frame_difference is not None:
                    sign = "+" if m.frame_difference >= 0 else ""
                    diff_str = f" Δ: {sign}{m.frame_difference} frames"
                lines.append(
                    f"    GT: {gt_str} → Detected: {det_str} [{m.event_type_name}]"
                )
                lines.append(
                    f"        {diff_str}, duration: {dur_str}"
                )
            # TP by type
            if result.tp_by_type:
                tp_type_str = ", ".join(f"{t}: {c}" for t, c in sorted(result.tp_by_type.items()))
                lines.append(f"    By Type: {tp_type_str}")
        else:
            lines.append(f"  ✓ TRUE POSITIVES: 0")

        lines.append("")

        # False positives with frame details
        if result.false_positives:
            lines.append(f"  ✗ FALSE POSITIVES: {len(result.false_positives)} (detected but not in ground truth)")
            lines.append("  " + "-" * 56)
            for fp in result.false_positives:
                frame_str = f"frame {fp.start_frame}" if fp.start_frame is not None else "frame ?"
                dur_str = f"{fp.duration_frames} frames ({fp.duration_seconds:.1f}s)"
                lines.append(
                    f"    @ {frame_str} ({fp.timestamp:.1f}s) [{fp.event_type_name}] "
                    f"duration: {dur_str}"
                )
            # FP by type
            if result.fp_by_type:
                fp_type_str = ", ".join(f"{t}: {c}" for t, c in sorted(result.fp_by_type.items()))
                lines.append(f"    By Type: {fp_type_str}")
        else:
            lines.append(f"  ✗ FALSE POSITIVES: 0")

        lines.append("")

        # False negatives with frame details
        if result.false_negatives:
            lines.append(f"  ✗ FALSE NEGATIVES: {len(result.false_negatives)} (in ground truth but not detected)")
            lines.append("  " + "-" * 56)
            for fn in result.false_negatives:
                frame_str = f"frame {fn.frame}" if fn.frame is not None else "frame ?"
                time_str = f" ({fn.timestamp:.1f}s)" if fn.timestamp is not None else ""
                lines.append(f"    @ {frame_str}{time_str} (MISSED)")
        else:
            lines.append(f"  ✗ FALSE NEGATIVES: 0")

        lines.append("")

        # Metrics
        lines.append(f"  METRICS:")
        lines.append(
            f"    Precision: {result.precision*100:.1f}% | "
            f"Recall: {result.recall*100:.1f}% | "
            f"F1: {result.f1_score*100:.1f}%"
        )
        lines.append("")

    # Overall summary
    lines.append("=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Processing stats
    lines.append("PROCESSING:")
    lines.append(f"  Videos in dataset: {summary.total_videos}")
    lines.append(f"  Videos with Ground Truth: {summary.videos_with_ground_truth}")
    lines.append(f"  Videos Processed: {summary.videos_processed}")
    lines.append(f"  Videos Skipped (missing data): {summary.videos_skipped}")
    lines.append("")

    # Event counts
    lines.append("EVENT COUNTS:")
    lines.append(f"  Total Ground Truth Events: {summary.total_ground_truth_events}")
    lines.append(f"  Total Detected Events: {summary.total_detected_events}")
    lines.append(f"  Total True Positives: {summary.total_true_positives}")
    lines.append(f"  Total False Positives: {summary.total_false_positives}")
    lines.append(f"  Total False Negatives: {summary.total_false_negatives}")
    lines.append("")

    # Event type breakdown
    summary.calculate_type_breakdown()
    if summary.total_by_type:
        lines.append("DETECTION TYPE BREAKDOWN:")
        lines.append("-" * 40)
        for type_name in sorted(summary.total_by_type.keys()):
            total = summary.total_by_type.get(type_name, 0)
            tp = summary.tp_by_type.get(type_name, 0)
            fp = summary.fp_by_type.get(type_name, 0)

            # Calculate precision for this type
            if tp + fp > 0:
                type_precision = (tp / (tp + fp)) * 100
            else:
                type_precision = 0.0

            lines.append(f"  {type_name}:")
            lines.append(f"    Total Detected: {total}")
            lines.append(f"    True Positives: {tp}")
            lines.append(f"    False Positives: {fp}")
            lines.append(f"    Type Precision: {type_precision:.1f}%")
            lines.append("")

    # Overall metrics
    lines.append("OVERALL METRICS:")
    lines.append("-" * 40)
    lines.append(f"  Precision: {summary.overall_precision*100:.1f}%")
    lines.append(f"  Recall: {summary.overall_recall*100:.1f}%")
    lines.append(f"  F1 Score: {summary.overall_f1*100:.1f}%")
    lines.append("")

    # Visual summary bar
    lines.append("VISUAL SUMMARY:")
    lines.append("-" * 40)
    total_events = summary.total_true_positives + summary.total_false_positives + summary.total_false_negatives
    if total_events > 0:
        tp_bar = "█" * int(summary.total_true_positives / total_events * 30)
        fp_bar = "░" * int(summary.total_false_positives / total_events * 30)
        fn_bar = "▒" * int(summary.total_false_negatives / total_events * 30)
        lines.append(f"  TP: {tp_bar} ({summary.total_true_positives})")
        lines.append(f"  FP: {fp_bar} ({summary.total_false_positives})")
        lines.append(f"  FN: {fn_bar} ({summary.total_false_negatives})")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def save_report(report: str, output_dir: Path, summary: 'OverallTestSummary' = None) -> Path:
    """
    Save report to timestamped file with descriptive naming.

    Creates two files:
    1. A timestamped file with F1 score: `test_report_F1-XX.X_YYYYMMDD_HHMMSS.txt`
    2. A `LATEST_report.txt` that's always the most recent (easy to find)

    Args:
        report: Report string content
        output_dir: Directory to save report
        summary: Optional summary to extract F1 score for filename

    Returns:
        Path to saved report file (the timestamped one)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

    # Include F1 score in filename if available
    if summary is not None:
        f1_pct = summary.overall_f1 * 100
        filename = f"test_report_F1-{f1_pct:.1f}_{timestamp}.txt"
    else:
        filename = f"test_report_{timestamp}.txt"

    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to {filepath}")

    # Also save as LATEST_report.txt for easy access
    latest_path = output_dir / "LATEST_report.txt"
    with open(latest_path, 'w') as f:
        f.write(report)
    logger.info(f"Latest report also saved to {latest_path}")

    return filepath


def save_summary_csv(
    results: Dict[str, TestResult],
    summary: OverallTestSummary,
    output_dir: Path
) -> Path:
    """
    Save test summary as simple CSV - one row per player.

    CSV columns:
        player_name, gt_events, detected_events, true_positives,
        false_positives, false_negatives, precision, recall, f1_score

    Args:
        results: Dict of player_name -> TestResult
        summary: Overall aggregated summary
        output_dir: Directory to save CSV

    Returns:
        Path to saved CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "test_summary.csv"

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'player_name', 'gt_events', 'detected_events',
            'true_positives', 'false_positives', 'false_negatives',
            'precision', 'recall', 'f1_score'
        ])

        # Per-player rows (sorted alphabetically)
        for player_name in sorted(results.keys()):
            result = results[player_name]
            writer.writerow([
                player_name,
                result.ground_truth_count,
                result.detected_count,
                len(result.true_positives),
                len(result.false_positives),
                len(result.false_negatives),
                f"{result.precision:.3f}",
                f"{result.recall:.3f}",
                f"{result.f1_score:.3f}"
            ])

        # Summary row at bottom
        writer.writerow([])  # Empty row separator
        writer.writerow([
            'TOTAL',
            summary.total_ground_truth_events,
            summary.total_detected_events,
            summary.total_true_positives,
            summary.total_false_positives,
            summary.total_false_negatives,
            f"{summary.overall_precision:.3f}",
            f"{summary.overall_recall:.3f}",
            f"{summary.overall_f1:.3f}"
        ])

    logger.info(f"Summary CSV saved to {filepath}")
    return filepath


def save_events_csv(
    results: Dict[str, TestResult],
    output_dir: Path
) -> Path:
    """
    Save detailed event-level results as CSV - one row per event.

    CSV columns (with frame numbers for debugging):
        player_name, source, frame, timestamp_s, event_type, duration_frames,
        duration_s, result, matched_frame, matched_timestamp_s, frame_diff, time_diff_s

    Args:
        results: Dict of player_name -> TestResult
        output_dir: Directory to save CSV

    Returns:
        Path to saved CSV file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "test_events.csv"

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header with frame columns
        writer.writerow([
            'player_name', 'source', 'frame', 'timestamp_s', 'event_type',
            'duration_frames', 'duration_s', 'result', 'matched_frame',
            'matched_timestamp_s', 'frame_diff', 'time_diff_s'
        ])

        for player_name in sorted(results.keys()):
            result = results[player_name]

            # Track which GT frames were matched
            matched_gt_frames = {m.ground_truth_frame for m in result.true_positives}

            # Ground truth events
            for gt_event in result.ground_truth_events:
                if gt_event.frame in matched_gt_frames:
                    # Find the matching detection
                    match = next(m for m in result.true_positives
                                 if m.ground_truth_frame == gt_event.frame)
                    writer.writerow([
                        player_name, 'ground_truth',
                        gt_event.frame if gt_event.frame is not None else '',
                        f"{gt_event.timestamp:.1f}" if gt_event.timestamp is not None else '',
                        '',  # event_type
                        '',  # duration_frames
                        '',  # duration_s
                        'TP',
                        match.detected_frame if match.detected_frame is not None else '',
                        f"{match.detected_time:.1f}",
                        match.frame_difference if match.frame_difference is not None else '',
                        f"{match.time_difference:.1f}"
                    ])
                else:
                    # False negative - missed
                    writer.writerow([
                        player_name, 'ground_truth',
                        gt_event.frame if gt_event.frame is not None else '',
                        f"{gt_event.timestamp:.1f}" if gt_event.timestamp is not None else '',
                        '', '', '', 'FN', '', '', '', ''
                    ])

            # Detected events - True positives
            for match in result.true_positives:
                event = match.loss_event
                dur_frames = event.duration_frames if event else ''
                dur_secs = f"{event.duration_seconds:.1f}" if event else ''
                writer.writerow([
                    player_name, 'detected',
                    match.detected_frame if match.detected_frame is not None else '',
                    f"{match.detected_time:.1f}",
                    match.event_type_name,
                    dur_frames,
                    dur_secs,
                    'TP',
                    match.ground_truth_frame if match.ground_truth_frame is not None else '',
                    f"{match.ground_truth_time:.1f}",
                    match.frame_difference if match.frame_difference is not None else '',
                    f"{match.time_difference:.1f}"
                ])

            # Detected events - False positives
            for fp in result.false_positives:
                writer.writerow([
                    player_name, 'detected',
                    fp.start_frame if fp.start_frame is not None else '',
                    f"{fp.timestamp:.1f}",
                    fp.event_type_name,
                    fp.duration_frames,
                    f"{fp.duration_seconds:.1f}",
                    'FP', '', '', '', ''
                ])

    logger.info(f"Events CSV saved to {filepath}")
    return filepath


def save_csv_results(
    results: Dict[str, TestResult],
    summary: OverallTestSummary,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Save test results to CSV files (summary + events).

    This is the main function to call for CSV output.

    Args:
        results: Dict of player_name -> TestResult
        summary: Overall aggregated summary
        output_dir: Directory to save CSVs

    Returns:
        Tuple of (summary_csv_path, events_csv_path)
    """
    summary_path = save_summary_csv(results, summary, output_dir)
    events_path = save_events_csv(results, output_dir)
    return summary_path, events_path
