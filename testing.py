"""
Testing module for Figure-8 Ball Control Detection validation.

Compares detected loss events against manually annotated ground truth
with configurable time tolerance.

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

from .detection.data_structures import LossEvent

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TOLERANCE = 0.5  # seconds


@dataclass
class MatchedEvent:
    """A detected event matched to ground truth."""
    detected_time: float
    ground_truth_time: float
    time_difference: float  # detected - ground_truth


@dataclass
class TestResult:
    """Results from comparing detection to ground truth for one player."""
    player_name: str

    # Ground truth info
    ground_truth_events: List[float]  # List of start timestamps
    ground_truth_count: int

    # Detection info
    detected_events: List[float]  # List of start timestamps
    detected_count: int

    # Matching results
    true_positives: List[MatchedEvent] = field(default_factory=list)
    false_positives: List[float] = field(default_factory=list)  # Detected but not in GT
    false_negatives: List[float] = field(default_factory=list)  # GT but not detected

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


def load_ground_truth(csv_path: str) -> Dict[str, List[float]]:
    """
    Load ground truth annotations from CSV file.

    Args:
        csv_path: Path to ground_truth.csv

    Returns:
        Dict mapping player_name -> list of loss event start times (sorted)

    Example:
        {
            'abdullah_nasib': [12.5, 25.0, 40.1],
            'ali_buraq': [8.3, 15.7],
            'archie_post': [],  # No events
        }
    """
    ground_truth: Dict[str, List[float]] = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            player_name = row['player_name'].strip()
            event_number = int(row['event_number'])

            # Initialize player if not seen
            if player_name not in ground_truth:
                ground_truth[player_name] = []

            # event_number=0 means explicitly no events
            if event_number == 0:
                continue

            # Parse start time
            start_time_str = row.get('loss_start_time', '').strip()
            if start_time_str:
                try:
                    start_time = float(start_time_str)
                    ground_truth[player_name].append(start_time)
                except ValueError:
                    logger.warning(f"Invalid time for {player_name}: {start_time_str}")

    # Sort times for each player
    for player in ground_truth:
        ground_truth[player].sort()

    logger.info(f"Loaded ground truth for {len(ground_truth)} players")
    return ground_truth


def compare_events(
    detected: List[LossEvent],
    ground_truth: List[float],
    tolerance: float = DEFAULT_TOLERANCE
) -> Tuple[List[MatchedEvent], List[float], List[float]]:
    """
    Compare detected events against ground truth with tolerance.

    Uses greedy matching: for each ground truth event, find the closest
    unmatched detection within tolerance.

    Args:
        detected: List of detected LossEvent objects
        ground_truth: List of ground truth start timestamps (sorted)
        tolerance: Maximum time difference for a match (default 0.5s)

    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
        - true_positives: List of MatchedEvent with matched pairs
        - false_positives: List of detected times not matching any GT
        - false_negatives: List of GT times not matched by any detection
    """
    # Extract start timestamps from detected events
    detected_times = sorted([e.start_timestamp for e in detected])
    gt_times = sorted(ground_truth)

    # Track which detections have been matched
    matched_detections = set()
    matched_gt = set()

    true_positives: List[MatchedEvent] = []

    # For each ground truth event, find best matching detection
    for gt_idx, gt_time in enumerate(gt_times):
        best_match_idx = None
        best_match_diff = float('inf')

        for det_idx, det_time in enumerate(detected_times):
            if det_idx in matched_detections:
                continue

            diff = abs(det_time - gt_time)
            if diff <= tolerance and diff < best_match_diff:
                best_match_idx = det_idx
                best_match_diff = diff

        if best_match_idx is not None:
            # Found a match
            matched_detections.add(best_match_idx)
            matched_gt.add(gt_idx)
            true_positives.append(MatchedEvent(
                detected_time=detected_times[best_match_idx],
                ground_truth_time=gt_time,
                time_difference=detected_times[best_match_idx] - gt_time
            ))

    # False positives: detections not matched to any GT
    false_positives = [
        detected_times[i] for i in range(len(detected_times))
        if i not in matched_detections
    ]

    # False negatives: GT events not matched by any detection
    false_negatives = [
        gt_times[i] for i in range(len(gt_times))
        if i not in matched_gt
    ]

    return true_positives, false_positives, false_negatives


def create_test_result(
    player_name: str,
    detected_events: List[LossEvent],
    ground_truth_times: List[float],
    tolerance: float = DEFAULT_TOLERANCE
) -> TestResult:
    """
    Create a TestResult for a single player.

    Args:
        player_name: Player identifier
        detected_events: List of detected LossEvent objects
        ground_truth_times: List of ground truth timestamps
        tolerance: Time tolerance for matching

    Returns:
        TestResult with all metrics calculated
    """
    detected_times = [e.start_timestamp for e in detected_events]

    true_positives, false_positives, false_negatives = compare_events(
        detected_events, ground_truth_times, tolerance
    )

    return TestResult(
        player_name=player_name,
        ground_truth_events=ground_truth_times,
        ground_truth_count=len(ground_truth_times),
        detected_events=detected_times,
        detected_count=len(detected_times),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )


def generate_report(
    results: Dict[str, TestResult],
    summary: OverallTestSummary,
    tolerance: float = DEFAULT_TOLERANCE
) -> str:
    """
    Generate formatted test report string.

    Args:
        results: Dict of player_name -> TestResult
        summary: Overall aggregated summary
        tolerance: Tolerance used for matching

    Returns:
        Formatted report string ready for console/file output
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("FIGURE-8 BALL CONTROL DETECTION - TEST RESULTS")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Tolerance: +/- {tolerance}s")
    lines.append("=" * 60)
    lines.append("")

    # Per-player results
    for player_name in sorted(results.keys()):
        result = results[player_name]
        lines.append(f"Player: {player_name}")
        lines.append(f"  Ground Truth Events: {result.ground_truth_count}")
        lines.append(f"  Detected Events: {result.detected_count}")

        # True positives with time mapping
        if result.true_positives:
            tp_strs = [
                f"{m.ground_truth_time:.1f}s->{m.detected_time:.1f}s"
                for m in result.true_positives
            ]
            lines.append(f"  True Positives: {len(result.true_positives)} ({', '.join(tp_strs)})")
        else:
            lines.append(f"  True Positives: 0")

        # False positives
        if result.false_positives:
            fp_strs = [f"{t:.1f}s" for t in result.false_positives]
            lines.append(f"  False Positives: {len(result.false_positives)} ({', '.join(fp_strs)})")
        else:
            lines.append(f"  False Positives: 0")

        # False negatives
        if result.false_negatives:
            fn_strs = [f"{t:.1f}s" for t in result.false_negatives]
            lines.append(f"  False Negatives: {len(result.false_negatives)} ({', '.join(fn_strs)})")
        else:
            lines.append(f"  False Negatives: 0")

        # Metrics
        lines.append(
            f"  Precision: {result.precision*100:.1f}% | "
            f"Recall: {result.recall*100:.1f}% | "
            f"F1: {result.f1_score*100:.1f}%"
        )
        lines.append("")

    # Overall summary
    lines.append("=" * 60)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Videos in PLAYERS dict: {summary.total_videos}")
    lines.append(f"Videos with Ground Truth: {summary.videos_with_ground_truth}")
    lines.append(f"Videos Processed: {summary.videos_processed}")
    lines.append(f"Videos Skipped (missing data): {summary.videos_skipped}")
    lines.append("")
    lines.append(f"Total Ground Truth Events: {summary.total_ground_truth_events}")
    lines.append(f"Total Detected Events: {summary.total_detected_events}")
    lines.append(f"Total True Positives: {summary.total_true_positives}")
    lines.append(f"Total False Positives: {summary.total_false_positives}")
    lines.append(f"Total False Negatives: {summary.total_false_negatives}")
    lines.append("")
    lines.append(f"Overall Precision: {summary.overall_precision*100:.1f}%")
    lines.append(f"Overall Recall: {summary.overall_recall*100:.1f}%")
    lines.append(f"Overall F1 Score: {summary.overall_f1*100:.1f}%")
    lines.append("=" * 60)

    return "\n".join(lines)


def save_report(report: str, output_dir: Path) -> Path:
    """
    Save report to timestamped file.

    Args:
        report: Report string content
        output_dir: Directory to save report

    Returns:
        Path to saved report file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_report_{timestamp}.txt"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to {filepath}")
    return filepath
