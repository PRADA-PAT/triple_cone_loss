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


# =============================================================================
# Report Generation Helper Functions
# =============================================================================

def _format_ascii_table(headers: List[str], rows: List[List[str]], min_widths: Optional[List[int]] = None) -> str:
    """
    Format data as ASCII table with aligned columns.

    Args:
        headers: Column header names
        rows: List of rows, each row is a list of cell values
        min_widths: Optional minimum column widths

    Returns:
        Formatted ASCII table string
    """
    if not rows:
        return ""

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    # Apply minimum widths if provided
    if min_widths:
        for i, mw in enumerate(min_widths):
            if i < len(widths):
                widths[i] = max(widths[i], mw)

    # Build table
    lines = []

    # Header row
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"

    lines.append(header_line)
    lines.append(separator)

    # Data rows
    for row in rows:
        row_cells = []
        for i, cell in enumerate(row):
            if i < len(widths):
                row_cells.append(str(cell).ljust(widths[i]))
        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


def _categorize_false_positives(
    all_results: Dict[str, 'TestResult'],
    frame_tolerance: int = DEFAULT_FRAME_TOLERANCE
) -> Dict[str, List[Tuple[str, 'DetectedEventInfo']]]:
    """
    Categorize false positives across all players into categories.

    Categories:
    - detection_errors: Very long duration (>5s) - likely algorithm issues
    - tolerance_mismatch: Close to a GT event but outside tolerance window
    - unclassified: Require manual review

    Args:
        all_results: Dict of player_name -> TestResult
        frame_tolerance: Frame tolerance used for matching

    Returns:
        Dict with keys 'detection_errors', 'tolerance_mismatch', 'unclassified'
        Each value is list of (player_name, DetectedEventInfo) tuples
    """
    categorized = {
        'detection_errors': [],
        'tolerance_mismatch': [],
        'unclassified': []
    }

    extended_tolerance = frame_tolerance * 2  # e.g., 90 frames for tolerance_mismatch check

    for player_name, result in all_results.items():
        for fp in result.false_positives:
            # Check if this FP is close to any GT event (tolerance mismatch)
            is_tolerance_mismatch = False
            for gt_event in result.ground_truth_events:
                if gt_event.frame is not None and fp.start_frame is not None:
                    diff = abs(fp.start_frame - gt_event.frame)
                    if frame_tolerance < diff <= extended_tolerance:
                        is_tolerance_mismatch = True
                        break

            if is_tolerance_mismatch:
                categorized['tolerance_mismatch'].append((player_name, fp))
            elif fp.duration_seconds > 5.0:
                # Very long detections are likely algorithm errors
                categorized['detection_errors'].append((player_name, fp))
            else:
                categorized['unclassified'].append((player_name, fp))

    return categorized


def _generate_executive_summary(summary: 'OverallTestSummary') -> List[str]:
    """Generate executive summary section."""
    lines = []

    lines.append("=" * 80)
    lines.append("1. EXECUTIVE SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("The Ball Control Detection System analyzes Triple Cone drill videos to detect")
    lines.append("when a player loses control of the ball. Two types of loss events are detected:")
    lines.append("")
    lines.append("  - BALL_BEHIND    : Ball stays behind player relative to movement direction")
    lines.append("  - OUT_OF_BOUNDS  : Ball exits video frame (boundary violation)")
    lines.append("")
    lines.append("CURRENT PERFORMANCE:")
    lines.append("-" * 40)
    lines.append(f"  F1 Score:     {summary.overall_f1*100:.1f}%")
    lines.append(f"  Precision:    {summary.overall_precision*100:.1f}%")
    lines.append(f"  Recall:       {summary.overall_recall*100:.1f}%")
    lines.append("")
    lines.append(f"  True Positives:   {summary.total_true_positives}")
    lines.append(f"  False Positives:  {summary.total_false_positives}")
    lines.append(f"  False Negatives:  {summary.total_false_negatives}")
    lines.append("")

    return lines


def _generate_detection_config_section() -> List[str]:
    """Generate detection configuration section with current thresholds."""
    lines = []

    lines.append("=" * 80)
    lines.append("2. DETECTION CONFIGURATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("BALL-BEHIND DETECTION:")
    lines.append("  - Behind Threshold:        20.0 pixels (ball must be this far behind hip)")
    lines.append("  - Sustained Frames:        10 frames (~0.33s at 30fps)")
    lines.append("  - Movement Threshold:      3.0 pixels (min hip movement for direction)")
    lines.append("  - Direction Consistency:   70% (required for sustained detection)")
    lines.append("")
    lines.append("BOUNDARY DETECTION:")
    lines.append("  - Edge Margin:             50 pixels")
    lines.append("  - Min Timestamp:           3.0 seconds (skip early frames)")
    lines.append("")
    lines.append("EVENT FILTERING:")
    lines.append("  - Min Event Duration:      15 frames (~0.5s at 30fps)")
    lines.append("")
    lines.append("TURNING ZONES (Detection Suppressed):")
    lines.append("  - Zone Radius:             68px base radius (elliptical)")
    lines.append("  - Y Stretch Factor:        5.0x (for camera perspective)")
    lines.append("")

    return lines


def _generate_performance_metrics(summary: 'OverallTestSummary') -> List[str]:
    """Generate performance metrics section with type breakdown."""
    lines = []

    lines.append("=" * 80)
    lines.append("3. PERFORMANCE METRICS")
    lines.append("=" * 80)
    lines.append("")

    # Calculate type breakdown
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
            lines.append(f"    Total Detected:    {total}")
            lines.append(f"    True Positives:    {tp}")
            lines.append(f"    False Positives:   {fp}")
            lines.append(f"    Type Precision:    {type_precision:.1f}%")
            lines.append("")

    # Player performance distribution
    perfect = 0
    partial = 0
    no_detection = 0
    fp_only = 0

    for result in summary.per_player_results.values():
        if result.f1_score == 1.0 and result.ground_truth_count > 0:
            perfect += 1
        elif result.f1_score == 1.0 and result.ground_truth_count == 0 and result.detected_count == 0:
            perfect += 1  # Clean run correctly detected
        elif 0 < result.f1_score < 1.0:
            partial += 1
        elif result.f1_score == 0 and result.detected_count > 0 and result.ground_truth_count == 0:
            fp_only += 1
        elif result.f1_score == 0:
            no_detection += 1

    lines.append("PLAYER PERFORMANCE DISTRIBUTION:")
    lines.append("-" * 40)
    lines.append(f"  Perfect Detection (100% F1):  {perfect} players")
    lines.append(f"  Partial Detection:            {partial} players")
    lines.append(f"  No Detection:                 {no_detection} players")
    lines.append(f"  False Positives Only:         {fp_only} players")
    lines.append("")

    return lines


def _generate_fp_analysis_section(
    results: Dict[str, 'TestResult'],
    frame_tolerance: int = DEFAULT_FRAME_TOLERANCE
) -> List[str]:
    """Generate categorized false positives analysis section."""
    lines = []

    lines.append("=" * 80)
    lines.append("4. FALSE POSITIVES ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Count total FPs
    total_fp = sum(len(r.false_positives) for r in results.values())
    lines.append(f"Total False Positives: {total_fp}")
    lines.append("")

    # Categorize FPs
    categorized = _categorize_false_positives(results, frame_tolerance)

    lines.append("CATEGORIZATION:")
    lines.append("-" * 40)
    lines.append("")

    # [A] Detection Errors
    lines.append("[A] DETECTION ERRORS (Algorithm Issues) - {} events".format(
        len(categorized['detection_errors'])))
    lines.append("=" * 60)
    lines.append("These are incorrectly detected events due to algorithm limitations.")
    lines.append("")

    if categorized['detection_errors']:
        headers = ["Player", "Time", "Type", "Duration", "Issue"]
        rows = []
        for player_name, fp in categorized['detection_errors']:
            rows.append([
                player_name.upper(),
                f"{fp.timestamp:.1f}s",
                fp.event_type_name,
                f"{fp.duration_seconds:.1f}s",
                "Long duration detection"
            ])
        lines.append(_format_ascii_table(headers, rows))
    else:
        lines.append("None detected.")
    lines.append("")

    # [B] Tolerance Mismatch
    lines.append("[B] TOLERANCE MISMATCH - {} events".format(
        len(categorized['tolerance_mismatch'])))
    lines.append("=" * 60)
    lines.append("Events detected but outside the tolerance window.")
    lines.append("")

    if categorized['tolerance_mismatch']:
        headers = ["Player", "Time", "Type", "Duration"]
        rows = []
        for player_name, fp in categorized['tolerance_mismatch']:
            rows.append([
                player_name.upper(),
                f"{fp.timestamp:.1f}s",
                fp.event_type_name,
                f"{fp.duration_seconds:.1f}s"
            ])
        lines.append(_format_ascii_table(headers, rows))
    else:
        lines.append("None detected.")
    lines.append("")

    # [C] Unclassified
    lines.append("[C] UNCLASSIFIED - {} events".format(
        len(categorized['unclassified'])))
    lines.append("=" * 60)
    lines.append("Require manual video review to determine cause.")
    lines.append("")

    if categorized['unclassified']:
        headers = ["Player", "Time", "Type", "Duration"]
        rows = []
        for player_name, fp in categorized['unclassified']:
            rows.append([
                player_name.upper(),
                f"{fp.timestamp:.1f}s",
                fp.event_type_name,
                f"{fp.duration_seconds:.1f}s"
            ])
        lines.append(_format_ascii_table(headers, rows))
    else:
        lines.append("None detected.")
    lines.append("")

    return lines


def _generate_fn_analysis_section(results: Dict[str, 'TestResult']) -> List[str]:
    """Generate false negatives analysis section with reasons."""
    lines = []

    lines.append("=" * 80)
    lines.append("5. FALSE NEGATIVES ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # Collect all FNs
    all_fns = []
    for player_name, result in results.items():
        for fn in result.false_negatives:
            all_fns.append((player_name, fn))

    lines.append(f"Total False Negatives: {len(all_fns)}")
    lines.append("")

    if all_fns:
        lines.append("MISSED EVENTS:")
        lines.append("-" * 40)
        lines.append("")

        headers = ["Player", "GT Frame", "GT Time", "Possible Reason"]
        rows = []
        for player_name, fn in all_fns:
            frame_str = str(fn.frame) if fn.frame is not None else "?"
            time_str = f"{fn.timestamp:.1f}s" if fn.timestamp is not None else "?"
            # Heuristic reason based on timestamp
            reason = "Requires review"
            if fn.timestamp is not None:
                if fn.timestamp < 5.0:
                    reason = "Early in video (startup)"
                elif fn.timestamp > 35.0:
                    reason = "Late in video (drill ending)"

            rows.append([
                player_name.upper(),
                frame_str,
                time_str,
                reason
            ])

        lines.append(_format_ascii_table(headers, rows))
    else:
        lines.append("No missed events - all ground truth events were detected!")
    lines.append("")

    return lines


def _generate_known_issues_section() -> List[str]:
    """Generate known issues and bugs section."""
    lines = []

    lines.append("=" * 80)
    lines.append("6. KNOWN ISSUES & BUGS")
    lines.append("=" * 80)
    lines.append("")

    lines.append("[ISSUE-001] TURNING ZONE SUPPRESSION")
    lines.append("-" * 40)
    lines.append("Severity: Medium")
    lines.append("Status: Under Review")
    lines.append("")
    lines.append("Description:")
    lines.append("  Some valid loss events inside turning zones may be suppressed.")
    lines.append("  The elliptical zones around cones filter out detections that")
    lines.append("  occur during expected tight turns.")
    lines.append("")
    lines.append("Mitigation:")
    lines.append("  Review false negatives that occur near cone positions.")
    lines.append("")

    lines.append("[ISSUE-002] BALL-BEHIND DIRECTION SENSITIVITY")
    lines.append("-" * 40)
    lines.append("Severity: Low")
    lines.append("Status: Monitoring")
    lines.append("")
    lines.append("Description:")
    lines.append("  Ball-behind detection relies on hip movement direction.")
    lines.append("  Brief stationary moments can affect direction calculation.")
    lines.append("")

    return lines


def _generate_recommendations_section(
    summary: 'OverallTestSummary',
    categorized_fps: Dict[str, List]
) -> List[str]:
    """Generate recommendations based on test results."""
    lines = []

    lines.append("=" * 80)
    lines.append("7. RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("")

    lines.append("IMMEDIATE ACTIONS:")
    lines.append("-" * 40)

    # Dynamic recommendations based on results
    if len(categorized_fps.get('detection_errors', [])) > 0:
        lines.append("1. Review long-duration false positives for algorithm tuning")

    if summary.total_false_negatives > 0:
        lines.append("2. Analyze missed events - check if inside turning zones")

    if summary.overall_precision < 0.5:
        lines.append("3. Consider increasing detection thresholds to reduce false positives")

    if summary.overall_recall < 0.7:
        lines.append("4. Review detection sensitivity - may be missing valid events")

    lines.append("")

    lines.append("DETECTION IMPROVEMENTS:")
    lines.append("-" * 40)
    lines.append("1. Consider adaptive thresholds based on player speed")
    lines.append("2. Add ball trajectory analysis for edge cases")
    lines.append("3. Implement velocity-based confirmation for ball-behind detection")
    lines.append("")

    lines.append("GROUND TRUTH UPDATES NEEDED:")
    lines.append("-" * 40)
    lines.append("Review unclassified false positives - some may be valid events")
    lines.append("that should be added to ground truth.")
    lines.append("")

    return lines


def _generate_detailed_results_by_category(results: Dict[str, 'TestResult']) -> List[str]:
    """Generate player results grouped by performance category."""
    lines = []

    lines.append("=" * 80)
    lines.append("8. DETAILED PLAYER RESULTS")
    lines.append("=" * 80)
    lines.append("")

    # Categorize players
    perfect = []
    partial = []
    no_tp = []

    for player_name, result in sorted(results.items()):
        if result.f1_score == 1.0:
            perfect.append((player_name, result))
        elif result.f1_score > 0:
            partial.append((player_name, result))
        else:
            no_tp.append((player_name, result))

    # Perfect Detection
    lines.append("PERFECT DETECTION (F1 = 100%):")
    lines.append("-" * 40)
    if perfect:
        for player_name, result in perfect:
            if result.ground_truth_count == 0 and result.detected_count == 0:
                lines.append(f"  - {player_name.upper()}: No events (correctly detected as clean run)")
            else:
                for match in result.true_positives:
                    gt_time = f"{match.ground_truth_time:.1f}s" if match.ground_truth_time else "?"
                    det_time = f"{match.detected_time:.1f}s"
                    lines.append(
                        f"  - {player_name.upper()}: GT: {gt_time} -> Detected: {det_time} "
                        f"[{match.event_type_name}]"
                    )
    else:
        lines.append("  None")
    lines.append("")

    # Partial Detection
    lines.append("PARTIAL DETECTION (0% < F1 < 100%):")
    lines.append("-" * 40)
    if partial:
        for player_name, result in partial:
            tp_count = len(result.true_positives)
            fp_count = len(result.false_positives)
            fn_count = len(result.false_negatives)
            lines.append(
                f"  - {player_name.upper()}: F1: {result.f1_score*100:.1f}% "
                f"({tp_count} TP, {fp_count} FP, {fn_count} FN)"
            )
    else:
        lines.append("  None")
    lines.append("")

    # No True Positives
    lines.append("NO TRUE POSITIVES (F1 = 0%):")
    lines.append("-" * 40)
    if no_tp:
        for player_name, result in no_tp:
            fp_count = len(result.false_positives)
            fn_count = len(result.false_negatives)
            lines.append(f"  - {player_name.upper()}: {fp_count} FP, {fn_count} FN")
    else:
        lines.append("  None")
    lines.append("")

    return lines


def generate_report(
    results: Dict[str, TestResult],
    summary: OverallTestSummary,
    frame_tolerance: int = DEFAULT_FRAME_TOLERANCE
) -> str:
    """
    Generate comprehensive test report with executive summary and per-player details.

    Report structure (matching f8_loss format):
    1. Executive Summary - Overall metrics at a glance
    2. Detection Configuration - All threshold values
    3. Performance Metrics - Type breakdown and distribution
    4. False Positives Analysis - Categorized FP analysis
    5. False Negatives Analysis - Missed events with reasons
    6. Known Issues & Bugs - Algorithm limitations
    7. Recommendations - Suggested improvements
    8. Detailed Player Results - Summary by performance category
    9. Per-Player Debug Sections - Full TP/FP/FN details for each player

    Args:
        results: Dict of player_name -> TestResult
        summary: Overall aggregated summary
        frame_tolerance: Frame tolerance used for matching

    Returns:
        Formatted report string ready for console/file output
    """
    lines = []

    # ==========================================================================
    # HEADER
    # ==========================================================================
    lines.append("=" * 80)
    lines.append("LOSS OF CONTROL (LOC) - TRIPLE CONE DRILL DETECTION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Detection System Version: 1.0")
    lines.append(f"Tolerance: +/- {frame_tolerance} frames (~{frame_tolerance/30.0:.1f}s at 30fps)")
    lines.append("=" * 80)
    lines.append("")

    # ==========================================================================
    # TABLE OF CONTENTS
    # ==========================================================================
    lines.append("TABLE OF CONTENTS")
    lines.append("-" * 80)
    lines.append("1. Executive Summary")
    lines.append("2. Detection Configuration")
    lines.append("3. Performance Metrics")
    lines.append("4. False Positives Analysis")
    lines.append("5. False Negatives Analysis")
    lines.append("6. Known Issues & Bugs")
    lines.append("7. Recommendations")
    lines.append("8. Detailed Player Results (by category)")
    lines.append("9. Per-Player Debug Sections (full details)")
    lines.append("")

    # ==========================================================================
    # 1. EXECUTIVE SUMMARY
    # ==========================================================================
    lines.extend(_generate_executive_summary(summary))

    # ==========================================================================
    # 2. DETECTION CONFIGURATION
    # ==========================================================================
    lines.extend(_generate_detection_config_section())

    # ==========================================================================
    # 3. PERFORMANCE METRICS
    # ==========================================================================
    lines.extend(_generate_performance_metrics(summary))

    # ==========================================================================
    # 4. FALSE POSITIVES ANALYSIS
    # ==========================================================================
    lines.extend(_generate_fp_analysis_section(results, frame_tolerance))

    # ==========================================================================
    # 5. FALSE NEGATIVES ANALYSIS
    # ==========================================================================
    lines.extend(_generate_fn_analysis_section(results))

    # ==========================================================================
    # 6. KNOWN ISSUES & BUGS
    # ==========================================================================
    lines.extend(_generate_known_issues_section())

    # ==========================================================================
    # 7. RECOMMENDATIONS
    # ==========================================================================
    categorized_fps = _categorize_false_positives(results, frame_tolerance)
    lines.extend(_generate_recommendations_section(summary, categorized_fps))

    # ==========================================================================
    # 8. DETAILED PLAYER RESULTS (by category)
    # ==========================================================================
    lines.extend(_generate_detailed_results_by_category(results))

    # ==========================================================================
    # 9. PER-PLAYER DEBUG SECTIONS (full TP/FP/FN details)
    # ==========================================================================
    lines.append("=" * 80)
    lines.append("9. PER-PLAYER DEBUG SECTIONS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Detailed breakdown for each player - use these sections to debug")
    lines.append("detection results while watching the corresponding video.")
    lines.append("")

    for player_name in sorted(results.keys()):
        result = results[player_name]
        lines.append(f"{'─' * 80}")
        lines.append(f"PLAYER: {player_name.upper()}")
        lines.append(f"{'─' * 80}")
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
                    diff_str = f"Δ: {sign}{m.frame_difference} frames"
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

    # ==========================================================================
    # VISUAL SUMMARY (at the end)
    # ==========================================================================
    lines.append("=" * 80)
    lines.append("VISUAL SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    total_events = summary.total_true_positives + summary.total_false_positives + summary.total_false_negatives
    if total_events > 0:
        tp_bar = "█" * max(1, int(summary.total_true_positives / total_events * 40))
        fp_bar = "░" * max(1, int(summary.total_false_positives / total_events * 40))
        fn_bar = "▒" * max(1, int(summary.total_false_negatives / total_events * 40))
        lines.append(f"  TP: {tp_bar} ({summary.total_true_positives})")
        lines.append(f"  FP: {fp_bar} ({summary.total_false_positives})")
        lines.append(f"  FN: {fn_bar} ({summary.total_false_negatives})")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
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
