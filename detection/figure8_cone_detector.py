"""
Figure-8 Cone Role Detection and Gate Passage Tracking.

This module handles:
1. Automatic cone role identification from positions
2. Manual cone annotation loading (preferred)
3. Gate passage detection
4. Drill phase and direction tracking

Cone arrangement (left to right on horizontal line):
[START] ---- [PAIR1_L] [PAIR1_R] ---- [PAIR2_L] [PAIR2_R]
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .config import Figure8DrillConfig
from .data_structures import (
    ConeRole, DrillDirection, DrillPhase,
    GatePassage, FrameData
)

logger = logging.getLogger(__name__)


class Figure8ConeDetector:
    """
    Detects cone roles and tracks gate passages for Figure-8 drill.

    Usage:
        detector = Figure8ConeDetector(config)
        cone_roles = detector.identify_cone_roles(cone_df, frame_id=0)
        detector.setup_gates(cone_roles)

        # Per-frame tracking
        passage = detector.detect_gate_passage(prev_ball_pos, curr_ball_pos, frame_id, timestamp)
        phase = detector.get_current_phase(ball_pos)
    """

    def __init__(self, config: Figure8DrillConfig, parquet_dir: Optional[str] = None):
        """
        Initialize with Figure-8 drill config.

        Args:
            config: Figure-8 drill configuration
            parquet_dir: Optional path to parquet directory containing
                        cone_annotations.json for manual annotations
        """
        self.config = config
        self.parquet_dir = Path(parquet_dir) if parquet_dir else None
        self._cone_roles: List[ConeRole] = []
        self._gate1_center: Optional[Tuple[float, float]] = None
        self._gate2_center: Optional[Tuple[float, float]] = None
        self._gate1_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self._gate2_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self._start_cone_pos: Optional[Tuple[float, float]] = None

        # State tracking
        self._current_direction = DrillDirection.STATIONARY
        self._current_phase = DrillPhase.AT_START
        self._lap_count = 0
        self._gate_passages: List[GatePassage] = []

        # Gate crossing tracking
        self._last_g1_side: Optional[str] = None  # "left" or "right"
        self._last_g2_side: Optional[str] = None

        logger.info("Figure8ConeDetector initialized")

    def _load_manual_annotations(self) -> Optional[List[ConeRole]]:
        """
        Load manual cone annotations from JSON file if available.

        Returns:
            List of ConeRole objects or None if no annotations found
        """
        if self.parquet_dir is None:
            return None

        annotation_file = self.parquet_dir / "cone_annotations.json"
        if not annotation_file.exists():
            return None

        try:
            with open(annotation_file) as f:
                data = json.load(f)

            cones = data.get("cones", {})
            roles = []

            for i, role_name in enumerate(["start", "gate1_left", "gate1_right",
                                           "gate2_left", "gate2_right"]):
                if role_name not in cones:
                    logger.warning(f"Missing cone role in annotations: {role_name}")
                    return None

                cone_data = cones[role_name]
                roles.append(ConeRole(
                    cone_id=i + 1,  # Assign sequential IDs for manual annotations
                    role=role_name,
                    field_x=cone_data["px"],  # Use pixel coordinates from JSON
                    field_y=cone_data["py"],
                ))

            logger.info(f"Loaded manual cone annotations from {annotation_file}")
            return roles

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading annotations: {e}")
            return None

    def identify_cone_roles(
        self,
        cone_df: pd.DataFrame,
        frame_id: int = 0
    ) -> List[ConeRole]:
        """
        Identify cone roles from their positions.

        First checks for manual annotations (cone_annotations.json).
        If not found, falls back to automatic detection.

        Algorithm for automatic detection:
        1. Get all cone positions for the frame
        2. Sort by field_x (horizontal position)
        3. Identify the start cone (leftmost, alone)
        4. Identify pair 1 (next two closest cones)
        5. Identify pair 2 (remaining two cones)

        Args:
            cone_df: Cone detection DataFrame
            frame_id: Frame to analyze (default: 0 for first frame)

        Returns:
            List of ConeRole assignments
        """
        # Try manual annotations first
        manual_roles = self._load_manual_annotations()
        if manual_roles:
            self._cone_roles = manual_roles
            return manual_roles

        # Fall back to automatic detection
        logger.info("No manual annotations found, using automatic detection")

        # Get cones for this frame
        frame_cones = cone_df[cone_df['frame_id'] == frame_id].copy()

        if len(frame_cones) < 5:
            logger.warning(f"Expected 5 cones, found {len(frame_cones)}")

        # Sort by field_x position (left to right)
        frame_cones = frame_cones.sort_values('field_center_x')

        cone_positions = []
        for _, row in frame_cones.iterrows():
            cone_positions.append({
                'object_id': int(row['object_id']),
                'field_x': row['field_center_x'],
                'field_y': row['field_center_y'],
            })

        # Compute distances between adjacent cones
        distances = []
        for i in range(len(cone_positions) - 1):
            d = np.sqrt(
                (cone_positions[i+1]['field_x'] - cone_positions[i]['field_x'])**2 +
                (cone_positions[i+1]['field_y'] - cone_positions[i]['field_y'])**2
            )
            distances.append({
                'index': i,
                'distance': d,
                'cone1_id': cone_positions[i]['object_id'],
                'cone2_id': cone_positions[i+1]['object_id'],
            })

        logger.debug(f"Adjacent cone distances: {distances}")

        # Identify cone roles based on distance patterns
        # Expected pattern: [large gap] [small gap (gate1)] [medium gap] [small gap (gate2)]
        # OR: [START] -- [PAIR1] -- [PAIR2]
        roles = self._assign_roles_by_clustering(cone_positions, distances)

        self._cone_roles = roles
        return roles

    def _assign_roles_by_clustering(
        self,
        cone_positions: List[Dict],
        distances: List[Dict]
    ) -> List[ConeRole]:
        """
        Assign cone roles using distance clustering.

        The two smallest distances are gate widths (within pairs).
        The two largest distances are gaps between groups.
        """
        if len(cone_positions) < 5:
            # Handle fewer cones gracefully
            return self._assign_roles_fallback(cone_positions)

        # Sort distances
        sorted_distances = sorted(distances, key=lambda x: x['distance'])

        # Two smallest = gate widths (pairs)
        gate_gaps = sorted_distances[:2]
        gate_gap_indices = {d['index'] for d in gate_gaps}

        # Identify which indices are gate pairs
        # Gate 1 should be index 1-2 (cones 2,3), Gate 2 should be index 3-4 (cones 4,5)
        # Index 0 is start-to-pair1, index 2 is pair1-to-pair2

        roles = []

        # Cone 0 (leftmost) is always start
        roles.append(ConeRole(
            cone_id=cone_positions[0]['object_id'],
            role="start",
            field_x=cone_positions[0]['field_x'],
            field_y=cone_positions[0]['field_y'],
        ))

        # Determine gate assignments based on gap pattern
        if 1 in gate_gap_indices:
            # Cones 1,2 form gate 1
            roles.append(ConeRole(
                cone_id=cone_positions[1]['object_id'],
                role="gate1_left",
                field_x=cone_positions[1]['field_x'],
                field_y=cone_positions[1]['field_y'],
            ))
            roles.append(ConeRole(
                cone_id=cone_positions[2]['object_id'],
                role="gate1_right",
                field_x=cone_positions[2]['field_x'],
                field_y=cone_positions[2]['field_y'],
            ))
        else:
            # Assume standard positions for gate 1
            roles.append(ConeRole(
                cone_id=cone_positions[1]['object_id'],
                role="gate1_left",
                field_x=cone_positions[1]['field_x'],
                field_y=cone_positions[1]['field_y'],
            ))
            roles.append(ConeRole(
                cone_id=cone_positions[2]['object_id'],
                role="gate1_right",
                field_x=cone_positions[2]['field_x'],
                field_y=cone_positions[2]['field_y'],
            ))

        if len(cone_positions) >= 5:
            if 3 in gate_gap_indices:
                # Cones 3,4 form gate 2
                roles.append(ConeRole(
                    cone_id=cone_positions[3]['object_id'],
                    role="gate2_left",
                    field_x=cone_positions[3]['field_x'],
                    field_y=cone_positions[3]['field_y'],
                ))
                roles.append(ConeRole(
                    cone_id=cone_positions[4]['object_id'],
                    role="gate2_right",
                    field_x=cone_positions[4]['field_x'],
                    field_y=cone_positions[4]['field_y'],
                ))
            else:
                roles.append(ConeRole(
                    cone_id=cone_positions[3]['object_id'],
                    role="gate2_left",
                    field_x=cone_positions[3]['field_x'],
                    field_y=cone_positions[3]['field_y'],
                ))
                roles.append(ConeRole(
                    cone_id=cone_positions[4]['object_id'],
                    role="gate2_right",
                    field_x=cone_positions[4]['field_x'],
                    field_y=cone_positions[4]['field_y'],
                ))

        logger.info(f"Assigned cone roles: {[r.role for r in roles]}")
        return roles

    def _assign_roles_fallback(self, cone_positions: List[Dict]) -> List[ConeRole]:
        """Fallback role assignment when cone count is unexpected."""
        roles = []
        role_names = ["start", "gate1_left", "gate1_right", "gate2_left", "gate2_right"]

        for i, cone in enumerate(cone_positions):
            role = role_names[i] if i < len(role_names) else f"extra_{i}"
            roles.append(ConeRole(
                cone_id=cone['object_id'],
                role=role,
                field_x=cone['field_x'],
                field_y=cone['field_y'],
            ))

        return roles

    def setup_gates(self, cone_roles: Optional[List[ConeRole]] = None):
        """
        Set up gate geometry from cone roles.

        Must be called after identify_cone_roles() or with explicit roles.
        """
        roles = cone_roles or self._cone_roles
        if not roles:
            raise ValueError("No cone roles available. Call identify_cone_roles() first.")

        role_map = {r.role: r for r in roles}

        # Start cone position
        if 'start' in role_map:
            self._start_cone_pos = (role_map['start'].field_x, role_map['start'].field_y)

        # Gate 1 setup
        if 'gate1_left' in role_map and 'gate1_right' in role_map:
            g1_left = role_map['gate1_left']
            g1_right = role_map['gate1_right']

            self._gate1_center = (
                (g1_left.field_x + g1_right.field_x) / 2,
                (g1_left.field_y + g1_right.field_y) / 2,
            )
            self._gate1_line = (
                (g1_left.field_x, g1_left.field_y),
                (g1_right.field_x, g1_right.field_y),
            )

            # Update config
            self.config.gate1_cone_ids = (g1_left.cone_id, g1_right.cone_id)

        # Gate 2 setup
        if 'gate2_left' in role_map and 'gate2_right' in role_map:
            g2_left = role_map['gate2_left']
            g2_right = role_map['gate2_right']

            self._gate2_center = (
                (g2_left.field_x + g2_right.field_x) / 2,
                (g2_left.field_y + g2_right.field_y) / 2,
            )
            self._gate2_line = (
                (g2_left.field_x, g2_left.field_y),
                (g2_right.field_x, g2_right.field_y),
            )

            # Update config
            self.config.gate2_cone_ids = (g2_left.cone_id, g2_right.cone_id)

        self.config.update_gate_definitions()

        logger.info(f"Gates setup complete:")
        logger.info(f"  Gate 1 center: {self._gate1_center}")
        logger.info(f"  Gate 2 center: {self._gate2_center}")

    def detect_gate_passage(
        self,
        prev_ball_pos: Tuple[float, float],
        curr_ball_pos: Tuple[float, float],
        frame_id: int,
        timestamp: float,
        player_pos: Tuple[float, float],
        ball_controlled: bool
    ) -> Optional[GatePassage]:
        """
        Detect if ball passed through a gate between frames.

        Args:
            prev_ball_pos: Ball position in previous frame (field coords)
            curr_ball_pos: Ball position in current frame (field coords)
            frame_id: Current frame number
            timestamp: Current timestamp
            player_pos: Player position (field coords)
            ball_controlled: Whether ball was under control

        Returns:
            GatePassage if passage detected, None otherwise
        """
        # Check Gate 1
        if self._gate1_line:
            g1_crossed = self._check_line_crossing(
                prev_ball_pos, curr_ball_pos, self._gate1_line
            )
            if g1_crossed:
                direction = self._determine_direction(prev_ball_pos, curr_ball_pos)
                quality = self._calculate_passage_quality(
                    curr_ball_pos, self._gate1_center, self._gate1_line
                )
                passage = GatePassage(
                    gate_id="G1",
                    direction=direction,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    ball_position=curr_ball_pos,
                    player_position=player_pos,
                    ball_controlled=ball_controlled,
                    passage_quality=quality,
                )
                self._gate_passages.append(passage)
                logger.debug(f"Gate 1 passage: {direction.value} at frame {frame_id}")
                return passage

        # Check Gate 2
        if self._gate2_line:
            g2_crossed = self._check_line_crossing(
                prev_ball_pos, curr_ball_pos, self._gate2_line
            )
            if g2_crossed:
                direction = self._determine_direction(prev_ball_pos, curr_ball_pos)
                quality = self._calculate_passage_quality(
                    curr_ball_pos, self._gate2_center, self._gate2_line
                )
                passage = GatePassage(
                    gate_id="G2",
                    direction=direction,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    ball_position=curr_ball_pos,
                    player_position=player_pos,
                    ball_controlled=ball_controlled,
                    passage_quality=quality,
                )
                self._gate_passages.append(passage)
                logger.debug(f"Gate 2 passage: {direction.value} at frame {frame_id}")
                return passage

        return None

    def _check_line_crossing(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float],
        gate_line: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> bool:
        """
        Check if movement from prev_pos to curr_pos crosses the gate line.

        Uses line segment intersection algorithm.
        """
        p1, p2 = gate_line
        p3, p4 = prev_pos, curr_pos

        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        # Check if line segments intersect
        if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
            # Also verify the crossing is within gate width
            # (not around the cones)
            crossing_point = self._get_intersection_point(
                (p3, p4), (p1, p2)
            )
            if crossing_point:
                gate_center = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                gate_width = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                dist_from_center = np.sqrt(
                    (crossing_point[0]-gate_center[0])**2 +
                    (crossing_point[1]-gate_center[1])**2
                )
                return dist_from_center <= gate_width / 2 + self.config.gate_passage_margin

        return False

    def _get_intersection_point(
        self,
        line1: Tuple[Tuple[float, float], Tuple[float, float]],
        line2: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Get intersection point of two line segments."""
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom

        px = x1 + t*(x2-x1)
        py = y1 + t*(y2-y1)

        return (px, py)

    def _determine_direction(
        self,
        prev_pos: Tuple[float, float],
        curr_pos: Tuple[float, float]
    ) -> DrillDirection:
        """Determine direction of movement based on X coordinate change."""
        dx = curr_pos[0] - prev_pos[0]
        if abs(dx) < 5:  # Small threshold for stationary
            return DrillDirection.STATIONARY
        return DrillDirection.FORWARD if dx > 0 else DrillDirection.BACKWARD

    def _calculate_passage_quality(
        self,
        ball_pos: Tuple[float, float],
        gate_center: Tuple[float, float],
        gate_line: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> float:
        """
        Calculate quality of gate passage (0-1).

        Higher score = closer to gate center.
        """
        gate_width = np.sqrt(
            (gate_line[1][0] - gate_line[0][0])**2 +
            (gate_line[1][1] - gate_line[0][1])**2
        )

        dist_from_center = np.sqrt(
            (ball_pos[0] - gate_center[0])**2 +
            (ball_pos[1] - gate_center[1])**2
        )

        # Score: 1.0 at center, 0.0 at edge
        quality = max(0.0, 1.0 - (dist_from_center / (gate_width / 2)))
        return quality

    def get_current_phase(
        self,
        ball_pos: Tuple[float, float],
        direction: DrillDirection
    ) -> DrillPhase:
        """
        Determine current drill phase based on ball position.

        Args:
            ball_pos: Current ball position (field coords)
            direction: Current movement direction

        Returns:
            Current DrillPhase
        """
        if not self._gate1_center or not self._gate2_center:
            return DrillPhase.AT_START

        ball_x = ball_pos[0]

        # Define phase boundaries based on gate positions
        g1_x = self._gate1_center[0]
        g2_x = self._gate2_center[0]
        start_x = self._start_cone_pos[0] if self._start_cone_pos else g1_x - 200

        # Buffer zone around gates
        gate_buffer = self.config.gate_passage_margin

        if ball_x < g1_x - gate_buffer:
            if direction == DrillDirection.FORWARD:
                return DrillPhase.APPROACHING_G1
            else:
                return DrillPhase.AT_START

        elif abs(ball_x - g1_x) <= gate_buffer:
            return DrillPhase.PASSING_G1

        elif g1_x + gate_buffer < ball_x < g2_x - gate_buffer:
            if direction == DrillDirection.FORWARD:
                return DrillPhase.APPROACHING_G2
            else:
                return DrillPhase.BETWEEN_GATES

        elif abs(ball_x - g2_x) <= gate_buffer:
            return DrillPhase.PASSING_G2

        else:  # ball_x > g2_x + gate_buffer
            return DrillPhase.AT_TURN

    def update_lap_count(self, passage: GatePassage):
        """
        Update lap count based on gate passage.

        A lap is complete when player passes G1 → G2 → G2 → G1 (back to start).
        """
        if len(self._gate_passages) < 4:
            return

        # Check last 4 passages for complete lap pattern
        last_four = self._gate_passages[-4:]
        expected_pattern = [
            ("G1", DrillDirection.FORWARD),
            ("G2", DrillDirection.FORWARD),
            ("G2", DrillDirection.BACKWARD),
            ("G1", DrillDirection.BACKWARD),
        ]

        matches = all(
            p.gate_id == exp[0] and p.direction == exp[1]
            for p, exp in zip(last_four, expected_pattern)
        )

        if matches:
            self._lap_count += 1
            logger.info(f"Lap {self._lap_count} completed!")

    @property
    def cone_roles(self) -> List[ConeRole]:
        """Get assigned cone roles."""
        return self._cone_roles

    @property
    def gate_passages(self) -> List[GatePassage]:
        """Get all recorded gate passages."""
        return self._gate_passages

    @property
    def lap_count(self) -> int:
        """Get current lap count."""
        return self._lap_count

    def reset(self):
        """Reset tracking state for new detection run."""
        self._current_direction = DrillDirection.STATIONARY
        self._current_phase = DrillPhase.AT_START
        self._lap_count = 0
        self._gate_passages = []
        self._last_g1_side = None
        self._last_g2_side = None
