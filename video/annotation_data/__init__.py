"""
Data structures and loaders for Triple Cone annotation.
"""

from .structures import (
    BallPositionResult,
    IntentionPositionResult,
    EdgeZoneStatus,
    BallTrackingState,
    TurnEvent,
    ConeData,
)

from .loaders import (
    read_parquet_safe,
    load_cone_positions_from_parquet,
    load_all_cone_positions,
    load_ball_data,
    load_pose_data,
    prepare_pose_lookup,
)

__all__ = [
    # Structures
    'BallPositionResult',
    'IntentionPositionResult',
    'EdgeZoneStatus',
    'BallTrackingState',
    'TurnEvent',
    'ConeData',
    # Loaders
    'read_parquet_safe',
    'load_cone_positions_from_parquet',
    'load_all_cone_positions',
    'load_ball_data',
    'load_pose_data',
    'prepare_pose_lookup',
]
