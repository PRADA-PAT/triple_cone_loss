"""
Data loaders for Triple Cone annotation.

Handles loading and parsing parquet files for cone, ball, and pose data.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .structures import ConeData


def read_parquet_safe(parquet_path: Path) -> pd.DataFrame:
    """
    Read parquet file with fallback for uint32 dictionary encoding issue.

    Some parquet files use uint32 dictionary indices which pandas/pyarrow
    doesn't support directly. This function handles that case by using
    pyarrow to decode dictionaries first.
    """
    try:
        # Try normal pandas read first
        return pd.read_parquet(parquet_path)
    except Exception as e:
        if "unsigned dictionary indices" in str(e) or "uint32" in str(e):
            # Fallback: use pyarrow and decode dictionaries manually
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_path)

            # Decode dictionary columns to regular columns
            new_columns = []
            for i, field in enumerate(table.schema):
                col = table.column(i)
                if pa.types.is_dictionary(field.type):
                    # For ChunkedArray, decode each chunk and combine
                    decoded_chunks = [chunk.dictionary_decode() for chunk in col.chunks]
                    new_col = pa.chunked_array(decoded_chunks)
                    new_columns.append(new_col)
                else:
                    new_columns.append(col)

            # Rebuild table with decoded columns
            new_table = pa.table(
                {field.name: new_columns[i] for i, field in enumerate(table.schema)}
            )
            return new_table.to_pandas()
        else:
            raise


def load_cone_positions_from_parquet(parquet_path: Path) -> Tuple[ConeData, ConeData, ConeData]:
    """
    Load cone positions and bounding box dimensions from parquet file.

    Returns (cone1, cone2, cone3) sorted by X (left to right).
    CONE1 = leftmost (HOME), CONE2 = center, CONE3 = rightmost.
    Each cone includes center position and mean bbox dimensions.
    """
    cone_df = read_parquet_safe(parquet_path)

    # Filter out NaN object_ids
    cone_df = cone_df[cone_df['object_id'].notna()]

    # Group by object_id and get mean position + dimensions
    cones = []
    for obj_id in sorted(cone_df['object_id'].unique()):
        obj_data = cone_df[cone_df['object_id'] == obj_id]
        mean_x = obj_data['center_x'].mean()
        mean_y = obj_data['center_y'].mean()
        mean_width = obj_data['width'].mean() if 'width' in obj_data.columns else 15.0
        mean_height = obj_data['height'].mean() if 'height' in obj_data.columns else 15.0

        # Skip positions with NaN values
        if pd.notna(mean_x) and pd.notna(mean_y):
            cones.append(ConeData(
                center_x=mean_x,
                center_y=mean_y,
                width=mean_width if pd.notna(mean_width) else 15.0,
                height=mean_height if pd.notna(mean_height) else 15.0,
            ))

    # Sort by X position (left to right = CONE1, CONE2, CONE3)
    cones.sort(key=lambda c: c.center_x)

    if len(cones) < 3:
        raise ValueError(f"Expected 3 cones, found {len(cones)}")

    return (cones[0], cones[1], cones[2])


def load_ball_data(parquet_path: Path) -> pd.DataFrame:
    """Load ball detection data including interpolated flag for off-screen detection."""
    df = read_parquet_safe(parquet_path)
    cols = ['frame_id', 'x1', 'y1', 'x2', 'y2', 'confidence']
    if 'interpolated' in df.columns:
        cols.append('interpolated')
    return df[cols].copy()


def load_pose_data(parquet_path: Path) -> pd.DataFrame:
    """Load pose keypoint data."""
    df = read_parquet_safe(parquet_path)
    return df[['frame_idx', 'person_id', 'keypoint_name', 'x', 'y', 'confidence']].copy()


def prepare_pose_lookup(pose_df: pd.DataFrame) -> Dict[int, Dict[int, Dict[str, Tuple[float, float, float]]]]:
    """Create efficient lookup for pose data."""
    lookup = {}
    for _, row in pose_df.iterrows():
        frame_id = int(row['frame_idx'])
        person_id = int(row['person_id'])
        keypoint = row['keypoint_name']

        if frame_id not in lookup:
            lookup[frame_id] = {}
        if person_id not in lookup[frame_id]:
            lookup[frame_id][person_id] = {}

        lookup[frame_id][person_id][keypoint] = (
            float(row['x']),
            float(row['y']),
            float(row['confidence'])
        )
    return lookup
