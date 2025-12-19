# Video module - Video generation with loss events marked
"""
Video generation tools for creating annotated drill videos.

This module contains:
- annotate_with_json_cones: Primary video annotation using static JSON cone positions
- annotate_videos: Alternative annotation using parquet cone detection
"""

from .annotate_with_json_cones import (
    annotate_video_with_json_cones,
    convert_to_h264,
    get_available_videos
)
from .annotate_videos import annotate_video

__all__ = [
    'annotate_video_with_json_cones',
    'convert_to_h264',
    'get_available_videos',
    'annotate_video',
]
