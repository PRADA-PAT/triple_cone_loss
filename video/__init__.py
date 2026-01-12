# Video module - Video generation with loss events marked
"""
Video generation tools for creating annotated drill videos.

This module contains:
- annotate_triple_cone: Primary video annotation with full debug overlay
- annotate_videos: Basic annotation using parquet cone detection
"""

from .annotate_videos import annotate_video

__all__ = [
    'annotate_video',
]
