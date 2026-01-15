# Video module - Video generation with loss events marked
"""
Video generation tools for creating annotated drill videos.

This module contains:
- annotate_triple_cone: Primary video annotation with full debug overlay
- annotation_config: Configuration for annotation styles
- annotation_utils: Video utilities (H.264 conversion, video discovery)
- annotation_data: Data structures and loaders
- annotation_analysis: Detection algorithms
- annotation_drawing: Drawing functions
"""

from .annotate_triple_cone import annotate_triple_cone_video
from .annotation_config import TripleConeAnnotationConfig, scale_config_for_resolution
from .annotation_utils import convert_to_h264, get_available_videos

__all__ = [
    'annotate_triple_cone_video',
    'TripleConeAnnotationConfig',
    'scale_config_for_resolution',
    'convert_to_h264',
    'get_available_videos',
]
