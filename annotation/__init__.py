# Annotation module - Cone annotation and visualization tools
"""
Tools for annotating cone positions and visualizing drill data.

This module contains:
- ConeAnnotator: Interactive GUI tool for marking cone bounding boxes
- DrillVisualizer: Debug visualization for detection overlays
- annotate_cones: Utility functions for cone annotation
"""

from .cone_annotator import ConeAnnotator
from .drill_visualizer import DrillVisualizer

__all__ = [
    'ConeAnnotator',
    'DrillVisualizer',
]
