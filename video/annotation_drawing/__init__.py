"""
Drawing functions for Triple Cone annotation.
"""

from .sidebar import (
    draw_sidebar_section_header,
    draw_sidebar_row,
    draw_sidebar,
)

from .primitives import (
    draw_bbox,
    draw_triple_cone_markers,
    draw_skeleton,
    get_momentum_color,
    draw_momentum_arrow,
    get_ball_momentum_color,
    draw_ball_momentum_arrow,
    calculate_ball_vertical_deviation,
    draw_intention_arrow,
    draw_dashed_line,
    draw_debug_axes,
    draw_edge_zones,
)

from .indicators import (
    draw_ball_position_indicator,
    draw_intention_position_indicator,
    draw_behind_counter,
    draw_intention_behind_counter,
    draw_edge_counter,
    draw_off_screen_indicator,
    draw_return_counter,
    draw_unified_tracking_indicator,
    draw_vertical_deviation_counter,
)

__all__ = [
    # Sidebar
    'draw_sidebar_section_header',
    'draw_sidebar_row',
    'draw_sidebar',
    # Primitives
    'draw_bbox',
    'draw_triple_cone_markers',
    'draw_skeleton',
    'get_momentum_color',
    'draw_momentum_arrow',
    'get_ball_momentum_color',
    'draw_ball_momentum_arrow',
    'calculate_ball_vertical_deviation',
    'draw_intention_arrow',
    'draw_dashed_line',
    'draw_debug_axes',
    'draw_edge_zones',
    # Indicators
    'draw_ball_position_indicator',
    'draw_intention_position_indicator',
    'draw_behind_counter',
    'draw_intention_behind_counter',
    'draw_edge_counter',
    'draw_off_screen_indicator',
    'draw_return_counter',
    'draw_unified_tracking_indicator',
    'draw_vertical_deviation_counter',
]
