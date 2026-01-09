"""
Turning Zones Module for Triple Cone Drill Analysis.

Defines elliptical turning zones around cones where players turn during drills.
Zones are elliptical (not circular) to compensate for camera perspective distortion.

Key components:
- TurningZone: Ellipse geometry with point-in-zone detection
- TripleConeZoneConfig: Configuration for 3-zone triple cone drills
- TripleConeZoneSet: Container for 3 zones (CONE1, CONE2, CONE3)
- create_triple_cone_zones(): Factory for triple cone zones
- draw_triple_cone_zones(): Video visualization

Usage:
    from detection.turning_zones import create_triple_cone_zones

    # Cone positions from parquet analysis (mean positions)
    cone1 = (467, 801)   # LEFT/HOME
    cone2 = (1393, 791)  # CENTER
    cone3 = (2316, 778)  # RIGHT

    zones = create_triple_cone_zones(cone1, cone2, cone3)

    if zones.is_in_turning_zone(ball_x, ball_y):
        print(f"Ball in: {zones.get_zone_at_point(ball_x, ball_y)}")
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import OpenCV for drawing functions
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TurningZone:
    """
    Elliptical turning zone for Triple Cone drill.

    The ellipse is defined by center point, semi-axes, and optional rotation.
    Camera perspective distortion is handled via stretch factors applied
    during zone creation (semi_major vs semi_minor).

    Attributes:
        name: Zone identifier ("CONE1", "CONE2", or "CONE3")
        center_px: X-coordinate of ellipse center (pixels)
        center_py: Y-coordinate of ellipse center (pixels)
        semi_major: Semi-major axis length (pixels) - typically horizontal
        semi_minor: Semi-minor axis length (pixels) - typically vertical
        rotation_deg: Rotation angle in degrees (0 = axes aligned with frame)
    """
    name: str
    center_px: float
    center_py: float
    semi_major: float
    semi_minor: float
    rotation_deg: float = 0.0

    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is inside the ellipse.

        Uses standard ellipse equation with rotation:
        For point (x', y') translated to ellipse center and rotated:
        (x'/a)² + (y'/b)² <= 1

        Args:
            x: X-coordinate to test (pixels)
            y: Y-coordinate to test (pixels)

        Returns:
            True if point is inside or on the ellipse boundary
        """
        # Translate point to ellipse-centered coordinates
        dx = x - self.center_px
        dy = y - self.center_py

        # Apply rotation (rotate point by negative angle to align with ellipse axes)
        theta = math.radians(self.rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Rotated coordinates
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t

        # Check ellipse equation: (x/a)² + (y/b)² <= 1
        if self.semi_major == 0 or self.semi_minor == 0:
            return False

        normalized = (x_rot / self.semi_major) ** 2 + (y_rot / self.semi_minor) ** 2
        return normalized <= 1.0

    def distance_to_center(self, x: float, y: float) -> float:
        """Calculate distance from point to zone center."""
        return math.sqrt((x - self.center_px) ** 2 + (y - self.center_py) ** 2)

    def get_boundary_points(self, num_points: int = 64) -> List[Tuple[int, int]]:
        """
        Generate points along the ellipse boundary.

        Args:
            num_points: Number of points to generate around ellipse

        Returns:
            List of (x, y) integer tuples forming ellipse boundary
        """
        points = []
        theta_rad = math.radians(self.rotation_deg)
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        for i in range(num_points):
            angle = 2 * math.pi * i / num_points

            # Parametric ellipse (before rotation)
            x_local = self.semi_major * math.cos(angle)
            y_local = self.semi_minor * math.sin(angle)

            # Apply rotation
            x_rot = x_local * cos_t - y_local * sin_t
            y_rot = x_local * sin_t + y_local * cos_t

            # Translate to center
            x_final = int(self.center_px + x_rot)
            y_final = int(self.center_py + y_rot)

            points.append((x_final, y_final))

        return points

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'name': self.name,
            'center_px': self.center_px,
            'center_py': self.center_py,
            'semi_major': self.semi_major,
            'semi_minor': self.semi_minor,
            'rotation_deg': self.rotation_deg,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TurningZone':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            name=data['name'],
            center_px=data['center_px'],
            center_py=data['center_py'],
            semi_major=data['semi_major'],
            semi_minor=data['semi_minor'],
            rotation_deg=data.get('rotation_deg', 0.0),
        )


# =============================================================================
# TRIPLE CONE DRILL ZONE STRUCTURES
# =============================================================================

@dataclass
class TripleConeZoneConfig:
    """
    Configuration for creating triple cone drill turning zones.

    Triple cone drill has 3 cones in a horizontal line:
    - CONE1 (LEFT): Home cone where player starts/returns
    - CONE2 (CENTER): Middle cone
    - CONE3 (RIGHT): Far right cone

    All 3 cones are turn points in the drill sequence:
    CONE1 → CONE2(turn) → CONE1(turn) → CONE3(turn) → CONE1(turn) → repeat

    Attributes:
        cone1_zone_radius: Base radius for CONE1/HOME zone (pixels)
        cone2_zone_radius: Base radius for CONE2/CENTER zone (pixels)
        cone3_zone_radius: Base radius for CONE3/RIGHT zone (pixels)
        stretch_x: Horizontal stretch factor (default: 1.0)
        stretch_y: Vertical compression factor for side-view camera (default: 5.0)
        cone1_zone_rotation: Rotation of CONE1 zone ellipse (degrees)
        cone2_zone_rotation: Rotation of CONE2 zone ellipse (degrees)
        cone3_zone_rotation: Rotation of CONE3 zone ellipse (degrees)
    """
    cone1_zone_radius: float = 150.0  # HOME cone
    cone2_zone_radius: float = 150.0  # CENTER cone
    cone3_zone_radius: float = 150.0  # RIGHT cone
    stretch_x: float = 1.0
    stretch_y: float = 5.0  # Heavy horizontal stretch for side-view camera
    cone1_zone_rotation: float = 0.0
    cone2_zone_rotation: float = 0.0
    cone3_zone_rotation: float = 0.0

    @classmethod
    def default(cls) -> 'TripleConeZoneConfig':
        """Create default configuration with equal zone sizes."""
        return cls()

    @classmethod
    def for_overhead_camera(cls) -> 'TripleConeZoneConfig':
        """Configuration for nearly overhead camera (less distortion)."""
        return cls(stretch_y=1.1)

    @classmethod
    def for_tilted_camera(cls) -> 'TripleConeZoneConfig':
        """Configuration for tilted camera (more distortion)."""
        return cls(stretch_y=1.5)

    @classmethod
    def small_zones(cls) -> 'TripleConeZoneConfig':
        """Configuration with smaller, tighter zones."""
        return cls(
            cone1_zone_radius=80.0,
            cone2_zone_radius=80.0,
            cone3_zone_radius=80.0
        )

    @classmethod
    def large_zones(cls) -> 'TripleConeZoneConfig':
        """Configuration with larger, more generous zones."""
        return cls(
            cone1_zone_radius=200.0,
            cone2_zone_radius=200.0,
            cone3_zone_radius=200.0
        )


@dataclass
class TripleConeZoneSet:
    """
    Container for all three turning zones in a Triple Cone drill.

    Triple cone drill pattern:
    ```
      CONE1 (LEFT)     CONE2 (CENTER)     CONE3 (RIGHT)
         "HOME"
           │
           ├──────────→ TURN ←────────┐
           │             │            │
           │←────────────┘            │
           │ TURN                     │
           ├───────────────────────→ TURN
           │                          │
           │←─────────────────────────┘
           │ TURN
         END (1 rep)
    ```

    Attributes:
        cone1_zone: Elliptical zone around CONE1 (HOME/LEFT)
        cone2_zone: Elliptical zone around CONE2 (CENTER)
        cone3_zone: Elliptical zone around CONE3 (RIGHT)
        config: Configuration used to create these zones
    """
    cone1_zone: TurningZone  # LEFT/HOME
    cone2_zone: TurningZone  # CENTER
    cone3_zone: TurningZone  # RIGHT
    config: TripleConeZoneConfig = field(default_factory=TripleConeZoneConfig)

    def get_zone_at_point(self, x: float, y: float) -> Optional[str]:
        """
        Get the name of the zone containing this point.

        Checks zones in order: CONE1, CONE2, CONE3.

        Args:
            x: X-coordinate (pixels)
            y: Y-coordinate (pixels)

        Returns:
            "CONE1", "CONE2", "CONE3", or None if point is not in any zone
        """
        if self.cone1_zone.contains_point(x, y):
            return "CONE1"
        if self.cone2_zone.contains_point(x, y):
            return "CONE2"
        if self.cone3_zone.contains_point(x, y):
            return "CONE3"
        return None

    def is_in_turning_zone(self, x: float, y: float) -> bool:
        """Check if point is in any turning zone."""
        return self.get_zone_at_point(x, y) is not None

    def get_all_zones(self) -> List[TurningZone]:
        """Get list of all zones."""
        return [self.cone1_zone, self.cone2_zone, self.cone3_zone]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'cone1_zone': self.cone1_zone.to_dict(),
            'cone2_zone': self.cone2_zone.to_dict(),
            'cone3_zone': self.cone3_zone.to_dict(),
            'config': {
                'cone1_zone_radius': self.config.cone1_zone_radius,
                'cone2_zone_radius': self.config.cone2_zone_radius,
                'cone3_zone_radius': self.config.cone3_zone_radius,
                'stretch_x': self.config.stretch_x,
                'stretch_y': self.config.stretch_y,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TripleConeZoneSet':
        """Create from dictionary (JSON deserialization)."""
        config_data = data.get('config', {})
        config = TripleConeZoneConfig(
            cone1_zone_radius=config_data.get('cone1_zone_radius', 150.0),
            cone2_zone_radius=config_data.get('cone2_zone_radius', 150.0),
            cone3_zone_radius=config_data.get('cone3_zone_radius', 150.0),
            stretch_x=config_data.get('stretch_x', 1.0),
            stretch_y=config_data.get('stretch_y', 5.0),
        )
        return cls(
            cone1_zone=TurningZone.from_dict(data['cone1_zone']),
            cone2_zone=TurningZone.from_dict(data['cone2_zone']),
            cone3_zone=TurningZone.from_dict(data['cone3_zone']),
            config=config,
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_triple_cone_zones(
    cone1_pos: Tuple[float, float],
    cone2_pos: Tuple[float, float],
    cone3_pos: Tuple[float, float],
    config: Optional[TripleConeZoneConfig] = None
) -> TripleConeZoneSet:
    """
    Factory function to create turning zones for Triple Cone drill.

    Creates three elliptical zones, one at each cone:
    - CONE1 zone: At LEFT/HOME cone (where player starts and returns)
    - CONE2 zone: At CENTER cone
    - CONE3 zone: At RIGHT cone

    Triple cone drill pattern:
    CONE1 → CONE2(turn) → CONE1(turn) → CONE3(turn) → CONE1(turn) → repeat

    The ellipse axes are determined by the base radius and stretch factors:
    - semi_major = radius * stretch_x (horizontal)
    - semi_minor = radius / stretch_y (vertical, compressed for side-view camera)

    Args:
        cone1_pos: (x, y) pixel position of CONE1 (LEFT/HOME)
        cone2_pos: (x, y) pixel position of CONE2 (CENTER)
        cone3_pos: (x, y) pixel position of CONE3 (RIGHT)
        config: TripleConeZoneConfig for radii and stretch factors (uses default if None)

    Returns:
        TripleConeZoneSet containing CONE1, CONE2, and CONE3 zones
    """
    if config is None:
        config = TripleConeZoneConfig.default()

    # CONE1 zone: HOME/LEFT cone
    cone1_zone = TurningZone(
        name="CONE1",
        center_px=cone1_pos[0],
        center_py=cone1_pos[1],
        semi_major=config.cone1_zone_radius * config.stretch_x,  # Horizontal (wider)
        semi_minor=config.cone1_zone_radius / config.stretch_y,  # Vertical (compressed)
        rotation_deg=config.cone1_zone_rotation,
    )

    # CONE2 zone: CENTER cone
    cone2_zone = TurningZone(
        name="CONE2",
        center_px=cone2_pos[0],
        center_py=cone2_pos[1],
        semi_major=config.cone2_zone_radius * config.stretch_x,
        semi_minor=config.cone2_zone_radius / config.stretch_y,
        rotation_deg=config.cone2_zone_rotation,
    )

    # CONE3 zone: RIGHT cone
    cone3_zone = TurningZone(
        name="CONE3",
        center_px=cone3_pos[0],
        center_py=cone3_pos[1],
        semi_major=config.cone3_zone_radius * config.stretch_x,
        semi_minor=config.cone3_zone_radius / config.stretch_y,
        rotation_deg=config.cone3_zone_rotation,
    )

    return TripleConeZoneSet(
        cone1_zone=cone1_zone,
        cone2_zone=cone2_zone,
        cone3_zone=cone3_zone,
        config=config,
    )


# =============================================================================
# DRAWING FUNCTIONS
# =============================================================================

# Default colors (BGR format for OpenCV)
ZONE_HIGHLIGHT_COLOR = (0, 255, 255)  # Bright Yellow

# Triple Cone zone colors
CONE1_ZONE_COLOR = (200, 200, 0)      # Teal/Cyan (HOME)
CONE2_ZONE_COLOR = (200, 100, 200)    # Purple/Magenta (CENTER)
CONE3_ZONE_COLOR = (100, 200, 200)    # Orange/Yellow (RIGHT)


def draw_turning_zone(
    frame: np.ndarray,
    zone: TurningZone,
    color: Tuple[int, int, int] = CONE1_ZONE_COLOR,
    alpha: float = 0.25,
    x_offset: int = 0,
    highlight: bool = False,
    highlight_color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2,
) -> None:
    """
    Draw a turning zone ellipse on a video frame with transparency.

    Uses cv2.ellipse() for the shape and cv2.addWeighted() for transparency.

    Args:
        frame: Video frame to draw on (modified in place)
        zone: TurningZone to draw
        color: BGR color for normal state
        alpha: Transparency (0.0 = invisible, 1.0 = opaque)
        x_offset: Horizontal offset for sidebar (matches existing pattern)
        highlight: If True, use highlight_color and increased opacity
        highlight_color: Color when highlighted (default: bright yellow)
        thickness: Border thickness when not filled
    """
    if not HAS_CV2:
        return

    # Determine drawing parameters
    draw_color = highlight_color if highlight and highlight_color else color
    if highlight and highlight_color is None:
        draw_color = ZONE_HIGHLIGHT_COLOR
    draw_alpha = min(alpha + 0.15, 0.6) if highlight else alpha

    # Calculate ellipse center with offset
    center = (int(zone.center_px) + x_offset, int(zone.center_py))
    axes = (int(zone.semi_major), int(zone.semi_minor))
    angle = zone.rotation_deg

    # Create overlay for transparency
    overlay = frame.copy()

    # Draw filled ellipse on overlay
    cv2.ellipse(overlay, center, axes, angle, 0, 360, draw_color, -1)

    # Blend overlay with original frame
    cv2.addWeighted(overlay, draw_alpha, frame, 1 - draw_alpha, 0, frame)

    # Draw ellipse border (always visible)
    border_thickness = thickness + 1 if highlight else thickness
    cv2.ellipse(frame, center, axes, angle, 0, 360, draw_color, border_thickness)


def draw_triple_cone_zones(
    frame: np.ndarray,
    zones: TripleConeZoneSet,
    ball_position: Optional[Tuple[float, float]],
    x_offset: int = 0,
    cone1_color: Tuple[int, int, int] = CONE1_ZONE_COLOR,
    cone2_color: Tuple[int, int, int] = CONE2_ZONE_COLOR,
    cone3_color: Tuple[int, int, int] = CONE3_ZONE_COLOR,
    highlight_color: Tuple[int, int, int] = ZONE_HIGHLIGHT_COLOR,
    alpha: float = 0.25,
) -> Optional[str]:
    """
    Draw all three triple cone turning zones with ball-in-zone highlighting.

    Args:
        frame: Video frame to draw on
        zones: TripleConeZoneSet containing all three zones
        ball_position: (x, y) of ball center, or None if not detected
        x_offset: Sidebar offset
        cone1_color: Color for CONE1 (HOME/LEFT) zone
        cone2_color: Color for CONE2 (CENTER) zone
        cone3_color: Color for CONE3 (RIGHT) zone
        highlight_color: Color when ball is inside zone
        alpha: Base transparency

    Returns:
        Name of zone containing ball ("CONE1", "CONE2", "CONE3", or None)
    """
    if not HAS_CV2:
        return None

    # Determine if ball is in any zone
    active_zone = None
    if ball_position is not None:
        active_zone = zones.get_zone_at_point(ball_position[0], ball_position[1])

    # Draw CONE1 zone (HOME/LEFT)
    draw_turning_zone(
        frame,
        zones.cone1_zone,
        color=cone1_color,
        alpha=alpha,
        x_offset=x_offset,
        highlight=(active_zone == "CONE1"),
        highlight_color=highlight_color,
    )

    # Draw CONE2 zone (CENTER)
    draw_turning_zone(
        frame,
        zones.cone2_zone,
        color=cone2_color,
        alpha=alpha,
        x_offset=x_offset,
        highlight=(active_zone == "CONE2"),
        highlight_color=highlight_color,
    )

    # Draw CONE3 zone (RIGHT)
    draw_turning_zone(
        frame,
        zones.cone3_zone,
        color=cone3_color,
        alpha=alpha,
        x_offset=x_offset,
        highlight=(active_zone == "CONE3"),
        highlight_color=highlight_color,
    )

    return active_zone
