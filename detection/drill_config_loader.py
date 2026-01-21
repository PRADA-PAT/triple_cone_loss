"""Loader for drill type configurations."""
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .data_structures import ConeType, ConeDefinition, DrillTypeConfig, DetectedCone


class DrillConfigLoader:
    """Loads and provides access to drill type configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize loader with optional custom config path.

        Args:
            config_path: Path to config YAML. Defaults to drill_types_config.yaml
                        in the same directory as this module.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "drill_types_config.yaml"

        self._config_path = Path(config_path)
        self._drill_types: Dict[str, DrillTypeConfig] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load and parse the config file."""
        with open(self._config_path) as f:
            raw = yaml.safe_load(f)

        for drill_id, drill_data in raw["drill_types"].items():
            cones = [
                ConeDefinition(
                    position=c["position"],
                    type=ConeType(c["type"]),
                    label=c["label"]
                )
                for c in drill_data["cones"]
            ]

            self._drill_types[drill_id] = DrillTypeConfig(
                id=drill_id,
                name=drill_data["name"],
                cone_count=drill_data["cone_count"],
                cones=cones
            )

    def get_drill_type(self, drill_id: str) -> DrillTypeConfig:
        """Get config for a specific drill type.

        Args:
            drill_id: The drill type identifier (e.g., "triple_cone")

        Returns:
            DrillTypeConfig for the requested drill type

        Raises:
            ValueError: If drill_id is not found
        """
        if drill_id not in self._drill_types:
            raise ValueError(f"Unknown drill type: {drill_id}")
        return self._drill_types[drill_id]

    def list_drill_types(self) -> List[str]:
        """List all available drill type IDs.

        Returns:
            List of drill type identifiers
        """
        return list(self._drill_types.keys())

    def get_by_cone_count(self, count: int) -> List[DrillTypeConfig]:
        """Find drill types matching a cone count.

        Args:
            count: Number of cones to match

        Returns:
            List of DrillTypeConfig objects with matching cone count
        """
        return [d for d in self._drill_types.values() if d.cone_count == count]


def assign_cones_to_config(
    detected_positions: List[tuple],
    drill_config: DrillTypeConfig
) -> List[DetectedCone]:
    """Match detected cone positions to config definitions.

    Sorts detected positions left-to-right by X coordinate and assigns
    each to the corresponding cone definition from the config.

    Args:
        detected_positions: List of (x, y) tuples for detected cone positions
        drill_config: DrillTypeConfig specifying expected cones

    Returns:
        List of DetectedCone objects with positions linked to definitions

    Raises:
        ValueError: If number of detected positions doesn't match config
    """
    if len(detected_positions) != drill_config.cone_count:
        raise ValueError(
            f"Expected {drill_config.cone_count} cones, "
            f"detected {len(detected_positions)}"
        )

    # Sort left-to-right by x coordinate
    sorted_positions = sorted(detected_positions, key=lambda p: p[0])

    # Assign each position to its config definition
    return [
        DetectedCone(position=pos, definition=cone_def)
        for pos, cone_def in zip(sorted_positions, drill_config.cones)
    ]
