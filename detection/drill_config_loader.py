"""Loader for drill type configurations."""
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .data_structures import ConeType, ConeDefinition, DrillTypeConfig


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
