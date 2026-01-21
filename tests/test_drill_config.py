"""Tests for multi-drill configuration system."""
import pytest
from detection.data_structures import ConeType, ConeDefinition, DrillTypeConfig, DetectedCone


class TestConeType:
    """Tests for ConeType enum."""

    def test_cone_type_values(self):
        """Test all cone types exist."""
        assert ConeType.TURN.value == "turn"
        assert ConeType.AREA.value == "area"
        assert ConeType.WEAVE.value == "weave"

    def test_cone_type_from_string(self):
        """Test creating ConeType from string."""
        assert ConeType("turn") == ConeType.TURN
        assert ConeType("area") == ConeType.AREA
        assert ConeType("weave") == ConeType.WEAVE


class TestConeDefinition:
    """Tests for ConeDefinition dataclass."""

    def test_cone_definition_creation(self):
        """Test creating a cone definition."""
        cone = ConeDefinition(position=0, type=ConeType.TURN, label="turn_cone_1")
        assert cone.position == 0
        assert cone.type == ConeType.TURN
        assert cone.label == "turn_cone_1"


class TestDrillTypeConfig:
    """Tests for DrillTypeConfig dataclass."""

    def test_drill_type_config_creation(self):
        """Test creating a drill type config."""
        cones = [
            ConeDefinition(position=0, type=ConeType.TURN, label="turn_cone_3"),
            ConeDefinition(position=1, type=ConeType.TURN, label="turn_cone_2"),
            ConeDefinition(position=2, type=ConeType.TURN, label="turn_cone_1"),
        ]
        config = DrillTypeConfig(
            id="triple_cone",
            name="Triple Cone Drill",
            cone_count=3,
            cones=cones
        )
        assert config.id == "triple_cone"
        assert config.name == "Triple Cone Drill"
        assert config.cone_count == 3
        assert len(config.cones) == 3

    def test_get_turn_cones(self):
        """Test filtering turn cones."""
        cones = [
            ConeDefinition(position=0, type=ConeType.TURN, label="turn_cone_1"),
            ConeDefinition(position=1, type=ConeType.AREA, label="area_cone_1"),
            ConeDefinition(position=2, type=ConeType.AREA, label="area_cone_2"),
        ]
        config = DrillTypeConfig(id="test", name="Test", cone_count=3, cones=cones)

        turn_cones = config.get_turn_cones()
        assert len(turn_cones) == 1
        assert turn_cones[0].label == "turn_cone_1"

    def test_get_area_cones(self):
        """Test filtering area cones."""
        cones = [
            ConeDefinition(position=0, type=ConeType.TURN, label="turn_cone_1"),
            ConeDefinition(position=1, type=ConeType.AREA, label="area_cone_1"),
            ConeDefinition(position=2, type=ConeType.AREA, label="area_cone_2"),
        ]
        config = DrillTypeConfig(id="test", name="Test", cone_count=3, cones=cones)

        area_cones = config.get_area_cones()
        assert len(area_cones) == 2
        assert area_cones[0].label == "area_cone_1"

    def test_get_weave_cones(self):
        """Test filtering weave cones."""
        cones = [
            ConeDefinition(position=0, type=ConeType.TURN, label="turn_cone_1"),
            ConeDefinition(position=1, type=ConeType.WEAVE, label="cone_2"),
            ConeDefinition(position=2, type=ConeType.WEAVE, label="cone_3"),
            ConeDefinition(position=3, type=ConeType.TURN, label="turn_cone_2"),
        ]
        config = DrillTypeConfig(id="test", name="Test", cone_count=4, cones=cones)

        weave_cones = config.get_weave_cones()
        assert len(weave_cones) == 2


class TestDetectedCone:
    """Tests for DetectedCone dataclass."""

    def test_detected_cone_creation(self):
        """Test creating a detected cone."""
        definition = ConeDefinition(position=0, type=ConeType.TURN, label="turn_cone_1")
        detected = DetectedCone(position=(100.0, 200.0), definition=definition)

        assert detected.position == (100.0, 200.0)
        assert detected.definition.label == "turn_cone_1"
        assert detected.definition.type == ConeType.TURN
