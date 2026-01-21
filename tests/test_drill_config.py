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


from pathlib import Path


class TestDrillTypesConfigFile:
    """Tests for drill_types_config.yaml file."""

    def test_config_file_exists(self):
        """Test that the config file exists."""
        config_path = Path(__file__).parent.parent / "detection" / "drill_types_config.yaml"
        assert config_path.exists(), f"Config file not found at {config_path}"

    def test_config_file_has_drill_types(self):
        """Test that config file has drill_types key."""
        import yaml
        config_path = Path(__file__).parent.parent / "detection" / "drill_types_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "drill_types" in config
        assert len(config["drill_types"]) >= 1

    def test_triple_cone_defined(self):
        """Test that triple_cone drill is defined."""
        import yaml
        config_path = Path(__file__).parent.parent / "detection" / "drill_types_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "triple_cone" in config["drill_types"]
        tc = config["drill_types"]["triple_cone"]
        assert tc["cone_count"] == 3
        assert len(tc["cones"]) == 3


class TestDrillConfigLoader:
    """Tests for DrillConfigLoader class."""

    def test_loader_initialization(self):
        """Test loader initializes and loads config."""
        from detection.drill_config_loader import DrillConfigLoader
        loader = DrillConfigLoader()
        assert loader is not None

    def test_list_drill_types(self):
        """Test listing available drill types."""
        from detection.drill_config_loader import DrillConfigLoader
        loader = DrillConfigLoader()
        drill_types = loader.list_drill_types()
        assert "triple_cone" in drill_types
        assert "seven_cone_weave" in drill_types
        assert "chest_control" in drill_types

    def test_get_drill_type_triple_cone(self):
        """Test getting triple_cone config."""
        from detection.drill_config_loader import DrillConfigLoader
        loader = DrillConfigLoader()
        config = loader.get_drill_type("triple_cone")

        assert config.id == "triple_cone"
        assert config.name == "Triple Cone Drill"
        assert config.cone_count == 3
        assert len(config.cones) == 3
        assert all(c.type == ConeType.TURN for c in config.cones)

    def test_get_drill_type_seven_cone(self):
        """Test getting seven_cone_weave config."""
        from detection.drill_config_loader import DrillConfigLoader
        loader = DrillConfigLoader()
        config = loader.get_drill_type("seven_cone_weave")

        assert config.cone_count == 7
        assert len(config.get_turn_cones()) == 2
        assert len(config.get_weave_cones()) == 5

    def test_get_drill_type_chest_control(self):
        """Test getting chest_control config."""
        from detection.drill_config_loader import DrillConfigLoader
        loader = DrillConfigLoader()
        config = loader.get_drill_type("chest_control")

        assert config.cone_count == 5
        assert len(config.get_turn_cones()) == 1
        assert len(config.get_area_cones()) == 4

    def test_get_unknown_drill_type_raises(self):
        """Test that unknown drill type raises ValueError."""
        from detection.drill_config_loader import DrillConfigLoader
        loader = DrillConfigLoader()

        with pytest.raises(ValueError, match="Unknown drill type"):
            loader.get_drill_type("nonexistent_drill")

    def test_get_by_cone_count(self):
        """Test finding drills by cone count."""
        from detection.drill_config_loader import DrillConfigLoader
        loader = DrillConfigLoader()

        three_cone_drills = loader.get_by_cone_count(3)
        assert len(three_cone_drills) >= 1
        assert any(d.id == "triple_cone" for d in three_cone_drills)

    def test_custom_config_path(self, tmp_path):
        """Test loading from custom config path."""
        from detection.drill_config_loader import DrillConfigLoader

        # Create a minimal config file
        config_content = """
drill_types:
  test_drill:
    name: "Test Drill"
    cone_count: 2
    cones:
      - position: 0
        type: turn
        label: cone_1
      - position: 1
        type: turn
        label: cone_2
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        loader = DrillConfigLoader(config_path=config_file)
        assert "test_drill" in loader.list_drill_types()
