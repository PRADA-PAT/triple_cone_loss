"""Tests for horizontal distance escape detection."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video.annotation_config import TripleConeAnnotationConfig
from video.annotation_analysis.ball_position import calculate_horizontal_distance


@pytest.fixture
def config():
    return TripleConeAnnotationConfig()


def test_none_ball_returns_none(config):
    result = calculate_horizontal_distance(None, (500, 400), config)
    assert result is None


def test_none_hip_returns_none(config):
    result = calculate_horizontal_distance((500, 400), None, config)
    assert result is None


def test_safe_zone(config):
    # Ball 30px from hip (< 80px warning threshold)
    result = calculate_horizontal_distance((530, 400), (500, 400), config)
    assert result.zone == "SAFE"
    assert result.distance == 30.0
    assert result.color == config.H_DIST_SAFE_COLOR


def test_warning_zone(config):
    # Ball 100px from hip (80-120px)
    result = calculate_horizontal_distance((600, 400), (500, 400), config)
    assert result.zone == "WARNING"
    assert result.distance == 100.0
    assert result.color == config.H_DIST_WARNING_COLOR


def test_escaped_zone(config):
    # Ball 150px from hip (> 120px)
    result = calculate_horizontal_distance((650, 400), (500, 400), config)
    assert result.zone == "ESCAPED"
    assert result.distance == 150.0
    assert result.color == config.H_DIST_ESCAPED_COLOR


def test_distance_is_absolute(config):
    # Ball to LEFT of hip — distance should still be positive
    result = calculate_horizontal_distance((400, 400), (500, 400), config)
    assert result.distance == 100.0
    assert result.zone == "WARNING"


def test_exact_warning_boundary(config):
    # Exactly at warning threshold — should be WARNING
    result = calculate_horizontal_distance((580, 400), (500, 400), config)
    assert result.zone == "WARNING"
    assert result.distance == 80.0


def test_exact_escaped_boundary(config):
    # Exactly at escaped threshold — should be ESCAPED
    result = calculate_horizontal_distance((620, 400), (500, 400), config)
    assert result.zone == "ESCAPED"
    assert result.distance == 120.0
