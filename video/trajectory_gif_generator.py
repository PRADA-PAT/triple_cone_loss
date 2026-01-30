#!/usr/bin/env python3
"""
Ball Trajectory GIF Generator for VLM Validation

Generates animated GIFs showing ball trajectory relative to cone positions.
Designed to be fed to Vision Language Models (VLMs) to validate whether
the model can identify the Triple Cone drill pattern:
- CONE1 -> CONE2 (turn) -> CONE1 (turn) -> CONE3 (turn) -> CONE1

Usage:
    python video/trajectory_gif_generator.py <player_name>
    python video/trajectory_gif_generator.py --list  # List available players

Output:
    video/output/<player_name>_trajectory.gif
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.data_loader import load_triple_cone_layout_from_parquet, read_parquet_safe


# =============================================================================
# CONFIGURATION
# =============================================================================

class TrajectoryConfig:
    """Configuration for trajectory GIF generation."""

    # Output settings
    FPS = 30  # Real-time playback (matches video)
    DPI = 100
    FIGURE_SIZE = (10, 8)

    # Colors
    CONE_COLOR = '#FF8C00'  # Dark orange
    TRAJECTORY_COLOR = '#1E90FF'  # Dodger blue
    START_COLOR = '#00FF00'  # Green
    END_COLOR = '#FF0000'  # Red
    CURRENT_BALL_COLOR = '#FFD700'  # Gold
    BACKGROUND_COLOR = '#F0F8FF'  # Alice blue (light)

    # Marker sizes
    CONE_SIZE = 200
    START_SIZE = 300
    END_SIZE = 300
    CURRENT_BALL_SIZE = 150
    TRAJECTORY_LINE_WIDTH = 2.0

    # Padding for plot bounds (as fraction of data range)
    PADDING_FRACTION = 0.15

    # Smoothing settings (Savitzky-Golay filter)
    SMOOTHING_ENABLED = True
    SMOOTHING_WINDOW = 15  # Must be odd, larger = more smoothing
    SMOOTHING_POLYORDER = 3  # Polynomial order, preserves turns better

    # Color gradient for time-based PNG
    GRADIENT_COLORMAP = 'coolwarm'  # Blue (start) -> Red (end)


# =============================================================================
# DATA LOADING
# =============================================================================

def find_player_data_dir(player_name: str) -> Optional[Path]:
    """
    Find the data directory for a player.

    Handles various naming conventions:
    - "Alex mochar" -> "Drill_1_Triple cone Turn _dubaiacademy_Alex mochar"
    - Full directory names

    Args:
        player_name: Player name or partial name

    Returns:
        Path to player's data directory, or None if not found
    """
    base_dir = Path(__file__).parent.parent / "video_detection_pose_ball_cones"

    if not base_dir.exists():
        print(f"Data directory not found: {base_dir}")
        return None

    # Try exact match first
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            if player_name.lower() in subdir.name.lower():
                return subdir

    return None


def list_available_players() -> List[str]:
    """List all available players in the data directory."""
    base_dir = Path(__file__).parent.parent / "video_detection_pose_ball_cones"

    if not base_dir.exists():
        return []

    players = []
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir():
            # Extract player name from directory name
            # Format: "Drill_1_Triple cone Turn _dubaiacademy_<player_name>"
            name = subdir.name
            if "_dubaiacademy_" in name:
                player_name = name.split("_dubaiacademy_")[-1]
                players.append(player_name)
            else:
                players.append(name)

    return players


def load_ball_trajectory(ball_parquet_path: Path) -> pd.DataFrame:
    """
    Load ball positions and compute base center coordinates.

    Args:
        ball_parquet_path: Path to football parquet file

    Returns:
        DataFrame with columns: frame_id, x, y (base center)
    """
    df = read_parquet_safe(ball_parquet_path)

    # Compute base center (bottom center of bounding box)
    # x = center of bbox, y = bottom of bbox (y2)
    df['base_x'] = (df['x1'] + df['x2']) / 2
    df['base_y'] = df['y2']  # Bottom of bounding box

    # Filter out interpolated positions (ball not actually detected)
    if 'interpolated' in df.columns:
        df = df[df['interpolated'] == False].copy()

    # Sort by frame
    df = df.sort_values('frame_id').reset_index(drop=True)

    return df[['frame_id', 'base_x', 'base_y']].copy()


def load_cone_positions(cone_parquet_path: Path) -> List[Tuple[float, float, str]]:
    """
    Load cone positions with labels.

    Note: Parquet data sorts cones by X position (left to right),
    but in the actual drill, HOME (CONE1) is on the RIGHT side.

    Physical layout (camera view):
        LEFT          CENTER         RIGHT
        CONE3         CONE2          CONE1 (HOME)

    Drill pattern: CONE1(HOME) → CONE2 → CONE1 → CONE3 → CONE1

    Args:
        cone_parquet_path: Path to cone parquet file

    Returns:
        List of (x, y, label) tuples for each cone
    """
    layout = load_triple_cone_layout_from_parquet(str(cone_parquet_path))

    # Cones from parquet are sorted left-to-right by X position
    # But HOME (CONE1) is actually on the RIGHT in the physical drill
    return [
        (layout.cone1_x, layout.cone1_y, "CONE3"),           # Leftmost
        (layout.cone2_x, layout.cone2_y, "CONE2\n(CENTER)"), # Center
        (layout.cone3_x, layout.cone3_y, "CONE1\n(HOME)"),   # Rightmost = HOME
    ]


# =============================================================================
# SMOOTHING
# =============================================================================

def smooth_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    config: TrajectoryConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Savitzky-Golay smoothing to trajectory coordinates.

    This filter reduces noise/jitter while preserving the shape of turns,
    which is important for drill pattern recognition.

    Args:
        x: X coordinates array
        y: Y coordinates array
        config: Configuration with smoothing parameters

    Returns:
        Tuple of (smoothed_x, smoothed_y)
    """
    if not config.SMOOTHING_ENABLED or len(x) < config.SMOOTHING_WINDOW:
        return x, y

    # Ensure window size is valid (must be odd and <= data length)
    window = min(config.SMOOTHING_WINDOW, len(x))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return x, y  # Not enough data for meaningful smoothing

    polyorder = min(config.SMOOTHING_POLYORDER, window - 1)

    try:
        x_smooth = savgol_filter(x, window, polyorder)
        y_smooth = savgol_filter(y, window, polyorder)
        return x_smooth, y_smooth
    except Exception as e:
        print(f"Warning: Smoothing failed ({e}), using raw data")
        return x, y


# =============================================================================
# VISUALIZATION
# =============================================================================

def normalize_coordinates(
    ball_df: pd.DataFrame,
    cones: List[Tuple[float, float, str]],
    config: TrajectoryConfig
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, str]], Tuple[float, float, float, float]]:
    """
    Normalize coordinates to fit nicely in plot.

    Args:
        ball_df: Ball trajectory DataFrame
        cones: List of cone positions
        config: Configuration

    Returns:
        (ball_x_normalized, ball_y_normalized, cones_normalized, bounds)
    """
    # Collect all points for bounds calculation
    all_x = list(ball_df['base_x'].values) + [c[0] for c in cones]
    all_y = list(ball_df['base_y'].values) + [c[1] for c in cones]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add padding
    range_x = max_x - min_x
    range_y = max_y - min_y

    pad_x = range_x * config.PADDING_FRACTION
    pad_y = range_y * config.PADDING_FRACTION

    bounds = (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)

    # Normalize to 0-1 range
    def norm_x(x):
        return (x - (min_x - pad_x)) / (range_x + 2 * pad_x)

    def norm_y(y):
        return (y - (min_y - pad_y)) / (range_y + 2 * pad_y)

    ball_x_norm = np.array([norm_x(x) for x in ball_df['base_x']])
    ball_y_norm = np.array([norm_y(y) for y in ball_df['base_y']])

    cones_norm = [(norm_x(c[0]), norm_y(c[1]), c[2]) for c in cones]

    return ball_x_norm, ball_y_norm, cones_norm, bounds


def generate_trajectory_gif(
    player_name: str,
    output_path: Optional[Path] = None,
    config: TrajectoryConfig = None
) -> Path:
    """
    Generate trajectory GIF for a player.

    Args:
        player_name: Name of the player
        output_path: Optional output path (default: video/output/<player>_trajectory.gif)
        config: Optional configuration

    Returns:
        Path to generated GIF
    """
    if config is None:
        config = TrajectoryConfig()

    # Find player data
    data_dir = find_player_data_dir(player_name)
    if data_dir is None:
        raise ValueError(f"Player '{player_name}' not found. Use --list to see available players.")

    # Find parquet files
    ball_parquets = list(data_dir.glob("*_football.parquet"))
    cone_parquets = list(data_dir.glob("*_cone.parquet"))

    if not ball_parquets:
        raise FileNotFoundError(f"No ball parquet found in {data_dir}")
    if not cone_parquets:
        raise FileNotFoundError(f"No cone parquet found in {data_dir}")

    ball_parquet = ball_parquets[0]
    cone_parquet = cone_parquets[0]

    print(f"Loading data for {player_name}...")
    print(f"  Ball parquet: {ball_parquet.name}")
    print(f"  Cone parquet: {cone_parquet.name}")

    # Load data
    ball_df = load_ball_trajectory(ball_parquet)
    cones = load_cone_positions(cone_parquet)

    print(f"  Loaded {len(ball_df)} ball positions")
    print(f"  Loaded {len(cones)} cone positions")

    # Normalize coordinates
    ball_x, ball_y, cones_norm, bounds = normalize_coordinates(ball_df, cones, config)
    frame_ids = ball_df['frame_id'].values

    # Apply smoothing to reduce jitter
    if config.SMOOTHING_ENABLED:
        print(f"  Applying Savitzky-Golay smoothing (window={config.SMOOTHING_WINDOW})...")
        ball_x, ball_y = smooth_trajectory(ball_x, ball_y, config)

    # Setup output path
    if output_path is None:
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        # Clean player name for filename
        clean_name = player_name.replace(" ", "_").replace("/", "_")
        output_path = output_dir / f"{clean_name}_trajectory.gif"

    print(f"Generating GIF with {len(ball_df)} frames...")

    # Create figure
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, facecolor=config.BACKGROUND_COLOR)
    ax.set_facecolor(config.BACKGROUND_COLOR)

    # Set axis limits (normalized 0-1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Invert Y axis (video coordinates: Y increases downward)
    ax.invert_yaxis()

    # Remove axis ticks (cleaner for VLM)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw cones (static)
    for cx, cy, label in cones_norm:
        ax.scatter(cx, cy, s=config.CONE_SIZE, c=config.CONE_COLOR,
                   marker='^', zorder=5, edgecolors='black', linewidths=1)
        ax.annotate(label, (cx, cy), textcoords="offset points",
                    xytext=(0, -25), ha='center', fontsize=10, fontweight='bold')

    # Draw start marker (static)
    start_scatter = ax.scatter(ball_x[0], ball_y[0], s=config.START_SIZE,
                               c=config.START_COLOR, marker='*', zorder=10,
                               edgecolors='black', linewidths=1, label='Start')

    # Initialize animated elements
    trajectory_line, = ax.plot([], [], c=config.TRAJECTORY_COLOR,
                                linewidth=config.TRAJECTORY_LINE_WIDTH,
                                alpha=0.7, zorder=3)
    current_ball = ax.scatter([], [], s=config.CURRENT_BALL_SIZE,
                              c=config.CURRENT_BALL_COLOR, marker='o',
                              edgecolors='black', linewidths=1, zorder=8)
    end_scatter = ax.scatter([], [], s=config.END_SIZE, c=config.END_COLOR,
                             marker='o', zorder=10, edgecolors='black',
                             linewidths=1, label='End')

    # Frame counter text
    frame_text = ax.text(0.02, 0.02, '', transform=ax.transAxes,
                         fontsize=12, verticalalignment='bottom',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Title
    ax.set_title(f"Ball Trajectory - {player_name}\nTriple Cone Drill",
                 fontsize=14, fontweight='bold', pad=10)

    # Legend
    legend_elements = [
        mpatches.Patch(color=config.CONE_COLOR, label='Cones'),
        mpatches.Patch(color=config.START_COLOR, label='Start'),
        mpatches.Patch(color=config.END_COLOR, label='End'),
        mpatches.Patch(color=config.TRAJECTORY_COLOR, label='Path'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Animation update function
    def update(frame_idx):
        # Draw trajectory up to current frame
        trajectory_line.set_data(ball_x[:frame_idx+1], ball_y[:frame_idx+1])

        # Update current ball position
        current_ball.set_offsets(np.array([[ball_x[frame_idx], ball_y[frame_idx]]]))

        # Update frame counter
        frame_text.set_text(f'Frame: {frame_ids[frame_idx]} ({frame_idx+1}/{len(ball_df)})')

        # Show end marker on last frame
        if frame_idx == len(ball_df) - 1:
            end_scatter.set_offsets(np.array([[ball_x[frame_idx], ball_y[frame_idx]]]))
        else:
            end_scatter.set_offsets(np.zeros((0, 2)))

        return trajectory_line, current_ball, frame_text, end_scatter

    # Create animation
    # Use interval to control playback speed (ms between frames)
    interval_ms = 1000 / config.FPS  # e.g., 33.3ms for 30fps

    anim = FuncAnimation(
        fig, update, frames=len(ball_df),
        interval=interval_ms, blit=True, repeat=True
    )

    # Save GIF
    print(f"Saving GIF to {output_path}...")
    anim.save(str(output_path), writer='pillow', fps=config.FPS, dpi=config.DPI)

    plt.close(fig)

    print(f"GIF saved successfully: {output_path}")
    print(f"  Duration: {len(ball_df) / config.FPS:.1f} seconds at {config.FPS} fps")

    # Save color-gradient PNG (optimized for VLM analysis)
    gradient_png_path = output_path.with_name(
        output_path.stem + "_gradient.png"
    )
    save_color_gradient_png(ball_x, ball_y, cones_norm, player_name, gradient_png_path, config)

    return output_path


def save_color_gradient_png(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    cones_norm: List[Tuple[float, float, str]],
    player_name: str,
    output_path: Path,
    config: TrajectoryConfig
):
    """
    Save a static PNG with color-gradient trajectory (blue->red by time).

    This is optimized for VLM analysis: overlapping path segments from
    multiple repetitions are distinguishable by their color (time).
    """
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, facecolor=config.BACKGROUND_COLOR)
    ax.set_facecolor(config.BACKGROUND_COLOR)

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    # Create line segments for color gradient
    # Each segment connects consecutive points
    points = np.array([ball_x, ball_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create color array based on time (0 to 1)
    colors = np.linspace(0, 1, len(segments))

    # Create LineCollection with colormap
    lc = LineCollection(
        segments,
        cmap=config.GRADIENT_COLORMAP,
        norm=plt.Normalize(0, 1),
        linewidth=config.TRAJECTORY_LINE_WIDTH,
        alpha=0.8,
        zorder=3
    )
    lc.set_array(colors)
    ax.add_collection(lc)

    # Add colorbar to show time progression
    cbar = plt.colorbar(lc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Time (Start → End)', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Start', 'Middle', 'End'])

    # Draw cones
    for cx, cy, label in cones_norm:
        ax.scatter(cx, cy, s=config.CONE_SIZE, c=config.CONE_COLOR,
                   marker='^', zorder=5, edgecolors='black', linewidths=1.5)
        ax.annotate(label, (cx, cy), textcoords="offset points",
                    xytext=(0, -25), ha='center', fontsize=10, fontweight='bold')

    # Start and end markers
    ax.scatter(ball_x[0], ball_y[0], s=config.START_SIZE, c=config.START_COLOR,
               marker='*', zorder=10, edgecolors='black', linewidths=1.5, label='Start')
    ax.scatter(ball_x[-1], ball_y[-1], s=config.END_SIZE, c=config.END_COLOR,
               marker='o', zorder=10, edgecolors='black', linewidths=1.5, label='End')

    # Title
    ax.set_title(f"Ball Trajectory - {player_name}\nTriple Cone Drill (Color = Time)",
                 fontsize=14, fontweight='bold', pad=10)

    # Legend for markers only (colorbar explains path)
    legend_elements = [
        mpatches.Patch(color=config.CONE_COLOR, label='Cones'),
        mpatches.Patch(color=config.START_COLOR, label='Start'),
        mpatches.Patch(color=config.END_COLOR, label='End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config.DPI, facecolor=config.BACKGROUND_COLOR)
    plt.close(fig)

    print(f"Color-gradient PNG saved: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate ball trajectory GIF for VLM validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video/trajectory_gif_generator.py "Alex mochar"
  python video/trajectory_gif_generator.py --list

Output:
  video/output/<player_name>_trajectory.gif
  video/output/<player_name>_trajectory.png (static view)
        """
    )

    parser.add_argument(
        'player_name',
        nargs='?',
        help='Name of the player (partial match supported)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available players'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Custom output path for GIF'
    )

    args = parser.parse_args()

    if args.list:
        players = list_available_players()
        if players:
            print("Available players:")
            for p in players:
                print(f"  - {p}")
        else:
            print("No players found in data directory")
        return

    if not args.player_name:
        parser.print_help()
        return

    try:
        generate_trajectory_gif(args.player_name, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
