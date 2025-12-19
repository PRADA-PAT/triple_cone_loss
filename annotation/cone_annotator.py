"""
Manual Cone Annotation Tool for Figure-8 Drills.

Provides an OpenCV-based GUI to manually annotate cone bounding boxes
in video frames, replacing unreliable automatic cone detection.

Features:
- Click-and-drag to draw bounding boxes (like the parquet format)
- Visual guide showing cone formation layout
- Shows rectangle preview while drawing

Usage:
    from cone_annotator import ConeAnnotator

    annotator = ConeAnnotator(video_path, output_dir)
    annotator.run()  # Interactive annotation session
"""
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Cone roles in click order (right-to-left direction of play)
CONE_ROLES = ["start", "gate1_right", "gate1_left", "gate2_right", "gate2_left"]
ROLE_LABELS = ["START", "Gate1-R", "Gate1-L", "Gate2-R", "Gate2-L"]
ROLE_COLORS = {
    "start": (0, 255, 0),         # Green
    "gate1_left": (0, 165, 255),  # Orange (BGR)
    "gate1_right": (0, 165, 255), # Orange
    "gate2_left": (255, 165, 0),  # Blue (BGR)
    "gate2_right": (255, 165, 0), # Blue
}

# Minimum bounding box size (pixels)
MIN_BBOX_SIZE = 8


class BoundingBox:
    """Represents a cone bounding box annotation."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        # Normalize coordinates (ensure x1 < x2, y1 < y2)
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def is_valid(self) -> bool:
        """Check if bbox meets minimum size requirement."""
        return self.width >= MIN_BBOX_SIZE and self.height >= MIN_BBOX_SIZE

    def to_dict(self) -> Dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2
        }


class ConeAnnotator:
    """
    Interactive cone annotation tool using OpenCV with bounding boxes.

    Features:
    - Click and drag to draw bounding boxes
    - Visual guide showing Figure-8 cone layout
    """

    def __init__(self, video_path: str, output_dir: str):
        """
        Initialize annotator.

        Args:
            video_path: Path to video file
            output_dir: Directory to save cone_annotations.json
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "cone_annotations.json"

        # Annotation state
        self.bboxes: List[BoundingBox] = []
        self.frame: Optional[np.ndarray] = None
        self.original_frame: Optional[np.ndarray] = None
        self.window_name = "Cone Annotator - Draw Bounding Boxes"

        # Drawing state
        self.drawing = False
        self.start_point: Optional[Tuple[int, int]] = None  # In IMAGE coordinates
        self.current_point: Optional[Tuple[int, int]] = None  # In IMAGE coordinates

        # Window size (will be set when window is created)
        self.window_width = 1280
        self.window_height = 720

        # Scale factors for coordinate conversion
        self.scale_x = 1.0
        self.scale_y = 1.0

    def load_frame(self) -> bool:
        """Load first frame from video."""
        if not self.video_path.exists():
            print(f"Video not found: {self.video_path}")
            return False

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            print(f"Cannot open video: {self.video_path}")
            return False

        ret, self.frame = cap.read()
        cap.release()

        if not ret or self.frame is None:
            print("Failed to read first frame")
            return False

        self.original_frame = self.frame.copy()

        # Calculate scale factors
        img_h, img_w = self.original_frame.shape[:2]
        self.scale_x = img_w / self.window_width
        self.scale_y = img_h / self.window_height

        return True

    def display_to_image(self, dx: int, dy: int) -> Tuple[int, int]:
        """Convert display coordinates to image coordinates."""
        img_x = int(dx * self.scale_x)
        img_y = int(dy * self.scale_y)
        return (img_x, img_y)

    def image_to_display(self, ix: int, iy: int) -> Tuple[int, int]:
        """Convert image coordinates to display coordinates."""
        dx = int(ix / self.scale_x)
        dy = int(iy / self.scale_y)
        return (dx, dy)

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """Handle mouse events for bounding box drawing."""

        # Left-click for drawing bounding boxes
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.bboxes) < 5:
                self.drawing = True
                # Convert to image coordinates
                self.start_point = self.display_to_image(x, y)
                self.current_point = self.start_point

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_point = self.display_to_image(x, y)
                self.draw_annotations()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                end_point = self.display_to_image(x, y)
                bbox = BoundingBox(
                    self.start_point[0], self.start_point[1],
                    end_point[0], end_point[1]
                )

                if bbox.is_valid():
                    self.bboxes.append(bbox)
                    print(f"  Added {CONE_ROLES[len(self.bboxes)-1]}: "
                          f"bbox=({bbox.x1},{bbox.y1})-({bbox.x2},{bbox.y2}), "
                          f"center=({bbox.center[0]},{bbox.center[1]})")
                else:
                    print(f"  Box too small (min {MIN_BBOX_SIZE}px), try again")

                self.start_point = None
                self.current_point = None
                self.draw_annotations()

    def draw_guide_diagram(self, frame: np.ndarray):
        """Draw the cone formation guide diagram in top-right corner."""
        h, w = frame.shape[:2]

        # Diagram dimensions and position
        dw, dh = 380, 180
        margin = 10
        dx, dy = w - dw - margin, margin

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Border
        cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), (100, 100, 100), 1)

        # Title
        cv2.putText(frame, "FIGURE-8 CONE LAYOUT", (dx + 10, dy + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Direction arrow (right to left)
        cv2.putText(frame, "Direction of play", (dx + 180, dy + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.arrowedLine(frame, (dx + 170, dy + 47), (dx + 110, dy + 47),
                       (180, 180, 180), 2, tipLength=0.3)

        # Cone positions in diagram (right-to-left: START on right, Gate2 on left)
        # Order matches click order: START, G1-R, G1-L, G2-R, G2-L
        cone_y = dy + 95
        cone_positions = [
            (dx + 340, cone_y),        # 1: START (rightmost)
            (dx + 230, cone_y + 20),   # 2: G1-R (bottom of gate1)
            (dx + 230, cone_y - 20),   # 3: G1-L (top of gate1)
            (dx + 90, cone_y + 20),    # 4: G2-R (bottom of gate2)
            (dx + 90, cone_y - 20),    # 5: G2-L (top of gate2, leftmost)
        ]

        # Draw gate lines first (behind cones)
        cv2.line(frame, cone_positions[1], cone_positions[2],
                (0, 165, 255), 2)  # Gate 1
        cv2.line(frame, cone_positions[3], cone_positions[4],
                (255, 165, 0), 2)  # Gate 2

        # Draw path line (dashed effect)
        path_y = cone_y
        for i in range(dx + 60, dx + 340, 20):
            cv2.line(frame, (i, path_y), (i + 10, path_y), (80, 80, 80), 1)

        # Draw cones
        current_idx = len(self.bboxes)
        for i, (cx, cy) in enumerate(cone_positions):
            role = CONE_ROLES[i]
            color = ROLE_COLORS[role]

            # Highlight current cone to mark
            if i == current_idx:
                # Pulsing highlight effect
                cv2.circle(frame, (cx, cy), 22, (255, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 18, color, -1)
                cv2.circle(frame, (cx, cy), 18, (255, 255, 255), 2)
            elif i < current_idx:
                # Already marked - solid with checkmark
                cv2.circle(frame, (cx, cy), 15, color, -1)
                cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 1)
                # Checkmark
                cv2.line(frame, (cx - 5, cy), (cx - 1, cy + 4), (255, 255, 255), 2)
                cv2.line(frame, (cx - 1, cy + 4), (cx + 6, cy - 5), (255, 255, 255), 2)
            else:
                # Not yet marked - hollow
                cv2.circle(frame, (cx, cy), 15, color, 2)

            # Cone number
            num_text = str(i + 1)
            if i >= current_idx:
                cv2.putText(frame, num_text, (cx - 4, cy + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Labels below cones (matching new order)
        labels = ["1:START", "2:G1-R", "3:G1-L", "4:G2-R", "5:G2-L"]
        label_y = dy + 130
        label_x_offsets = [dx + 315, dx + 205, dx + 205, dx + 65, dx + 65]
        label_y_offsets = [0, 15, -15, 15, -15]

        for i, (lx, ly_off) in enumerate(zip(label_x_offsets, label_y_offsets)):
            color = ROLE_COLORS[CONE_ROLES[i]]
            if i == current_idx:
                color = (255, 255, 255)  # Highlight current
            cv2.putText(frame, labels[i], (lx, label_y + ly_off),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Gate labels
        cv2.putText(frame, "GATE 1", (dx + 215, dy + 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        cv2.putText(frame, "GATE 2", (dx + 75, dy + 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        # Current instruction
        if current_idx < 5:
            inst = f">> Mark: {labels[current_idx]} <<"
            cv2.putText(frame, inst, (dx + 10, dy + dh - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def draw_annotations(self):
        """Draw current annotations on frame."""
        if self.original_frame is None:
            return

        # Create a working copy of the frame
        work_frame = self.original_frame.copy()

        # Draw completed bounding boxes on work frame (in image coordinates)
        for i, bbox in enumerate(self.bboxes):
            role = CONE_ROLES[i]
            color = ROLE_COLORS[role]

            # Draw rectangle
            cv2.rectangle(work_frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                         color, 2)

            # Draw center point
            cx, cy = bbox.center
            cv2.circle(work_frame, (cx, cy), 4, color, -1)

            # Draw label with background
            label = f"{i+1}:{ROLE_LABELS[i]}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(work_frame, (bbox.x1, bbox.y1 - th - 8),
                         (bbox.x1 + tw + 4, bbox.y1), color, -1)
            cv2.putText(work_frame, label, (bbox.x1 + 2, bbox.y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw gate lines
        if len(self.bboxes) >= 3:
            c1 = self.bboxes[1].center
            c2 = self.bboxes[2].center
            cv2.line(work_frame, c1, c2, ROLE_COLORS["gate1_left"], 2)

        if len(self.bboxes) >= 5:
            c1 = self.bboxes[3].center
            c2 = self.bboxes[4].center
            cv2.line(work_frame, c1, c2, ROLE_COLORS["gate2_left"], 2)

        # Draw preview rectangle while dragging
        if self.drawing and self.start_point and self.current_point:
            next_role = CONE_ROLES[len(self.bboxes)]
            color = ROLE_COLORS[next_role]
            cv2.rectangle(work_frame, self.start_point, self.current_point, color, 2)

            # Show dimensions
            temp_bbox = BoundingBox(
                self.start_point[0], self.start_point[1],
                self.current_point[0], self.current_point[1]
            )
            dim_text = f"{temp_bbox.width}x{temp_bbox.height}px"
            cv2.putText(work_frame, dim_text,
                       (self.current_point[0] + 10, self.current_point[1] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Resize to window size
        display = cv2.resize(work_frame, (self.window_width, self.window_height),
                            interpolation=cv2.INTER_LINEAR)

        # Draw UI elements (on display frame)
        # Instructions panel
        cv2.rectangle(display, (0, 0), (350, 85), (0, 0, 0), -1)

        if len(self.bboxes) < 5:
            next_role = CONE_ROLES[len(self.bboxes)]
            color = ROLE_COLORS[next_role]
            cv2.putText(display, f"Draw box: {ROLE_LABELS[len(self.bboxes)]}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, f"({len(self.bboxes)}/5 cones)",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(display, "All 5 cones marked!",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press 's' to save",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Controls hint
        cv2.putText(display, "s:save  r:reset  z:undo  q:quit",
                   (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Draw the guide diagram
        self.draw_guide_diagram(display)

        cv2.imshow(self.window_name, display)

    def save_annotations(self) -> bool:
        """Save annotations to JSON file."""
        if len(self.bboxes) != 5:
            print("Need exactly 5 cone bounding boxes to save")
            return False

        # Build cone data with bounding boxes and pixel centers
        cones = {}
        for i, role in enumerate(CONE_ROLES):
            bbox = self.bboxes[i]
            px, py = bbox.center
            cones[role] = {
                "bbox": bbox.to_dict(),
                "px": px,
                "py": py
            }

        annotation = {
            "video": self.video_path.name,
            "annotated_at": datetime.now().isoformat(),
            "cones": cones
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(annotation, f, indent=2)

        print(f"\nSaved annotations to: {self.output_file}")
        return True

    def reset(self):
        """Reset annotation state."""
        self.bboxes = []
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.draw_annotations()
        print("  Reset all annotations")

    def undo(self):
        """Undo last bounding box."""
        if self.bboxes:
            self.bboxes.pop()
            print(f"  Removed last bbox (was {CONE_ROLES[len(self.bboxes)]})")
            self.draw_annotations()

    def validate_annotations(self) -> List[str]:
        """Validate current annotations (orientation-agnostic)."""
        warnings = []

        if len(self.bboxes) < 5:
            warnings.append(f"Need 5 cones, only have {len(self.bboxes)}")
            return warnings

        # Check for overlapping/too-close cones
        for i in range(len(self.bboxes)):
            for j in range(i+1, len(self.bboxes)):
                b1, b2 = self.bboxes[i], self.bboxes[j]
                c1, c2 = b1.center, b2.center
                dist = np.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)
                if dist < 30:
                    warnings.append(f"Cones {i+1} and {j+1} centers very close ({dist:.0f}px)")

        centers_x = [b.center[0] for b in self.bboxes]

        # Determine direction: left-to-right or right-to-left
        # START should be at one end, Gate2 at the other
        start_x = centers_x[0]
        gate1_mid_x = (centers_x[1] + centers_x[2]) / 2
        gate2_mid_x = (centers_x[3] + centers_x[4]) / 2

        # Check logical ordering (either direction is fine)
        # START -> Gate1 -> Gate2 should be monotonic
        left_to_right = start_x < gate1_mid_x < gate2_mid_x
        right_to_left = start_x > gate1_mid_x > gate2_mid_x

        if not (left_to_right or right_to_left):
            warnings.append("Cone order seems inconsistent: START -> Gate1 -> Gate2 should be in a line")

        # Check gate pairs are vertically aligned (form actual gates)
        gate1_x_diff = abs(centers_x[1] - centers_x[2])
        gate2_x_diff = abs(centers_x[3] - centers_x[4])

        if gate1_x_diff > 200:
            warnings.append(f"Gate1 cones too far apart horizontally ({gate1_x_diff:.0f}px)")
        if gate2_x_diff > 200:
            warnings.append(f"Gate2 cones too far apart horizontally ({gate2_x_diff:.0f}px)")

        return warnings

    def run(self) -> bool:
        """Run interactive annotation session."""
        if self.output_file.exists():
            print(f"Annotation already exists: {self.output_file}")
            response = input("Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                return False

        if not self.load_frame():
            return False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print(f"\nAnnotating: {self.video_path.name}")
        print("=" * 50)
        print("Draw bounding boxes around 5 cones (see guide in top-right):")
        print("  1. START     - Green cone (rightmost)")
        print("  2. Gate1-R   - Orange cone (right of Gate 1)")
        print("  3. Gate1-L   - Orange cone (left of Gate 1)")
        print("  4. Gate2-R   - Blue cone (right of Gate 2)")
        print("  5. Gate2-L   - Blue cone (left of Gate 2, leftmost)")
        print()
        print("Controls:")
        print("  Mouse drag   : Draw bounding box")
        print("  s            : Save annotations")
        print("  r            : Reset all")
        print("  z            : Undo last")
        print("  q            : Quit")
        print("=" * 50)

        self.draw_annotations()
        saved = False

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
            elif key == ord('z'):
                self.undo()
            elif key == ord('s'):
                if len(self.bboxes) == 5:
                    warnings = self.validate_annotations()
                    if warnings:
                        print("\nWarnings:")
                        for w in warnings:
                            print(f"  - {w}")
                        response = input("Save anyway? [y/N]: ").strip().lower()
                        if response != 'y':
                            continue

                    if self.save_annotations():
                        saved = True
                        break
                else:
                    print(f"Need 5 bounding boxes, only have {len(self.bboxes)}")

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        return saved


def load_cone_annotations(parquet_dir: str) -> Optional[Dict]:
    """Load manual cone annotations if they exist."""
    annotation_file = Path(parquet_dir) / "cone_annotations.json"

    if not annotation_file.exists():
        return None

    with open(annotation_file) as f:
        return json.load(f)


def get_annotation_status(parquet_base: str, video_dir: str, players: dict) -> Dict[str, str]:
    """Check annotation status for all players."""
    parquet_base = Path(parquet_base)
    video_dir = Path(video_dir)

    status = {}
    for player, video_file in players.items():
        annotation_file_f8 = parquet_base / f"{player}_f8" / "cone_annotations.json"
        annotation_file = parquet_base / player / "cone_annotations.json"
        video_path = video_dir / video_file

        if annotation_file_f8.exists() or annotation_file.exists():
            status[player] = "annotated"
        elif video_path.exists():
            status[player] = "pending"
        else:
            status[player] = "no_video"

    return status
