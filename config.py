"""
Configuration settings for the Virtual Mouse & Air Writing System.
Centralized parameters for easy customization.
"""

from dataclasses import dataclass, field
from typing import Tuple, Dict, List


@dataclass
class CameraConfig:
    """Camera capture settings."""
    width: int = 1280
    height: int = 720
    fps: int = 30
    camera_id: int = 0


@dataclass
class HandTrackingConfig:
    """MediaPipe hand tracking parameters."""
    max_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    model_complexity: int = 1


@dataclass
class MouseConfig:
    """Virtual mouse control settings."""
    smoothing_factor: float = 0.3
    click_threshold: float = 0.04
    double_click_interval: float = 0.3
    scroll_sensitivity: float = 50
    frame_reduction: int = 100  # Border margin for mouse movement


@dataclass
class CanvasConfig:
    """Air canvas drawing settings."""
    default_color: Tuple[int, int, int] = (0, 255, 0)
    brush_thickness: int = 8
    eraser_thickness: int = 40
    colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'red': (0, 0, 255),
        'yellow': (0, 255, 255),
        'purple': (255, 0, 255),
        'white': (255, 255, 255),
    })


@dataclass
class GestureConfig:
    """Gesture recognition thresholds."""
    pinch_threshold: float = 0.05
    fist_threshold: float = 0.15
    spread_threshold: float = 0.25
    gesture_hold_frames: int = 3


@dataclass
class UIConfig:
    """User interface settings."""
    header_height: int = 80
    button_size: Tuple[int, int] = (80, 50)
    font_scale: float = 0.6
    font_thickness: int = 2
    fps_position: Tuple[int, int] = (20, 50)


@dataclass
class Config:
    """Master configuration container."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    hand_tracking: HandTrackingConfig = field(default_factory=HandTrackingConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    ui: UIConfig = field(default_factory=UIConfig)


# Global configuration instance
config = Config()
