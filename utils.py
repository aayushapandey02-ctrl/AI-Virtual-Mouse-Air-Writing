"""
Utility functions for the Virtual Mouse & Air Writing System.
"""

import time
import numpy as np
from collections import deque
from typing import Tuple, List, Optional


class FPSCounter:
    """Calculates and smooths FPS readings."""
    
    def __init__(self, buffer_size: int = 30):
        self.timestamps = deque(maxlen=buffer_size)
        self.fps = 0.0
    
    def update(self) -> float:
        """Update FPS calculation with current timestamp."""
        current_time = time.time()
        self.timestamps.append(current_time)
        
        if len(self.timestamps) >= 2:
            time_diff = self.timestamps[-1] - self.timestamps[0]
            if time_diff > 0:
                self.fps = (len(self.timestamps) - 1) / time_diff
        
        return self.fps
    
    def get_fps(self) -> float:
        """Return current smoothed FPS."""
        return self.fps


class SmoothingFilter:
    """Exponential moving average filter for smooth cursor movement."""
    
    def __init__(self, smoothing_factor: float = 0.3):
        self.smoothing_factor = smoothing_factor
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
    
    def apply(self, x: float, y: float) -> Tuple[float, float]:
        """Apply smoothing to coordinates."""
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y
        
        smooth_x = self.prev_x + self.smoothing_factor * (x - self.prev_x)
        smooth_y = self.prev_y + self.smoothing_factor * (y - self.prev_y)
        
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y
    
    def reset(self):
        """Reset filter state."""
        self.prev_x = None
        self.prev_y = None


class GestureBuffer:
    """Buffers gestures to prevent accidental triggers."""
    
    def __init__(self, required_frames: int = 3):
        self.required_frames = required_frames
        self.gesture_counts: dict = {}
    
    def update(self, gesture: str) -> bool:
        """
        Update buffer with detected gesture.
        Returns True if gesture is confirmed (held for required frames).
        """
        # Reset counts for other gestures
        for g in list(self.gesture_counts.keys()):
            if g != gesture:
                self.gesture_counts[g] = 0
        
        # Increment count for current gesture
        self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
        
        return self.gesture_counts[gesture] >= self.required_frames
    
    def reset(self):
        """Reset all gesture counts."""
        self.gesture_counts.clear()


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_angle(p1: Tuple[float, float], p2: Tuple[float, float], 
                    p3: Tuple[float, float]) -> float:
    """Calculate angle at p2 formed by p1-p2-p3."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    return np.degrees(angle)


def map_coordinates(x: float, y: float, 
                    src_range: Tuple[float, float, float, float],
                    dst_range: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Map coordinates from source range to destination range.
    Ranges are (x_min, y_min, x_max, y_max).
    """
    src_x_min, src_y_min, src_x_max, src_y_max = src_range
    dst_x_min, dst_y_min, dst_x_max, dst_y_max = dst_range
    
    # Normalize to 0-1
    norm_x = (x - src_x_min) / (src_x_max - src_x_min + 1e-6)
    norm_y = (y - src_y_min) / (src_y_max - src_y_min + 1e-6)
    
    # Clip to bounds
    norm_x = np.clip(norm_x, 0, 1)
    norm_y = np.clip(norm_y, 0, 1)
    
    # Map to destination
    dst_x = dst_x_min + norm_x * (dst_x_max - dst_x_min)
    dst_y = dst_y_min + norm_y * (dst_y_max - dst_y_min)
    
    return dst_x, dst_y
