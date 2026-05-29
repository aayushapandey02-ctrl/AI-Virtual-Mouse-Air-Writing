"""
Air Canvas - Draw in the air using hand gestures.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime

from hand_tracker import HandTracker, HandLandmarks
from gesture_recognizer import Gesture, GestureResult
from config import config


@dataclass
class DrawingStroke:
    """Represents a single drawing stroke."""
    points: List[Tuple[int, int]] = field(default_factory=list)
    color: Tuple[int, int, int] = (0, 255, 0)
    thickness: int = 8
    
    def add_point(self, point: Tuple[int, int]):
        """Add a point to the stroke."""
        self.points.append(point)


class AirCanvas:
    """
    Air drawing canvas that tracks finger movement and renders drawings.
    
    Features:
    - Multiple pen colors
    - Eraser mode
    - Clear screen
    - Save canvas
    - Undo functionality
    """
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        
        # Canvas layers
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Drawing state
        self.is_drawing = False
        self.current_stroke: Optional[DrawingStroke] = None
        self.strokes: List[DrawingStroke] = []
        self.prev_point: Optional[Tuple[int, int]] = None
        
        # Pen settings
        self.pen_color = config.canvas.default_color
        self.pen_thickness = config.canvas.brush_thickness
        self.eraser_mode = False
        self.eraser_thickness = config.canvas.eraser_thickness
        
        # Color palette
        self.colors = config.canvas.colors
        self.color_names = list(self.colors.keys())
        self.current_color_idx = 0
        
        # Drawing zone (exclude UI header area)
        self.draw_zone_top = config.ui.header_height + 20
    
    def process(self, hand_data: HandLandmarks, 
                gesture_result: GestureResult) -> np.ndarray:
        """
        Process hand gesture and update canvas.
        
        Args:
            hand_data: Hand landmarks
            gesture_result: Recognized gesture
            
        Returns:
            Current canvas state
        """
        gesture = gesture_result.gesture
        index_tip = gesture_result.details['index_tip']
        
        # Only draw with pointer gesture (index finger only)
        if gesture == Gesture.POINTER:
            # Check if in drawing zone
            if index_tip[1] > self.draw_zone_top:
                if not self.is_drawing:
                    self._start_stroke(index_tip)
                else:
                    self._continue_stroke(index_tip)
            else:
                self._end_stroke()
        
        # Peace sign to cycle colors
        elif gesture == Gesture.PEACE:
            if not self.is_drawing:
                self._cycle_color()
            self._end_stroke()
        
        # Fist to toggle eraser
        elif gesture == Gesture.FIST:
            self.eraser_mode = not self.eraser_mode
            self._end_stroke()
        
        # Open palm to pause drawing
        elif gesture == Gesture.OPEN_PALM:
            self._end_stroke()
        
        # Thumb up to undo
        elif gesture == Gesture.THUMB_UP:
            self.undo()
            self._end_stroke()
        
        else:
            self._end_stroke()
        
        return self.canvas
    
    def _start_stroke(self, point: Tuple[int, int]):
        """Start a new drawing stroke."""
        self.is_drawing = True
        
        if self.eraser_mode:
            color = (0, 0, 0)
            thickness = self.eraser_thickness
        else:
            color = self.pen_color
            thickness = self.pen_thickness
        
        self.current_stroke = DrawingStroke(
            color=color,
            thickness=thickness
        )
        self.current_stroke.add_point(point)
        self.prev_point = point
    
    def _continue_stroke(self, point: Tuple[int, int]):
        """Continue current drawing stroke."""
        if self.current_stroke is None:
            self._start_stroke(point)
            return
        
        self.current_stroke.add_point(point)
        
        # Draw line on canvas
        if self.prev_point is not None:
            cv2.line(
                self.canvas,
                self.prev_point,
                point,
                self.current_stroke.color,
                self.current_stroke.thickness,
                cv2.LINE_AA
            )
        
        self.prev_point = point
    
    def _end_stroke(self):
        """End current drawing stroke."""
        if self.current_stroke is not None and len(self.current_stroke.points) > 0:
            self.strokes.append(self.current_stroke)
        
        self.current_stroke = None
        self.is_drawing = False
        self.prev_point = None
    
    def _cycle_color(self):
        """Cycle to next pen color."""
        self.current_color_idx = (self.current_color_idx + 1) % len(self.color_names)
        color_name = self.color_names[self.current_color_idx]
        self.pen_color = self.colors[color_name]
        self.eraser_mode = False
    
    def set_color(self, color_name: str):
        """Set pen color by name."""
        if color_name in self.colors:
            self.pen_color = self.colors[color_name]
            self.current_color_idx = self.color_names.index(color_name)
            self.eraser_mode = False
    
    def set_thickness(self, thickness: int):
        """Set pen thickness."""
        self.pen_thickness = max(1, min(thickness, 50))
    
    def toggle_eraser(self):
        """Toggle eraser mode."""
        self.eraser_mode = not self.eraser_mode
    
    def clear(self):
        """Clear the canvas."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.strokes.clear()
        self._end_stroke()
    
    def undo(self):
        """Undo last stroke."""
        if len(self.strokes) > 0:
            self.strokes.pop()
            self._redraw_canvas()
    
    def _redraw_canvas(self):
        """Redraw all strokes on canvas."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for stroke in self.strokes:
            if len(stroke.points) < 2:
                continue
            
            for i in range(1, len(stroke.points)):
                cv2.line(
                    self.canvas,
                    stroke.points[i-1],
                    stroke.points[i],
                    stroke.color,
                    stroke.thickness,
                    cv2.LINE_AA
                )
    
    def save(self, filename: Optional[str] = None) -> str:
        """
        Save canvas to file.
        
        Returns:
            Saved filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"air_drawing_{timestamp}.png"
        
        # Create white background version
        white_bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Overlay drawing
        mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        mask = mask > 0
        white_bg[mask] = self.canvas[mask]
        
        cv2.imwrite(filename, white_bg)
        return filename
    
    def get_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Overlay canvas drawing on camera frame.
        
        Args:
            frame: Camera frame
            
        Returns:
            Frame with canvas overlay
        """
        # Create mask from canvas
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Invert mask for background
        mask_inv = cv2.bitwise_not(mask)
        
        # Black out drawing area in frame
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        
        # Take drawing from canvas
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        
        # Combine
        result = cv2.add(frame_bg, canvas_fg)
        
        return result
    
    def get_drawing_region(self) -> np.ndarray:
        """Get the current drawing as an image for recognition."""
        # Find bounding box of drawing
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        
        if coords is None:
            return np.zeros((28, 28), dtype=np.uint8)
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(self.width - x, w + 2 * padding)
        h = min(self.height - y, h + 2 * padding)
        
        # Extract region
        region = gray[y:y+h, x:x+w]
        
        return region
