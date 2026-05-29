"""
Virtual mouse control using hand gestures.
"""

import pyautogui
import time
import numpy as np
from typing import Optional, Tuple
from enum import Enum, auto

from hand_tracker import HandTracker, HandLandmarks
from gesture_recognizer import GestureRecognizer, Gesture, GestureResult
from utils import SmoothingFilter, map_coordinates
from config import config


class ClickState(Enum):
    """Mouse click states."""
    IDLE = auto()
    CLICK_STARTED = auto()
    CLICKED = auto()
    DOUBLE_CLICK_WAIT = auto()


class MouseController:
    """
    Controls system mouse using hand gestures.
    
    Gestures:
    - Pointer (index only): Move cursor
    - Pinch: Left click
    - Peace sign: Right click  
    - Three fingers: Double click
    - Fist: Drag start
    - Open palm: Drag end
    """
    
    def __init__(self):
        # Disable PyAutoGUI failsafe and set instant movement
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Smoothing filter for cursor movement
        self.smoother = SmoothingFilter(config.mouse.smoothing_factor)
        
        # Click handling
        self.click_state = ClickState.IDLE
        self.last_click_time = 0
        self.is_dragging = False
        
        # Previous positions for scroll
        self.prev_y: Optional[float] = None
        
        # Frame dimensions (set when processing)
        self.frame_width = config.camera.width
        self.frame_height = config.camera.height
        
    def process(self, hand_data: HandLandmarks, 
                gesture_result: GestureResult) -> dict:
        """
        Process hand data and control mouse accordingly.
        
        Returns:
            dict with action performed and cursor position
        """
        action = {'type': 'none', 'position': (0, 0)}
        
        gesture = gesture_result.gesture
        index_tip = gesture_result.details['index_tip']
        
        # Calculate cursor position from index finger
        cursor_x, cursor_y = self._calculate_cursor_position(index_tip)
        action['position'] = (cursor_x, cursor_y)
        
        # Handle different gestures
        if gesture == Gesture.POINTER:
            # Move cursor
            self._move_cursor(cursor_x, cursor_y)
            action['type'] = 'move'
            self._reset_click_state()
            
        elif gesture == Gesture.PINCH:
            # Left click
            if self.click_state == ClickState.IDLE:
                self._left_click()
                action['type'] = 'left_click'
                self.click_state = ClickState.CLICKED
                self.last_click_time = time.time()
                
        elif gesture == Gesture.PEACE:
            # Right click or scroll
            if self.click_state == ClickState.IDLE:
                self._right_click()
                action['type'] = 'right_click'
                self.click_state = ClickState.CLICKED
            # Check for scroll gesture
            elif self.prev_y is not None:
                delta_y = index_tip[1] - self.prev_y
                if abs(delta_y) > 10:
                    self._scroll(int(delta_y * -0.5))
                    action['type'] = 'scroll'
                    
        elif gesture == Gesture.THREE_FINGERS:
            # Double click
            if self.click_state == ClickState.IDLE:
                self._double_click()
                action['type'] = 'double_click'
                self.click_state = ClickState.CLICKED
                
        elif gesture == Gesture.FIST:
            # Start drag
            if not self.is_dragging:
                pyautogui.mouseDown()
                self.is_dragging = True
                action['type'] = 'drag_start'
            else:
                self._move_cursor(cursor_x, cursor_y)
                action['type'] = 'dragging'
                
        elif gesture == Gesture.OPEN_PALM:
            # End drag
            if self.is_dragging:
                pyautogui.mouseUp()
                self.is_dragging = False
                action['type'] = 'drag_end'
            self._reset_click_state()
            
        else:
            self._reset_click_state()
        
        # Update previous position for scroll
        self.prev_y = index_tip[1]
        
        return action
    
    def _calculate_cursor_position(self, fingertip: Tuple[int, int]) -> Tuple[int, int]:
        """Map fingertip position to screen coordinates."""
        x, y = fingertip
        
        # Define the active tracking region (with margins)
        margin = config.mouse.frame_reduction
        src_range = (margin, margin, 
                     self.frame_width - margin, 
                     self.frame_height - margin)
        
        # Map to screen coordinates
        dst_range = (0, 0, self.screen_width, self.screen_height)
        
        screen_x, screen_y = map_coordinates(x, y, src_range, dst_range)
        
        # Apply smoothing
        smooth_x, smooth_y = self.smoother.apply(screen_x, screen_y)
        
        return int(smooth_x), int(smooth_y)
    
    def _move_cursor(self, x: int, y: int):
        """Move cursor to position."""
        # Clamp to screen bounds
        x = max(0, min(x, self.screen_width - 1))
        y = max(0, min(y, self.screen_height - 1))
        pyautogui.moveTo(x, y, _pause=False)
    
    def _left_click(self):
        """Perform left click."""
        pyautogui.click(_pause=False)
    
    def _right_click(self):
        """Perform right click."""
        pyautogui.rightClick(_pause=False)
    
    def _double_click(self):
        """Perform double click."""
        pyautogui.doubleClick(_pause=False)
    
    def _scroll(self, amount: int):
        """Scroll by amount."""
        pyautogui.scroll(amount, _pause=False)
    
    def _reset_click_state(self):
        """Reset click state after gesture ends."""
        current_time = time.time()
        if current_time - self.last_click_time > config.mouse.double_click_interval:
            self.click_state = ClickState.IDLE
    
    def set_frame_dimensions(self, width: int, height: int):
        """Update frame dimensions for coordinate mapping."""
        self.frame_width = width
        self.frame_height = height
