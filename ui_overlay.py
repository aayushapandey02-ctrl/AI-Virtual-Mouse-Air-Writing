"""
User interface overlay for the Virtual Mouse & Air Writing System.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

from config import config
from gesture_recognizer import Gesture


@dataclass
class Button:
    """UI Button definition."""
    x: int
    y: int
    width: int
    height: int
    label: str
    color: Tuple[int, int, int]
    is_active: bool = False
    
    def contains(self, px: int, py: int) -> bool:
        """Check if point is inside button."""
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)


class UIOverlay:
    """
    Renders the user interface overlay on the camera feed.
    
    Includes:
    - Mode indicator
    - Color palette
    - Tool buttons
    - FPS display
    - Gesture feedback
    """
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.header_height = config.ui.header_height
        
        # Create buttons
        self.buttons: List[Button] = []
        self._create_buttons()
        
        # State
        self.current_mode = 'mouse'
        self.current_color = 'green'
        self.eraser_active = False
        self.fps = 0.0
        self.gesture_text = ''
        
        # Colors
        self.bg_color = (30, 30, 30)
        self.text_color = (255, 255, 255)
        self.accent_color = (0, 200, 255)
        
    def _create_buttons(self):
        """Create UI buttons."""
        btn_width, btn_height = config.ui.button_size
        margin = 10
        x_offset = margin
        y = 15
        
        # Mode buttons
        self.buttons.append(Button(
            x=x_offset, y=y, width=btn_width, height=btn_height,
            label='Mouse', color=(100, 100, 100)
        ))
        x_offset += btn_width + margin
        
        self.buttons.append(Button(
            x=x_offset, y=y, width=btn_width, height=btn_height,
            label='Draw', color=(100, 100, 100)
        ))
        x_offset += btn_width + margin * 3
        
        # Color buttons
        for color_name, color_value in config.canvas.colors.items():
            self.buttons.append(Button(
                x=x_offset, y=y, width=40, height=btn_height,
                label='', color=color_value
            ))
            x_offset += 50
        
        # Tool buttons
        x_offset += margin * 2
        
        self.buttons.append(Button(
            x=x_offset, y=y, width=btn_width, height=btn_height,
            label='Eraser', color=(150, 150, 150)
        ))
        x_offset += btn_width + margin
        
        self.buttons.append(Button(
            x=x_offset, y=y, width=btn_width, height=btn_height,
            label='Clear', color=(0, 0, 200)
        ))
        x_offset += btn_width + margin
        
        self.buttons.append(Button(
            x=x_offset, y=y, width=btn_width, height=btn_height,
            label='Save', color=(0, 150, 0)
        ))
    
    def render(self, frame: np.ndarray, 
               mode: str = 'mouse',
               fps: float = 0.0,
               gesture: Optional[Gesture] = None,
               volume: Optional[float] = None,
               brightness: Optional[int] = None,
               recognized_text: str = '') -> np.ndarray:
        """
        Render UI overlay on frame.
        
        Args:
            frame: Camera frame
            mode: Current mode ('mouse' or 'draw')
            fps: Current FPS
            gesture: Current gesture
            volume: Volume level (0-1)
            brightness: Brightness level (0-100)
            recognized_text: Recognized handwritten text
            
        Returns:
            Frame with UI overlay
        """
        self.current_mode = mode
        self.fps = fps
        
        # Draw semi-transparent header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, self.header_height), 
                     self.bg_color, -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw buttons
        frame = self._draw_buttons(frame)
        
        # Draw FPS
        fps_text = f'FPS: {int(fps)}'
        cv2.putText(frame, fps_text, 
                   (self.width - 100, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                   self.accent_color, 2)
        
        # Draw mode indicator
        mode_text = f'Mode: {mode.upper()}'
        cv2.putText(frame, mode_text,
                   (self.width - 100, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   self.text_color, 1)
        
        # Draw gesture feedback
        if gesture and gesture != Gesture.NONE:
            gesture_text = gesture.name.replace('_', ' ').title()
            cv2.putText(frame, gesture_text,
                       (20, self.height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                       self.accent_color, 2)
        
        # Draw volume/brightness indicators
        if volume is not None:
            frame = self._draw_volume_bar(frame, volume)
        
        if brightness is not None:
            frame = self._draw_brightness_bar(frame, brightness)
        
        # Draw recognized text
        if recognized_text:
            cv2.putText(frame, f'Text: {recognized_text}',
                       (20, self.height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 255), 2)
        
        # Draw instructions
        frame = self._draw_instructions(frame)
        
        return frame
    
    def _draw_buttons(self, frame: np.ndarray) -> np.ndarray:
        """Draw all UI buttons."""
        for btn in self.buttons:
            # Determine if active
            is_active = False
            if btn.label == 'Mouse' and self.current_mode == 'mouse':
                is_active = True
            elif btn.label == 'Draw' and self.current_mode == 'draw':
                is_active = True
            elif btn.label == 'Eraser' and self.eraser_active:
                is_active = True
            
            # Draw button background
            color = btn.color
            if is_active:
                color = tuple(min(c + 50, 255) for c in color)
            
            cv2.rectangle(frame,
                         (btn.x, btn.y),
                         (btn.x + btn.width, btn.y + btn.height),
                         color, -1)
            
            # Draw border
            border_color = (255, 255, 255) if is_active else (100, 100, 100)
            cv2.rectangle(frame,
                         (btn.x, btn.y),
                         (btn.x + btn.width, btn.y + btn.height),
                         border_color, 2)
            
            # Draw label
            if btn.label:
                text_size = cv2.getTextSize(
                    btn.label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )[0]
                text_x = btn.x + (btn.width - text_size[0]) // 2
                text_y = btn.y + (btn.height + text_size[1]) // 2
                cv2.putText(frame, btn.label,
                           (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           self.text_color, 1)
        
        return frame
    
    def _draw_volume_bar(self, frame: np.ndarray, level: float) -> np.ndarray:
        """Draw volume indicator bar."""
        bar_width = 30
        bar_height = 150
        x = 30
        y = self.height // 2 - bar_height // 2
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     (50, 50, 50), -1)
        
        # Level
        level_height = int(bar_height * level)
        cv2.rectangle(frame,
                     (x, y + bar_height - level_height),
                     (x + bar_width, y + bar_height),
                     (0, 255, 0), -1)
        
        # Label
        cv2.putText(frame, 'VOL', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        cv2.putText(frame, f'{int(level * 100)}%',
                   (x - 5, y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        return frame
    
    def _draw_brightness_bar(self, frame: np.ndarray, level: int) -> np.ndarray:
        """Draw brightness indicator bar."""
        bar_width = 30
        bar_height = 150
        x = self.width - 60
        y = self.height // 2 - bar_height // 2
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height),
                     (50, 50, 50), -1)
        
        # Level
        level_height = int(bar_height * level / 100)
        cv2.rectangle(frame,
                     (x, y + bar_height - level_height),
                     (x + bar_width, y + bar_height),
                     (255, 255, 0), -1)
        
        # Label
        cv2.putText(frame, 'BRT', (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        cv2.putText(frame, f'{level}%',
                   (x - 5, y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        return frame
    
    def _draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw gesture instructions."""
        instructions = [
            "M - Mouse Mode | D - Draw Mode",
            "C - Clear | S - Save | Q - Quit",
            "V - Volume | B - Brightness"
        ]
        
        y_start = self.height - 20
        for i, text in enumerate(reversed(instructions)):
            y = y_start - i * 20
            cv2.putText(frame, text,
                       (self.width // 2 - 150, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (150, 150, 150), 1)
        
        return frame
    
    def check_button_click(self, x: int, y: int) -> Optional[str]:
        """Check if a button was clicked."""
        for btn in self.buttons:
            if btn.contains(x, y):
                return btn.label
        return None
    
    def set_eraser_active(self, active: bool):
        """Set eraser button state."""
        self.eraser_active = active
