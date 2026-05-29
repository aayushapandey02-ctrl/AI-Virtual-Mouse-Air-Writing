"""
AI-Powered Virtual Mouse & Air Writing System

Main application entry point with all features integrated.
"""

import cv2
import numpy as np
import time
from typing import Optional
from enum import Enum, auto

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer, Gesture
from mouse_controller import MouseController
from air_canvas import AirCanvas
from system_controls import SystemControls
from character_recognizer import CharacterRecognizer
from ui_overlay import UIOverlay
from utils import FPSCounter
from config import config


class AppMode(Enum):
    """Application operating modes."""
    MOUSE = auto()
    DRAW = auto()
    VOLUME = auto()
    BRIGHTNESS = auto()


class VirtualMouseApp:
    """
    Main application class integrating all components.
    
    Features:
    - Virtual mouse control with gestures
    - Air drawing canvas
    - Volume/brightness control
    - Character recognition
    - Clean modern UI
    """
    
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(config.camera.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.camera.fps)
        
        # Get actual dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize components
        self.hand_tracker = HandTracker()
        self.gesture_recognizer = GestureRecognizer()
        self.mouse_controller = MouseController()
        self.mouse_controller.set_frame_dimensions(self.frame_width, self.frame_height)
        
        self.air_canvas = AirCanvas(self.frame_width, self.frame_height)
        self.system_controls = SystemControls()
        self.character_recognizer = CharacterRecognizer()
        self.ui_overlay = UIOverlay(self.frame_width, self.frame_height)
        
        # Performance tracking
        self.fps_counter = FPSCounter()
        
        # Application state
        self.mode = AppMode.MOUSE
        self.running = True
        self.show_landmarks = True
        
        # Recognition state
        self.recognized_text = ''
        self.last_recognition_time = 0
        self.recognition_interval = 2.0  # Seconds
        
        print("=" * 50)
        print("Virtual Mouse & Air Writing System Initialized")
        print("=" * 50)
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"Camera FPS: {config.camera.fps}")
        print("=" * 50)
    
    def run(self):
        """Main application loop."""
        print("\nControls:")
        print("  M - Mouse mode")
        print("  D - Draw mode")
        print("  V - Volume control mode")
        print("  B - Brightness control mode")
        print("  C - Clear canvas")
        print("  S - Save canvas")
        print("  R - Recognize text")
        print("  L - Toggle landmarks")
        print("  Q - Quit")
        print("-" * 50)
        
        while self.running:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Update FPS
            fps = self.fps_counter.update()
            
            # Process hand tracking
            hand_data = self.hand_tracker.process_frame(frame)
            
            # Process gesture and mode-specific logic
            gesture = None
            volume_level = None
            brightness_level = None
            
            if hand_data:
                # Recognize gesture
                gesture_result = self.gesture_recognizer.recognize(hand_data)
                gesture = gesture_result.gesture
                
                # Mode-specific processing
                if self.mode == AppMode.MOUSE:
                    self._process_mouse_mode(hand_data, gesture_result)
                
                elif self.mode == AppMode.DRAW:
                    self._process_draw_mode(hand_data, gesture_result, frame)
                
                elif self.mode == AppMode.VOLUME:
                    volume_level = self._process_volume_mode(hand_data, gesture_result)
                
                elif self.mode == AppMode.BRIGHTNESS:
                    brightness_level = self._process_brightness_mode(hand_data, gesture_result)
                
                # Draw hand landmarks
                if self.show_landmarks:
                    frame = self.hand_tracker.draw_landmarks(frame, hand_data)
            
            # Overlay canvas in draw mode
            if self.mode == AppMode.DRAW:
                frame = self.air_canvas.get_overlay(frame)
            
            # Render UI
            mode_name = self.mode.name.lower()
            frame = self.ui_overlay.render(
                frame,
                mode=mode_name,
                fps=fps,
                gesture=gesture,
                volume=volume_level,
                brightness=brightness_level,
                recognized_text=self.recognized_text
            )
            
            # Display
            cv2.imshow('Virtual Mouse & Air Writing', frame)
            
            # Handle keyboard input
            self._handle_keyboard()
        
        self._cleanup()
    
    def _process_mouse_mode(self, hand_data, gesture_result):
        """Process mouse control mode."""
        action = self.mouse_controller.process(hand_data, gesture_result)
        
        # Mode switching gesture (open palm held)
        if gesture_result.gesture == Gesture.OPEN_PALM:
            pass  # Could add gesture-based mode switching here
    
    def _process_draw_mode(self, hand_data, gesture_result, frame):
        """Process drawing mode."""
        self.air_canvas.process(hand_data, gesture_result)
        
        # Update UI eraser state
        self.ui_overlay.set_eraser_active(self.air_canvas.eraser_mode)
        
        # Check for button clicks via gesture
        if gesture_result.gesture == Gesture.PINCH:
            index_tip = gesture_result.details['index_tip']
            button_clicked = self.ui_overlay.check_button_click(*index_tip)
            
            if button_clicked:
                self._handle_button_action(button_clicked)
        
        # Auto-recognize text periodically
        current_time = time.time()
        if current_time - self.last_recognition_time > self.recognition_interval:
            self._recognize_drawing()
            self.last_recognition_time = current_time
    
    def _process_volume_mode(self, hand_data, gesture_result) -> float:
        """Process volume control mode."""
        if gesture_result.gesture in [Gesture.POINTER, Gesture.PEACE, Gesture.PINCH]:
            level = self.system_controls.process_gesture(hand_data, 'volume')
            return level
        return self.system_controls.get_volume()
    
    def _process_brightness_mode(self, hand_data, gesture_result) -> int:
        """Process brightness control mode."""
        if gesture_result.gesture in [Gesture.POINTER, Gesture.PEACE, Gesture.PINCH]:
            level = self.system_controls.process_gesture(hand_data, 'brightness')
            return int(level * 100)
        return self.system_controls.get_brightness()
    
    def _handle_button_action(self, button_label: str):
        """Handle UI button actions."""
        if button_label == 'Mouse':
            self.mode = AppMode.MOUSE
        elif button_label == 'Draw':
            self.mode = AppMode.DRAW
        elif button_label == 'Clear':
            self.air_canvas.clear()
            self.recognized_text = ''
        elif button_label == 'Save':
            filename = self.air_canvas.save()
            print(f"Saved canvas to: {filename}")
        elif button_label == 'Eraser':
            self.air_canvas.toggle_eraser()
        elif button_label == '':  # Color buttons have no label
            pass  # Handle color selection
    
    def _recognize_drawing(self):
        """Recognize text from current drawing."""
        drawing_region = self.air_canvas.get_drawing_region()
        if drawing_region.any():
            text = self.character_recognizer.get_text(drawing_region)
            if text:
                self.recognized_text = text
    
    def _handle_keyboard(self):
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or Escape
            self.running = False
        
        elif key == ord('m'):
            self.mode = AppMode.MOUSE
            print("Switched to Mouse Mode")
        
        elif key == ord('d'):
            self.mode = AppMode.DRAW
            print("Switched to Draw Mode")
        
        elif key == ord('v'):
            self.mode = AppMode.VOLUME
            print("Switched to Volume Control Mode")
        
        elif key == ord('b'):
            self.mode = AppMode.BRIGHTNESS
            print("Switched to Brightness Control Mode")
        
        elif key == ord('c'):
            self.air_canvas.clear()
            self.recognized_text = ''
            print("Canvas cleared")
        
        elif key == ord('s'):
            filename = self.air_canvas.save()
            print(f"Canvas saved to: {filename}")
        
        elif key == ord('r'):
            self._recognize_drawing()
            print(f"Recognized text: {self.recognized_text}")
        
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
            print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
        
        elif key == ord('u'):
            self.air_canvas.undo()
            print("Undo last stroke")
    
    def _cleanup(self):
        """Clean up resources."""
        print("\nShutting down...")
        self.cap.release()
        self.hand_tracker.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


def main():
    """Application entry point."""
    print("\n" + "=" * 50)
    print("  AI-Powered Virtual Mouse & Air Writing System")
    print("=" * 50 + "\n")
    
    try:
        app = VirtualMouseApp()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
