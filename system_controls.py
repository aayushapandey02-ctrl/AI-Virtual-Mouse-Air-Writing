"""
System controls for volume and brightness using gestures.
"""

import numpy as np
from typing import Optional, Tuple
import platform

from hand_tracker import HandTracker, HandLandmarks
from utils import calculate_distance


class SystemControls:
    """
    Controls system volume and brightness using finger distance gestures.
    
    Uses distance between thumb and index finger to control levels.
    """
    
    def __init__(self):
        self.system = platform.system()
        self._init_audio_control()
        self._init_brightness_control()
        
        # Control state
        self.current_volume = 0.5
        self.current_brightness = 50
        self.control_active = False
        
        # Distance calibration
        self.min_distance = 0.02  # Normalized
        self.max_distance = 0.25
    
    def _init_audio_control(self):
        """Initialize audio control based on platform."""
        self.volume_control = None
        
        if self.system == 'Windows':
            try:
                from ctypes import cast, POINTER
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
                
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                )
                self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))
                self.min_vol, self.max_vol, _ = self.volume_control.GetVolumeRange()
            except Exception as e:
                print(f"Could not initialize volume control: {e}")
        
        elif self.system == 'Darwin':  # macOS
            # Use osascript for macOS
            pass
        
        elif self.system == 'Linux':
            # Use pactl or amixer for Linux
            pass
    
    def _init_brightness_control(self):
        """Initialize brightness control."""
        self.brightness_available = False
        
        try:
            import screen_brightness_control as sbc
            self.brightness_available = True
            self.sbc = sbc
        except Exception as e:
            print(f"Brightness control not available: {e}")
    
    def set_volume(self, level: float):
        """
        Set system volume.
        
        Args:
            level: Volume level from 0.0 to 1.0
        """
        level = np.clip(level, 0, 1)
        self.current_volume = level
        
        if self.system == 'Windows' and self.volume_control:
            try:
                vol = self.min_vol + level * (self.max_vol - self.min_vol)
                self.volume_control.SetMasterVolumeLevel(vol, None)
            except Exception:
                pass
        
        elif self.system == 'Darwin':
            import subprocess
            vol_percent = int(level * 100)
            subprocess.run(
                ['osascript', '-e', f'set volume output volume {vol_percent}'],
                capture_output=True
            )
        
        elif self.system == 'Linux':
            import subprocess
            vol_percent = int(level * 100)
            subprocess.run(
                ['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f'{vol_percent}%'],
                capture_output=True
            )
    
    def set_brightness(self, level: float):
        """
        Set screen brightness.
        
        Args:
            level: Brightness level from 0.0 to 1.0
        """
        if not self.brightness_available:
            return
        
        level = np.clip(level, 0.1, 1.0)  # Minimum 10% brightness
        self.current_brightness = int(level * 100)
        
        try:
            self.sbc.set_brightness(self.current_brightness)
        except Exception:
            pass
    
    def process_gesture(self, hand_data: HandLandmarks, 
                        control_type: str = 'volume') -> float:
        """
        Process hand gesture to control volume or brightness.
        
        Uses thumb-index distance for control.
        
        Args:
            hand_data: Hand landmarks
            control_type: 'volume' or 'brightness'
            
        Returns:
            Current level (0-1)
        """
        landmarks = hand_data.landmarks
        
        # Get thumb and index positions
        thumb_tip = landmarks[HandTracker.THUMB_TIP]
        index_tip = landmarks[HandTracker.INDEX_TIP]
        
        # Calculate normalized distance
        distance = calculate_distance(thumb_tip[:2], index_tip[:2])
        
        # Map distance to level
        level = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        level = np.clip(level, 0, 1)
        
        if control_type == 'volume':
            self.set_volume(level)
        elif control_type == 'brightness':
            self.set_brightness(level)
        
        return level
    
    def get_volume(self) -> float:
        """Get current volume level (0-1)."""
        if self.system == 'Windows' and self.volume_control:
            try:
                vol = self.volume_control.GetMasterVolumeLevel()
                return (vol - self.min_vol) / (self.max_vol - self.min_vol)
            except Exception:
                pass
        return self.current_volume
    
    def get_brightness(self) -> int:
        """Get current brightness level (0-100)."""
        if self.brightness_available:
            try:
                return self.sbc.get_brightness()[0]
            except Exception:
                pass
        return self.current_brightness
