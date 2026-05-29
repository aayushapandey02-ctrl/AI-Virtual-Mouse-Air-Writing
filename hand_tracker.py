"""
Hand detection and landmark tracking using MediaPipe.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple, NamedTuple
from dataclasses import dataclass

from config import config


@dataclass
class HandLandmarks:
    """Container for processed hand landmark data."""
    landmarks: List[Tuple[float, float, float]]  # Normalized (x, y, z)
    pixel_coords: List[Tuple[int, int]]  # Pixel coordinates
    handedness: str  # 'Left' or 'Right'
    confidence: float
    
    def get_landmark(self, idx: int) -> Tuple[float, float, float]:
        """Get specific landmark by MediaPipe index."""
        return self.landmarks[idx]
    
    def get_pixel_coord(self, idx: int) -> Tuple[int, int]:
        """Get pixel coordinates for specific landmark."""
        return self.pixel_coords[idx]


class HandTracker:
    """
    Real-time hand tracking using MediaPipe Hands.
    Provides landmark detection and basic hand analysis.
    """
    
    # MediaPipe landmark indices
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.hand_tracking.max_hands,
            min_detection_confidence=config.hand_tracking.min_detection_confidence,
            min_tracking_confidence=config.hand_tracking.min_tracking_confidence,
            model_complexity=config.hand_tracking.model_complexity
        )
        
        self.frame_shape: Optional[Tuple[int, int]] = None
    
    def process_frame(self, frame: np.ndarray) -> Optional[HandLandmarks]:
        """
        Process a frame and extract hand landmarks.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            HandLandmarks object if hand detected, None otherwise
        """
        self.frame_shape = frame.shape[:2]
        h, w = self.frame_shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Process first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0]
        
        # Extract landmarks
        landmarks = []
        pixel_coords = []
        
        for lm in hand_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))
            px = int(lm.x * w)
            py = int(lm.y * h)
            pixel_coords.append((px, py))
        
        return HandLandmarks(
            landmarks=landmarks,
            pixel_coords=pixel_coords,
            handedness=handedness.label,
            confidence=handedness.score
        )
    
    def draw_landmarks(self, frame: np.ndarray, 
                       hand_data: Optional[HandLandmarks],
                       draw_connections: bool = True) -> np.ndarray:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: Image to draw on
            hand_data: HandLandmarks object
            draw_connections: Whether to draw connections between landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        if hand_data is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Recreate MediaPipe landmark format for drawing
        landmark_list = []
        for (x, y, z) in hand_data.landmarks:
            landmark = type('Landmark', (), {'x': x, 'y': y, 'z': z})()
            landmark_list.append(landmark)
        
        # Create NormalizedLandmarkList-like object
        class LandmarkContainer:
            def __init__(self, landmarks):
                self.landmark = landmarks
        
        container = LandmarkContainer(landmark_list)
        
        if draw_connections:
            pass
            '''self.mp_draw.draw_landmarks(
                frame,
                container,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )'''
        else:
            # Draw only points
            for px, py in hand_data.pixel_coords:
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
        
        return frame
    
    def get_fingertip_positions(self, hand_data: HandLandmarks) -> dict:
        """Get positions of all fingertips."""
        return {
            'thumb': hand_data.pixel_coords[self.THUMB_TIP],
            'index': hand_data.pixel_coords[self.INDEX_TIP],
            'middle': hand_data.pixel_coords[self.MIDDLE_TIP],
            'ring': hand_data.pixel_coords[self.RING_TIP],
            'pinky': hand_data.pixel_coords[self.PINKY_TIP]
        }
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()
