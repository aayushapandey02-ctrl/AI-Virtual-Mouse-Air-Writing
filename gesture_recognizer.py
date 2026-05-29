"""
Gesture recognition system for hand pose classification.
"""

import numpy as np
from typing import Optional, List, Tuple
from enum import Enum, auto
from dataclasses import dataclass

from hand_tracker import HandTracker, HandLandmarks
from utils import calculate_distance, calculate_angle, GestureBuffer
from config import config


class Gesture(Enum):
    """Enumeration of recognized gestures."""
    NONE = auto()
    POINTER = auto()           # Index finger up only
    PEACE = auto()             # Index and middle up (V sign)
    FIST = auto()              # All fingers closed
    OPEN_PALM = auto()         # All fingers spread
    PINCH = auto()             # Thumb and index touching
    THUMB_INDEX_PINCH = auto() # Specific pinch for clicking
    THREE_FINGERS = auto()     # Index, middle, ring up
    FOUR_FINGERS = auto()      # All except thumb
    THUMB_UP = auto()          # Thumb up, others closed
    SCROLL_UP = auto()         # Index up, thumb out
    SCROLL_DOWN = auto()       # Index and middle together, pointing down
    OK_SIGN = auto()           # Thumb and index circle
    GRAB = auto()              # Closing fist motion


@dataclass
class GestureResult:
    """Result of gesture recognition."""
    gesture: Gesture
    confidence: float
    finger_states: List[bool]  # [thumb, index, middle, ring, pinky]
    details: dict


class GestureRecognizer:
    """
    Recognizes hand gestures from landmark positions.
    Uses geometric analysis of finger positions and angles.
    """
    
    def __init__(self):
        self.gesture_buffer = GestureBuffer(config.gesture.gesture_hold_frames)
        self.prev_gesture = Gesture.NONE
        self.prev_landmarks: Optional[HandLandmarks] = None
    
    def recognize(self, hand_data: HandLandmarks) -> GestureResult:
        """
        Analyze hand landmarks and recognize gesture.
        
        Args:
            hand_data: HandLandmarks from tracker
            
        Returns:
            GestureResult with recognized gesture and metadata
        """
        finger_states = self._get_finger_states(hand_data)
        gesture = self._classify_gesture(hand_data, finger_states)
        confidence = self._calculate_confidence(hand_data, gesture)
        
        # Apply temporal smoothing
        if self.gesture_buffer.update(gesture.name):
            confirmed_gesture = gesture
        else:
            confirmed_gesture = self.prev_gesture
        
        details = self._get_gesture_details(hand_data, gesture)
        
        self.prev_gesture = confirmed_gesture
        self.prev_landmarks = hand_data
        
        return GestureResult(
            gesture=confirmed_gesture,
            confidence=confidence,
            finger_states=finger_states,
            details=details
        )
    
    def _get_finger_states(self, hand_data: HandLandmarks) -> List[bool]:
        """
        Determine which fingers are extended.
        Returns list of booleans for [thumb, index, middle, ring, pinky].
        """
        landmarks = hand_data.landmarks
        
        # Finger tip and pip indices
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [3, 6, 10, 14, 18]
        finger_mcps = [2, 5, 9, 13, 17]
        
        states = []
        
        # Thumb - check horizontal distance from palm
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        
        # For thumb, check if tip is further from palm center than IP joint
        if hand_data.handedness == 'Right':
            thumb_extended = thumb_tip[0] < thumb_ip[0]
        else:
            thumb_extended = thumb_tip[0] > thumb_ip[0]
        states.append(thumb_extended)
        
        # Other fingers - compare tip y to pip y (lower y = higher position)
        for tip_idx, pip_idx in zip(finger_tips[1:], finger_pips[1:]):
            tip_y = landmarks[tip_idx][1]
            pip_y = landmarks[pip_idx][1]
            mcp_y = landmarks[finger_mcps[finger_tips.index(tip_idx)]][1]
            
            # Finger is extended if tip is above (lower y) than pip
            extended = tip_y < pip_y
            states.append(extended)
        
        return states
    
    def _classify_gesture(self, hand_data: HandLandmarks, 
                          finger_states: List[bool]) -> Gesture:
        """Classify gesture based on finger states and positions."""
        landmarks = hand_data.landmarks
        thumb, index, middle, ring, pinky = finger_states
        
        # Check for pinch (thumb and index close together)
        thumb_tip = landmarks[HandTracker.THUMB_TIP]
        index_tip = landmarks[HandTracker.INDEX_TIP]
        pinch_distance = calculate_distance(thumb_tip[:2], index_tip[:2])
        
        if pinch_distance < config.gesture.pinch_threshold:
            return Gesture.PINCH
        
        # Count extended fingers
        extended_count = sum(finger_states)
        
        # Fist - no fingers extended
        if extended_count == 0:
            return Gesture.FIST
        
        # Thumb up
        if thumb and not any([index, middle, ring, pinky]):
            return Gesture.THUMB_UP
        
        # Pointer - only index extended
        if index and not any([thumb, middle, ring, pinky]):
            return Gesture.POINTER
        
        # Peace sign - index and middle extended
        if index and middle and not any([thumb, ring, pinky]):
            return Gesture.PEACE
        
        # Three fingers
        if index and middle and ring and not any([thumb, pinky]):
            return Gesture.THREE_FINGERS
        
        # Four fingers
        if index and middle and ring and pinky and not thumb:
            return Gesture.FOUR_FINGERS
        
        # Open palm - all fingers extended
        if all(finger_states):
            return Gesture.OPEN_PALM
        
        # Check for OK sign (thumb and index forming circle, others extended)
        if pinch_distance < config.gesture.pinch_threshold * 1.5:
            if middle and ring and pinky:
                return Gesture.OK_SIGN
        
        return Gesture.NONE
    
    def _calculate_confidence(self, hand_data: HandLandmarks, 
                              gesture: Gesture) -> float:
        """Calculate confidence score for recognized gesture."""
        # Base confidence from hand detection
        base_confidence = hand_data.confidence
        
        # Adjust based on gesture clarity
        # This is a simplified confidence calculation
        return base_confidence * 0.9
    
    def _get_gesture_details(self, hand_data: HandLandmarks, 
                             gesture: Gesture) -> dict:
        """Get additional details about the gesture."""
        landmarks = hand_data.landmarks
        
        details = {
            'index_tip': hand_data.pixel_coords[HandTracker.INDEX_TIP],
            'thumb_tip': hand_data.pixel_coords[HandTracker.THUMB_TIP],
            'palm_center': self._calculate_palm_center(hand_data),
        }
        
        # Add pinch distance for pinch gestures
        if gesture in [Gesture.PINCH, Gesture.OK_SIGN]:
            thumb_tip = landmarks[HandTracker.THUMB_TIP]
            index_tip = landmarks[HandTracker.INDEX_TIP]
            details['pinch_distance'] = calculate_distance(
                thumb_tip[:2], index_tip[:2]
            )
        
        # Add scroll info for peace gesture
        if gesture == Gesture.PEACE:
            index_tip = landmarks[HandTracker.INDEX_TIP]
            middle_tip = landmarks[HandTracker.MIDDLE_TIP]
            details['finger_spread'] = calculate_distance(
                index_tip[:2], middle_tip[:2]
            )
        
        return details
    
    def _calculate_palm_center(self, hand_data: HandLandmarks) -> Tuple[int, int]:
        """Calculate approximate center of palm."""
        # Use wrist and base of middle finger
        wrist = hand_data.pixel_coords[HandTracker.WRIST]
        middle_mcp = hand_data.pixel_coords[HandTracker.MIDDLE_MCP]
        
        center_x = (wrist[0] + middle_mcp[0]) // 2
        center_y = (wrist[1] + middle_mcp[1]) // 2
        
        return (center_x, center_y)
    
    def get_pinch_distance(self, hand_data: HandLandmarks) -> float:
        """Get normalized distance between thumb and index finger."""
        landmarks = hand_data.landmarks
        thumb_tip = landmarks[HandTracker.THUMB_TIP]
        index_tip = landmarks[HandTracker.INDEX_TIP]
        return calculate_distance(thumb_tip[:2], index_tip[:2])
    
    def get_finger_distance(self, hand_data: HandLandmarks, 
                            finger1_idx: int, finger2_idx: int) -> float:
        """Get distance between any two landmarks."""
        landmarks = hand_data.landmarks
        p1 = landmarks[finger1_idx]
        p2 = landmarks[finger2_idx]
        return calculate_distance(p1[:2], p2[:2])
