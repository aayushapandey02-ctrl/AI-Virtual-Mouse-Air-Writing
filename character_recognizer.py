"""
AI-based handwritten character recognition.
Uses a CNN model for recognizing drawn characters.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import os


class CharacterRecognizer:
    """
    Recognizes handwritten characters from the air canvas.
    Uses TensorFlow/Keras CNN model trained on EMNIST dataset.
    """
    
    # Character labels for EMNIST balanced dataset
    LABELS = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._create_simple_model()
    
    def _load_model(self, model_path: str):
        """Load pre-trained model."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            self.model_loaded = True
        except Exception as e:
            print(f"Could not load model: {e}")
            self._create_simple_model()
    
    def _create_simple_model(self):
        """Create a simple CNN model for character recognition."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            self.model = keras.Sequential([
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(self.LABELS), activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model_loaded = True
            print("Created character recognition model (untrained)")
            
        except ImportError:
            print("TensorFlow not available. Character recognition disabled.")
            self.model_loaded = False
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess drawing for recognition.
        
        Args:
            image: Grayscale drawing image
            
        Returns:
            Preprocessed 28x28 image
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find contours and bounding box
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return np.zeros((28, 28), dtype=np.float32)
        
        # Get bounding box of all contours
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        
        # Extract and pad to square
        roi = image[y:y+h, x:x+w]
        
        # Make square
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = roi
        
        # Resize to 28x28
        resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize character in image.
        
        Args:
            image: Drawing image
            
        Returns:
            Tuple of (predicted_character, confidence)
        """
        if not self.model_loaded:
            return ('?', 0.0)
        
        # Preprocess
        processed = self.preprocess(image)
        
        # Add batch and channel dimensions
        input_image = processed.reshape(1, 28, 28, 1)
        
        # Predict
        try:
            predictions = self.model.predict(input_image, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]
            
            if predicted_idx < len(self.LABELS):
                character = self.LABELS[predicted_idx]
            else:
                character = '?'
            
            return (character, float(confidence))
            
        except Exception as e:
            print(f"Recognition error: {e}")
            return ('?', 0.0)
    
    def recognize_multiple(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Recognize multiple characters/segments in image.
        
        Returns:
            List of (character, confidence) tuples
        """
        if not self.model_loaded:
            return []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find contours for segmentation
        contours, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return []
        
        # Sort contours left to right
        bboxes = [cv2.boundingRect(c) for c in contours]
        sorted_indices = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0])
        
        results = []
        for idx in sorted_indices:
            x, y, w, h = bboxes[idx]
            
            # Skip very small contours
            if w < 10 or h < 10:
                continue
            
            # Extract segment
            segment = gray[y:y+h, x:x+w]
            
            # Recognize
            char, conf = self.recognize(segment)
            results.append((char, conf))
        
        return results
    
    def get_text(self, image: np.ndarray, min_confidence: float = 0.5) -> str:
        """
        Get recognized text from image.
        
        Args:
            image: Drawing image
            min_confidence: Minimum confidence threshold
            
        Returns:
            Recognized text string
        """
        results = self.recognize_multiple(image)
        
        text = ''
        for char, conf in results:
            if conf >= min_confidence:
                text += char
        
        return text
