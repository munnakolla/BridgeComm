"""
Custom Emotion Detection Service
Uses the trained FER-2013 model instead of DeepFace for faster inference.
"""

import os
import json
import base64
import io
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

# TensorFlow imports
CUSTOM_MODEL_AVAILABLE = False
emotion_model = None
emotion_interpreter = None  # TFLite interpreter
class_mapping = None

try:
    import tensorflow as tf
    from tensorflow import keras
    
    # Look in backend/models directory (same as ASL models)
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "models", "emotion_model.h5")
    TFLITE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               "models", "emotion_model.tflite")
    MAPPING_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                "models", "emotion_class_mapping.json")
    
    # Try TFLite first (more compatible)
    if os.path.exists(TFLITE_PATH):
        emotion_interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        emotion_interpreter.allocate_tensors()
        CUSTOM_MODEL_AVAILABLE = True
        print(f"Custom emotion TFLite model loaded from {TFLITE_PATH}")
    elif os.path.exists(MODEL_PATH):
        try:
            emotion_model = keras.models.load_model(MODEL_PATH, compile=False)
            CUSTOM_MODEL_AVAILABLE = True
            print(f"Custom emotion model loaded from {MODEL_PATH}")
        except Exception as load_err:
            print(f"Could not load emotion H5 model: {load_err}")
    else:
        print(f"Custom emotion model not found at {MODEL_PATH} or {TFLITE_PATH}, using DeepFace")
    
    # Load class mapping
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        print(f"Emotion class mapping loaded")
    else:
        # Default mapping
        class_mapping = {
            "index_to_class": {
                "0": "angry", "1": "disgust", "2": "fear",
                "3": "happy", "4": "neutral", "5": "sad", "6": "surprise"
            }
        }
        
except Exception as e:
    print(f"Could not load custom emotion model: {e}")


class CustomEmotionService:
    """Service for emotion detection using custom trained model."""
    
    def __init__(self):
        self.model = emotion_model
        self.interpreter = emotion_interpreter
        self.class_mapping = class_mapping
        self.img_size = 48  # FER-2013 image size
        self.available = CUSTOM_MODEL_AVAILABLE
        
        # Emotion to text responses
        self.emotion_responses = {
            "angry": "I sense frustration. How can I help?",
            "disgust": "Something seems unpleasant. Let me assist.",
            "fear": "You seem worried. I'm here to help.",
            "happy": "Great to see you in good spirits!",
            "neutral": "How can I assist you today?",
            "sad": "I'm sorry you're feeling down. How can I help?",
            "surprise": "Something unexpected? Let me know what you need."
        }
        
        if self.class_mapping and "emotion_responses" in self.class_mapping:
            self.emotion_responses.update(self.class_mapping["emotion_responses"])
    
    def preprocess_image(self, image_base64: str) -> Optional[np.ndarray]:
        """Preprocess image for emotion detection."""
        try:
            # Handle data URL prefix
            if image_base64.startswith('data:'):
                image_base64 = image_base64.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Resize to model input size
            image = image.resize((self.img_size, self.img_size))
            
            # Convert to numpy array and normalize
            img_array = np.array(image, dtype=np.float32) / 255.0
            
            # Add batch and channel dimensions
            img_array = np.expand_dims(img_array, axis=0)  # Batch dimension
            img_array = np.expand_dims(img_array, axis=-1)  # Channel dimension
            
            return img_array
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    async def detect_emotion(self, image_base64: str) -> Dict[str, Any]:
        """
        Detect emotion from facial image.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Dict with emotion, confidence, and all emotion scores
        """
        if not self.available:
            return {
                "error": "Custom emotion model not available",
                "dominant_emotion": "neutral",
                "confidence": 0.5
            }
        
        if self.model is None and self.interpreter is None:
            return {
                "error": "No emotion model loaded",
                "dominant_emotion": "neutral",
                "confidence": 0.5
            }
        
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_base64)
            if img_array is None:
                return {
                    "error": "Failed to preprocess image",
                    "dominant_emotion": "neutral",
                    "confidence": 0.5
                }
            
            # Make prediction using TFLite or Keras model
            if self.interpreter is not None:
                # Use TFLite interpreter
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.interpreter.set_tensor(input_details[0]['index'], img_array)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(output_details[0]['index'])[0]
            else:
                # Use Keras model
                predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Get emotion scores
            index_to_class = self.class_mapping.get("index_to_class", {})
            emotion_scores = {}
            
            for i, score in enumerate(predictions):
                emotion_name = index_to_class.get(str(i), f"emotion_{i}")
                emotion_scores[emotion_name] = float(score * 100)  # As percentage
            
            # Get dominant emotion
            dominant_idx = np.argmax(predictions)
            dominant_emotion = index_to_class.get(str(dominant_idx), "neutral")
            confidence = float(predictions[dominant_idx])
            
            return {
                "dominant_emotion": dominant_emotion,
                "emotions": emotion_scores,
                "confidence": confidence,
                "response": self.emotion_responses.get(dominant_emotion, "How can I assist you?"),
                "face_detected": True
            }
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return {
                "error": str(e),
                "dominant_emotion": "neutral",
                "confidence": 0.5,
                "face_detected": False
            }


# Singleton instance
_custom_emotion_service: Optional[CustomEmotionService] = None


def get_custom_emotion_service() -> CustomEmotionService:
    """Get the custom emotion service singleton."""
    global _custom_emotion_service
    if _custom_emotion_service is None:
        _custom_emotion_service = CustomEmotionService()
    return _custom_emotion_service
