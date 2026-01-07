"""
ASL Gesture Detection Service
Uses the trained ASL CNN model for alphabet letter recognition.
"""

import os
import json
import base64
import io
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from PIL import Image

# TensorFlow imports
ASL_MODEL_AVAILABLE = False
asl_model = None
asl_class_mapping = None
asl_interpreter = None  # TFLite interpreter

try:
    import tensorflow as tf
    from tensorflow import keras
    
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "models", "asl_cnn_model.h5")
    TFLITE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                               "models", "asl_cnn_model.tflite")
    MAPPING_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                "models", "asl_class_mapping.json")
    
    # Try TFLite first (more compatible)
    if os.path.exists(TFLITE_PATH):
        asl_interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
        asl_interpreter.allocate_tensors()
        ASL_MODEL_AVAILABLE = True
        print(f"ASL TFLite model loaded from {TFLITE_PATH}")
    elif os.path.exists(MODEL_PATH):
        # Fallback to H5 model with custom objects
        try:
            asl_model = keras.models.load_model(MODEL_PATH, compile=False)
            ASL_MODEL_AVAILABLE = True
            print(f"ASL CNN model loaded from {MODEL_PATH}")
        except Exception as load_err:
            print(f"Failed to load H5 model: {load_err}")
            # Try loading with legacy format
            try:
                import tf_keras
                asl_model = tf_keras.models.load_model(MODEL_PATH, compile=False)
                ASL_MODEL_AVAILABLE = True
                print(f"ASL CNN model loaded with tf_keras from {MODEL_PATH}")
            except Exception as legacy_err:
                print(f"Also failed with tf_keras: {legacy_err}")
    else:
        print(f"ASL CNN model not found at {MODEL_PATH} or {TFLITE_PATH}")
    
    # Load class mapping
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, 'r') as f:
            asl_class_mapping = json.load(f)
        print(f"ASL class mapping loaded: {len(asl_class_mapping.get('index_to_class', {}))} classes")
    else:
        # Default mapping for ASL alphabet
        asl_class_mapping = {
            "index_to_class": {str(i): chr(65 + i) for i in range(26)}
        }
        asl_class_mapping["index_to_class"]["26"] = "del"
        asl_class_mapping["index_to_class"]["27"] = "nothing"
        asl_class_mapping["index_to_class"]["28"] = "space"
        
except Exception as e:
    print(f"Could not load ASL CNN model: {e}")

# Try to import OpenCV for hand detection
cv2 = None
try:
    import cv2 as cv2_import
    cv2 = cv2_import
except ImportError:
    print("Warning: OpenCV not available for ASL service")

# Try to import MediaPipe for hand detection using tasks API
mp_hands_detector = None
try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp_import
    
    # Use the Hands detector from MediaPipe Tasks if available
    # For now, we'll skip hand detection if solutions is not available
    # The gesture recognizer already handles hand detection internally
    print("MediaPipe Tasks available for ASL service")
except Exception as e:
    print(f"MediaPipe Tasks not available for ASL: {e}")


class ASLGestureService:
    """Service for ASL alphabet gesture detection using trained CNN model."""
    
    def __init__(self):
        self.model = asl_model
        self.interpreter = asl_interpreter  # TFLite interpreter
        self.class_mapping = asl_class_mapping
        self.img_size = 64  # Model was trained on 64x64 images based on error
        self.available = ASL_MODEL_AVAILABLE
        self.hands_detector = None  # Disabled for now - will use full frame
        
        # Common ASL letter sequences to form words
        self.common_words = {
            "HELLO": "Hello",
            "HI": "Hi",
            "YES": "Yes",
            "NO": "No",
            "HELP": "Help",
            "PLEASE": "Please",
            "THANKS": "Thanks",
            "THANK": "Thank",
            "YOU": "You",
            "ME": "Me",
            "LOVE": "Love",
            "GOOD": "Good",
            "BAD": "Bad",
            "SORRY": "Sorry",
            "OK": "OK",
            "WATER": "Water",
            "FOOD": "Food",
            "HOME": "Home",
            "WORK": "Work",
        }
        
        print(f"ASLGestureService initialized. Model available: {self.available}")
    
    def _extract_hand_region(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Extract the hand region from an image using MediaPipe.
        Returns cropped hand image and bounding box info.
        """
        if self.hands_detector is None or cv2 is None:
            return None
            
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands_detector.process(rgb_image)
            
            if not results.multi_hand_landmarks:
                return None
            
            # Get first hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Calculate bounding box
            h, w, _ = image.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min = int(max(0, min(x_coords) - 20))
            x_max = int(min(w, max(x_coords) + 20))
            y_min = int(max(0, min(y_coords) - 20))
            y_max = int(min(h, max(y_coords) + 20))
            
            # Crop hand region
            hand_crop = image[y_min:y_max, x_min:x_max]
            
            return hand_crop, {
                "x": x_min, "y": y_min, 
                "width": x_max - x_min, 
                "height": y_max - y_min
            }
            
        except Exception as e:
            print(f"Hand extraction error: {e}")
            return None
    
    def preprocess_image(self, image_base64: str) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Preprocess image for ASL gesture detection.
        Extracts hand region and resizes for model input.
        
        Returns:
            Tuple of (preprocessed image array, metadata) or None if failed
        """
        try:
            # Handle data URL prefix
            if image_base64.startswith('data:'):
                image_base64 = image_base64.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image = np.array(pil_image)
            
            # Convert RGB to BGR for OpenCV
            if cv2 is not None:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            metadata = {"hand_detected": False, "original_size": image.shape[:2]}
            
            # Try to extract hand region
            hand_result = self._extract_hand_region(image)
            if hand_result is not None:
                hand_crop, bbox = hand_result
                metadata["hand_detected"] = True
                metadata["hand_bbox"] = bbox
                image = hand_crop
            
            # Resize to model input size
            if cv2 is not None:
                image = cv2.resize(image, (self.img_size, self.img_size))
            else:
                pil_resized = Image.fromarray(image).resize((self.img_size, self.img_size))
                image = np.array(pil_resized)
            
            # Normalize to [0, 1]
            img_array = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, metadata
            
        except Exception as e:
            print(f"Error preprocessing ASL image: {e}")
            return None
    
    async def recognize_letter(self, image_base64: str) -> Dict[str, Any]:
        """
        Recognize ASL alphabet letter from image.
        
        Returns:
            Dict with detected letter, confidence, and alternatives
        """
        if not self.available:
            return {
                "error": "ASL model not available",
                "letter": None,
                "confidence": 0
            }
        
        if self.model is None and self.interpreter is None:
            return {
                "error": "No ASL model loaded",
                "letter": None,
                "confidence": 0
            }
        
        try:
            # Preprocess image
            result = self.preprocess_image(image_base64)
            if result is None:
                return {
                    "error": "Failed to preprocess image",
                    "letter": None,
                    "confidence": 0
                }
            
            img_array, metadata = result
            
            # Run prediction using TFLite or Keras model
            if self.interpreter is not None:
                # Use TFLite interpreter
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()
                
                self.interpreter.set_tensor(input_details[0]['index'], img_array)
                self.interpreter.invoke()
                predictions = self.interpreter.get_tensor(output_details[0]['index'])
            else:
                # Use Keras model
                predictions = self.model.predict(img_array, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[::-1][:5]
            
            results = []
            for idx in top_indices:
                letter = self.class_mapping.get("index_to_class", {}).get(str(idx), "?")
                confidence = float(predictions[0][idx])
                results.append({"letter": letter, "confidence": confidence})
            
            # Primary result
            top_letter = results[0]["letter"] if results else None
            top_confidence = results[0]["confidence"] if results else 0
            
            print(f"ASL recognized: {top_letter} ({top_confidence:.2%})")
            
            return {
                "letter": top_letter,
                "confidence": top_confidence,
                "alternatives": results[1:4],  # Top 3 alternatives
                "hand_detected": metadata.get("hand_detected", False),
                "hand_bbox": metadata.get("hand_bbox")
            }
            
        except Exception as e:
            print(f"ASL recognition error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "letter": None,
                "confidence": 0
            }
    
    async def recognize_sequence(self, images_base64: List[str], 
                                  interval_ms: int = 500) -> Dict[str, Any]:
        """
        Recognize a sequence of ASL letters from multiple images.
        Attempts to form words from detected letters.
        
        Args:
            images_base64: List of base64 encoded images
            interval_ms: Time interval between frames in milliseconds
        
        Returns:
            Dict with detected letters, formed word/text, and confidence
        """
        if not self.available:
            return {"error": "ASL model not available", "letters": [], "text": ""}
        
        detected_letters = []
        
        for img_b64 in images_base64:
            result = await self.recognize_letter(img_b64)
            if result.get("letter") and result.get("confidence", 0) > 0.5:
                detected_letters.append({
                    "letter": result["letter"],
                    "confidence": result["confidence"]
                })
        
        # Remove consecutive duplicates (finger spelling holds same letter)
        cleaned_letters = []
        for item in detected_letters:
            if not cleaned_letters or cleaned_letters[-1]["letter"] != item["letter"]:
                cleaned_letters.append(item)
        
        # Form text from letters
        letters_str = "".join([l["letter"] for l in cleaned_letters 
                               if l["letter"] not in ["del", "nothing"]])
        
        # Replace "space" with actual space
        letters_str = letters_str.replace("space", " ")
        
        # Check for known words
        formed_word = None
        for word_key, word_value in self.common_words.items():
            if letters_str.upper() == word_key:
                formed_word = word_value
                break
        
        return {
            "letters": cleaned_letters,
            "raw_text": letters_str,
            "formed_word": formed_word or letters_str,
            "confidence": np.mean([l["confidence"] for l in cleaned_letters]) if cleaned_letters else 0
        }


# Singleton instance
_asl_service_instance = None

def get_asl_gesture_service() -> ASLGestureService:
    """Get singleton ASL gesture service instance."""
    global _asl_service_instance
    if _asl_service_instance is None:
        _asl_service_instance = ASLGestureService()
    return _asl_service_instance
