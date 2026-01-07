"""
Vision Service
Handles sign language detection and gesture recognition using MediaPipe and OpenCV.
Uses DeepFace for accurate emotion detection.
Also provides Azure AI Vision integration for scene description and OCR.
"""

import base64
import io
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import httpx

from app.core.config import get_settings

# Try to import dependencies
cv2 = None
mp = None
MEDIAPIPE_AVAILABLE = False
DEEPFACE_AVAILABLE = False
GestureRecognizer = None
gesture_recognizer = None

# Import OpenCV
try:
    import cv2 as cv2_import
    cv2 = cv2_import
except ImportError:
    print("Warning: OpenCV not available")

# Import DeepFace for emotion detection (fallback if custom model not available)
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("DeepFace loaded successfully for emotion detection.")
except ImportError as e:
    print(f"Warning: DeepFace not available: {e}")

# Import custom emotion service (trained on FER-2013)
CUSTOM_EMOTION_AVAILABLE = False
custom_emotion_service = None
try:
    from app.services.custom_emotion_service import get_custom_emotion_service, CUSTOM_MODEL_AVAILABLE
    if CUSTOM_MODEL_AVAILABLE:
        custom_emotion_service = get_custom_emotion_service()
        CUSTOM_EMOTION_AVAILABLE = True
        print("Custom emotion model loaded (trained on FER-2013)")
except ImportError as e:
    print(f"Custom emotion service not available: {e}")

# Import ASL gesture service (trained CNN for alphabet)
ASL_GESTURE_AVAILABLE = False
asl_gesture_service = None
try:
    from app.services.asl_gesture_service import get_asl_gesture_service, ASL_MODEL_AVAILABLE
    if ASL_MODEL_AVAILABLE:
        asl_gesture_service = get_asl_gesture_service()
        ASL_GESTURE_AVAILABLE = True
        print("ASL CNN model loaded (trained on ASL Alphabet dataset)")
except ImportError as e:
    print(f"ASL gesture service not available: {e}")

# Import MediaPipe Gesture Recognizer
try:
    import mediapipe as mp_import
    mp = mp_import
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    
    # Check if gesture recognizer model exists
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "models", "gesture_recognizer.task")
    
    if os.path.exists(MODEL_PATH):
        # Initialize gesture recognizer
        base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2
        )
        gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(options)
        MEDIAPIPE_AVAILABLE = True
        print(f"MediaPipe Gesture Recognizer loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Gesture model not found at {MODEL_PATH}. Using fallback.")
        MEDIAPIPE_AVAILABLE = False
except Exception as e:
    print(f"Warning: MediaPipe Gesture Recognizer not available: {e}")


class VisionService:
    """Service for computer vision operations including gesture and emotion detection.
    
    Uses LOCAL models only:
    - MediaPipe Gesture Recognizer for common gestures (thumbs up, wave, etc.)
    - ASL CNN model for alphabet letter recognition
    - Custom CNN (FER-2013 trained) for emotion detection
    - Falls back to DeepFace if custom model not available
    - OpenCV for image processing
    """
    
    def __init__(self):
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        self.deepface_available = DEEPFACE_AVAILABLE
        self.custom_emotion_available = CUSTOM_EMOTION_AVAILABLE
        self.custom_emotion_service = custom_emotion_service
        self.asl_gesture_available = ASL_GESTURE_AVAILABLE
        self.asl_gesture_service = asl_gesture_service
        self.settings = get_settings()
        
        # Gesture mappings for sign language interpretation
        self.gesture_to_text = {
            # MediaPipe Gesture Recognizer outputs
            "None": "No gesture detected",
            "Closed_Fist": "Yes / Agree / Power",
            "Open_Palm": "Stop / Hello / Wait",
            "Pointing_Up": "Attention / Up / One",
            "Thumb_Down": "No / Bad / Disagree",
            "Thumb_Up": "Good / Okay / Yes",
            "Victory": "Peace / Victory / Two",
            "ILoveYou": "I love you",
            # Fallback mappings
            "open_hand": "Hello / Stop",
            "fist": "Yes / Agree",
            "pointing": "That / There",
            "peace": "Peace / Two / Victory",
            "thumbs_up": "Good / Okay / Yes",
            "thumbs_down": "Bad / No / Disagree",
        }
        
        self.gesture_recognizer = gesture_recognizer
        print(f"VisionService initialized. CustomEmotion: {self.custom_emotion_available}, DeepFace: {self.deepface_available}, MediaPipe Gestures: {self.mediapipe_available}, ASL CNN: {self.asl_gesture_available}")
    
    def _decode_image(self, image_base64: str) -> Optional[np.ndarray]:
        """Decode base64 image to numpy array."""
        if cv2 is None:
            return None
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    async def detect_emotion_deepface(self, image_base64: str) -> Dict[str, Any]:
        """
        Detect facial emotion using custom model (FER-2013) or DeepFace (fallback).
        
        Returns:
            Dict with dominant_emotion, emotion scores, and face region
        """
        # Try custom emotion model first (faster, trained on FER-2013)
        if self.custom_emotion_available and self.custom_emotion_service:
            try:
                result = await self.custom_emotion_service.detect_emotion(image_base64)
                if not result.get("error"):
                    print("Used custom emotion model for detection")
                    return result
            except Exception as e:
                print(f"Custom emotion model failed, falling back to DeepFace: {e}")
        
        # Fallback to DeepFace
        if not self.deepface_available:
            return {"error": "DeepFace not available", "dominant_emotion": "neutral"}
        
        try:
            # Decode image
            image_data = base64.b64decode(image_base64)
            
            # Save temporarily for DeepFace
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(image_data)
                temp_path = f.name
            
            try:
                # Analyze with DeepFace
                result = DeepFace.analyze(
                    img_path=temp_path,
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                # DeepFace returns a list if multiple faces
                if isinstance(result, list):
                    result = result[0] if result else {}
                
                return {
                    "dominant_emotion": result.get("dominant_emotion", "neutral"),
                    "emotions": result.get("emotion", {}),
                    "face_detected": True,
                    "confidence": max(result.get("emotion", {}).values()) / 100 if result.get("emotion") else 0.5
                }
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"DeepFace error: {e}")
            return {
                "dominant_emotion": "neutral",
                "emotions": {},
                "face_detected": False,
                "error": str(e)
            }
    
    async def recognize_gesture_mediapipe(self, image_base64: str) -> Dict[str, Any]:
        """
        Recognize hand gestures using MediaPipe Gesture Recognizer.
        
        Returns:
            Dict with detected gestures, hand landmarks, and confidence
        """
        if not self.mediapipe_available or self.gesture_recognizer is None:
            print("MediaPipe not available or gesture_recognizer is None")
            return {"error": "MediaPipe Gesture Recognizer not available", "gestures": []}
        
        try:
            # Decode image - handle potential data URL prefix
            if image_base64.startswith('data:'):
                # Remove data URL prefix (e.g., "data:image/jpeg;base64,")
                image_base64 = image_base64.split(',', 1)[1]
            
            image_data = base64.b64decode(image_base64)
            print(f"Image data size: {len(image_data)} bytes")
            
            image = Image.open(io.BytesIO(image_data))
            print(f"Image size: {image.size}, mode: {image.mode}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            print(f"Image array shape: {image_array.shape}")
            
            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)
            
            # Recognize gestures
            result = self.gesture_recognizer.recognize(mp_image)
            print(f"MediaPipe result - gestures: {len(result.gestures) if result.gestures else 0}, hands: {len(result.handedness) if result.handedness else 0}")
            
            gestures_detected = []
            hands_data = []
            
            # Minimum confidence threshold for accepting a gesture
            MIN_CONFIDENCE_THRESHOLD = 0.65  # Only accept gestures with 65%+ confidence
            
            if result.gestures:
                for i, gesture_list in enumerate(result.gestures):
                    if gesture_list:
                        # Get top gesture
                        top_gesture = gesture_list[0]
                        gesture_name = top_gesture.category_name
                        confidence = top_gesture.score
                        
                        print(f"Raw detected gesture: {gesture_name} with confidence {confidence:.2f}")
                        
                        # Skip low-confidence gestures
                        if confidence < MIN_CONFIDENCE_THRESHOLD:
                            print(f"Skipping gesture {gesture_name} - confidence {confidence:.2f} below threshold {MIN_CONFIDENCE_THRESHOLD}")
                            continue
                        
                        # Skip "None" gesture (hand detected but no clear gesture)
                        if gesture_name == "None":
                            print(f"Skipping 'None' gesture - hand detected but no clear gesture")
                            # Still track that a hand was detected
                            hands_data.append({
                                "gesture": "None",
                                "handedness": "Unknown",
                                "confidence": confidence,
                                "note": "Hand detected but gesture unclear"
                            })
                            continue
                        
                        print(f"ACCEPTED gesture: {gesture_name} with confidence {confidence:.2f}")
                        
                        gestures_detected.append({
                            "gesture": gesture_name,
                            "text": self.gesture_to_text.get(gesture_name, gesture_name),
                            "confidence": confidence
                        })
                        
                        # Get handedness
                        handedness = "Unknown"
                        if result.handedness and i < len(result.handedness):
                            handedness = result.handedness[i][0].category_name
                        
                        hands_data.append({
                            "gesture": gesture_name,
                            "handedness": handedness,
                            "confidence": confidence
                        })
            else:
                print("No gestures detected in image - no hands found")
            
            return {
                "gestures_detected": len(gestures_detected),
                "gestures": gestures_detected,
                "hands": hands_data,
                "primary_gesture": gestures_detected[0] if gestures_detected else None
            }
            
        except Exception as e:
            print(f"MediaPipe gesture recognition error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "gestures_detected": 0,
                "gestures": [],
                "error": str(e)
            }
    
    async def recognize_asl_letter(self, image_base64: str) -> Dict[str, Any]:
        """
        Recognize ASL alphabet letter using trained CNN model.
        
        This is for fingerspelling (A-Z letters, space, delete, nothing).
        
        Returns:
            Dict with detected letter, confidence, and alternatives
        """
        if not self.asl_gesture_available or self.asl_gesture_service is None:
            return {"error": "ASL CNN model not available", "letter": None}
        
        try:
            result = await self.asl_gesture_service.recognize_letter(image_base64)
            return result
        except Exception as e:
            print(f"ASL letter recognition error: {e}")
            return {"error": str(e), "letter": None, "confidence": 0}
    
    async def recognize_combined(self, image_base64: str) -> Dict[str, Any]:
        """
        Combined gesture recognition using both MediaPipe and ASL CNN model.
        
        - MediaPipe: Detects common gestures (thumbs up, wave, pointing, etc.)
        - ASL CNN: Detects alphabet letters for fingerspelling
        
        Returns results from both systems for best coverage.
        """
        result = {
            "mediapipe_gesture": None,
            "asl_letter": None,
            "combined_text": "",
            "confidence": 0,
            "method_used": None
        }
        
        # Try MediaPipe first for common gestures
        mp_result = await self.recognize_gesture_mediapipe(image_base64)
        if mp_result.get("primary_gesture") and mp_result["primary_gesture"].get("confidence", 0) > 0.65:
            gesture = mp_result["primary_gesture"]
            result["mediapipe_gesture"] = gesture
            result["combined_text"] = gesture.get("text", gesture.get("gesture", ""))
            result["confidence"] = gesture.get("confidence", 0)
            result["method_used"] = "mediapipe"
        
        # Also try ASL CNN for alphabet letters
        if self.asl_gesture_available:
            asl_result = await self.recognize_asl_letter(image_base64)
            if asl_result.get("letter") and asl_result.get("confidence", 0) > 0.5:
                result["asl_letter"] = asl_result
                
                # If no good MediaPipe result, use ASL letter
                if not result["method_used"] or asl_result.get("confidence", 0) > result["confidence"]:
                    letter = asl_result.get("letter", "")
                    if letter not in ["nothing", "del"]:
                        result["combined_text"] = f"Letter: {letter}"
                        result["confidence"] = asl_result.get("confidence", 0)
                        result["method_used"] = "asl_cnn"
        
        # If neither worked, note that
        if not result["method_used"]:
            result["combined_text"] = "No gesture detected"
            result["method_used"] = "none"
        
        return result

    async def extract_hand_landmarks(
        self,
        image_base64: str
    ) -> Dict[str, Any]:
        """
        Extract hand landmarks and detect gestures using MediaPipe Gesture Recognizer.
        Falls back to simple detection if MediaPipe is not available.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Dictionary with hand landmarks and detected gestures
        """
        # Minimum confidence for accepting gestures
        MIN_CONFIDENCE = 0.65
        
        # Try MediaPipe Gesture Recognizer first
        if self.mediapipe_available and self.gesture_recognizer:
            mp_result = await self.recognize_gesture_mediapipe(image_base64)
            
            # Get all detected gestures (already filtered for confidence in recognize_gesture_mediapipe)
            gestures_info = mp_result.get("gestures", [])
            hands_info = mp_result.get("hands", [])
            
            # Filter for high-confidence, real gestures only
            valid_gestures = [
                g for g in gestures_info 
                if g.get("gesture") != "None" and g.get("confidence", 0) >= MIN_CONFIDENCE
            ]
            
            if valid_gestures:
                # Convert to expected format
                hands_data = []
                for gesture_info in valid_gestures:
                    hands_data.append({
                        "hand_index": len(hands_data),
                        "gesture": gesture_info.get("gesture", "None"),
                        "gesture_text": gesture_info.get("text", ""),
                        "confidence": gesture_info.get("confidence", 0),
                        "handedness": "Right"
                    })
                
                return {
                    "hands_detected": len(hands_data),
                    "hands": hands_data,
                    "gestures": [h["gesture"] for h in hands_data]
                }
            
            # If hands were detected but no clear gestures
            if hands_info:
                return {
                    "hands_detected": len(hands_info),
                    "hands": [{
                        "hand_index": 0,
                        "gesture": "None",
                        "gesture_text": "Hand detected but gesture not clear enough",
                        "confidence": hands_info[0].get("confidence", 0) if hands_info else 0,
                        "handedness": hands_info[0].get("handedness", "Unknown") if hands_info else "Unknown"
                    }],
                    "gestures": []  # No valid gestures
                }
        
        # Fallback to simple skin detection
        if cv2 is None:
            return {"error": "OpenCV not available", "hands": []}
        
        image = self._decode_image(image_base64)
        if image is None:
            return {"error": "Failed to decode image", "hands": []}
        
        # Use simple skin color detection for hand detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        hands_data = []
        if contours:
            # Get the largest contour (assumed to be hand)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only process if contour is large enough
            if area > 5000:
                # Get convex hull and defects for gesture detection
                hull = cv2.convexHull(largest_contour, returnPoints=False)
                
                # Count fingers using convexity defects
                finger_count = 0
                gesture = "unknown_gesture"
                
                if len(hull) > 3 and len(largest_contour) > 3:
                    try:
                        defects = cv2.convexityDefects(largest_contour, hull)
                        if defects is not None:
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(largest_contour[s][0])
                                end = tuple(largest_contour[e][0])
                                far = tuple(largest_contour[f][0])
                                
                                # Calculate angle
                                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                                
                                if b * c != 0:
                                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
                                    
                                    # Count as finger if angle is less than 90 degrees
                                    if angle <= np.pi / 2 and d > 5000:
                                        finger_count += 1
                    except Exception:
                        pass
                
                # Map finger count to gesture
                if finger_count == 0:
                    gesture = "Closed_Fist"
                elif finger_count == 1:
                    gesture = "Pointing_Up"
                elif finger_count == 2:
                    gesture = "Victory"
                elif finger_count >= 4:
                    gesture = "Open_Palm"
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                hand_info = {
                    "hand_index": 0,
                    "handedness": "Right",
                    "confidence": 0.6,
                    "gesture": gesture,
                    "gesture_text": self.gesture_to_text.get(gesture, gesture),
                    "finger_count": finger_count + 1,
                    "bounding_box": {"x": x, "y": y, "width": w, "height": h}
                }
                
                hands_data.append(hand_info)
        
        return {
            "hands_detected": len(hands_data),
            "hands": hands_data,
            "gestures": [h["gesture"] for h in hands_data if h.get("gesture")]
        }
    
    def _get_gesture_text(self, gesture: str) -> str:
        """Map gesture name to human-readable text."""
        return self.gesture_to_text.get(gesture, gesture.replace("_", " ").title())
    
    async def extract_face_data(
        self,
        image_base64: str
    ) -> Dict[str, Any]:
        """
        Extract face data and analyze expressions using DeepFace (if available).
        Falls back to simple detection if DeepFace is not available.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Dictionary with face data including expressions
        """
        # Try DeepFace first for accurate emotion detection
        if self.deepface_available:
            deepface_result = await self.detect_emotion_deepface(image_base64)
            
            if deepface_result.get("face_detected") or deepface_result.get("dominant_emotion"):
                return {
                    "faces_detected": 1,
                    "faces": [{
                        "expression": deepface_result.get("dominant_emotion", "neutral"),
                        "emotions": deepface_result.get("emotions", {}),
                        "confidence": deepface_result.get("confidence", 0.5),
                        "gaze_direction": "looking_center"
                    }]
                }
        
        # Fallback to simple OpenCV detection
        if cv2 is None:
            return {"error": "OpenCV not available", "faces": []}
        
        image = self._decode_image(image_base64)
        if image is None:
            return {"error": "Failed to decode image", "faces": []}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try to use Haar cascade for face detection
        faces_data = []
        try:
            # Load face cascade classifier
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_roi_gray = gray[y:y+h, x:x+w]
                
                # Analyze expression based on face proportions
                expression = self._estimate_expression_simple(face_roi_gray)
                
                face_info = {
                    "bounding_box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "expression": expression,
                    "confidence": 0.6,
                    "gaze_direction": "looking_center"
                }
                
                faces_data.append(face_info)
                
        except Exception as e:
            print(f"Face detection error: {e}")
            return {"error": str(e), "faces": []}
        
        return {
            "faces_detected": len(faces_data),
            "faces": faces_data
        }
    
    def _estimate_expression_simple(self, face_gray: np.ndarray) -> str:
        """
        Estimate facial expression using improved image analysis.
        Analyzes multiple facial regions for better accuracy.
        """
        try:
            # Get face dimensions
            h, w = face_gray.shape
            
            # Analyze different facial regions
            # Forehead/eyebrow region (upper third)
            forehead_region = face_gray[0:int(h*0.35), int(w*0.2):int(w*0.8)]
            # Eye region (middle)
            eye_region = face_gray[int(h*0.2):int(h*0.45), int(w*0.1):int(w*0.9)]
            # Mouth region (lower third of face)
            mouth_region = face_gray[int(h*0.55):int(h*0.9), int(w*0.25):int(w*0.75)]
            
            # Calculate features
            mouth_std = np.std(mouth_region)
            mouth_mean = np.mean(mouth_region)
            eye_std = np.std(eye_region)
            eye_mean = np.mean(eye_region)
            forehead_std = np.std(forehead_region)
            
            # Score different expressions
            scores = {}
            
            # Happy: High contrast in mouth (teeth visible), slightly closed eyes
            scores['happy'] = 0
            if mouth_std > 35:
                scores['happy'] += 2
            if mouth_mean > 110:
                scores['happy'] += 1
            if eye_std < 35:
                scores['happy'] += 1
                
            # Sad: Low brightness in mouth region, droopy features
            scores['sad'] = 0
            if mouth_mean < 90:
                scores['sad'] += 2
            if mouth_std < 30:
                scores['sad'] += 1
            if eye_mean < 100:
                scores['sad'] += 1
                
            # Surprised: Wide eyes (high eye region std), open mouth
            scores['surprised'] = 0
            if eye_std > 40:
                scores['surprised'] += 2
            if mouth_std > 45:
                scores['surprised'] += 1
            if forehead_std > 30:
                scores['surprised'] += 1
                
            # Angry: Furrowed brow, tense features
            scores['angry'] = 0
            if forehead_std > 35:
                scores['angry'] += 2
            if eye_std > 35 and mouth_std < 35:
                scores['angry'] += 1
                
            # Neutral: Balanced features
            scores['neutral'] = 1  # Base score
            if 85 < mouth_mean < 115 and 25 < mouth_std < 40:
                scores['neutral'] += 2
            
            # Find highest scoring expression
            best_expression = max(scores, key=scores.get)
            
            # Only return non-neutral if significantly higher
            if best_expression != 'neutral' and scores[best_expression] < 2:
                return "neutral"
                
            return best_expression
            
        except Exception as e:
            print(f"Expression estimation error: {e}")
            return "neutral"
    
    async def analyze_sign_language_frame(
        self,
        image_base64: str
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single frame for sign language detection.
        Uses both MediaPipe gestures and ASL CNN model for maximum coverage.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Complete analysis including hands, face, gestures, and ASL letters
        """
        # Handle potential data URL prefix
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',', 1)[1]
            print("Stripped data URL prefix from image")
        
        print(f"Analyzing frame, base64 length: {len(image_base64)} chars")
        
        # Run gesture recognition (both MediaPipe and ASL CNN)
        combined_gesture = await self.recognize_combined(image_base64)
        
        # Also get detailed hand landmarks
        hand_data = await self.extract_hand_landmarks(image_base64)
        face_data = await self.extract_face_data(image_base64)

        error = None
        if hand_data.get("error"):
            error = hand_data.get("error")
        elif face_data.get("error"):
            error = face_data.get("error")
        
        # Build gestures list from combined recognition
        gestures = []
        if combined_gesture.get("mediapipe_gesture"):
            mp_g = combined_gesture["mediapipe_gesture"]
            gestures.append({
                "gesture": mp_g.get("gesture", "Unknown"),
                "text": mp_g.get("text", ""),
                "confidence": mp_g.get("confidence", 0),
                "source": "mediapipe"
            })
        
        if combined_gesture.get("asl_letter") and combined_gesture["asl_letter"].get("letter"):
            asl = combined_gesture["asl_letter"]
            letter = asl.get("letter", "")
            if letter not in ["nothing"]:
                gestures.append({
                    "gesture": f"ASL_{letter}",
                    "text": f"Letter {letter}" if letter != "space" else "[space]",
                    "confidence": asl.get("confidence", 0),
                    "source": "asl_cnn"
                })
        
        # Combine into comprehensive analysis
        return {
            "hands": {
                **hand_data,
                "hands_detected": len(gestures) > 0 or hand_data.get("hands_detected", 0) > 0
            },
            "face": face_data,
            "combined_gestures": gestures,
            "asl_letter": combined_gesture.get("asl_letter"),
            "primary_text": combined_gesture.get("combined_text", "No gesture detected"),
            "recognition_method": combined_gesture.get("method_used", "none"),
            "expression": face_data.get("faces", [{}])[0].get("expression", "unknown") if face_data.get("faces") else "unknown",
            "error": error,
        }


# Singleton instance
_vision_service: Optional[VisionService] = None


def get_vision_service() -> VisionService:
    """Get the vision service singleton."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service
