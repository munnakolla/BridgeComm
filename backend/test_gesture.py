"""
Test script to verify MediaPipe Gesture Recognizer is working.
Run from backend directory: python test_gesture.py
"""
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

import base64
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "gesture_recognizer.task")

print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

# Initialize gesture recognizer
base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.GestureRecognizerOptions(
    base_options=base_options,
    num_hands=2
)
gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(options)
print("Gesture recognizer created!")

# Create a simple test image - a white background (no hands expected)
test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=test_image)

result = gesture_recognizer.recognize(mp_image)
print(f"Result on blank image - gestures: {len(result.gestures) if result.gestures else 0}")

# Try with a different size image
test_image2 = np.ones((1080, 1920, 3), dtype=np.uint8) * 200
mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=test_image2)

result2 = gesture_recognizer.recognize(mp_image2)
print(f"Result on larger blank image - gestures: {len(result2.gestures) if result2.gestures else 0}")

print("\n=== MediaPipe Gesture Recognizer is working! ===")
print("Supported gestures: Thumb_Up, Thumb_Down, Open_Palm, Closed_Fist, Victory, Pointing_Up, ILoveYou, None")
print("\nNote: 'None' means a hand is detected but no clear gesture.")
print("If you're seeing 'no gestures detected', make sure your hand is:")
print("  1. Clearly visible in the frame")
print("  2. Well-lit (not in shadow)")
print("  3. Close enough to the camera")
print("  4. Showing a clear gesture (thumbs up, open palm, etc.)")
