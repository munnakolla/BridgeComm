"""
Continuous Gesture Detection Service
Handles video frame sequence analysis and sentence formation from gestures.
"""

from typing import List, Dict, Any, Optional
import base64
from collections import deque
from datetime import datetime, timedelta


class GestureSequenceAnalyzer:
    """Analyzes sequences of gestures and forms sentences."""
    
    def __init__(self, window_seconds: float = 2.5, min_confidence: float = 0.65):
        """
        Initialize gesture sequence analyzer.
        
        Args:
            window_seconds: Time window for gesture sequence (seconds)
            min_confidence: Minimum confidence threshold for gestures (65% default)
        """
        self.window_seconds = window_seconds
        self.min_confidence = min_confidence
        self.gesture_history: deque = deque(maxlen=30)  # Last 30 gestures for better context
        
        # Gesture to single word mappings - CLEAR AND CONSISTENT
        self.gesture_words = {
            # MediaPipe gestures - single clear words
            "Thumb_Up": "good",
            "Thumb_Down": "no",
            "Open_Palm": "hello",
            "Closed_Fist": "yes",
            "Victory": "peace",
            "Pointing_Up": "look",
            "ILoveYou": "love you",
            # Additional gestures
            "wave": "hello",
            "wave_goodbye": "goodbye",
            "ok_sign": "okay",
            "raise_hand": "help",
        }
        
        # Gesture sequences that form natural phrases
        self.gesture_phrases = {
            # Greetings
            ("Open_Palm", "Thumb_Up"): "Hello, good to see you!",
            ("Open_Palm", "ILoveYou"): "Hello, I love you!",
            # Requests
            ("Pointing_Up", "Open_Palm"): "Look at this, please.",
            ("Open_Palm", "Closed_Fist"): "Please help me.",
            # Agreements
            ("Thumb_Up", "Thumb_Up"): "Very good!",
            ("Thumb_Up", "Closed_Fist"): "I agree!",
            ("Thumb_Down", "Closed_Fist"): "I disagree.",
            ("Thumb_Down", "Open_Palm"): "No, please stop.",
            # Farewells
            ("ILoveYou", "Open_Palm"): "I love you, goodbye!",
            ("Thumb_Up", "Open_Palm"): "Okay, goodbye!",
            # Combined expressions
            ("Victory", "Thumb_Up"): "Peace and love!",
            ("Victory", "ILoveYou"): "Peace and love!",
        }
    
    def add_gesture(self, gesture: str, confidence: float, timestamp: Optional[datetime] = None) -> None:
        """
        Add a detected gesture to the history.
        
        Args:
            gesture: Gesture name
            confidence: Detection confidence
            timestamp: When gesture was detected
        """
        # Filter out low-confidence gestures and "None" gestures
        if gesture == "None":
            print(f"GestureSequence: Skipping 'None' gesture (hand detected but no clear gesture)")
            return
            
        if confidence < self.min_confidence:
            print(f"GestureSequence: Skipping gesture '{gesture}' - confidence {confidence:.2f} below threshold {self.min_confidence}")
            return
        
        print(f"GestureSequence: ACCEPTING gesture '{gesture}' with confidence {confidence:.2f}")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Avoid duplicate consecutive gestures
        if self.gesture_history and self.gesture_history[-1]['gesture'] == gesture:
            # Update timestamp of last gesture
            self.gesture_history[-1]['timestamp'] = timestamp
            self.gesture_history[-1]['confidence'] = max(
                self.gesture_history[-1]['confidence'], 
                confidence
            )
        else:
            self.gesture_history.append({
                'gesture': gesture,
                'confidence': confidence,
                'timestamp': timestamp
            })
    
    def get_recent_gestures(self) -> List[Dict[str, Any]]:
        """Get gestures within the time window."""
        if not self.gesture_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=self.window_seconds)
        recent = [
            g for g in self.gesture_history 
            if g['timestamp'] >= cutoff_time
        ]
        
        return recent
    
    def form_sentence(self) -> str:
        """
        Form a sentence from recent gesture sequence.
        
        Returns:
            Natural language sentence
        """
        recent_gestures = self.get_recent_gestures()
        
        if not recent_gestures:
            return ""
        
        # Extract gesture names
        gesture_names = [g['gesture'] for g in recent_gestures]
        
        # Check for matching phrase patterns
        for i in range(len(gesture_names) - 1):
            pair = (gesture_names[i], gesture_names[i + 1])
            if pair in self.gesture_phrases:
                return self.gesture_phrases[pair]
        
        # Check for triple patterns
        if len(gesture_names) >= 3:
            triple = (gesture_names[-3], gesture_names[-2], gesture_names[-1])
            # Could add triple patterns here
        
        # Fall back to word-by-word translation
        words = []
        for gesture in gesture_names:
            word = self.gesture_words.get(gesture, gesture.lower().replace("_", " "))
            if word and word not in words[-1:]:  # Avoid consecutive duplicates
                words.append(word)
        
        if not words:
            return ""
        
        # Capitalize first letter and add period
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:] + "."
        
        return sentence
    
    def get_gesture_summary(self) -> Dict[str, Any]:
        """
        Get summary of gesture sequence.
        
        Returns:
            Dict with gestures, confidence, and formed sentence
        """
        recent = self.get_recent_gestures()
        
        if not recent:
            return {
                "gestures": [],
                "sentence": "",
                "confidence": 0.0,
                "gesture_count": 0
            }
        
        avg_confidence = sum(g['confidence'] for g in recent) / len(recent)
        gesture_names = [g['gesture'] for g in recent]
        
        return {
            "gestures": gesture_names,
            "sentence": self.form_sentence(),
            "confidence": avg_confidence,
            "gesture_count": len(recent)
        }
    
    def clear(self) -> None:
        """Clear gesture history."""
        self.gesture_history.clear()


# Global session storage for continuous detection
_gesture_sessions: Dict[str, GestureSequenceAnalyzer] = {}


def get_gesture_session(session_id: str) -> GestureSequenceAnalyzer:
    """
    Get or create a gesture session.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        GestureSequenceAnalyzer instance
    """
    if session_id not in _gesture_sessions:
        _gesture_sessions[session_id] = GestureSequenceAnalyzer()
    return _gesture_sessions[session_id]


def clear_gesture_session(session_id: str) -> None:
    """Clear a gesture session."""
    if session_id in _gesture_sessions:
        del _gesture_sessions[session_id]
