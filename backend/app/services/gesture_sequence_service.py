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
        
        # Gesture to word mappings - expanded vocabulary
        self.gesture_words = {
            "Thumb_Up": "yes",
            "Thumb_Down": "no",
            "Open_Palm": "stop",
            "Closed_Fist": "agree",
            "Victory": "peace",
            "Pointing_Up": "attention",
            "ILoveYou": "love you",
            # Common sign language phrases
            "wave": "hello",
            "wave_goodbye": "goodbye",
            "prayer_hands": "thank you",
            "ok_sign": "okay",
            "raise_hand": "help",
            # Additional gestures for better communication
            "clap": "good job",
            "fist_bump": "agreement",
            "crossed_arms": "no way",
            "shrug": "don't know",
            "point_left": "that way",
            "point_right": "this way",
            "point_down": "below",
            "phone_gesture": "call me",
            "time_check": "what time",
            "money_gesture": "money",
            "eating_gesture": "hungry",
            "drinking_gesture": "thirsty",
            "sleeping_gesture": "tired",
        }
        
        # Gesture sequences that form phrases - expanded patterns
        self.gesture_phrases = {
            # Greetings
            ("wave", "Open_Palm"): "Hello, how are you?",
            ("wave", "Thumb_Up"): "Hello, I'm good",
            ("wave", "ILoveYou"): "Hello, nice to see you",
            # Requests and needs
            ("Pointing_Up", "Open_Palm"): "Look at this",
            ("Open_Palm", "Closed_Fist"): "Please help me",
            ("raise_hand", "Open_Palm"): "I need help please",
            ("eating_gesture", "Thumb_Up"): "I'm hungry",
            ("drinking_gesture", "Thumb_Up"): "I'm thirsty",
            ("sleeping_gesture", "Closed_Fist"): "I'm very tired",
            # Agreements and disagreements
            ("Thumb_Up", "Thumb_Up"): "Very good",
            ("Thumb_Up", "Closed_Fist"): "I completely agree",
            ("Thumb_Down", "Closed_Fist"): "I disagree strongly",
            ("Thumb_Down", "Open_Palm"): "No, please stop",
            ("crossed_arms", "Thumb_Down"): "Absolutely not",
            # Farewells
            ("ILoveYou", "wave"): "I love you, goodbye",
            ("prayer_hands", "wave"): "Thank you, goodbye",
            ("Thumb_Up", "wave"): "Okay, goodbye",
            ("wave_goodbye", "ILoveYou"): "Goodbye, love you",
            # Questions and uncertainty
            ("shrug", "Open_Palm"): "I don't know, sorry",
            ("Pointing_Up", "time_check"): "What time is it?",
            ("point_left", "Pointing_Up"): "Is it over there?",
            # Complex interactions
            ("phone_gesture", "Thumb_Up"): "Call me later",
            ("money_gesture", "Thumb_Down"): "Too expensive",
            ("Victory", "Thumb_Up"): "Peace and love",
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
