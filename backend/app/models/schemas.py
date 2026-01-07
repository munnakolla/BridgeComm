"""
Pydantic models for API request/response schemas.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class InputMode(str, Enum):
    """Input mode for disabled users."""
    SIGN = "sign"
    EYE = "eye"
    TOUCH = "touch"
    BEHAVIOR = "behavior"


class FeedbackType(str, Enum):
    """Type of feedback provided by user."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


# =============================================================================
# SYMBOL MODELS
# =============================================================================

class Symbol(BaseModel):
    """Represents a visual symbol/icon."""
    id: str = Field(..., description="Unique symbol identifier")
    name: str = Field(..., description="Symbol name")
    url: str = Field(..., description="URL to symbol image")
    category: Optional[str] = Field(None, description="Symbol category")


# =============================================================================
# SPEECH TO TEXT
# =============================================================================

class SpeechToTextRequest(BaseModel):
    """Request for speech-to-text conversion."""
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio data")
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    language: str = Field(default="en-US", description="Language code")
    user_id: Optional[str] = Field(None, description="User ID for personalization")


class SpeechToTextResponse(BaseModel):
    """Response from speech-to-text conversion."""
    text: str = Field(..., description="Recognized text")
    confidence: float = Field(..., description="Recognition confidence (0-1)")
    language: str = Field(..., description="Detected language")
    duration_ms: Optional[int] = Field(None, description="Audio duration in milliseconds")


# =============================================================================
# TEXT TO SYMBOLS
# =============================================================================

class TextToSymbolsRequest(BaseModel):
    """Request for text to symbols conversion."""
    text: str = Field(..., description="Input text to convert")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    simplify: bool = Field(default=True, description="Whether to simplify text first")
    max_symbols: int = Field(default=10, description="Maximum number of symbols")


class TextToSymbolsResponse(BaseModel):
    """Response from text to symbols conversion."""
    original_text: str = Field(..., description="Original input text")
    simplified_text: str = Field(..., description="Simplified text")
    symbols: List[Symbol] = Field(..., description="Mapped visual symbols")
    keywords: List[str] = Field(..., description="Extracted keywords")
    confidence: float = Field(..., description="Overall confidence score")


# =============================================================================
# SIGN LANGUAGE TO INTENT
# =============================================================================

class SignToIntentRequest(BaseModel):
    """Request for sign language to intent recognition."""
    image_base64: Optional[str] = Field(None, description="Base64 encoded image")
    image_url: Optional[str] = Field(None, description="URL to image")
    video_base64: Optional[str] = Field(None, description="Base64 encoded video")
    video_url: Optional[str] = Field(None, description="URL to video")
    user_id: Optional[str] = Field(None, description="User ID for personalization")


class SignToIntentResponse(BaseModel):
    """Response from sign language recognition."""
    intent: str = Field(..., description="Detected intent label")
    text: str = Field(..., description="Natural language interpretation")
    confidence: float = Field(..., description="Recognition confidence")
    gestures_detected: List[str] = Field(default=[], description="Individual gestures detected")
    hand_landmarks: Optional[Dict[str, Any]] = Field(None, description="Hand landmark data")


# =============================================================================
# BEHAVIOR TO INTENT
# =============================================================================

class BehaviorData(BaseModel):
    """Behavioral input data from frontend."""
    touch_patterns: Optional[List[Dict[str, Any]]] = Field(None, description="Touch interaction patterns")
    eye_tracking: Optional[Dict[str, Any]] = Field(None, description="Eye tracking data")
    facial_expressions: Optional[Dict[str, Any]] = Field(None, description="Facial expression data")
    motion_data: Optional[List[Dict[str, Any]]] = Field(None, description="Device motion data")
    interaction_sequence: Optional[List[str]] = Field(None, description="Sequence of UI interactions")


class BehaviorToIntentRequest(BaseModel):
    """Request for behavior to intent recognition."""
    behavior_data: BehaviorData = Field(..., description="Behavioral input data")
    context: Optional[str] = Field(None, description="Current app context")
    user_id: Optional[str] = Field(None, description="User ID for personalization")


class BehaviorToIntentResponse(BaseModel):
    """Response from behavior recognition."""
    intent: str = Field(..., description="Detected intent label")
    text: str = Field(..., description="Natural language interpretation")
    confidence: float = Field(..., description="Recognition confidence")
    behavior_type: str = Field(..., description="Type of behavior detected")
    features_extracted: Dict[str, Any] = Field(default={}, description="Extracted behavioral features")


# =============================================================================
# GENERATE TEXT (NLG)
# =============================================================================

class GenerateTextRequest(BaseModel):
    """Request for natural language generation."""
    intent: str = Field(..., description="Intent to convert to text")
    context: Optional[str] = Field(None, description="Conversation context")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    style: str = Field(default="natural", description="Text style: natural, formal, simple")


class GenerateTextResponse(BaseModel):
    """Response from text generation."""
    text: str = Field(..., description="Generated natural language text")
    intent: str = Field(..., description="Original intent")
    confidence: float = Field(..., description="Generation confidence")
    alternatives: Optional[List[str]] = Field(None, description="Alternative phrasings")


# =============================================================================
# FEEDBACK
# =============================================================================

class FeedbackRequest(BaseModel):
    """Request for submitting feedback for model improvement."""
    user_id: str = Field(..., description="User ID")
    session_id: str = Field(..., description="Session ID")
    interaction_id: str = Field(..., description="Interaction ID")
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    original_output: str = Field(..., description="Original system output")
    corrected_output: Optional[str] = Field(None, description="User-corrected output")
    input_mode: InputMode = Field(..., description="Input mode used")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class FeedbackResponse(BaseModel):
    """Response from feedback submission."""
    success: bool = Field(..., description="Whether feedback was recorded")
    feedback_id: str = Field(..., description="Unique feedback ID")
    message: str = Field(..., description="Status message")
    personalization_updated: bool = Field(default=False, description="Whether user model was updated")


# =============================================================================
# USER PROFILE
# =============================================================================

class UserProfile(BaseModel):
    """User profile with personalization settings."""
    user_id: str = Field(..., description="Unique user ID")
    display_name: Optional[str] = Field(None, description="Display name")
    communication_mode: InputMode = Field(default=InputMode.SIGN)
    preferences: Dict[str, Any] = Field(default={})
    personalization_data: Dict[str, Any] = Field(default={})
    created_at: Optional[str] = Field(None)
    updated_at: Optional[str] = Field(None)


class UserProfileUpdate(BaseModel):
    """Update user profile request."""
    display_name: Optional[str] = None
    communication_mode: Optional[InputMode] = None
    preferences: Optional[Dict[str, Any]] = None


# =============================================================================
# TEXT TO SPEECH
# =============================================================================

class TextToSpeechRequest(BaseModel):
    """Request for text to speech conversion."""
    text: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="en-US-JennyNeural", description="Voice name")
    rate: float = Field(default=1.0, description="Speech rate (0.5-2.0)")
    pitch: float = Field(default=1.0, description="Speech pitch (0.5-2.0)")


class TextToSpeechResponse(BaseModel):
    """Response from text to speech conversion."""
    audio_url: str = Field(..., description="URL to generated audio file")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio")
    duration_ms: int = Field(..., description="Audio duration in milliseconds")
    format: str = Field(default="mp3", description="Audio format")


# =============================================================================
# HEALTH CHECK
# =============================================================================

class HealthResponse(BaseModel):
    """API health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(default={}, description="Status of dependent services")
