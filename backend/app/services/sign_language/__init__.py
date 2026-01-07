"""
Sign Language Recognition Services
==================================
Integrates pretrained video-based sign language models for inference.

Models:
- SignVLM (Primary): Vision-Language Transformer for video-to-gloss
- I3D (Fallback): Inflated 3D ConvNet for video classification
- Pose LSTM (Low-latency): MediaPipe landmarks → LSTM classifier

Usage:
    from app.services.sign_language import get_sign_language_service
    
    service = get_sign_language_service()
    result = await service.recognize_from_video(video_base64)
"""

from .sign_language_service import SignLanguageRecognitionService, get_sign_language_service
from .i3d_service import I3DInferenceService, get_i3d_service
from .pose_lstm_service import PoseLSTMService, get_pose_lstm_service

__all__ = [
    "SignLanguageRecognitionService",
    "get_sign_language_service",
    "I3DInferenceService", 
    "get_i3d_service",
    "PoseLSTMService",
    "get_pose_lstm_service",
]
