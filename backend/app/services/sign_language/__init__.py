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

# Check if PyTorch is available for video-based models
PYTORCH_AVAILABLE = False
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Video-based sign language recognition (I3D, Pose-LSTM) disabled.")
    print("         Install with: pip install torch torchvision")

# Conditionally import based on PyTorch availability
if PYTORCH_AVAILABLE:
    from .sign_language_service import SignLanguageRecognitionService, get_sign_language_service
    from .i3d_service import I3DInferenceService, get_i3d_service
    from .pose_lstm_service import PoseLSTMService, get_pose_lstm_service
else:
    # Stub implementations for when PyTorch is not available
    class SignLanguageRecognitionService:
        """Stub service when PyTorch is not available."""
        def __init__(self):
            self.available = False
            print("SignLanguageRecognitionService: PyTorch not available")
        
        def get_status(self):
            return {
                "service": "SignLanguageRecognitionService",
                "status": "pytorch_not_installed",
                "models": {},
                "config": {"error": "Install PyTorch: pip install torch torchvision"},
                "vocabulary_size": 0
            }
        
        async def recognize_from_video(self, *args, **kwargs):
            return {
                "recognized_words": [],
                "sentence": "Video recognition requires PyTorch. Install with: pip install torch torchvision",
                "confidence": 0.0,
                "error": "pytorch_not_installed"
            }
        
        async def recognize_from_base64_video(self, *args, **kwargs):
            return await self.recognize_from_video()
        
        async def recognize_from_base64_frame(self, *args, **kwargs):
            return {
                "recognized_words": [],
                "sentence": "",
                "confidence": 0.0,
                "status": "error",
                "error": "pytorch_not_installed"
            }
        
        def set_groq_service(self, groq_service):
            pass
    
    class I3DInferenceService:
        """Stub I3D service."""
        def __init__(self):
            self.available = False
        
        async def recognize_from_frames(self, *args, **kwargs):
            return {"error": "pytorch_not_installed", "recognized_words": [], "confidence": 0.0}
    
    class PoseLSTMService:
        """Stub Pose-LSTM service."""
        def __init__(self):
            self.available = False
        
        async def recognize_from_frames(self, *args, **kwargs):
            return {"error": "pytorch_not_installed", "recognized_words": [], "confidence": 0.0}
    
    _stub_service = None
    
    def get_sign_language_service():
        global _stub_service
        if _stub_service is None:
            _stub_service = SignLanguageRecognitionService()
        return _stub_service
    
    def get_i3d_service():
        return I3DInferenceService()
    
    def get_pose_lstm_service():
        return PoseLSTMService()

__all__ = [
    "SignLanguageRecognitionService",
    "get_sign_language_service",
    "I3DInferenceService", 
    "get_i3d_service",
    "PoseLSTMService",
    "get_pose_lstm_service",
    "PYTORCH_AVAILABLE",
]

