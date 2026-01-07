"""Services module initialization."""

from .speech_service import SpeechService, get_speech_service
from .openai_service import OpenAIService, get_openai_service
from .symbol_service import SymbolService, get_symbol_service
from .vision_service import VisionService, get_vision_service
from .user_service import UserService, get_user_service
from .storage_service import StorageService, get_storage_service

# Try to import custom emotion service (available after training)
try:
    from .custom_emotion_service import CustomEmotionService, get_custom_emotion_service
except ImportError:
    CustomEmotionService = None
    get_custom_emotion_service = None

__all__ = [
    "SpeechService",
    "get_speech_service",
    "OpenAIService",
    "get_openai_service",
    "SymbolService",
    "get_symbol_service",
    "VisionService",
    "get_vision_service",
    "UserService",
    "get_user_service",
    "StorageService",
    "get_storage_service",
    "CustomEmotionService",
    "get_custom_emotion_service",
]
