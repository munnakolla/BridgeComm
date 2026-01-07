"""API Routes initialization."""

from .speech import router as speech_router
from .symbols import router as symbols_router
from .sign_language import router as sign_language_router
from .sign_video import router as sign_video_router
from .behavior import router as behavior_router
from .text_generation import router as text_generation_router
from .feedback import router as feedback_router
from .users import router as users_router
from .translator import router as translator_router

__all__ = [
    "speech_router",
    "symbols_router",
    "sign_language_router",
    "sign_video_router",
    "behavior_router",
    "text_generation_router",
    "feedback_router",
    "users_router",
    "translator_router",
]
