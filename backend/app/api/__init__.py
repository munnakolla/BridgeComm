"""API module initialization."""

from .routes import (
    speech_router,
    symbols_router,
    sign_language_router,
    sign_video_router,
    behavior_router,
    text_generation_router,
    feedback_router,
    users_router,
    translator_router,
)

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
