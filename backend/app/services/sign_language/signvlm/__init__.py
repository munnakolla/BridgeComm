"""
SignVLM - Vision Language Model for Sign Language Recognition
=============================================================

A Pre-trained Large Vision Model for Sign Language Recognition based on:
"SignVLM: A Pre-trained Large Vision Model for Sign Language Recognition"
by Hamzah Luqman

Repository: https://github.com/Hamzah-Luqman/signVLM

This module provides:
- EVLTransformer: Main model architecture using CLIP backbone + temporal decoder
- Video preprocessing for sign language recognition
- Inference service for real-time sign detection

Supported datasets:
- WLASL (Word-Level American Sign Language)
- KArSL (King Abdullah Arabic Sign Language)
- AUTSL (Ankara University Turkish Sign Language)
- LSA64 (Argentinian Sign Language)
"""

from .signvlm_service import SignVLMService, get_signvlm_service

__all__ = ["SignVLMService", "get_signvlm_service"]
