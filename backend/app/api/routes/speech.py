"""
Speech API Routes
Endpoints for speech-to-text and text-to-speech operations.
Uses Groq Whisper API for fast speech recognition.
"""

import base64
import random
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional

from app.models.schemas import (
    SpeechToTextRequest,
    SpeechToTextResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
)
from app.services.groq_service import get_groq_service
from app.services.local_whisper_service import get_local_whisper_service
from app.services import get_storage_service

# Demo responses when Azure Speech is unavailable
DEMO_TRANSCRIPTIONS = [
    "Hello, how are you today?",
    "I need some help please.",
    "Thank you very much!",
    "I am feeling happy.",
    "Can I have some water?",
    "I want to go home.",
    "Yes, I agree with that.",
    "No, I do not want that.",
    "Good morning!",
    "Please help me.",
]

router = APIRouter(prefix="/azure", tags=["Speech"])


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(request: SpeechToTextRequest):
    """
    Convert speech audio to text using Groq Whisper API (fast!).
    Falls back to local Whisper if Groq is unavailable.
    
    Accepts either:
    - audio_base64: Base64 encoded audio data
    - audio_url: URL to an audio file (must be accessible)
    
    Returns recognized text with confidence score.
    """
    # Try Groq first (faster), fall back to local Whisper
    try:
        groq_service = get_groq_service()
        if groq_service.api_key:
            speech_service = groq_service
            service_name = "Groq Whisper"
        else:
            speech_service = get_local_whisper_service()
            service_name = "Local Whisper"
    except Exception as e:
        speech_service = get_local_whisper_service()
        service_name = "Local Whisper"
    
    try:
        if not request.audio_base64 and not request.audio_url:
            raise HTTPException(
                status_code=400,
                detail="Either audio_base64 or audio_url must be provided"
            )
        
        if request.audio_url:
            raise HTTPException(
                status_code=501,
                detail="audio_url not yet implemented. Please use audio_base64."
            )
        
        # Convert language code (e.g., en-US -> en)
        lang = request.language[:2] if len(request.language) > 2 else request.language
        
        print(f"Using {service_name} for speech-to-text")
        text, confidence = await speech_service.speech_to_text(
            audio_base64=request.audio_base64,
            language=lang
        )
        
        if not text:
            return SpeechToTextResponse(
                text="",
                confidence=0.0,
                language=request.language,
                duration_ms=None
            )
        
        return SpeechToTextResponse(
            text=text,
            confidence=confidence,
            language=request.language,
            duration_ms=None
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Speech recognition failed: {str(e)}"
        )


@router.post("/speech-to-text/upload", response_model=SpeechToTextResponse)
async def speech_to_text_upload(
    file: UploadFile = File(...),
    language: str = Form(default="en-US"),
    user_id: Optional[str] = Form(default=None)
):
    """
    Convert uploaded audio file to text using local Whisper (FREE).
    
    Accepts audio file upload directly (WAV, MP3, M4A, etc.).
    """
    speech_service = get_local_whisper_service()
    
    try:
        # Read uploaded file
        audio_data = await file.read()
        
        # Convert language code
        lang = language[:2] if len(language) > 2 else language
        
        text, confidence = await speech_service.speech_to_text(
            audio_data=audio_data,
            language=lang
        )
        
        return SpeechToTextResponse(
            text=text,
            confidence=confidence,
            language=language,
            duration_ms=None
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text-to-speech", response_model=TextToSpeechResponse)
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech using gTTS (FREE).
    
    Returns URL to generated audio file and optionally base64 encoded audio.
    """
    speech_service = get_local_whisper_service()
    storage_service = get_storage_service()
    
    try:
        # Generate speech audio
        audio_bytes, duration_ms = await speech_service.text_to_speech(
            text=request.text,
            voice=request.voice,
            speed=request.rate  # Map rate to speed
        )
        
        # Upload to blob storage
        audio_url = await storage_service.upload_audio(
            audio_data=audio_bytes,
            file_format="mp3"
        )
        
        # Optionally include base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        return TextToSpeechResponse(
            audio_url=audio_url,
            audio_base64=audio_base64,
            duration_ms=duration_ms,
            format="mp3"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
