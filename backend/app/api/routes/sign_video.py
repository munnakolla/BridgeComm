"""
Video Sign Language Recognition API Routes
===========================================
Enhanced endpoints for video-based sign language recognition
using pretrained deep learning models (I3D, Pose-LSTM).

This module provides:
- Video-to-sentence sign language translation
- Frame streaming recognition
- Model status and health checks

These routes complement the existing sign_language.py routes
which use MediaPipe gesture recognition.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import base64
import tempfile
import os

# Import the new sign language recognition service
from app.services.sign_language import get_sign_language_service
from app.services.groq_service import GroqService


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class VideoRecognitionRequest(BaseModel):
    """Request for video-based sign language recognition."""
    video_base64: str = Field(..., description="Base64 encoded video data")
    use_groq_correction: bool = Field(
        default=True,
        description="Use Groq LLM for sentence correction"
    )
    user_id: Optional[str] = Field(None, description="User ID for personalization")


class VideoRecognitionResponse(BaseModel):
    """Response from video-based sign language recognition."""
    recognized_words: List[str] = Field(
        default=[],
        description="List of recognized sign language words"
    )
    sentence: str = Field(
        ...,
        description="Natural language sentence (Groq-corrected if enabled)"
    )
    confidence: float = Field(
        ...,
        description="Average recognition confidence (0-1)"
    )
    frames_processed: Optional[int] = Field(
        None,
        description="Number of video frames processed"
    )
    windows_processed: Optional[int] = Field(
        None,
        description="Number of sliding windows processed"
    )
    models_used: Optional[List[str]] = Field(
        None,
        description="Models used for recognition (I3D, Pose-LSTM)"
    )
    raw_sentence: Optional[str] = Field(
        None,
        description="Raw sentence before Groq correction"
    )
    error: Optional[str] = Field(None, description="Error message if any")


class StreamFrameRequest(BaseModel):
    """Request for streaming frame recognition."""
    frame_base64: str = Field(..., description="Base64 encoded image frame")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")


class StreamFrameResponse(BaseModel):
    """Response from streaming frame recognition."""
    recognized_words: List[str] = Field(default=[])
    sentence: str = Field(default="")
    confidence: float = Field(default=0.0)
    status: str = Field(
        ...,
        description="Status: buffering, recognized, no_sign_detected, error"
    )
    frames_buffered: Optional[int] = Field(None)
    frames_needed: Optional[int] = Field(None)
    model: Optional[str] = Field(None)


class ModelStatusResponse(BaseModel):
    """Response with model status information."""
    service: str = Field(default="SignLanguageRecognitionService")
    status: str = Field(..., description="ready, models_not_loaded, error")
    models: dict = Field(..., description="Status of each model")
    config: dict = Field(..., description="Service configuration")
    vocabulary_size: int = Field(..., description="Number of words in vocabulary")


# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter(prefix="/sign-video", tags=["Sign Language Video Recognition"])


# Singleton Groq service for sentence correction
_groq_service: Optional[GroqService] = None

def get_groq_service() -> Optional[GroqService]:
    """Get or create Groq service singleton."""
    global _groq_service
    if _groq_service is None:
        try:
            from app.core.config import get_settings
            settings = get_settings()
            if settings.groq_api_key:
                _groq_service = GroqService()
        except Exception as e:
            print(f"Warning: Could not initialize Groq service: {e}")
    return _groq_service


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """
    Get the status of sign language recognition models.
    
    Returns information about:
    - Model availability and loading status
    - Service configuration
    - Vocabulary size
    
    Use this to check if models are ready before sending recognition requests.
    """
    try:
        service = get_sign_language_service()
        status = service.get_status()
        return ModelStatusResponse(**status)
        
    except Exception as e:
        return ModelStatusResponse(
            service="SignLanguageRecognitionService",
            status="error",
            models={},
            config={},
            vocabulary_size=0
        )


@router.post("/recognize", response_model=VideoRecognitionResponse)
async def recognize_video(request: VideoRecognitionRequest):
    """
    Recognize sign language from a video.
    
    This endpoint:
    1. Extracts frames from the video at 15 FPS
    2. Processes frames through I3D video model
    3. Falls back to Pose-LSTM if confidence is low
    4. Sends recognized words to Groq LLM for sentence correction
    5. Returns the final natural language sentence
    
    **Input:**
    - video_base64: Base64 encoded video (MP4, WebM, etc.)
    - use_groq_correction: Whether to use Groq for grammar correction (default: true)
    
    **Processing Flow:**
    ```
    Video → OpenCV (15 FPS) → I3D/Pose-LSTM → Words → Groq LLM → Sentence
    ```
    
    **Output:**
    ```json
    {
      "recognized_words": ["HELLO", "HOW", "YOU"],
      "sentence": "Hello, how are you?",
      "confidence": 0.92
    }
    ```
    """
    try:
        service = get_sign_language_service()
        
        # Set Groq service for sentence correction
        if request.use_groq_correction:
            groq = get_groq_service()
            if groq:
                service.set_groq_service(groq)
        
        # Recognize from video
        result = await service.recognize_from_base64_video(
            request.video_base64,
            use_groq_correction=request.use_groq_correction
        )
        
        return VideoRecognitionResponse(
            recognized_words=result.get("recognized_words", []),
            sentence=result.get("sentence", ""),
            confidence=result.get("confidence", 0.0),
            frames_processed=result.get("frames_processed"),
            windows_processed=result.get("windows_processed"),
            models_used=result.get("models_used"),
            raw_sentence=result.get("raw_sentence"),
            error=result.get("error")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize/upload", response_model=VideoRecognitionResponse)
async def recognize_video_upload(
    file: UploadFile = File(..., description="Video file (MP4, WebM, etc.)"),
    use_groq_correction: bool = Form(default=True),
    user_id: Optional[str] = Form(default=None)
):
    """
    Recognize sign language from an uploaded video file.
    
    Same as /recognize but accepts file upload instead of base64.
    
    Supported formats: MP4, WebM, AVI, MOV
    Recommended: 2-5 seconds of video, 720p or lower for faster processing
    """
    try:
        # Read video file
        video_data = await file.read()
        
        if len(video_data) < 1000:
            raise HTTPException(status_code=400, detail="Video file is too small")
        
        service = get_sign_language_service()
        
        # Set Groq service
        if use_groq_correction:
            groq = get_groq_service()
            if groq:
                service.set_groq_service(groq)
        
        # Recognize
        result = await service.recognize_from_video(
            video_data,
            use_groq_correction=use_groq_correction
        )
        
        return VideoRecognitionResponse(
            recognized_words=result.get("recognized_words", []),
            sentence=result.get("sentence", ""),
            confidence=result.get("confidence", 0.0),
            frames_processed=result.get("frames_processed"),
            windows_processed=result.get("windows_processed"),
            models_used=result.get("models_used"),
            raw_sentence=result.get("raw_sentence"),
            error=result.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream/frame", response_model=StreamFrameResponse)
async def process_stream_frame(request: StreamFrameRequest):
    """
    Process a single frame for real-time streaming recognition.
    
    This endpoint is designed for real-time camera streaming:
    1. Add each frame to an internal buffer
    2. When enough frames are buffered (1+ seconds), attempt recognition
    3. Return recognized words or buffering status
    
    Call this endpoint at ~15 FPS for best results.
    
    **Statuses:**
    - `buffering`: Not enough frames yet, keep sending
    - `recognized`: Sign detected, see recognized_words
    - `no_sign_detected`: No clear sign in current window
    - `error`: Processing error
    """
    try:
        service = get_sign_language_service()
        
        result = await service.recognize_from_base64_frame(request.frame_base64)
        
        return StreamFrameResponse(
            recognized_words=result.get("recognized_words", []),
            sentence=result.get("sentence", ""),
            confidence=result.get("confidence", 0.0),
            status=result.get("status", "error"),
            frames_buffered=result.get("frames_buffered"),
            frames_needed=result.get("frames_needed"),
            model=result.get("model")
        )
        
    except Exception as e:
        return StreamFrameResponse(
            recognized_words=[],
            sentence="",
            confidence=0.0,
            status="error",
            model=str(e)
        )


@router.post("/stream/clear")
async def clear_stream_buffer(session_id: Optional[str] = Form(default=None)):
    """
    Clear the frame buffer for streaming recognition.
    
    Call this when:
    - Starting a new sign language sentence
    - User wants to reset the recognition
    - Switching contexts
    """
    try:
        service = get_sign_language_service()
        service.clear_buffers()
        
        return {
            "success": True,
            "message": "Buffers cleared successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize/words-to-sentence")
async def words_to_sentence(
    words: List[str] = Form(..., description="List of recognized sign words"),
    use_groq: bool = Form(default=True)
):
    """
    Convert a list of sign language words/glosses to a natural sentence.
    
    This endpoint only does the Groq sentence correction step,
    useful if you have your own recognition pipeline.
    
    Example:
    - Input: ["HELLO", "HOW", "YOU"]
    - Output: "Hello, how are you?"
    """
    if not words:
        return {
            "sentence": "",
            "original_words": words,
            "corrected": False
        }
    
    # Simple sentence without Groq
    raw_sentence = " ".join(words)
    
    if not use_groq:
        return {
            "sentence": raw_sentence,
            "original_words": words,
            "corrected": False
        }
    
    # Use Groq for correction
    groq = get_groq_service()
    if not groq:
        return {
            "sentence": raw_sentence,
            "original_words": words,
            "corrected": False,
            "note": "Groq service not available"
        }
    
    try:
        service = get_sign_language_service()
        service.set_groq_service(groq)
        corrected = await service._correct_sentence_with_groq(words)
        
        return {
            "sentence": corrected or raw_sentence,
            "original_words": words,
            "raw_sentence": raw_sentence,
            "corrected": corrected is not None
        }
        
    except Exception as e:
        return {
            "sentence": raw_sentence,
            "original_words": words,
            "corrected": False,
            "error": str(e)
        }


@router.get("/vocabulary")
async def get_vocabulary():
    """
    Get the vocabulary of supported signs.
    
    Returns the list of all sign words that can be recognized.
    """
    try:
        from app.services.sign_language.config import load_vocabulary
        vocab = load_vocabulary()
        
        # Sort by index
        sorted_vocab = [(idx, word) for idx, word in vocab.items()]
        sorted_vocab.sort(key=lambda x: x[0])
        
        return {
            "vocabulary_size": len(vocab),
            "words": [word for _, word in sorted_vocab],
            "word_to_index": {word: idx for idx, word in vocab.items()},
            "index_to_word": vocab
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check for sign language recognition service.
    """
    try:
        service = get_sign_language_service()
        status = service.get_status()
        
        is_healthy = status.get("status") == "ready"
        
        return {
            "healthy": is_healthy,
            "service": "sign-video",
            "models_loaded": any(
                m.get("loaded", False) 
                for m in status.get("models", {}).values()
            ),
            "device": status.get("config", {}).get("device", "unknown")
        }
        
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }


# =============================================================================
# SIGNVLM ENDPOINTS (CLIP-based Vision Language Model)
# =============================================================================

# Lazy import SignVLM service
_signvlm_service = None

def get_signvlm_service():
    """Get or create SignVLM service singleton."""
    global _signvlm_service
    if _signvlm_service is None:
        try:
            from app.services.sign_language.signvlm import get_signvlm_service as _get_svc
            _signvlm_service = _get_svc()
        except Exception as e:
            print(f"Warning: Could not initialize SignVLM service: {e}")
    return _signvlm_service


class SignVLMRecognitionRequest(BaseModel):
    """Request for SignVLM video recognition."""
    video_base64: str = Field(..., description="Base64 encoded video data")
    top_k: int = Field(default=5, description="Number of top predictions to return")
    use_groq_correction: bool = Field(
        default=True,
        description="Use Groq LLM for sentence correction"
    )


class SignVLMFramesRequest(BaseModel):
    """Request for SignVLM recognition from frames."""
    frames_base64: List[str] = Field(
        ...,
        description="List of base64 encoded frame images",
        min_length=2
    )
    top_k: int = Field(default=5, description="Number of top predictions")
    use_groq_correction: bool = Field(default=True)


class SignVLMResponse(BaseModel):
    """Response from SignVLM recognition."""
    recognized_words: List[str] = Field(default=[])
    sentence: str = Field(default="")
    confidence: float = Field(default=0.0)
    predictions: Optional[List[Dict[str, Any]]] = Field(None)
    frames_processed: Optional[int] = Field(None)
    model: str = Field(default="SignVLM")
    raw_sentence: Optional[str] = Field(None)
    error: Optional[str] = Field(None)


class SignVLMStatusResponse(BaseModel):
    """Response with SignVLM model status."""
    service: str = Field(default="SignVLMService")
    status: str = Field(..., description="ready, model_not_loaded, backbone_missing")
    models: Dict[str, Any] = Field(default={})
    config: Dict[str, Any] = Field(default={})
    vocabulary_size: int = Field(default=0)


@router.get("/signvlm/status", response_model=SignVLMStatusResponse)
async def get_signvlm_status():
    """
    Get the status of SignVLM model.
    
    SignVLM is a CLIP-based Vision Language Model for sign language recognition.
    It uses a ViT-L/14 backbone from CLIP with a temporal decoder.
    
    Returns:
    - Model availability and loading status
    - Backbone configuration
    - Vocabulary size
    
    **Note:** If backbone is missing, run the download script first:
    ```bash
    python backend/app/services/sign_language/signvlm/download_models.py
    ```
    """
    try:
        service = get_signvlm_service()
        if service is None:
            return SignVLMStatusResponse(
                service="SignVLMService",
                status="service_not_available",
                models={},
                config={},
                vocabulary_size=0
            )
        
        status = service.get_status()
        return SignVLMStatusResponse(**status)
        
    except Exception as e:
        return SignVLMStatusResponse(
            service="SignVLMService",
            status=f"error: {str(e)}",
            models={},
            config={},
            vocabulary_size=0
        )


@router.post("/signvlm/recognize", response_model=SignVLMResponse)
async def signvlm_recognize_video(request: SignVLMRecognitionRequest):
    """
    Recognize sign language from video using SignVLM.
    
    **SignVLM Model:**
    - Uses CLIP ViT-L/14 backbone (frozen)
    - Temporal cross-attention decoder
    - Trained on sign language datasets (WLASL, KArSL, etc.)
    
    **Input:**
    - video_base64: Base64 encoded video (MP4, WebM)
    - top_k: Number of top predictions (default: 5)
    - use_groq_correction: Enable Groq LLM sentence correction
    
    **Processing:**
    ```
    Video → Extract Frames (15 FPS) → Normalize (CLIP) → 
    SignVLM Model → Top-K Predictions → [Groq Correction] → Result
    ```
    
    **Output:**
    ```json
    {
      "recognized_words": ["hello", "thank", "you"],
      "sentence": "Hello, thank you!",
      "confidence": 0.87,
      "predictions": [
        {"gloss": "hello", "confidence": 0.87},
        {"gloss": "help", "confidence": 0.05}
      ]
    }
    ```
    """
    try:
        service = get_signvlm_service()
        if service is None:
            return SignVLMResponse(
                error="SignVLM service not available",
                recognized_words=[],
                sentence="",
                confidence=0.0
            )
        
        # Decode video
        try:
            if request.video_base64.startswith('data:'):
                video_b64 = request.video_base64.split(',', 1)[1]
            else:
                video_b64 = request.video_base64
            video_bytes = base64.b64decode(video_b64)
        except Exception as e:
            return SignVLMResponse(
                error=f"Failed to decode video: {str(e)}",
                recognized_words=[],
                sentence="",
                confidence=0.0
            )
        
        # Recognize
        result = await service.recognize_video(video_bytes, top_k=request.top_k)
        
        # Optionally correct with Groq
        raw_sentence = result.get("sentence", "")
        if request.use_groq_correction and raw_sentence:
            groq = get_groq_service()
            if groq:
                try:
                    words = result.get("recognized_words", [])
                    if words:
                        corrected = await groq.generate(
                            f"Convert these sign language glosses into a natural English sentence. "
                            f"Only output the corrected sentence, nothing else.\n"
                            f"Glosses: {', '.join(words)}",
                            max_tokens=100
                        )
                        if corrected:
                            result["sentence"] = corrected.strip()
                            result["raw_sentence"] = raw_sentence
                except Exception as e:
                    print(f"Groq correction failed: {e}")
        
        return SignVLMResponse(**result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signvlm/recognize/frames", response_model=SignVLMResponse)
async def signvlm_recognize_frames(request: SignVLMFramesRequest):
    """
    Recognize sign language from a sequence of frames using SignVLM.
    
    Use this endpoint when you have pre-extracted frames (e.g., from webcam).
    
    **Input:**
    - frames_base64: List of base64 encoded images (minimum 2 frames)
    - top_k: Number of top predictions
    
    **Note:** For best results, provide 24-32 frames at ~15 FPS (1.5-2 seconds of video)
    """
    try:
        service = get_signvlm_service()
        if service is None:
            return SignVLMResponse(
                error="SignVLM service not available",
                recognized_words=[],
                sentence="",
                confidence=0.0
            )
        
        # Extract frames from base64 images
        frames = service.extract_frames_from_base64_images(request.frames_base64)
        
        if len(frames) < 2:
            return SignVLMResponse(
                error="Could not decode enough frames",
                recognized_words=[],
                sentence="",
                confidence=0.0
            )
        
        # Recognize
        result = await service.recognize_frames(frames, top_k=request.top_k)
        
        # Optionally correct with Groq
        raw_sentence = result.get("sentence", "")
        if request.use_groq_correction and raw_sentence:
            groq = get_groq_service()
            if groq:
                try:
                    words = result.get("recognized_words", [])
                    if words:
                        corrected = await groq.generate(
                            f"Convert these sign language glosses into a natural English sentence. "
                            f"Only output the corrected sentence, nothing else.\n"
                            f"Glosses: {', '.join(words)}",
                            max_tokens=100
                        )
                        if corrected:
                            result["sentence"] = corrected.strip()
                            result["raw_sentence"] = raw_sentence
                except Exception:
                    pass
        
        return SignVLMResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signvlm/recognize/upload", response_model=SignVLMResponse)
async def signvlm_recognize_upload(
    file: UploadFile = File(..., description="Video file (MP4, WebM, etc.)"),
    top_k: int = Form(default=5),
    use_groq_correction: bool = Form(default=True)
):
    """
    Recognize sign language from uploaded video using SignVLM.
    
    Same as /signvlm/recognize but accepts file upload.
    
    **Supported formats:** MP4, WebM, AVI, MOV
    **Recommended:** 2-3 seconds of video, 480p-720p
    """
    try:
        service = get_signvlm_service()
        if service is None:
            return SignVLMResponse(
                error="SignVLM service not available",
                recognized_words=[],
                sentence="",
                confidence=0.0
            )
        
        # Read video
        video_bytes = await file.read()
        
        if len(video_bytes) < 1000:
            return SignVLMResponse(
                error="Video file is too small",
                recognized_words=[],
                sentence="",
                confidence=0.0
            )
        
        # Recognize
        result = await service.recognize_video(video_bytes, top_k=top_k)
        
        # Correct with Groq
        raw_sentence = result.get("sentence", "")
        if use_groq_correction and raw_sentence:
            groq = get_groq_service()
            if groq:
                try:
                    words = result.get("recognized_words", [])
                    if words:
                        corrected = await groq.generate(
                            f"Convert these sign language glosses into a natural English sentence. "
                            f"Only output the corrected sentence, nothing else.\n"
                            f"Glosses: {', '.join(words)}",
                            max_tokens=100
                        )
                        if corrected:
                            result["sentence"] = corrected.strip()
                            result["raw_sentence"] = raw_sentence
                except Exception:
                    pass
        
        return SignVLMResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signvlm/vocabulary")
async def get_signvlm_vocabulary():
    """
    Get the vocabulary of signs recognized by SignVLM.
    
    Returns the list of all sign glosses that SignVLM can recognize,
    based on the loaded class mapping (WLASL-100 by default).
    """
    try:
        service = get_signvlm_service()
        if service is None:
            return {
                "vocabulary_size": 0,
                "words": [],
                "error": "SignVLM service not available"
            }
        
        vocab = service.class_mapping
        
        return {
            "vocabulary_size": len(vocab),
            "words": [vocab[i] for i in sorted(vocab.keys())],
            "index_to_word": vocab,
            "dataset": "WLASL-100"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signvlm/load")
async def load_signvlm_model():
    """
    Explicitly load the SignVLM model.
    
    Call this endpoint to pre-load the model before making recognition requests.
    The model will be loaded automatically on first use, but this can be slow.
    
    **Returns:**
    - success: Whether the model was loaded
    - message: Status message
    """
    try:
        service = get_signvlm_service()
        if service is None:
            return {
                "success": False,
                "message": "SignVLM service not available"
            }
        
        success = service.load_model()
        
        if success:
            return {
                "success": True,
                "message": "SignVLM model loaded successfully",
                "config": {
                    "backbone": service.config.backbone_name,
                    "num_classes": len(service.class_mapping),
                    "device": service.config.device
                }
            }
        else:
            return {
                "success": False,
                "message": "Failed to load SignVLM model. Check if backbone weights are downloaded.",
                "help": "Run: python backend/app/services/sign_language/signvlm/download_models.py"
            }
        
    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@router.get("/signvlm/health")
async def signvlm_health_check():
    """
    Health check for SignVLM service.
    """
    try:
        service = get_signvlm_service()
        if service is None:
            return {
                "healthy": False,
                "service": "signvlm",
                "error": "Service not available"
            }
        
        status = service.get_status()
        is_ready = status.get("status") == "ready"
        backbone_available = status.get("models", {}).get("backbone", {}).get("available", False)
        
        return {
            "healthy": backbone_available,
            "service": "signvlm",
            "model_loaded": is_ready,
            "backbone_available": backbone_available,
            "device": status.get("config", {}).get("device", "unknown")
        }
        
    except Exception as e:
        return {
            "healthy": False,
            "service": "signvlm",
            "error": str(e)
        }
