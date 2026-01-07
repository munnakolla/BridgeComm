"""
Translator API Routes
Provides endpoints for Azure AI Translator features.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.services.translator_service import get_translator_service


router = APIRouter(prefix="/azure", tags=["Translator"])


class TranslateRequest(BaseModel):
    """Request model for translation."""
    text: str
    to_language: str = "en"
    from_language: Optional[str] = None


class TranslateResponse(BaseModel):
    """Response model for translation."""
    original: str
    translated: str
    from_language: str
    to_language: str
    from_confidence: Optional[float] = None
    error: Optional[str] = None


class BatchTranslateRequest(BaseModel):
    """Request model for batch translation."""
    texts: List[str]
    to_language: str = "en"
    from_language: Optional[str] = None


class DetectLanguageRequest(BaseModel):
    """Request model for language detection."""
    text: str


class DetectLanguageResponse(BaseModel):
    """Response model for language detection."""
    language: str
    confidence: float
    is_translation_supported: Optional[bool] = None
    error: Optional[str] = None


@router.post("/translate", response_model=TranslateResponse)
async def translate_text(request: TranslateRequest):
    """
    Translate text to a target language.
    
    Useful for multi-language support in communication.
    """
    translator = get_translator_service()
    
    result = await translator.translate(
        text=request.text,
        to_language=request.to_language,
        from_language=request.from_language
    )
    
    return TranslateResponse(**result)


@router.post("/translate/batch", response_model=List[TranslateResponse])
async def translate_batch(request: BatchTranslateRequest):
    """
    Translate multiple texts at once.
    """
    translator = get_translator_service()
    
    results = await translator.translate_batch(
        texts=request.texts,
        to_language=request.to_language,
        from_language=request.from_language
    )
    
    return [TranslateResponse(**r) for r in results]


@router.post("/translate/detect", response_model=DetectLanguageResponse)
async def detect_language(request: DetectLanguageRequest):
    """
    Detect the language of text.
    """
    translator = get_translator_service()
    
    result = await translator.detect_language(request.text)
    
    return DetectLanguageResponse(**result)


@router.get("/translate/languages")
async def get_supported_languages():
    """Get list of supported languages for translation."""
    translator = get_translator_service()
    return await translator.get_supported_languages()


@router.get("/translate/status")
async def translator_status():
    """Check if Azure Translator is configured and available."""
    translator = get_translator_service()
    return {
        "translator_available": translator._has_translator
    }
