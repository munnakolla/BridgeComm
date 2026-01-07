"""
Vision API Routes
Provides endpoints for Azure AI Vision features like scene description and OCR.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from app.services.vision_service import get_vision_service


router = APIRouter(prefix="/vision", tags=["vision"])


class SceneDescriptionResponse(BaseModel):
    """Response model for scene description."""
    description: str
    details: List[str] = []
    confidence: float = 0.0
    error: Optional[str] = None


class OCRResponse(BaseModel):
    """Response model for OCR."""
    text: str
    lines: List[str] = []
    error: Optional[str] = None


@router.post("/describe", response_model=SceneDescriptionResponse)
async def describe_scene(
    image: UploadFile = File(...)
):
    """
    Analyze an image and return a natural language description.
    
    This endpoint uses Azure AI Vision to describe what's in the image,
    which can then be converted to speech for visually impaired users.
    """
    vision_service = get_vision_service()
    
    # Read image data
    image_data = await image.read()
    
    if not image_data:
        raise HTTPException(status_code=400, detail="No image data provided")
    
    # Get scene description
    result = await vision_service.describe_scene(image_data)
    
    return SceneDescriptionResponse(
        description=result.get("description", ""),
        details=result.get("details", []),
        confidence=result.get("confidence", 0.0),
        error=result.get("error")
    )


@router.post("/read-text", response_model=OCRResponse)
async def read_text(
    image: UploadFile = File(...)
):
    """
    Extract text from an image using OCR.
    
    Useful for reading signs, documents, or any text in the environment
    for users who need assistance.
    """
    vision_service = get_vision_service()
    
    # Read image data
    image_data = await image.read()
    
    if not image_data:
        raise HTTPException(status_code=400, detail="No image data provided")
    
    # Extract text
    result = await vision_service.read_text_from_image(image_data)
    
    return OCRResponse(
        text=result.get("text", ""),
        lines=result.get("lines", []),
        error=result.get("error")
    )


@router.get("/status")
async def vision_status():
    """Check if Azure Vision is configured and available."""
    vision_service = get_vision_service()
    return {
        "azure_vision_available": vision_service._has_azure_vision,
        "mediapipe_available": vision_service.mediapipe_available
    }
