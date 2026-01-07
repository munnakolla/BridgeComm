"""
Groq AI Routes
Endpoints for Groq-powered text processing, simplification, and symbol generation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from app.services.groq_service import get_groq_service

router = APIRouter(prefix="/groq", tags=["Groq AI"])


class TextProcessRequest(BaseModel):
    """Request for text processing."""
    text: str = Field(..., description="Text to process")
    task: str = Field(default="full", description="Task: full, simplify, emotion, symbols, intent")
    context: Optional[str] = Field(default=None, description="Additional context")


class TextProcessResponse(BaseModel):
    """Response from text processing."""
    original_text: str
    simplified_text: str
    emotion: str
    symbols: List[str]
    intent: str
    response: str


class TextToSymbolsRequest(BaseModel):
    """Request for text to symbols conversion."""
    text: str = Field(..., description="Text to convert")
    simplify: bool = Field(default=True, description="Whether to simplify")
    max_symbols: int = Field(default=5, description="Maximum symbols")
    user_id: Optional[str] = Field(default=None)


class TextToSymbolsResponse(BaseModel):
    """Response with symbols and simplified text."""
    original_text: str
    simplified_text: str
    emotion: str
    symbols: List[str]
    keywords: List[str] = []
    confidence: float = 0.95


class GenerateResponseRequest(BaseModel):
    """Request for response generation."""
    user_input: str = Field(..., description="User's input")
    context: Optional[str] = Field(default=None)
    user_emotion: Optional[str] = Field(default=None)


class GenerateResponseResponse(BaseModel):
    """Generated response."""
    response: str
    emotion: str = "neutral"


@router.post("/process", response_model=TextProcessResponse)
async def process_text(request: TextProcessRequest):
    """
    Process text using Groq LLM.
    
    Tasks:
    - full: Complete analysis (simplify, emotion, symbols, intent, response)
    - simplify: Just simplify the text
    - emotion: Detect emotion
    - symbols: Convert to emoji symbols
    - intent: Detect user intent
    """
    try:
        groq_service = get_groq_service()
        
        result = await groq_service.text_to_symbols(
            text=request.text,
            simplify=True
        )
        
        return TextProcessResponse(
            original_text=result.get("original_text", request.text),
            simplified_text=result.get("simplified_text", request.text),
            emotion=result.get("emotion", "neutral"),
            symbols=result.get("symbols", ["💬"]),
            intent=result.get("intent", "statement"),
            response=result.get("response", "")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/text-to-symbols", response_model=TextToSymbolsResponse)
async def text_to_symbols(request: TextToSymbolsRequest):
    """
    Convert text to simplified text and visual symbols using Groq AI.
    
    This is the main endpoint for the BridgeComm flow:
    1. Takes user text/speech
    2. Simplifies it
    3. Detects emotion
    4. Converts to visual symbols
    """
    try:
        groq_service = get_groq_service()
        
        result = await groq_service.text_to_symbols(
            text=request.text,
            simplify=request.simplify,
            max_symbols=request.max_symbols
        )
        
        # Extract keywords from simplified text
        keywords = result.get("simplified_text", "").split()[:5]
        
        return TextToSymbolsResponse(
            original_text=result.get("original_text", request.text),
            simplified_text=result.get("simplified_text", request.text),
            emotion=result.get("emotion", "neutral"),
            symbols=result.get("symbols", ["💬"]),
            keywords=keywords,
            confidence=0.95
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/generate-response", response_model=GenerateResponseResponse)
async def generate_response(request: GenerateResponseRequest):
    """
    Generate an appropriate response for the user using Groq AI.
    
    Takes into account:
    - User's input
    - Conversation context
    - Detected emotion
    """
    try:
        groq_service = get_groq_service()
        
        response = await groq_service.generate_response(
            user_input=request.user_input,
            context=request.context,
            user_emotion=request.user_emotion
        )
        
        return GenerateResponseResponse(
            response=response,
            emotion=request.user_emotion or "neutral"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Check if Groq service is configured and working."""
    try:
        groq_service = get_groq_service()
        
        if not groq_service.api_key:
            return {
                "status": "not_configured",
                "message": "GROQ_API_KEY not set in .env"
            }
        
        # Quick test
        result = await groq_service.process_text("Hello", task="emotion")
        
        return {
            "status": "healthy",
            "message": "Groq AI service is working",
            "test_result": result
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
