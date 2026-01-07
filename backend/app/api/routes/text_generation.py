"""
Text Generation API Routes
Endpoints for natural language generation.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    GenerateTextRequest,
    GenerateTextResponse,
)
from app.services import get_openai_service, get_user_service


router = APIRouter(prefix="/azure", tags=["Text Generation"])


@router.post("/generate-text", response_model=GenerateTextResponse)
async def generate_text(request: GenerateTextRequest):
    """
    Generate natural language text from an intent.
    
    This endpoint converts detected intents (from sign language or behavior)
    into natural, human-readable text that can be spoken or displayed.
    
    Supports different text styles:
    - natural: Everyday conversational language
    - formal: Polite, professional language
    - simple: Very simple, clear language
    - friendly: Warm, approachable language
    """
    openai_service = get_openai_service()
    user_service = get_user_service()
    
    try:
        # Get user preferences if available
        user_style = request.style
        if request.user_id:
            user = await user_service.get_user(request.user_id)
            if user and user.preferences:
                user_style = user.preferences.get("text_style", request.style)
        
        # Generate text
        generated_text = await openai_service.generate_natural_text(
            intent=request.intent,
            context=request.context,
            style=user_style
        )
        
        # Generate alternatives
        alternatives = []
        if request.style != "simple":
            simple_text = await openai_service.generate_natural_text(
                intent=request.intent,
                context=request.context,
                style="simple"
            )
            if simple_text != generated_text:
                alternatives.append(simple_text)
        
        if request.style != "formal":
            formal_text = await openai_service.generate_natural_text(
                intent=request.intent,
                context=request.context,
                style="formal"
            )
            if formal_text != generated_text and formal_text not in alternatives:
                alternatives.append(formal_text)
        
        return GenerateTextResponse(
            text=generated_text,
            intent=request.intent,
            confidence=0.92,  # High confidence for GPT generation
            alternatives=alternatives[:3] if alternatives else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/intent-to-full-message")
async def intent_to_full_message(
    intent: str,
    context: str = None,
    include_greeting: bool = False,
    include_politeness: bool = True
):
    """
    Convert a simple intent into a complete, polished message.
    
    This is useful for making communication more natural and complete.
    For example:
    - Input intent: "want water"
    - Output: "Excuse me, could I please have some water?"
    """
    openai_service = get_openai_service()
    
    try:
        # Build enhanced prompt
        style = "formal" if include_politeness else "natural"
        
        enhanced_intent = intent
        if include_greeting:
            enhanced_intent = f"greeting, then {intent}"
        
        if include_politeness:
            enhanced_intent = f"politely express: {intent}"
        
        text = await openai_service.generate_natural_text(
            intent=enhanced_intent,
            context=context,
            style=style
        )
        
        return {
            "original_intent": intent,
            "full_message": text,
            "style": style
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
