"""
Behavior API Routes
Endpoints for behavioral pattern recognition.
"""

from fastapi import APIRouter, HTTPException
import json

from app.models.schemas import (
    BehaviorToIntentRequest,
    BehaviorToIntentResponse,
)
from app.services import get_openai_service, get_user_service, get_vision_service


router = APIRouter(prefix="/azure", tags=["Behavior"])


@router.post("/behavior-to-intent", response_model=BehaviorToIntentResponse)
async def behavior_to_intent(request: BehaviorToIntentRequest):
    """
    Recognize behavioral patterns and convert to intent/text.
    
    This endpoint analyzes various behavioral inputs:
    - Touch patterns (taps, swipes, pressure patterns)
    - Eye tracking data (gaze direction, fixation points)
    - Facial expressions
    - Motion data from device sensors
    - UI interaction sequences
    
    Returns interpreted intent in natural language.
    """
    openai_service = get_openai_service()
    user_service = get_user_service()
    
    try:
        behavior_data = request.behavior_data
        
        # Build behavior description for AI analysis
        behavior_parts = []
        behavior_type = "unknown"
        extracted_features = {}
        
        # Analyze touch patterns
        if behavior_data.touch_patterns:
            behavior_type = "touch"
            touch_summary = _analyze_touch_patterns(behavior_data.touch_patterns)
            behavior_parts.append(f"Touch behavior: {touch_summary}")
            extracted_features["touch"] = touch_summary
        
        # Analyze eye tracking
        if behavior_data.eye_tracking:
            behavior_type = "eye" if behavior_type == "unknown" else "combined"
            eye_summary = _analyze_eye_tracking(behavior_data.eye_tracking)
            behavior_parts.append(f"Eye movement: {eye_summary}")
            extracted_features["eye"] = eye_summary
        
        # Analyze facial expressions
        if behavior_data.facial_expressions:
            behavior_type = "facial" if behavior_type == "unknown" else "combined"
            
            # Check if there's an image to analyze
            image_base64 = behavior_data.facial_expressions.get("image_base64")
            if image_base64:
                # Actually analyze the face image using vision service
                try:
                    vision_service = get_vision_service()
                    face_result = await vision_service.extract_face_data(image_base64)
                    
                    if face_result.get("faces") and len(face_result["faces"]) > 0:
                        expression = face_result["faces"][0].get("expression", "neutral")
                    else:
                        expression = "neutral"
                except Exception as e:
                    print(f"Face analysis error: {e}")
                    expression = "neutral"
            else:
                # Fallback to provided expression
                expression = behavior_data.facial_expressions.get("expression", "neutral")
            
            behavior_parts.append(f"Facial expression: {expression}")
            extracted_features["facial"] = expression
        
        # Analyze motion data
        if behavior_data.motion_data:
            behavior_type = "motion" if behavior_type == "unknown" else "combined"
            motion_summary = _analyze_motion_data(behavior_data.motion_data)
            behavior_parts.append(f"Motion pattern: {motion_summary}")
            extracted_features["motion"] = motion_summary
        
        # Analyze interaction sequence
        if behavior_data.interaction_sequence:
            behavior_type = "interaction" if behavior_type == "unknown" else "combined"
            sequence = " -> ".join(behavior_data.interaction_sequence)
            behavior_parts.append(f"Interaction sequence: {sequence}")
            extracted_features["sequence"] = behavior_data.interaction_sequence
        
        if not behavior_parts:
            return BehaviorToIntentResponse(
                intent="no_behavior",
                text="No behavioral input detected",
                confidence=0.0,
                behavior_type="none",
                features_extracted={}
            )
        
        # Get user personalization if available
        if request.user_id:
            personalization = await user_service.get_personalization_data(
                request.user_id
            )
            if personalization and personalization.get("gesture_patterns"):
                behavior_parts.append(
                    f"User's known patterns: {list(personalization.get('gesture_patterns', {}).keys())}"
                )
        
        # Combine behavior description
        behavior_description = ". ".join(behavior_parts)
        
        # Interpret using AI
        interpretation = await openai_service.interpret_behavior_pattern(
            behavior_description=behavior_description,
            context=request.context
        )
        
        return BehaviorToIntentResponse(
            intent=interpretation.get("intent", "unknown"),
            text=interpretation.get("text", "Unable to interpret behavior"),
            confidence=interpretation.get("confidence", 0.5),
            behavior_type=behavior_type,
            features_extracted=extracted_features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _analyze_touch_patterns(patterns: list) -> str:
    """Analyze touch patterns and return a summary description."""
    if not patterns:
        return "no touch detected"
    
    touch_types = []
    
    for pattern in patterns:
        touch_type = pattern.get("type", "tap")
        duration = pattern.get("duration", 0)
        
        if touch_type == "tap":
            touch_types.append("single tap")
        elif touch_type == "double_tap":
            touch_types.append("double tap")
        elif touch_type == "long_press":
            touch_types.append(f"long press ({duration}ms)")
        elif touch_type == "swipe":
            direction = pattern.get("direction", "unknown")
            touch_types.append(f"swipe {direction}")
        elif touch_type == "pinch":
            touch_types.append("pinch gesture")
        else:
            touch_types.append(touch_type)
    
    return ", ".join(touch_types)


def _analyze_eye_tracking(eye_data: dict) -> str:
    """Analyze eye tracking data and return a summary."""
    if not eye_data:
        return "no eye tracking data"
    
    gaze_direction = eye_data.get("gaze_direction", "center")
    fixation_duration = eye_data.get("fixation_duration", 0)
    target_element = eye_data.get("target_element", "unknown")
    
    parts = []
    
    if gaze_direction:
        parts.append(f"looking {gaze_direction}")
    
    if fixation_duration > 1000:
        parts.append(f"focused for {fixation_duration}ms")
    
    if target_element and target_element != "unknown":
        parts.append(f"on {target_element}")
    
    return " ".join(parts) if parts else "passive gaze"


def _analyze_motion_data(motion_data: list) -> str:
    """Analyze device motion data and return a summary."""
    if not motion_data:
        return "no motion detected"
    
    # Analyze motion patterns
    movements = []
    
    for motion in motion_data:
        motion_type = motion.get("type", "movement")
        intensity = motion.get("intensity", "low")
        direction = motion.get("direction")
        
        desc = f"{intensity} {motion_type}"
        if direction:
            desc += f" {direction}"
        
        movements.append(desc)
    
    # Summarize
    if len(movements) == 1:
        return movements[0]
    elif len(movements) <= 3:
        return ", then ".join(movements)
    else:
        return f"{len(movements)} motion events: {movements[0]} ... {movements[-1]}"
