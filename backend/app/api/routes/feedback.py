"""
Feedback API Routes
Endpoints for collecting user feedback and model personalization.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import uuid

from app.models.schemas import (
    FeedbackRequest,
    FeedbackResponse,
)
from app.services import get_user_service


router = APIRouter(prefix="/azure", tags=["Feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for AI model improvement.
    
    This endpoint collects feedback when the AI interpretation is:
    - Correct: The output was accurate
    - Incorrect: The output was wrong (user provides correction)
    - Partial: The output was partially correct
    
    Feedback is used for:
    1. Improving the user's personalized model
    2. Updating gesture/behavior pattern recognition
    3. Training data for future model improvements
    """
    user_service = get_user_service()
    
    try:
        feedback_id = str(uuid.uuid4())
        
        # Build feedback entry for storage
        feedback_entry = {
            "feedback_id": feedback_id,
            "session_id": request.session_id,
            "interaction_id": request.interaction_id,
            "feedback_type": request.feedback_type.value,
            "original_output": request.original_output,
            "corrected_output": request.corrected_output,
            "input_mode": request.input_mode.value,
            "additional_context": request.additional_context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update user's personalization
        personalization_updated = False
        
        if request.feedback_type.value == "correct":
            # Positive feedback - reinforce the pattern
            success = await user_service.update_personalization(
                user_id=request.user_id,
                feedback_entry=feedback_entry
            )
            
            # Record successful interaction
            await user_service.record_interaction(
                user_id=request.user_id,
                interaction_type=request.input_mode.value,
                success=True
            )
            
            personalization_updated = success
            
        elif request.feedback_type.value == "incorrect" and request.corrected_output:
            # Negative feedback with correction - learn the correction
            
            # Add to vocabulary preferences if it's a text correction
            vocabulary_update = None
            if request.corrected_output:
                # Extract key words from the correction
                words = request.corrected_output.lower().split()
                vocabulary_update = [w for w in words if len(w) > 2]
            
            # Update gesture pattern if applicable
            gesture_update = None
            if request.input_mode.value in ["sign", "behavior"]:
                # Store the corrected interpretation for this pattern
                pattern_key = f"{request.input_mode.value}_{request.interaction_id}"
                gesture_update = {
                    pattern_key: {
                        "correct_output": request.corrected_output,
                        "original_output": request.original_output,
                        "learned_at": datetime.utcnow().isoformat()
                    }
                }
            
            success = await user_service.update_personalization(
                user_id=request.user_id,
                feedback_entry=feedback_entry,
                vocabulary=vocabulary_update,
                gesture_pattern=gesture_update
            )
            
            # Record unsuccessful interaction
            await user_service.record_interaction(
                user_id=request.user_id,
                interaction_type=request.input_mode.value,
                success=False
            )
            
            personalization_updated = success
            
        else:
            # Partial or no correction provided
            await user_service.update_personalization(
                user_id=request.user_id,
                feedback_entry=feedback_entry
            )
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Feedback recorded successfully",
            personalization_updated=personalization_updated
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feedback/stats/{user_id}")
async def get_feedback_stats(user_id: str):
    """
    Get feedback statistics for a user.
    
    Returns summary of feedback history and accuracy trends.
    """
    user_service = get_user_service()
    
    try:
        user = await user_service.get_user(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        personalization = user.personalization_data or {}
        feedback_history = personalization.get("feedback_history", [])
        
        # Calculate stats
        total_feedback = len(feedback_history)
        correct_count = sum(1 for f in feedback_history if f.get("feedback_type") == "correct")
        incorrect_count = sum(1 for f in feedback_history if f.get("feedback_type") == "incorrect")
        partial_count = sum(1 for f in feedback_history if f.get("feedback_type") == "partial")
        
        accuracy_rate = correct_count / total_feedback if total_feedback > 0 else 0.0
        
        # Get recent trend (last 20 feedback entries)
        recent = feedback_history[-20:]
        recent_correct = sum(1 for f in recent if f.get("feedback_type") == "correct")
        recent_accuracy = recent_correct / len(recent) if recent else 0.0
        
        return {
            "user_id": user_id,
            "total_feedback": total_feedback,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "partial_count": partial_count,
            "accuracy_rate": accuracy_rate,
            "recent_accuracy": recent_accuracy,
            "improvement": recent_accuracy - accuracy_rate if total_feedback > 20 else 0.0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
