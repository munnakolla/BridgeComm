"""
User Profile API Routes
Endpoints for user management and personalization.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from app.models.schemas import (
    UserProfile,
    UserProfileUpdate,
    InputMode,
)
from app.services import get_user_service


router = APIRouter(prefix="/users", tags=["Users"])


@router.post("/", response_model=UserProfile)
async def create_user(
    display_name: Optional[str] = None,
    communication_mode: InputMode = InputMode.SIGN
):
    """
    Create a new user profile.
    
    Returns the created user with a unique ID.
    """
    user_service = get_user_service()
    
    try:
        user = await user_service.create_user(
            display_name=display_name,
            communication_mode=communication_mode
        )
        return user
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}", response_model=UserProfile)
async def get_user(user_id: str):
    """
    Get a user profile by ID.
    """
    user_service = get_user_service()
    
    user = await user_service.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.patch("/{user_id}", response_model=UserProfile)
async def update_user(user_id: str, update: UserProfileUpdate):
    """
    Update a user profile.
    
    Supports partial updates - only provided fields will be changed.
    """
    user_service = get_user_service()
    
    user = await user_service.update_user(user_id, update)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """
    Delete a user profile and all associated data.
    
    This action is irreversible.
    """
    user_service = get_user_service()
    
    success = await user_service.delete_user(user_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User deleted successfully"}


@router.get("/{user_id}/personalization")
async def get_user_personalization(user_id: str):
    """
    Get a user's personalization data.
    
    This includes learned patterns, preferences, and model customizations.
    """
    user_service = get_user_service()
    
    data = await user_service.get_personalization_data(user_id)
    
    if data is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": user_id,
        "personalization": data
    }


@router.get("/{user_id}/stats")
async def get_user_stats(user_id: str):
    """
    Get user statistics including usage and accuracy metrics.
    """
    user_service = get_user_service()
    
    user = await user_service.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Extract stats from user data
    # Note: In a real implementation, you'd query the raw Cosmos data
    return {
        "user_id": user_id,
        "member_since": user.created_at,
        "last_active": user.updated_at,
        "communication_mode": user.communication_mode.value,
        # Stats would be extracted from personalization_data
        "stats": {
            "total_sessions": 0,
            "total_interactions": 0,
            "accuracy": 0.0,
            "improvement": 0.0
        }
    }
