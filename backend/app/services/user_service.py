"""
User Service
Handles user profiles and personalization using Azure Cosmos DB.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from app.core.azure_clients import get_azure_clients
from app.core.config import get_settings
from app.models.schemas import UserProfile, UserProfileUpdate, InputMode


class UserService:
    """Service for user management and personalization."""
    
    def __init__(self):
        self.clients = get_azure_clients()
        self.settings = get_settings()
    
    def _get_container(self):
        """Get the Cosmos DB container for users."""
        return self.clients.cosmos_container
    
    async def create_user(
        self,
        user_id: Optional[str] = None,
        display_name: Optional[str] = None,
        communication_mode: InputMode = InputMode.SIGN
    ) -> UserProfile:
        """
        Create a new user profile.
        
        Args:
            user_id: Optional custom user ID
            display_name: User's display name
            communication_mode: Preferred communication mode
            
        Returns:
            Created UserProfile
        """
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        now = datetime.utcnow().isoformat()
        
        user_data = {
            "id": user_id,
            "user_id": user_id,
            "display_name": display_name,
            "communication_mode": communication_mode.value,
            "preferences": {
                "text_simplification_level": "simple",
                "symbol_size": "medium",
                "speech_rate": 1.0,
                "voice": "en-US-JennyNeural"
            },
            "personalization_data": {
                "gesture_patterns": {},
                "vocabulary_preferences": [],
                "feedback_history": [],
                "model_weights": {}
            },
            "stats": {
                "sessions_count": 0,
                "total_interactions": 0,
                "accuracy_score": 0.0
            },
            "created_at": now,
            "updated_at": now
        }
        
        container = self._get_container()
        container.create_item(body=user_data)
        
        return UserProfile(**user_data)
    
    async def get_user(self, user_id: str) -> Optional[UserProfile]:
        """
        Get a user profile by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            UserProfile or None if not found
        """
        container = self._get_container()
        
        try:
            response = container.read_item(item=user_id, partition_key=user_id)
            return UserProfile(**response)
        except Exception:
            return None
    
    async def update_user(
        self,
        user_id: str,
        update: UserProfileUpdate
    ) -> Optional[UserProfile]:
        """
        Update a user profile.
        
        Args:
            user_id: User ID
            update: Update data
            
        Returns:
            Updated UserProfile or None
        """
        container = self._get_container()
        
        try:
            # Get existing user
            user_data = container.read_item(item=user_id, partition_key=user_id)
            
            # Apply updates
            if update.display_name is not None:
                user_data["display_name"] = update.display_name
            if update.communication_mode is not None:
                user_data["communication_mode"] = update.communication_mode.value
            if update.preferences is not None:
                user_data["preferences"].update(update.preferences)
            
            user_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Save
            container.replace_item(item=user_id, body=user_data)
            
            return UserProfile(**user_data)
        except Exception:
            return None
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user profile.
        
        Args:
            user_id: User ID
            
        Returns:
            True if deleted, False otherwise
        """
        container = self._get_container()
        
        try:
            container.delete_item(item=user_id, partition_key=user_id)
            return True
        except Exception:
            return False
    
    async def get_personalization_data(
        self,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get user's personalization data for AI model adaptation.
        
        Args:
            user_id: User ID
            
        Returns:
            Personalization data dictionary
        """
        user = await self.get_user(user_id)
        if user:
            return user.personalization_data
        return None
    
    async def update_personalization(
        self,
        user_id: str,
        gesture_pattern: Optional[Dict[str, Any]] = None,
        vocabulary: Optional[List[str]] = None,
        feedback_entry: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update user's personalization data based on interactions.
        
        Args:
            user_id: User ID
            gesture_pattern: New gesture pattern to learn
            vocabulary: Vocabulary words to add
            feedback_entry: Feedback from user interaction
            
        Returns:
            True if updated successfully
        """
        container = self._get_container()
        
        try:
            user_data = container.read_item(item=user_id, partition_key=user_id)
            
            personalization = user_data.get("personalization_data", {})
            
            if gesture_pattern:
                patterns = personalization.get("gesture_patterns", {})
                patterns.update(gesture_pattern)
                personalization["gesture_patterns"] = patterns
            
            if vocabulary:
                vocab = personalization.get("vocabulary_preferences", [])
                vocab.extend([v for v in vocabulary if v not in vocab])
                personalization["vocabulary_preferences"] = vocab[-100:]  # Keep last 100
            
            if feedback_entry:
                history = personalization.get("feedback_history", [])
                history.append({
                    **feedback_entry,
                    "timestamp": datetime.utcnow().isoformat()
                })
                personalization["feedback_history"] = history[-500:]  # Keep last 500
            
            user_data["personalization_data"] = personalization
            user_data["updated_at"] = datetime.utcnow().isoformat()
            
            container.replace_item(item=user_id, body=user_data)
            return True
            
        except Exception:
            return False
    
    async def record_interaction(
        self,
        user_id: str,
        interaction_type: str,
        success: bool
    ) -> bool:
        """
        Record a user interaction for stats tracking.
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction
            success: Whether the interaction was successful
            
        Returns:
            True if recorded
        """
        container = self._get_container()
        
        try:
            user_data = container.read_item(item=user_id, partition_key=user_id)
            
            stats = user_data.get("stats", {})
            stats["total_interactions"] = stats.get("total_interactions", 0) + 1
            
            if success:
                # Update accuracy score (simple running average)
                current_accuracy = stats.get("accuracy_score", 0.5)
                total = stats.get("total_interactions", 1)
                stats["accuracy_score"] = (current_accuracy * (total - 1) + 1.0) / total
            else:
                current_accuracy = stats.get("accuracy_score", 0.5)
                total = stats.get("total_interactions", 1)
                stats["accuracy_score"] = (current_accuracy * (total - 1) + 0.0) / total
            
            user_data["stats"] = stats
            user_data["updated_at"] = datetime.utcnow().isoformat()
            
            container.replace_item(item=user_id, body=user_data)
            return True
            
        except Exception:
            return False


# Singleton instance
_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get the user service singleton."""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service
