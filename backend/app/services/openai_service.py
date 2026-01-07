"""
Azure OpenAI Service
Handles text simplification, intent generation, and natural language processing.
Uses Azure Language service as primary, with Azure OpenAI as secondary option.
Falls back to simple text processing when neither is configured.
"""

from typing import Optional, List, Dict, Any
import json
import re

from app.core.azure_clients import get_azure_clients
from app.core.config import get_settings


class OpenAIService:
    """Service for Azure OpenAI operations with Azure Language and fallback support."""
    
    def __init__(self):
        self.clients = get_azure_clients()
        self.settings = get_settings()
        self._has_openai = (
            self.settings.azure_openai_api_key is not None and 
            self.settings.azure_openai_endpoint is not None and
            self.clients.openai_client is not None
        )
        self._has_language = (
            self.settings.azure_language_key is not None and
            self.settings.azure_language_endpoint is not None
        )
        
        # Import language service for primary processing
        if self._has_language:
            from app.services.language_service import get_language_service
            self._language_service = get_language_service()
        else:
            self._language_service = None
    
    def _fallback_simplify(self, text: str, target_level: str) -> str:
        """Simple text processing fallback when OpenAI is not available."""
        # Remove complex punctuation
        text = re.sub(r'[;:\-—]', ' ', text)
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        simplified = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Remove filler words
            fillers = ['actually', 'basically', 'literally', 'honestly', 'obviously', 
                      'definitely', 'absolutely', 'certainly', 'probably', 'perhaps',
                      'however', 'therefore', 'furthermore', 'moreover', 'nevertheless']
            words = sentence.split()
            words = [w for w in words if w.lower() not in fillers]
            
            if target_level == "keywords_only":
                # Keep only nouns and verbs (rough heuristic)
                words = [w for w in words if len(w) > 3][:5]
                simplified.extend(words)
            else:
                simplified.append(' '.join(words))
        
        if target_level == "keywords_only":
            return ' '.join(simplified)
        return '. '.join(simplified) + '.' if simplified else text
    
    def _fallback_extract_keywords(self, text: str) -> List[str]:
        """Simple keyword extraction fallback."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'again', 'further',
                     'then', 'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if',
                     'because', 'until', 'while', 'this', 'that', 'these', 'those',
                     'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
                     'it', 'they', 'them', 'his', 'her', 'its', 'their'}
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        # Return unique keywords, preserving order
        seen = set()
        unique_keywords = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique_keywords.append(w)
        return unique_keywords[:7]
    
    async def simplify_text(
        self,
        text: str,
        target_level: str = "simple",
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Simplify text for communication-disabled users.
        Uses Azure Language service as primary, OpenAI as fallback.
        
        Args:
            text: Original text to simplify
            target_level: Simplification level (simple, very_simple, keywords_only)
            user_preferences: User-specific preferences for simplification
            
        Returns:
            Simplified text
        """
        # Try Azure Language service first (primary)
        if self._has_language and self._language_service:
            try:
                return await self._language_service.simplify_text(text, target_level)
            except Exception as e:
                print(f"Language service error, trying fallback: {e}")
        
        # If no AI services available, use basic fallback
        if not self._has_openai:
            return self._fallback_simplify(text, target_level)
        
        # Use OpenAI as secondary option
        system_prompt = """You are an expert at simplifying language for people with communication disabilities.
Your task is to convert complex sentences into simple, clear, and easy-to-understand language.

Rules:
1. Use short, simple sentences
2. Use common, everyday words
3. Remove unnecessary words and phrases
4. Keep the core meaning intact
5. Use present tense when possible
6. Avoid idioms and metaphors
7. Be direct and concrete

Output ONLY the simplified text, nothing else."""

        user_prompt = f"Simplify this text: \"{text}\""
        
        if target_level == "very_simple":
            user_prompt += "\nMake it extremely simple, use only basic vocabulary."
        elif target_level == "keywords_only":
            user_prompt += "\nReduce to essential keywords only, separated by spaces."
        
        response = self.clients.openai_client.chat.completions.create(
            model=self.settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    async def extract_keywords(self, text: str) -> List[str]:
        """
        Extract key words from text for symbol mapping.
        Uses Azure Language service as primary.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of keywords
        """
        # Try Azure Language service first (primary)
        if self._has_language and self._language_service:
            try:
                return await self._language_service.extract_key_phrases(text)
            except Exception as e:
                print(f"Language service error, trying fallback: {e}")
        
        # Use fallback if OpenAI not configured
        if not self._has_openai:
            return self._fallback_extract_keywords(text)
        
        system_prompt = """Extract the key action words and nouns from the given text.
Return them as a JSON array of strings.
Focus on words that can be represented by visual symbols/icons.
Include verbs (actions) and nouns (objects/people/places).
Maximum 5-7 keywords.
Return ONLY the JSON array, no explanation."""

        response = self.clients.openai_client.chat.completions.create(
            model=self.settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract keywords from: \"{text}\""}
            ],
            temperature=0.2,
            max_tokens=100
        )
        
        try:
            content = response.choices[0].message.content.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except (json.JSONDecodeError, IndexError):
            # Fallback: simple word extraction
            return text.lower().split()[:5]
    
    async def generate_natural_text(
        self,
        intent: str,
        context: Optional[str] = None,
        style: str = "natural"
    ) -> str:
        """
        Generate natural language text from an intent.
        
        Args:
            intent: The intent/meaning to express
            context: Conversation context
            style: Text style (natural, formal, simple, friendly)
            
        Returns:
            Generated natural language text
        """
        style_instructions = {
            "natural": "Use natural, everyday conversational language.",
            "formal": "Use polite, formal language appropriate for professional settings.",
            "simple": "Use very simple, clear language with basic vocabulary.",
            "friendly": "Use warm, friendly, and approachable language."
        }
        
        system_prompt = f"""You are helping communication-disabled users express themselves.
Convert the given intent or meaning into natural human language.
{style_instructions.get(style, style_instructions['natural'])}

Output ONLY the generated text, no explanations or quotes."""

        user_prompt = f"Express this intent as natural speech: \"{intent}\""
        if context:
            user_prompt += f"\n\nContext: {context}"
        
        # Use fallback if OpenAI not configured
        if not self._has_openai:
            # Simple fallback: just return the intent with basic formatting
            return intent.capitalize() if intent else ""
        
        response = self.clients.openai_client.chat.completions.create(
            model=self.settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    async def interpret_gesture_sequence(
        self,
        gestures: List[str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Interpret a sequence of detected gestures into intent and text.
        
        Args:
            gestures: List of detected gesture labels
            context: Current context
            
        Returns:
            Dict with intent, text, and confidence
        """
        system_prompt = """You are an expert at interpreting sign language and gesture patterns.
Given a sequence of detected gestures, determine the user's intent and generate natural text.

Return a JSON object with:
- "intent": A short label for the intent (e.g., "request_water", "greeting", "need_help")
- "text": Natural language expression of the intent
- "confidence": Your confidence level (0.0 to 1.0)

Return ONLY the JSON object."""

        gesture_str = " -> ".join(gestures)
        user_prompt = f"Interpret these gestures: {gesture_str}"
        if context:
            user_prompt += f"\nContext: {context}"
        
        # Use fallback if OpenAI not configured
        if not self._has_openai:
            # Simple gesture mapping fallback
            gesture_meanings = {
                "wave": "Hello",
                "point": "Look at that",
                "thumbs_up": "Yes, I agree",
                "thumbs_down": "No, I disagree",
                "open_palm": "Stop",
                "fist": "I need help",
                "peace": "I'm okay",
                "heart": "I love you",
            }
            for gesture in gestures:
                gesture_lower = gesture.lower()
                for key, value in gesture_meanings.items():
                    if key in gesture_lower:
                        return {"intent": key, "text": value, "confidence": 0.7}
            return {"intent": "gesture", "text": " ".join(gestures), "confidence": 0.5}
        
        response = self.clients.openai_client.chat.completions.create(
            model=self.settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except (json.JSONDecodeError, IndexError):
            return {
                "intent": "unknown",
                "text": "I'm trying to communicate something",
                "confidence": 0.3
            }
    
    async def interpret_behavior_pattern(
        self,
        behavior_description: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Interpret behavioral patterns into intent and text.
        
        Args:
            behavior_description: Description of the behavioral pattern
            context: Current context
            
        Returns:
            Dict with intent, text, and confidence
        """
        system_prompt = """You are an expert at interpreting behavioral patterns from users with communication disabilities.
Given a description of behavioral patterns (touch interactions, eye movements, facial expressions),
determine the user's likely intent and generate appropriate text.

Return a JSON object with:
- "intent": A short label for the intent
- "text": Natural language expression of the intent
- "confidence": Your confidence level (0.0 to 1.0)
- "behavior_type": Type of behavior (e.g., "touch", "eye", "facial", "combined")

Return ONLY the JSON object."""

        user_prompt = f"Interpret this behavior: {behavior_description}"
        if context:
            user_prompt += f"\nContext: {context}"
        
        # Use fallback if OpenAI not configured
        if not self._has_openai:
            # Simple behavior interpretation fallback
            behavior_lower = behavior_description.lower()
            
            # Handle facial expressions
            if "facial expression" in behavior_lower or "expression:" in behavior_lower:
                # Extract the detected expression
                if "happy" in behavior_lower:
                    return {"intent": "happy", "text": "I am feeling happy", "confidence": 0.8, "behavior_type": "facial"}
                if "sad" in behavior_lower:
                    return {"intent": "sad", "text": "I am feeling sad", "confidence": 0.8, "behavior_type": "facial"}
                if "angry" in behavior_lower:
                    return {"intent": "angry", "text": "I am feeling frustrated", "confidence": 0.8, "behavior_type": "facial"}
                if "surprised" in behavior_lower:
                    return {"intent": "surprised", "text": "That surprised me!", "confidence": 0.8, "behavior_type": "facial"}
                if "fearful" in behavior_lower or "fear" in behavior_lower:
                    return {"intent": "fearful", "text": "I am feeling scared", "confidence": 0.8, "behavior_type": "facial"}
                if "disgusted" in behavior_lower:
                    return {"intent": "disgusted", "text": "I do not like this", "confidence": 0.8, "behavior_type": "facial"}
                if "tired" in behavior_lower:
                    return {"intent": "tired", "text": "I am feeling tired", "confidence": 0.8, "behavior_type": "facial"}
                if "neutral" in behavior_lower:
                    return {"intent": "neutral", "text": "I am here and listening", "confidence": 0.7, "behavior_type": "facial"}
                # Default for unknown expressions
                return {"intent": "neutral", "text": "I am here and listening", "confidence": 0.5, "behavior_type": "facial"}
            
            if "tap" in behavior_lower or "touch" in behavior_lower:
                if "rapid" in behavior_lower or "multiple" in behavior_lower:
                    return {"intent": "urgent", "text": "I need attention", "confidence": 0.6, "behavior_type": "touch"}
                return {"intent": "select", "text": "I want this", "confidence": 0.6, "behavior_type": "touch"}
            if "swipe" in behavior_lower:
                if "left" in behavior_lower:
                    return {"intent": "reject", "text": "No, not this", "confidence": 0.6, "behavior_type": "touch"}
                if "right" in behavior_lower:
                    return {"intent": "accept", "text": "Yes, I like this", "confidence": 0.6, "behavior_type": "touch"}
            if "hold" in behavior_lower or "press" in behavior_lower:
                return {"intent": "emphasize", "text": "This is important", "confidence": 0.6, "behavior_type": "touch"}
            return {"intent": "interaction", "text": behavior_description, "confidence": 0.4, "behavior_type": "touch"}
        
        response = self.clients.openai_client.chat.completions.create(
            model=self.settings.azure_openai_deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        try:
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except (json.JSONDecodeError, IndexError):
            return {
                "intent": "unknown",
                "text": "I'm trying to express something",
                "confidence": 0.3,
                "behavior_type": "unknown"
            }


# Singleton instance
_openai_service: Optional[OpenAIService] = None


def get_openai_service() -> OpenAIService:
    """Get the OpenAI service singleton."""
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service
