"""
Groq AI Service
Uses Groq API for fast LLM inference and Whisper speech-to-text.
"""

import base64
import io
import json
import tempfile
import os
from typing import Optional, Tuple, Dict, Any, List
import httpx

from app.core.config import get_settings


class GroqService:
    """Service for Groq AI operations - LLM and Whisper."""
    
    BASE_URL = "https://api.groq.com/openai/v1"
    
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.groq_api_key
        
        if not self.api_key:
            print("Warning: GROQ_API_KEY not configured. Groq services will not work.")
        else:
            print("Groq AI service initialized successfully!")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def speech_to_text(
        self,
        audio_data: Optional[bytes] = None,
        audio_base64: Optional[str] = None,
        language: str = "en"
    ) -> Tuple[str, float]:
        """
        Convert speech to text using Groq's Whisper API.
        
        Args:
            audio_data: Raw audio bytes
            audio_base64: Base64 encoded audio
            language: Language code
            
        Returns:
            Tuple of (text, confidence)
        """
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not configured. Set it in .env file.")
        
        # Decode base64 if provided
        if audio_base64 and not audio_data:
            try:
                audio_data = base64.b64decode(audio_base64)
            except Exception as e:
                raise ValueError(f"Invalid base64 audio: {e}")
        
        if not audio_data:
            raise ValueError("No audio data provided")
        
        if len(audio_data) < 100:
            raise ValueError(f"Audio too small ({len(audio_data)} bytes)")
        
        print(f"Processing audio with Groq Whisper: {len(audio_data)} bytes")
        
        # Convert audio to proper format if needed
        audio_data = await self._prepare_audio(audio_data)
        
        # Save to temp file for upload
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(temp_path, "rb") as audio_file:
                    files = {
                        "file": ("audio.mp3", audio_file, "audio/mpeg"),
                    }
                    data = {
                        "model": "whisper-large-v3",
                        "language": language[:2] if len(language) > 2 else language,
                        "response_format": "verbose_json"
                    }
                    
                    response = await client.post(
                        f"{self.BASE_URL}/audio/transcriptions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files=files,
                        data=data
                    )
                
                if response.status_code != 200:
                    error_text = response.text
                    print(f"Groq Whisper error: {error_text}")
                    raise ValueError(f"Groq API error: {error_text}")
                
                result = response.json()
                text = result.get("text", "").strip()
                
                # Calculate confidence from segments if available
                confidence = 0.95
                segments = result.get("segments", [])
                if segments:
                    no_speech_probs = [s.get("no_speech_prob", 0.0) for s in segments]
                    if no_speech_probs:
                        avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)
                        confidence = max(0.1, 1.0 - avg_no_speech)
                
                print(f"Groq Whisper result: '{text}' (confidence: {confidence:.2f})")
                return text, confidence
                
        except httpx.TimeoutException:
            raise ValueError("Groq API timeout. Please try again.")
        except Exception as e:
            print(f"Groq Whisper error: {e}")
            raise ValueError(f"Speech recognition failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def _prepare_audio(self, audio_data: bytes) -> bytes:
        """Prepare audio for Groq Whisper API."""
        try:
            from pydub import AudioSegment
            
            audio_io = io.BytesIO(audio_data)
            
            # Try common formats
            formats = ['m4a', 'aac', 'mp4', 'webm', 'ogg', 'mp3', 'wav']
            
            for fmt in formats:
                try:
                    audio_io.seek(0)
                    audio = AudioSegment.from_file(audio_io, format=fmt)
                    
                    # Export as MP3
                    output = io.BytesIO()
                    audio.export(output, format="mp3")
                    print(f"Converted audio from {fmt} to MP3")
                    return output.getvalue()
                except:
                    continue
            
            # Try auto-detect
            audio_io.seek(0)
            audio = AudioSegment.from_file(audio_io)
            output = io.BytesIO()
            audio.export(output, format="mp3")
            return output.getvalue()
            
        except Exception as e:
            print(f"Audio conversion warning: {e}")
            return audio_data
    
    async def process_text(
        self,
        text: str,
        task: str = "simplify",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process text using Groq LLM for various tasks.
        
        Args:
            text: Input text to process
            task: Task type (simplify, emotion, symbols, full)
            context: Additional context
            
        Returns:
            Structured response with processed text
        """
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not configured")
        
        # Build prompt based on task
        if task == "full":
            system_prompt = """You are an AI assistant for BridgeComm, an assistive communication app.
Your job is to help translate between speech/text and visual symbols for people with disabilities.

Given user input, respond with a JSON object containing:
1. "original_text": The original input
2. "simplified_text": A clearer, simpler version (max 10 words)
3. "emotion": The detected emotion (happy, sad, angry, confused, neutral, excited, worried)
4. "symbols": Array of emoji symbols representing the message (max 5)
5. "intent": What the user wants (greeting, request, question, statement, help, emergency)
6. "response": A helpful, empathetic response (max 15 words)

ONLY respond with valid JSON, no additional text."""
        
        elif task == "simplify":
            system_prompt = """Simplify the following text for someone who needs clear, simple communication.
Respond ONLY with the simplified text, max 10 words. Be direct and clear."""
        
        elif task == "emotion":
            system_prompt = """Detect the emotion in the following text.
Respond with ONLY one word: happy, sad, angry, confused, neutral, excited, or worried."""
        
        elif task == "symbols":
            system_prompt = """Convert the following text to emoji symbols that represent the meaning.
Respond with ONLY 3-5 relevant emojis, nothing else."""
        
        elif task == "intent":
            system_prompt = """Identify the intent of the following text.
Respond with ONLY one word: greeting, request, question, statement, help, or emergency."""
        
        elif task == "keywords":
            system_prompt = """Extract the most important keywords from the text that can be represented as visual symbols/pictograms.
Focus on: nouns (objects, people, places), verbs (actions), and adjectives (descriptors).
Skip common words like: a, the, is, are, I, you, we, to, for, in, on, with, etc.

Respond ONLY with a JSON object: {"keywords": ["word1", "word2", "word3", ...]}
Extract up to 10 meaningful keywords. Return keywords in the order they appear in the sentence."""
        
        else:
            system_prompt = "You are a helpful assistant for an accessibility communication app."
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
        
        messages.append({"role": "user", "content": text})
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json={
                        "model": "llama-3.1-8b-instant",  # Fast model (updated from deprecated llama3-8b-8192)
                        "messages": messages,
                        "temperature": 0.3,  # More deterministic
                        "max_tokens": 500,
                    }
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    print(f"Groq LLM error: {error_text}")
                    raise ValueError(f"Groq API error: {error_text}")
                
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Parse based on task
                if task == "full":
                    try:
                        # Try to parse JSON response
                        parsed = json.loads(content)
                        return parsed
                    except json.JSONDecodeError:
                        # Fallback structure
                        return {
                            "original_text": text,
                            "simplified_text": content[:100],
                            "emotion": "neutral",
                            "symbols": ["💬"],
                            "intent": "statement",
                            "response": content[:100]
                        }
                
                elif task == "simplify":
                    return {"simplified_text": content}
                
                elif task == "emotion":
                    emotion = content.lower().strip()
                    valid_emotions = ["happy", "sad", "angry", "confused", "neutral", "excited", "worried"]
                    if emotion not in valid_emotions:
                        emotion = "neutral"
                    return {"emotion": emotion}
                
                elif task == "symbols":
                    # Extract emojis from response
                    import re
                    emojis = re.findall(r'[\U0001F300-\U0001F9FF]|[\U00002600-\U000027BF]', content)
                    return {"symbols": emojis[:5] if emojis else ["💬"]}
                
                elif task == "intent":
                    intent = content.lower().strip()
                    valid_intents = ["greeting", "request", "question", "statement", "help", "emergency"]
                    if intent not in valid_intents:
                        intent = "statement"
                    return {"intent": intent}
                
                elif task == "keywords":
                    # Parse JSON response with keywords
                    try:
                        parsed = json.loads(content)
                        keywords = parsed.get("keywords", [])
                        if isinstance(keywords, list):
                            return {"keywords": keywords}
                        return {"keywords": []}
                    except json.JSONDecodeError:
                        # Try to extract words from response
                        import re
                        words = re.findall(r'\b[a-zA-Z]{2,}\b', content)
                        return {"keywords": words[:10]}
                
                return {"result": content}
                
        except httpx.TimeoutException:
            raise ValueError("Groq API timeout. Please try again.")
        except Exception as e:
            print(f"Groq LLM error: {e}")
            raise ValueError(f"Text processing failed: {str(e)}")
    
    async def text_to_symbols(
        self,
        text: str,
        simplify: bool = True,
        max_symbols: int = 5
    ) -> Dict[str, Any]:
        """
        Convert text to simplified text and symbols.
        
        Args:
            text: Input text
            simplify: Whether to simplify the text
            max_symbols: Maximum number of symbols
            
        Returns:
            Dict with simplified_text, symbols, emotion
        """
        result = await self.process_text(text, task="full")
        
        # Ensure we have all fields
        return {
            "original_text": result.get("original_text", text),
            "simplified_text": result.get("simplified_text", text),
            "emotion": result.get("emotion", "neutral"),
            "symbols": result.get("symbols", ["💬"])[:max_symbols],
            "intent": result.get("intent", "statement"),
            "response": result.get("response", "")
        }
    
    async def generate_response(
        self,
        user_input: str,
        context: Optional[str] = None,
        user_emotion: Optional[str] = None
    ) -> str:
        """
        Generate an appropriate response for the user.
        
        Args:
            user_input: What the user said/typed
            context: Conversation context
            user_emotion: Detected user emotion
            
        Returns:
            Generated response text
        """
        system_prompt = f"""You are a compassionate AI assistant for BridgeComm, helping people with communication disabilities.

Your responses should be:
- Short (max 15 words)
- Clear and simple
- Empathetic and supportive
- Helpful and actionable

User's detected emotion: {user_emotion or 'neutral'}
"""
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
            messages.append({"role": "assistant", "content": f"Previous context: {context}"})
        
        messages.append({"role": "user", "content": user_input})
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 100,
                    }
                )
                
                if response.status_code != 200:
                    raise ValueError(f"Groq API error: {response.text}")
                
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
                
        except Exception as e:
            print(f"Groq response generation error: {e}")
            return "I understand. How can I help you?"
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        model: str = "llama-3.1-8b-instant"
    ) -> Dict[str, Any]:
        """
        General chat completion endpoint.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
            model: Model to use
            
        Returns:
            Dict with 'text' key containing response, or 'choices' from API
        """
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not configured")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._get_headers(),
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    print(f"Groq chat completion error: {error_text}")
                    raise ValueError(f"Groq API error: {error_text}")
                
                result = response.json()
                text = result["choices"][0]["message"]["content"].strip()
                
                return {
                    "text": text,
                    "choices": result.get("choices"),
                    "usage": result.get("usage")
                }
                
        except httpx.TimeoutException:
            raise ValueError("Groq API timeout. Please try again.")
        except Exception as e:
            print(f"Groq chat completion error: {e}")
            raise ValueError(f"Chat completion failed: {str(e)}")


# Singleton instance
_groq_service: Optional[GroqService] = None


def get_groq_service() -> GroqService:
    """Get Groq service singleton."""
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService()
    return _groq_service
