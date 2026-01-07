"""
Local Whisper Speech Service
Uses OpenAI Whisper locally (FREE - no API keys needed).
Falls back to gTTS for text-to-speech.
"""

import base64
import io
import tempfile
import os
from typing import Optional, Tuple
import warnings

# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


def get_whisper_model():
    """Load Whisper model lazily to save memory on startup."""
    global _whisper_model
    if '_whisper_model' not in globals() or _whisper_model is None:
        import whisper
        # Use "tiny" model for fastest recognition on CPU
        # Options: tiny, base, small, medium, large
        # tiny: fastest (~1s), good for short phrases (recommended for real-time)
        # base: balanced (~2-3s), good for longer speech
        # small: more accurate, slower (~5s)
        # medium/large: best accuracy but requires GPU
        print("Loading Whisper model (tiny)... This may take a moment on first run.")
        _whisper_model = whisper.load_model("tiny")
        print("Whisper model loaded successfully!")
    return _whisper_model


def convert_audio_for_whisper(audio_data: bytes) -> str:
    """
    Convert audio to format compatible with local Whisper.
    Returns path to temporary WAV file.
    """
    try:
        from pydub import AudioSegment
        
        audio_io = io.BytesIO(audio_data)
        
        # Try common mobile recording formats
        formats_to_try = ['m4a', 'aac', 'mp4', 'caf', 'webm', 'ogg', 'mp3', 'wav', '3gp']
        
        for fmt in formats_to_try:
            try:
                audio_io.seek(0)
                audio = AudioSegment.from_file(audio_io, format=fmt)
                
                # Convert to 16kHz mono WAV (optimal for Whisper)
                audio = audio.set_frame_rate(16000).set_channels(1)
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    audio.export(f.name, format="wav")
                    print(f"Converted from {fmt} to WAV for Whisper")
                    return f.name
            except Exception:
                continue
        
        # Try auto-detect
        audio_io.seek(0)
        audio = AudioSegment.from_file(audio_io)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio.export(f.name, format="wav")
            print("Converted audio to WAV using auto-detection")
            return f.name
        
    except ImportError:
        raise ValueError("pydub is required. Install: pip install pydub")
    except Exception as e:
        # If pydub fails, try saving raw data as wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            return f.name


class LocalWhisperService:
    """Speech service using local Whisper model (FREE)."""
    
    def __init__(self):
        self._model = None
        self._gtts_available = False
        self._check_gtts()
    
    def _check_gtts(self):
        """Check if gTTS is available for TTS."""
        try:
            from gtts import gTTS
            self._gtts_available = True
            print("gTTS available for text-to-speech")
        except ImportError:
            print("gTTS not available. Install with: pip install gtts")
            self._gtts_available = False
    
    @property
    def model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            self._model = get_whisper_model()
        return self._model
    
    async def speech_to_text(
        self,
        audio_data: Optional[bytes] = None,
        audio_base64: Optional[str] = None,
        language: str = "en"
    ) -> Tuple[str, float]:
        """
        Convert speech to text using local Whisper model.
        
        Args:
            audio_data: Raw audio bytes
            audio_base64: Base64 encoded audio
            language: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Tuple of (text, confidence)
        """
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
        
        print(f"Processing audio: {len(audio_data)} bytes")
        
        # Convert to WAV file
        temp_path = None
        try:
            temp_path = convert_audio_for_whisper(audio_data)
            
            # Use 2-letter language code
            lang = language[:2] if len(language) > 2 else language
            
            # Run Whisper transcription
            result = self.model.transcribe(
                temp_path,
                language=lang,
                fp16=False  # Use FP32 for CPU compatibility
            )
            
            text = result.get("text", "").strip()
            
            # Calculate confidence from segments
            confidence = 0.95  # Default high confidence
            segments = result.get("segments", [])
            if segments:
                # Average the no_speech_prob (lower = more confident)
                no_speech_probs = [s.get("no_speech_prob", 0.0) for s in segments]
                avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)
                confidence = max(0.1, 1.0 - avg_no_speech)
            
            print(f"Whisper result: '{text}' (confidence: {confidence:.2f})")
            return text, confidence
            
        except Exception as e:
            print(f"Whisper error: {e}")
            raise ValueError(f"Speech recognition failed: {str(e)}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def text_to_speech(
        self,
        text: str,
        voice: str = "en",
        speed: float = 1.0
    ) -> Tuple[bytes, int]:
        """
        Convert text to speech using gTTS (Google Text-to-Speech, free).
        
        Args:
            text: Text to convert
            voice: Language code (e.g., 'en', 'es', 'fr')
            speed: Speech speed (slow=True if < 0.8)
            
        Returns:
            Tuple of (audio_bytes, duration_ms)
        """
        if not self._gtts_available:
            raise ValueError("gTTS not installed. Run: pip install gtts")
        
        from gtts import gTTS
        
        try:
            # Extract language from voice parameter
            # Handle Azure-style voice names like "en-US-JennyNeural"
            if "-" in voice:
                lang = voice.split("-")[0]
            else:
                lang = voice[:2] if len(voice) > 2 else voice
            
            # Create TTS
            slow = speed < 0.8
            tts = gTTS(text=text, lang=lang, slow=slow)
            
            # Save to bytes
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_data = audio_io.getvalue()
            
            # Estimate duration (~60ms per character at normal speed)
            duration_ms = int(len(text) * 60 / speed)
            
            print(f"TTS generated: {len(audio_data)} bytes, ~{duration_ms}ms")
            return audio_data, duration_ms
            
        except Exception as e:
            raise ValueError(f"Text-to-speech failed: {e}")


# Singleton instance
_local_whisper_service: Optional[LocalWhisperService] = None


def get_local_whisper_service() -> LocalWhisperService:
    """Get local Whisper service singleton."""
    global _local_whisper_service
    if _local_whisper_service is None:
        _local_whisper_service = LocalWhisperService()
    return _local_whisper_service
