"""
Azure Speech Service
Handles speech-to-text and text-to-speech using Azure Cognitive Services.
"""

import base64
import io
import tempfile
import os
import wave
import struct
from typing import Optional, Tuple
import azure.cognitiveservices.speech as speechsdk

from app.core.azure_clients import get_azure_clients
from app.core.config import get_settings


def convert_audio_to_wav(audio_data: bytes) -> bytes:
    """
    Convert audio to WAV format that Azure Speech SDK can process.
    Supports multiple input formats including iOS/Android recordings.
    """
    import struct
    
    # Check if already a valid WAV file
    if len(audio_data) > 44 and audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
        # Validate WAV header
        try:
            # Check if format chunk exists
            fmt_offset = audio_data.find(b'fmt ')
            if fmt_offset != -1:
                # Read audio format (1 = PCM)
                audio_format = struct.unpack('<H', audio_data[fmt_offset+8:fmt_offset+10])[0]
                if audio_format == 1:  # PCM
                    print("Audio is valid PCM WAV, using as-is")
                    return audio_data
                else:
                    print(f"WAV has non-PCM format ({audio_format}), needs conversion")
        except Exception as e:
            print(f"WAV header validation failed: {e}")
    
    # Try to convert using pydub (handles M4A, AAC, MP4, WebM, etc.)
    try:
        from pydub import AudioSegment
        
        audio_io = io.BytesIO(audio_data)
        
        # Try common mobile recording formats
        formats_to_try = ['m4a', 'aac', 'mp4', 'caf', 'webm', 'ogg', 'mp3', 'wav', '3gp']
        
        for fmt in formats_to_try:
            try:
                audio_io.seek(0)
                audio = AudioSegment.from_file(audio_io, format=fmt)
                
                # Convert to Azure-compatible WAV: 16kHz, mono, 16-bit PCM
                audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                
                wav_io = io.BytesIO()
                audio.export(wav_io, format='wav')
                wav_data = wav_io.getvalue()
                print(f"Successfully converted from {fmt} format to WAV")
                return wav_data
            except Exception as e:
                continue
        
        # Try auto-detect as last resort
        audio_io.seek(0)
        audio = AudioSegment.from_file(audio_io)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        wav_io = io.BytesIO()
        audio.export(wav_io, format='wav')
        print("Successfully converted audio using auto-detection")
        return wav_io.getvalue()
        
    except ImportError:
        # Help operators fix missing dependency
        raise ValueError("pydub is required for audio conversion. Install pydub and ensure ffmpeg is available in PATH.")
    except Exception as e:
        print(f"Audio conversion with pydub failed: {e}")
    
    # If conversion fails, try to create a minimal valid WAV from raw PCM data
    print("Warning: Returning original audio data, may not be compatible")
    return audio_data


class SpeechService:
    """Service for Azure Speech operations."""
    
    def __init__(self):
        self.clients = get_azure_clients()
        self.settings = get_settings()
    
    async def speech_to_text(
        self,
        audio_data: Optional[bytes] = None,
        audio_base64: Optional[str] = None,
        language: str = "en-US"
    ) -> Tuple[str, float]:
        """
        Convert speech audio to text using Azure Speech-to-Text.
        
        Args:
            audio_data: Raw audio bytes
            audio_base64: Base64 encoded audio string
            language: Language code for recognition
            
        Returns:
            Tuple of (recognized_text, confidence_score)
        """
        # Decode base64 if provided
        if audio_base64 and not audio_data:
            try:
                audio_data = base64.b64decode(audio_base64)
            except Exception as e:
                raise ValueError(f"Invalid base64 audio data: {e}")
        
        if not audio_data:
            raise ValueError("No audio data provided")
        
        if len(audio_data) < 100:
            raise ValueError(f"Audio data too small ({len(audio_data)} bytes). Minimum 100 bytes required.")
        
        print(f"Received audio: {len(audio_data)} bytes, header: {audio_data[:min(20, len(audio_data))].hex()}")
        
        # Convert audio to WAV format if needed
        try:
            audio_data = convert_audio_to_wav(audio_data)
            print(f"After conversion: {len(audio_data)} bytes, header: {audio_data[:20].hex()}")
        except ValueError as e:
            # Re-raise ValueError from pydub missing
            raise
        except Exception as e:
            print(f"Audio conversion error: {e}")
            raise ValueError(f"Failed to convert audio format. Ensure audio is in supported format (WAV, M4A, MP4, etc.): {e}")
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            # Verify Azure Speech credentials
            if not self.settings.azure_speech_key:
                raise ValueError("Azure Speech API key not configured. Set AZURE_SPEECH_KEY in .env file.")
            
            # Configure speech recognition
            speech_config = speechsdk.SpeechConfig(
                subscription=self.settings.azure_speech_key,
                region=self.settings.azure_speech_region
            )
            speech_config.speech_recognition_language = language
            
            # Create audio config from file
            audio_config = speechsdk.audio.AudioConfig(filename=temp_path)
            
            # Create recognizer
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Perform recognition
            result = recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Extract confidence from detailed results if available
                confidence = 0.95  # Default high confidence
                if hasattr(result, 'best'):
                    confidence = result.best[0].confidence if result.best else 0.95
                print(f"Speech recognized: '{result.text}' (confidence: {confidence})")
                return result.text, confidence
            
            elif result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized from audio")
                return "", 0.0
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                error_details = cancellation.error_details if hasattr(cancellation, 'error_details') else "Unknown error"
                print(f"Speech recognition canceled: {cancellation.reason} - {error_details}")
                
                # Provide helpful error messages
                if "authentication" in error_details.lower() or "forbidden" in error_details.lower():
                    raise Exception(f"Azure Speech authentication failed. Check AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env: {error_details}")
                elif "timeout" in error_details.lower():
                    raise Exception(f"Speech recognition timed out. Audio may be too long or connection issues: {error_details}")
                else:
                    raise Exception(f"Speech recognition error: {error_details}")
            
            print("Unknown speech recognition result")
            return "", 0.0
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def text_to_speech(
        self,
        text: str,
        voice: str = "en-US-JennyNeural",
        rate: float = 1.0,
        pitch: float = 1.0
    ) -> Tuple[bytes, int]:
        """
        Convert text to speech using Azure Text-to-Speech.
        
        Args:
            text: Text to convert
            voice: Voice name to use
            rate: Speech rate multiplier
            pitch: Voice pitch multiplier
            
        Returns:
            Tuple of (audio_bytes, duration_ms)
        """
        speech_config = speechsdk.SpeechConfig(
            subscription=self.settings.azure_speech_key,
            region=self.settings.azure_speech_region
        )
        
        # Set output format to MP3
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        
        # Create SSML for better control
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice}">
                <prosody rate="{rate}" pitch="{self._pitch_to_percent(pitch)}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        # Create synthesizer (no audio output - we want the bytes)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None
        )
        
        # Synthesize
        result = synthesizer.speak_ssml_async(ssml).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            # Estimate duration (rough calculation based on text length)
            duration_ms = int(len(text) * 60)  # ~60ms per character
            return audio_data, duration_ms
        
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise Exception(f"Speech synthesis canceled: {cancellation.reason}")
        
        raise Exception("Speech synthesis failed")
    
    def _pitch_to_percent(self, pitch: float) -> str:
        """Convert pitch multiplier to SSML percentage format."""
        if pitch == 1.0:
            return "default"
        percent = int((pitch - 1.0) * 100)
        return f"{percent:+d}%"


# Singleton instance
_speech_service: Optional[SpeechService] = None


def get_speech_service() -> SpeechService:
    """Get the speech service singleton."""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService()
    return _speech_service
