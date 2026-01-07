"""
Startup validation checks for BridgeComm Backend.
Validates that all required dependencies and configurations are available.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple


def check_ffmpeg() -> Tuple[bool, str]:
    """Check if ffmpeg is available in PATH."""
    import shutil
    
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return True, f"✓ ffmpeg found at {ffmpeg_path}"
    return False, "✗ ffmpeg NOT found. Install ffmpeg and add to PATH for audio conversion."


def check_pydub() -> Tuple[bool, str]:
    """Check if pydub can be imported."""
    try:
        import pydub
        # pydub doesn't have __version__ attribute
        return True, "✓ pydub available"
    except ImportError:
        return False, "✗ pydub NOT installed. Run: pip install pydub"


def check_mediapipe() -> Tuple[bool, str]:
    """Check if MediaPipe is available."""
    try:
        import mediapipe
        return True, f"✓ MediaPipe {mediapipe.__version__} available"
    except ImportError:
        return False, "✗ MediaPipe NOT installed. Run: pip install mediapipe"


def check_opencv() -> Tuple[bool, str]:
    """Check if OpenCV is available."""
    try:
        import cv2
        return True, f"✓ OpenCV {cv2.__version__} available"
    except ImportError:
        return False, "✗ OpenCV NOT installed. Run: pip install opencv-python-headless"


def check_gesture_model() -> Tuple[bool, str]:
    """Check if gesture recognizer model file exists."""
    model_path = Path(__file__).parent.parent.parent / "models" / "gesture_recognizer.task"
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        return True, f"✓ Gesture model found at {model_path} ({size_mb:.1f} MB)"
    return False, f"✗ Gesture model NOT found at {model_path}"


def check_deepface() -> Tuple[bool, str]:
    """Check if DeepFace is available for emotion detection."""
    try:
        import deepface
        return True, "✓ DeepFace available for emotion detection"
    except ImportError:
        return False, "⚠ DeepFace NOT installed (optional). Install for better emotion detection: pip install deepface"


def check_whisper() -> Tuple[bool, str]:
    """Check if local Whisper is available for speech recognition."""
    try:
        import whisper
        return True, "✓ Local Whisper available for FREE speech-to-text"
    except ImportError:
        return False, "✗ Whisper NOT installed. Run: pip install openai-whisper"


def check_gtts() -> Tuple[bool, str]:
    """Check if gTTS is available for text-to-speech."""
    try:
        from gtts import gTTS
        return True, "✓ gTTS available for FREE text-to-speech"
    except ImportError:
        return False, "✗ gTTS NOT installed. Run: pip install gtts"


def check_groq() -> Tuple[bool, str]:
    """Check if Groq API is configured."""
    from app.core.config import get_settings
    
    try:
        settings = get_settings()
        if settings.groq_api_key:
            # Mask the key for display
            masked_key = settings.groq_api_key[:8] + "..." + settings.groq_api_key[-4:]
            return True, f"✓ Groq API configured ({masked_key})"
        return False, "⚠ GROQ_API_KEY not set (optional - will use local Whisper)"
    except Exception as e:
        return False, f"⚠ Groq config error: {str(e)}"


def check_azure_credentials() -> Tuple[bool, str]:
    """Check if required Azure credentials are configured."""
    from app.core.config import get_settings
    
    try:
        settings = get_settings()
        issues = []
        
        # Required: Speech
        if not settings.azure_speech_key:
            issues.append("AZURE_SPEECH_KEY missing")
        
        # Optional but recommended
        if not settings.azure_vision_key:
            issues.append("AZURE_VISION_KEY missing (optional)")
        if not settings.azure_language_key:
            issues.append("AZURE_LANGUAGE_KEY missing (optional)")
        
        if not issues:
            return True, "✓ Azure credentials configured (Speech: ✓, Vision: ✓, Language: ✓)"
        elif "AZURE_SPEECH_KEY missing" not in issues:
            return True, f"✓ Azure Speech configured. Optional: {', '.join(issues)}"
        else:
            return False, f"✗ Missing required credentials: {', '.join(issues)}"
    except Exception as e:
        return False, f"✗ Error loading configuration: {str(e)}"


def run_startup_checks(verbose: bool = True) -> bool:
    """
    Run all startup checks and print results.
    
    Args:
        verbose: Print detailed results
        
    Returns:
        True if all critical checks pass, False otherwise
    """
    checks = [
        ("Groq API", check_groq, False),  # Optional but recommended
        ("Azure Credentials", check_azure_credentials, False),  # Optional now with Groq
        ("Local Whisper", check_whisper, False),  # Fallback for speech-to-text
        ("gTTS", check_gtts, True),  # Critical for text-to-speech
        ("ffmpeg", check_ffmpeg, True),  # Critical for audio
        ("pydub", check_pydub, True),  # Critical for audio
        ("MediaPipe", check_mediapipe, True),  # Critical for gestures
        ("OpenCV", check_opencv, True),  # Critical for gestures
        ("Gesture Model", check_gesture_model, True),  # Critical for gestures
        ("DeepFace", check_deepface, False),  # Optional
    ]
    
    results: List[Tuple[str, bool, str]] = []
    critical_failures = []
    
    for name, check_func, is_critical in checks:
        success, message = check_func()
        results.append((name, success, message))
        
        if not success and is_critical:
            critical_failures.append(name)
    
    if verbose:
        print("\n" + "="*70)
        print("BridgeComm Backend - Startup Validation")
        print("="*70)
        
        for name, success, message in results:
            print(f"\n{message}")
        
        print("\n" + "="*70)
        
        if critical_failures:
            print(f"⚠ CRITICAL FAILURES: {', '.join(critical_failures)}")
            print("⚠ Some features may not work correctly.")
            print("="*70 + "\n")
        else:
            print("✓ All critical checks passed!")
            print("="*70 + "\n")
    
    return len(critical_failures) == 0


if __name__ == "__main__":
    """Run checks when executed directly."""
    success = run_startup_checks(verbose=True)
    sys.exit(0 if success else 1)
