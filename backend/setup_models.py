#!/usr/bin/env python3
"""
BridgeComm Model Setup Script

Downloads all required ML models for the BridgeComm backend.
Run this after cloning the repository to set up the AI models.

Usage:
    python setup_models.py           # Download all models
    python setup_models.py --verify  # Only verify existing models
    python setup_models.py --force   # Re-download all models
"""

import os
import sys
import urllib.request
import hashlib
import shutil
from pathlib import Path
import argparse
import ssl

# Disable SSL verification for some downloads (use with caution)
ssl._create_default_https_context = ssl._create_unverified_context

# Base paths
BACKEND_DIR = Path(__file__).parent
MODELS_DIR = BACKEND_DIR / "models"
SIGN_LANGUAGE_DIR = MODELS_DIR / "sign_language"
SIGNVLM_DIR = MODELS_DIR / "signvlm"

# Model definitions with download URLs
MODELS = {
    # MediaPipe Gesture Recognizer (required for hand gesture recognition)
    "gesture_recognizer.task": {
        "path": MODELS_DIR / "gesture_recognizer.task",
        "url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
        "description": "MediaPipe Gesture Recognizer",
        "size_mb": 5,
        "required": True,
    },
    
    # ASL CNN Model (trained for ASL alphabet recognition)
    "asl_cnn_model.h5": {
        "path": MODELS_DIR / "asl_cnn_model.h5",
        "url": None,  # Must be trained locally
        "description": "ASL Alphabet CNN Model (A-Z + special)",
        "size_mb": 50,
        "required": False,
        "training_script": "training/train_asl_cnn.py",
    },
    
    # ASL TFLite Model (optimized version)
    "asl_cnn_model.tflite": {
        "path": MODELS_DIR / "asl_cnn_model.tflite",
        "url": None,  # Generated from .h5
        "description": "ASL Alphabet TFLite Model",
        "size_mb": 15,
        "required": False,
        "training_script": "training/train_asl_cnn.py",
    },
    
    # Emotion Detection Model
    "emotion_model.h5": {
        "path": MODELS_DIR / "emotion_model.h5",
        "url": None,  # Must be trained locally
        "description": "Emotion Detection Model (7 emotions)",
        "size_mb": 30,
        "required": False,
        "training_script": "training/train_emotion_model.py",
    },
    
    # Emotion TFLite Model
    "emotion_model.tflite": {
        "path": MODELS_DIR / "emotion_model.tflite",
        "url": None,  # Generated from .h5
        "description": "Emotion Detection TFLite Model",
        "size_mb": 10,
        "required": False,
        "training_script": "training/train_emotion_model.py",
    },
    
    # Sign Language Video Models (for video-based sign recognition)
    "i3d_wlasl100_rgb.pth": {
        "path": SIGN_LANGUAGE_DIR / "i3d_wlasl100_rgb.pth",
        "url": None,  # Large model, needs manual download
        "description": "I3D Video Sign Language Model",
        "size_mb": 150,
        "required": False,
        "manual_download": "See backend/training/README.md for training instructions",
    },
    
    "pose_lstm_wlasl100.pth": {
        "path": SIGN_LANGUAGE_DIR / "pose_lstm_wlasl100.pth",
        "url": None,
        "description": "Pose LSTM Sign Language Model",
        "size_mb": 10,
        "required": False,
        "manual_download": "See backend/training/README.md for training instructions",
    },
    
    # SignVLM CLIP Model (very large)
    "ViT-L-14.pt": {
        "path": SIGNVLM_DIR / "ViT-L-14.pt",
        "url": None,  # Too large, manual download required
        "description": "CLIP ViT-L/14 Model for SignVLM",
        "size_mb": 890,
        "required": False,
        "manual_download": "Download from OpenAI CLIP repository if using SignVLM features",
    },
}

# Class mapping files (JSON configs - always needed)
CLASS_MAPPINGS = {
    "asl_class_mapping.json": {
        "path": MODELS_DIR / "asl_class_mapping.json",
        "content": '''{
    "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F",
    "6": "G", "7": "H", "8": "I", "9": "J", "10": "K", "11": "L",
    "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R",
    "18": "S", "19": "T", "20": "U", "21": "V", "22": "W", "23": "X",
    "24": "Y", "25": "Z", "26": "del", "27": "nothing", "28": "space"
}''',
        "description": "ASL Alphabet Class Mapping",
    },
    "emotion_class_mapping.json": {
        "path": MODELS_DIR / "emotion_class_mapping.json",
        "content": '''{
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "neutral",
    "5": "sad",
    "6": "surprise"
}''',
        "description": "Emotion Class Mapping",
    },
    "wlasl100_classes.json": {
        "path": SIGNVLM_DIR / "wlasl100_classes.json",
        "content": '''{"classes": ["hello", "thank you", "please", "sorry", "yes", "no", "help", "goodbye", "love", "friend"]}''',
        "description": "WLASL 100 Sign Classes",
    },
}


def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_status(name, status, details=""):
    icons = {"ok": "✓", "missing": "✗", "download": "↓", "skip": "○", "warn": "⚠"}
    icon = icons.get(status, "?")
    color_reset = ""
    print(f"  [{icon}] {name}" + (f" - {details}" if details else ""))


def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    try:
        print(f"\n  Downloading: {desc}")
        print(f"  URL: {url}")
        print(f"  Destination: {dest_path}")
        
        # Create parent directory
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def verify_models(force=False):
    """Verify or download all required models."""
    print_header("BridgeComm Model Setup")
    
    results = {"ok": 0, "downloaded": 0, "missing": 0, "optional_missing": 0}
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SIGN_LANGUAGE_DIR.mkdir(parents=True, exist_ok=True)
    SIGNVLM_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process class mappings first (always create)
    print("\n📋 Class Mapping Files:")
    for name, info in CLASS_MAPPINGS.items():
        path = info["path"]
        if not path.exists() or force:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(info["content"])
            print_status(name, "ok", "Created")
        else:
            print_status(name, "ok", "Exists")
        results["ok"] += 1
    
    # Process ML models
    print("\n🤖 ML Models:")
    for name, info in MODELS.items():
        path = info["path"]
        required = info.get("required", False)
        
        if path.exists() and not force:
            size_mb = path.stat().st_size / (1024 * 1024)
            print_status(name, "ok", f"Exists ({size_mb:.1f} MB)")
            results["ok"] += 1
            continue
        
        # Try to download if URL available
        if info.get("url"):
            if download_file(info["url"], path, info["description"]):
                print_status(name, "download", "Downloaded successfully")
                results["downloaded"] += 1
            else:
                if required:
                    print_status(name, "missing", "REQUIRED - Download failed!")
                    results["missing"] += 1
                else:
                    print_status(name, "warn", "Optional - Download failed")
                    results["optional_missing"] += 1
        else:
            # No URL available
            if required:
                print_status(name, "missing", f"REQUIRED - {info.get('manual_download', 'Manual setup needed')}")
                results["missing"] += 1
            else:
                training = info.get("training_script", "")
                manual = info.get("manual_download", "")
                hint = f"Train with {training}" if training else manual
                print_status(name, "skip", f"Optional - {hint}")
                results["optional_missing"] += 1
    
    # Summary
    print_header("Summary")
    print(f"  ✓ Ready:     {results['ok']}")
    print(f"  ↓ Downloaded: {results['downloaded']}")
    print(f"  ✗ Missing (required): {results['missing']}")
    print(f"  ○ Missing (optional): {results['optional_missing']}")
    
    if results["missing"] > 0:
        print("\n⚠️  Some required models are missing!")
        print("   The backend may not work correctly.")
        return False
    
    if results["optional_missing"] > 0:
        print("\n📝 Some optional models are missing.")
        print("   Core features will work. Advanced features may be limited.")
        print("\n   To train custom models, see: backend/training/README.md")
    
    print("\n✅ Setup complete! You can now run the backend server.")
    return True


def main():
    parser = argparse.ArgumentParser(description="BridgeComm Model Setup")
    parser.add_argument("--verify", action="store_true", help="Only verify existing models")
    parser.add_argument("--force", action="store_true", help="Re-download all models")
    args = parser.parse_args()
    
    if args.verify:
        print("Verification mode: checking existing models only")
    
    success = verify_models(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
