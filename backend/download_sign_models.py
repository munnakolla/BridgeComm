#!/usr/bin/env python3
"""
Standalone script to download sign language models.
This script has minimal dependencies to avoid TensorFlow import issues.
"""

import os
import urllib.request
import hashlib
from pathlib import Path

# Model configurations
MODELS_DIR = Path(__file__).parent / "models" / "sign_language"

MODEL_SOURCES = {
    "i3d_wlasl100_rgb.pth": {
        "url": "https://github.com/microsoft/computervision-recipes/releases/download/kinetics400_model/i3d_r50_kinetics.pth",
        "description": "I3D model pretrained on Kinetics-400 (to be fine-tuned for WLASL)",
        "size_mb": 150,
    },
    "pose_lstm_wlasl100.pth": {
        "url": None,  # We'll create a placeholder
        "description": "Pose-based LSTM model for sign language",
        "size_mb": 10,
    }
}

def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading {desc}...")
        print(f"  URL: {url}")
        print(f"  Destination: {dest_path}")
        
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # New line after progress
        print(f"  ✓ Downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False

def create_placeholder_model(dest_path: Path, model_type: str) -> bool:
    """Create a placeholder file indicating manual download is needed."""
    try:
        placeholder_content = f"""# Placeholder for {model_type}
# 
# To get actual model weights, you can:
# 1. Train the model using the training scripts in backend/training/
# 2. Download pretrained weights from Hugging Face or similar
# 3. Use the model architecture with random initialization (for testing)
#
# The application will work in demo mode with placeholder weights.
# Model type: {model_type}
"""
        dest_path.write_text(placeholder_content)
        print(f"  ✓ Created placeholder for {model_type}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create placeholder: {e}")
        return False

def main():
    print("=" * 60)
    print(" BridgeComm Sign Language Models Setup")
    print("=" * 60)
    print()
    
    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Models directory: {MODELS_DIR}")
    print()
    
    results = {"success": [], "failed": []}
    
    for model_name, model_info in MODEL_SOURCES.items():
        model_path = MODELS_DIR / model_name
        print(f"[{model_name}]")
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"  Already exists ({size_mb:.1f} MB)")
            results["success"].append(model_name)
            continue
        
        if model_info["url"]:
            success = download_file(model_info["url"], model_path, model_info["description"])
        else:
            print(f"  No direct download URL available")
            success = create_placeholder_model(model_path, model_info["description"])
        
        if success:
            results["success"].append(model_name)
        else:
            results["failed"].append(model_name)
        
        print()
    
    # Summary
    print("=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"  ✓ Successful: {len(results['success'])}")
    for name in results["success"]:
        print(f"      - {name}")
    
    if results["failed"]:
        print(f"  ✗ Failed: {len(results['failed'])}")
        for name in results["failed"]:
            print(f"      - {name}")
    
    print()
    print("Model setup complete!")
    print("You can now start the server with: uvicorn app.main:app --reload")
    
    return len(results["failed"]) == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
