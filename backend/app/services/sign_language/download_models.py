"""
Model Download Script
=====================
Downloads pretrained sign language recognition models.

Models:
1. I3D-WLASL: Video-based sign recognition (from WLASL dataset)
2. WLASL Vocabulary: Word mappings for the model

Usage:
    python -m app.services.sign_language.download_models
"""

import os
import sys
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
import shutil

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.services.sign_language.config import (
    MODELS_DIR,
    WLASL_100_VOCAB,
    get_config,
    save_vocabulary
)


# Model download URLs
MODEL_URLS = {
    # I3D pretrained on WLASL-100 (from sumedhsp/Sign-Language-Recognition)
    "i3d_wlasl_100": {
        "url": "https://github.com/sumedhsp/Sign-Language-Recognition/releases/download/v1.0/i3d_pretrained_100.pt",
        "filename": "i3d_wlasl_100.pth",
        "description": "I3D model pretrained on WLASL-100 (100 ASL words)"
    },
    # WLASL vocabulary JSON
    "wlasl_vocab_100": {
        "url": None,  # Generated locally
        "filename": "wlasl_vocab_100.json",
        "description": "WLASL-100 vocabulary mapping"
    },
    # Pose-LSTM (will be created as placeholder)
    "pose_lstm": {
        "url": None,  # Created locally as demo
        "filename": "pose_lstm_wlasl.pth",
        "description": "Pose-LSTM model for landmark-based recognition"
    }
}


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """
    Download a file with progress indication.
    
    Args:
        url: URL to download from
        dest_path: Destination file path
        description: Description for progress display
        
    Returns:
        True if successful
    """
    try:
        print(f"Downloading: {description or dest_path.name}")
        print(f"  URL: {url}")
        print(f"  Destination: {dest_path}")
        
        # Create parent directories
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = count * block_size * 100 / total_size
                sys.stdout.write(f"\r  Progress: {percent:.1f}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, str(dest_path), progress_hook)
        print("\n  ✓ Download complete!")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Download failed: {e}")
        return False


def create_vocabulary_file(dest_path: Path) -> bool:
    """
    Create the WLASL vocabulary file.
    
    Args:
        dest_path: Destination file path
        
    Returns:
        True if successful
    """
    try:
        print(f"Creating vocabulary file: {dest_path.name}")
        save_vocabulary(WLASL_100_VOCAB, dest_path)
        print(f"  ✓ Vocabulary created with {len(WLASL_100_VOCAB)} words")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to create vocabulary: {e}")
        return False


def create_placeholder_model(dest_path: Path, model_type: str) -> bool:
    """
    Create a placeholder model file for demo purposes.
    
    In production, this should be replaced with actual pretrained weights.
    
    Args:
        dest_path: Destination file path
        model_type: Type of model to create
        
    Returns:
        True if successful
    """
    try:
        import torch
        import torch.nn as nn
        
        print(f"Creating placeholder model: {dest_path.name}")
        
        if model_type == "pose_lstm":
            # Create a simple LSTM model matching PoseLSTMService architecture
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size=126, hidden_size=256, num_classes=100):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, 
                                       batch_first=True, bidirectional=True)
                    self.attention = nn.Sequential(
                        nn.Linear(hidden_size * 2, 128),
                        nn.Tanh(),
                        nn.Linear(128, 1),
                        nn.Softmax(dim=1)
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(hidden_size * 2, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, num_classes)
                    )
                
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    attn = self.attention(lstm_out)
                    attended = torch.sum(attn * lstm_out, dim=1)
                    return self.classifier(attended)
            
            model = SimpleLSTM()
            
            # Save state dict
            torch.save(model.state_dict(), dest_path)
            print(f"  ✓ Placeholder Pose-LSTM model created")
            print(f"    Note: Replace with actual pretrained weights for production")
            return True
        
        print(f"  ✗ Unknown model type: {model_type}")
        return False
        
    except ImportError:
        print(f"  ✗ PyTorch not available. Install with: pip install torch")
        return False
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        return False


def download_all_models(force: bool = False) -> Dict[str, bool]:
    """
    Download all required models.
    
    Args:
        force: Force re-download even if files exist
        
    Returns:
        Dict of model names to success status
    """
    print("=" * 60)
    print("BridgeComm Sign Language Model Downloader")
    print("=" * 60)
    print(f"\nModels directory: {MODELS_DIR}\n")
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for model_name, model_info in MODEL_URLS.items():
        dest_path = MODELS_DIR / model_info["filename"]
        description = model_info["description"]
        
        print(f"\n[{model_name}]")
        print(f"  Description: {description}")
        
        # Skip if file exists and not forcing
        if dest_path.exists() and not force:
            print(f"  ✓ Already exists: {dest_path}")
            results[model_name] = True
            continue
        
        url = model_info.get("url")
        
        if url:
            # Download from URL
            results[model_name] = download_file(url, dest_path, description)
        elif model_name == "wlasl_vocab_100":
            # Create vocabulary file
            results[model_name] = create_vocabulary_file(dest_path)
        elif model_name == "pose_lstm":
            # Create placeholder model
            results[model_name] = create_placeholder_model(dest_path, "pose_lstm")
        else:
            print(f"  ✗ No download URL or generator for {model_name}")
            results[model_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    print(f"\n{success_count}/{total_count} models ready")
    
    if success_count < total_count:
        print("\nSome models failed to download. The service will use available models.")
        print("You can re-run this script to retry failed downloads.")
    
    return results


def verify_models() -> Dict[str, bool]:
    """
    Verify that all required models are present and valid.
    
    Returns:
        Dict of model names to validity status
    """
    print("=" * 60)
    print("Model Verification")
    print("=" * 60)
    
    results = {}
    
    for model_name, model_info in MODEL_URLS.items():
        dest_path = MODELS_DIR / model_info["filename"]
        
        exists = dest_path.exists()
        valid = False
        size_mb = 0
        
        if exists:
            size_bytes = dest_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            # Basic validation
            if model_info["filename"].endswith(".json"):
                try:
                    with open(dest_path, 'r') as f:
                        json.load(f)
                    valid = True
                except:
                    valid = False
            elif model_info["filename"].endswith(".pth"):
                # PyTorch files should be > 1KB
                valid = size_bytes > 1024
            else:
                valid = size_bytes > 0
        
        results[model_name] = valid
        
        status = "✓" if valid else "✗"
        size_str = f"({size_mb:.1f} MB)" if exists else "(missing)"
        print(f"  {status} {model_name} {size_str}")
    
    return results


# CLI entrypoint
def main():
    """Main entry point for CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download pretrained sign language recognition models"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Only verify models, don't download"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        results = verify_models()
        sys.exit(0 if all(results.values()) else 1)
    else:
        results = download_all_models(force=args.force)
        sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
