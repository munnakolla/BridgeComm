"""
Download SignVLM Models
=======================

Script to download required models for SignVLM:
1. CLIP ViT-L/14 backbone weights
2. Optional: Pre-trained SignVLM checkpoints for various datasets

Usage:
    python download_signvlm_models.py

Models will be saved to: backend/models/signvlm/
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import ssl

# Disable SSL verification for some URLs (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context


# Model URLs
CLIP_MODELS = {
    "ViT-L-14": {
        "url": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "sha256": "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
        "size_mb": 890
    },
    "ViT-B-16": {
        "url": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "sha256": "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
        "size_mb": 335
    }
}

# WLASL-100 vocabulary (top 100 sign glosses)
WLASL100_VOCABULARY = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy",
    "cousin", "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all",
    "black", "cool", "finish", "hot", "like", "many", "mother", "now", "orange",
    "table", "thanksgiving", "what", "woman", "bed", "blue", "bowling", "can",
    "dog", "family", "friend", "hearing", "hello", "give", "language", "later",
    "man", "meet", "nothing", "right", "same", "shirt", "study", "think", "this",
    "thursday", "time", "tuesday", "want", "wednesday", "white", "why", "work",
    "birthday", "brown", "but", "bye", "change", "chicken", "cold", "color", "corn",
    "cow", "dance", "dark", "daughter", "day", "doctor", "eat", "enjoy", "face",
    "father", "fish", "forget", "friday", "full", "graduate", "green", "hat",
    "have", "he", "hungry", "hurt", "jacket", "know", "last", "learn", "letter",
    "lose", "medicine", "milk", "monday"
]

# Extended vocabulary for WLASL-300
WLASL300_VOCABULARY = WLASL100_VOCABULARY + [
    # Additional 200 glosses...
    "afternoon", "again", "airplane", "always", "animal", "apple", "aunt", "baby",
    "ball", "banana", "bathroom", "beautiful", "because", "bedroom", "bicycle",
    "bird", "boat", "body", "boy", "bread", "breakfast", "brother", "bus", "butter",
    "buy", "cake", "call", "car", "cat", "cheese", "child", "chocolate", "church",
    "city", "class", "clean", "clock", "close", "cloud", "coffee", "come", "cook",
    "cookie", "country", "cream", "cup", "daddy", "daughter", "dessert", "dinner",
    "dirty", "do", "door", "down", "dream", "dress", "drive", "early", "earth",
    "egg", "email", "evening", "every", "example", "excited", "excuse", "exercise",
    "expensive", "eye", "fall", "famous", "farm", "fast", "favorite", "feel",
    "finally", "find", "finger", "fire", "first", "floor", "flower", "fly",
    "food", "foot", "football", "forest", "free", "fruit", "fun", "funny",
    "future", "game", "garden", "get", "girl", "glass", "gloves", "god", "gold",
    # ... more classes can be added
]


def get_models_dir() -> Path:
    """Get the models directory path."""
    # This script is in: backend/app/services/sign_language/signvlm/
    # Models should be in: backend/models/signvlm/
    script_dir = Path(__file__).parent
    # Go up 4 levels to reach backend/
    backend_dir = script_dir.parent.parent.parent.parent
    models_dir = backend_dir / "models" / "signvlm"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def download_with_progress(url: str, destination: Path, expected_size_mb: int = None):
    """Download a file with progress bar."""
    
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            downloaded_mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()
        elif expected_size_mb:
            downloaded_mb = count * block_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {downloaded_mb:.1f}/{expected_size_mb} MB")
            sys.stdout.flush()
    
    try:
        urlretrieve(url, destination, progress_hook)
        print()  # New line after progress
        return True
    except URLError as e:
        print(f"\n  Error downloading: {e}")
        return False


def verify_checksum(file_path: Path, expected_sha256: str) -> bool:
    """Verify file SHA256 checksum."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest() == expected_sha256


def download_clip_backbone(model_name: str = "ViT-L-14", models_dir: Path = None) -> bool:
    """
    Download CLIP backbone weights.
    
    Args:
        model_name: CLIP model variant (ViT-L-14 or ViT-B-16)
        models_dir: Directory to save models
        
    Returns:
        True if successful
    """
    if models_dir is None:
        models_dir = get_models_dir()
    
    if model_name not in CLIP_MODELS:
        print(f"Unknown CLIP model: {model_name}")
        print(f"Available models: {list(CLIP_MODELS.keys())}")
        return False
    
    model_info = CLIP_MODELS[model_name]
    dest_path = models_dir / f"{model_name.replace('/', '-')}.pt"
    
    print(f"\n{'='*60}")
    print(f"Downloading CLIP {model_name} backbone")
    print(f"{'='*60}")
    print(f"  URL: {model_info['url']}")
    print(f"  Size: ~{model_info['size_mb']} MB")
    print(f"  Destination: {dest_path}")
    
    # Check if already exists and valid
    if dest_path.exists():
        print(f"  File already exists. Verifying checksum...")
        if verify_checksum(dest_path, model_info['sha256']):
            print(f"  ✓ Checksum valid. Skipping download.")
            return True
        else:
            print(f"  ✗ Checksum mismatch. Re-downloading...")
            dest_path.unlink()
    
    # Download
    print()
    success = download_with_progress(model_info['url'], dest_path, model_info['size_mb'])
    
    if success:
        # Verify
        print("  Verifying checksum...")
        if verify_checksum(dest_path, model_info['sha256']):
            print(f"  ✓ Download complete and verified!")
            return True
        else:
            print(f"  ✗ Checksum verification failed!")
            dest_path.unlink()
            return False
    
    return False


def create_class_mapping(models_dir: Path = None, dataset: str = "wlasl100"):
    """
    Create class mapping JSON file for the dataset.
    
    Args:
        models_dir: Directory to save mapping
        dataset: Dataset name (wlasl100, wlasl300)
    """
    if models_dir is None:
        models_dir = get_models_dir()
    
    vocab = WLASL100_VOCABULARY if dataset == "wlasl100" else WLASL300_VOCABULARY
    
    # Create index to class mapping
    mapping = {i: gloss for i, gloss in enumerate(vocab)}
    
    # Save as JSON
    mapping_path = models_dir / f"{dataset}_classes.json"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\n✓ Created class mapping: {mapping_path}")
    print(f"  Number of classes: {len(mapping)}")
    
    return mapping_path


def download_all_models():
    """Download all required models for SignVLM."""
    models_dir = get_models_dir()
    
    print("\n" + "="*60)
    print("SignVLM Model Downloader")
    print("="*60)
    print(f"Models will be saved to: {models_dir}")
    
    # Create class mappings
    print("\n[1/2] Creating class mappings...")
    create_class_mapping(models_dir, "wlasl100")
    
    # Download CLIP backbone
    print("\n[2/2] Downloading CLIP backbone...")
    success = download_clip_backbone("ViT-L-14", models_dir)
    
    if success:
        print("\n" + "="*60)
        print("✓ All downloads complete!")
        print("="*60)
        print("\nSignVLM is ready to use.")
        print("\nNote: For best results, you may want to fine-tune the model")
        print("on a sign language dataset. See the training scripts in the")
        print("original SignVLM repository: https://github.com/Hamzah-Luqman/signVLM")
    else:
        print("\n" + "="*60)
        print("✗ Some downloads failed")
        print("="*60)
        print("\nPlease check your internet connection and try again.")
    
    return success


if __name__ == "__main__":
    success = download_all_models()
    sys.exit(0 if success else 1)
