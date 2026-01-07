"""
Sign Language Model Configuration
=================================
Configuration for pretrained sign language recognition models.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Base paths
BACKEND_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = BACKEND_DIR / "models" / "sign_language"

# Create models directory if it doesn't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for a sign language model."""
    name: str
    model_path: Path
    checkpoint_url: Optional[str] = None
    vocab_path: Optional[Path] = None
    num_classes: int = 100
    input_size: tuple = (224, 224)
    num_frames: int = 32
    fps: int = 15
    enabled: bool = True


@dataclass
class SignLanguageConfig:
    """Master configuration for sign language recognition."""
    
    # I3D Model (Primary - video-based)
    i3d: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="I3D-WLASL",
        model_path=MODELS_DIR / "i3d_wlasl100_rgb.pth",  # Fixed: actual filename
        checkpoint_url="https://github.com/sumedhsp/Sign-Language-Recognition/releases/download/v1.0/i3d_pretrained_100.pt",
        vocab_path=None,  # Use embedded vocabulary
        num_classes=100,
        input_size=(224, 224),
        num_frames=64,  # I3D typically needs more frames
        fps=15,
        enabled=True
    ))
    
    # Pose-based LSTM (Low-latency fallback)
    pose_lstm: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="Pose-LSTM",
        model_path=MODELS_DIR / "pose_lstm_wlasl100.pth",  # Fixed: actual filename
        vocab_path=None,  # Use embedded vocabulary
        num_classes=100,
        input_size=(224, 224),
        num_frames=30,  # LSTM can work with fewer frames
        fps=15,
        enabled=True
    ))
    
    # Processing settings
    frame_extraction_fps: int = 15
    sliding_window_seconds: float = 2.0
    min_confidence_threshold: float = 0.3
    fallback_confidence_threshold: float = 0.6
    
    # Device settings
    use_cuda: bool = False  # CPU-first as per requirements
    device: str = "cpu"
    
    def __post_init__(self):
        """Set device based on CUDA availability if enabled."""
        if self.use_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
            except ImportError:
                pass


# WLASL-100 Vocabulary (most common ASL signs)
# This will be loaded from file if available, otherwise use this default
WLASL_100_VOCAB = {
    0: "book", 1: "drink", 2: "computer", 3: "before", 4: "chair",
    5: "go", 6: "clothes", 7: "who", 8: "candy", 9: "cousin",
    10: "deaf", 11: "fine", 12: "help", 13: "no", 14: "thin",
    15: "walk", 16: "year", 17: "yes", 18: "all", 19: "black",
    20: "cool", 21: "finish", 22: "hot", 23: "like", 24: "many",
    25: "mother", 26: "now", 27: "orange", 28: "table", 29: "thanksgiving",
    30: "what", 31: "woman", 32: "bed", 33: "blue", 34: "bowling",
    35: "can", 36: "dog", 37: "family", 38: "fish", 39: "graduate",
    40: "hat", 41: "hearing", 42: "hello", 43: "jacket", 44: "later",
    45: "man", 46: "meet", 47: "phone", 48: "pizza", 49: "school",
    50: "secretary", 51: "short", 52: "time", 53: "want", 54: "white",
    55: "afternoon", 56: "apple", 57: "basketball", 58: "birthday", 59: "brown",
    60: "but", 61: "cheat", 62: "city", 63: "cook", 64: "decide",
    65: "father", 66: "give", 67: "happy", 68: "hearing", 69: "hungry",
    70: "language", 71: "letter", 72: "medicine", 73: "money", 74: "movie",
    75: "name", 76: "nothing", 77: "nurse", 78: "pain", 79: "pig",
    80: "play", 81: "pretty", 82: "read", 83: "right", 84: "sad",
    85: "same", 86: "shirt", 87: "sister", 88: "sleep", 89: "slow",
    90: "student", 91: "study", 92: "teacher", 93: "thank", 94: "think",
    95: "water", 96: "work", 97: "wrong", 98: "yesterday", 99: "you"
}


def get_config() -> SignLanguageConfig:
    """Get the sign language configuration."""
    return SignLanguageConfig()


def load_vocabulary(vocab_path: Optional[Path] = None) -> Dict[int, str]:
    """Load vocabulary from file or return default."""
    if vocab_path and vocab_path.exists():
        import json
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            # Convert string keys to int if needed
            return {int(k): v for k, v in vocab.items()}
    return WLASL_100_VOCAB


def save_vocabulary(vocab: Dict[int, str], vocab_path: Path):
    """Save vocabulary to file."""
    import json
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)


def get_models_dir() -> Path:
    """Get the models directory path."""
    return MODELS_DIR
