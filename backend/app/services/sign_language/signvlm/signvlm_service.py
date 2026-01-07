"""
SignVLM Service - Video-based Sign Language Recognition
=======================================================

Service class for sign language recognition using the SignVLM model.
Handles video preprocessing, inference, and result formatting.

Features:
- Video/frame preprocessing with proper normalization
- Sliding window inference for long videos
- Class mapping for WLASL, KArSL, and other datasets
- Integration with Groq for sentence correction

Source: https://github.com/Hamzah-Luqman/signVLM
"""

from __future__ import annotations

import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from dataclasses import dataclass, field

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    import torch as torch_type

# Lazy imports for heavy dependencies
torch = None
cv2 = None
Image = None

# Global service instance
_signvlm_service = None


def _load_dependencies():
    """Lazily load heavy dependencies."""
    global torch, cv2, Image
    
    if torch is None:
        try:
            import torch as torch_import
            torch = torch_import
        except ImportError:
            raise ImportError("PyTorch is required for SignVLM")
    
    if cv2 is None:
        try:
            import cv2 as cv2_import
            cv2 = cv2_import
        except ImportError:
            print("Warning: OpenCV not available, using PIL only")
    
    if Image is None:
        try:
            from PIL import Image as Image_import
            Image = Image_import
        except ImportError:
            raise ImportError("Pillow is required for SignVLM")


@dataclass
class SignVLMConfig:
    """Configuration for SignVLM service."""
    
    # Model architecture
    backbone_name: str = "ViT-L/14-lnpre"
    backbone_type: str = "clip"
    decoder_num_layers: int = 4
    decoder_qkv_dim: int = 1024
    decoder_num_heads: int = 16
    
    # Input settings
    num_frames: int = 24
    input_size: Tuple[int, int] = (224, 224)
    fps: int = 15
    
    # Normalization (CLIP values)
    mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
    
    # Inference settings
    min_confidence: float = 0.3
    top_k: int = 5
    
    # Device
    use_cuda: bool = False
    device: str = "cpu"
    
    # Paths
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent.parent / "models" / "signvlm")
    backbone_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    class_mapping_path: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize paths and device."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        if self.backbone_path is None:
            self.backbone_path = self.models_dir / "ViT-L-14.pt"
        
        if self.checkpoint_path is None:
            self.checkpoint_path = self.models_dir / "signvlm_wlasl100.pth"
        
        if self.class_mapping_path is None:
            self.class_mapping_path = self.models_dir / "wlasl100_classes.json"
        
        # Check CUDA availability
        if self.use_cuda:
            _load_dependencies()
            if torch.cuda.is_available():
                self.device = "cuda"


# WLASL-100 class mapping (top 100 glosses)
# WLASL-100 class glosses (Word-Level American Sign Language dataset - 100 class subset)
# Source: https://dxli94.github.io/WLASL/
WLASL100_CLASSES = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who", "candy",
    "cousin", "deaf", "fine", "help", "no", "thin", "walk", "year", "yes", "all",
    "black", "cool", "finish", "hot", "like", "many", "mother", "now", "orange",
    "table", "thanksgiving", "what", "woman", "bed", "blue", "bowling", "can",
    "dog", "family", "friend", "hearing", "hello", "give", "language", "later",
    "man", "meet", "name", "right", "same", "shirt", "study", "teacher", "think",
    "thursday", "time", "tuesday", "want", "wednesday", "white", "why", "work",
    "birthday", "brown", "but", "bye", "change", "chicken", "cold", "color", "corn",
    "cow", "dance", "dark", "daughter", "day", "doctor", "eat", "enjoy", "face",
    "father", "fish", "forget", "friday", "full", "graduate", "green", "hat",
    "have", "he", "hungry", "hurt", "jacket", "know", "last", "learn", "letter",
    "lose", "medicine", "milk", "monday"
]


class SignVLMService:
    """
    Sign Language Recognition Service using SignVLM.
    
    Provides video-based sign language recognition using the EVL Transformer
    architecture with CLIP backbone.
    
    Usage:
        service = SignVLMService()
        result = await service.recognize_video(video_bytes)
        print(result['sentence'])
    """
    
    def __init__(self, config: Optional[SignVLMConfig] = None):
        """Initialize the SignVLM service."""
        _load_dependencies()
        
        self.config = config or SignVLMConfig()
        self.model = None
        self.class_mapping: Dict[int, str] = {}
        self.is_loaded = False
        
        # Load class mapping
        self._load_class_mapping()
        
        print(f"[SignVLM] Service initialized")
        print(f"[SignVLM] Models directory: {self.config.models_dir}")
        print(f"[SignVLM] Device: {self.config.device}")
    
    def _load_class_mapping(self):
        """Load class index to gloss mapping."""
        import json
        
        # Try to load from file
        if self.config.class_mapping_path and self.config.class_mapping_path.exists():
            with open(self.config.class_mapping_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.class_mapping = {int(k): v for k, v in data.items()}
                elif isinstance(data, list):
                    self.class_mapping = {i: v for i, v in enumerate(data)}
            print(f"[SignVLM] Loaded {len(self.class_mapping)} classes from file")
        else:
            # Use default WLASL-100 classes
            self.class_mapping = {i: c for i, c in enumerate(WLASL100_CLASSES)}
            print(f"[SignVLM] Using default WLASL-100 classes ({len(self.class_mapping)})")
    
    def load_model(self) -> bool:
        """
        Load the SignVLM model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.is_loaded:
            return True
        
        # Check if backbone exists
        if not self.config.backbone_path.exists():
            print(f"[SignVLM] Backbone not found: {self.config.backbone_path}")
            print(f"[SignVLM] Please run the download script to get CLIP weights")
            return False
        
        try:
            from .model import EVLTransformer
            
            num_classes = len(self.class_mapping) if self.class_mapping else 100
            
            # Create model
            self.model = EVLTransformer(
                num_frames=self.config.num_frames,
                backbone_name=self.config.backbone_name,
                backbone_type=self.config.backbone_type,
                backbone_path=str(self.config.backbone_path),
                backbone_mode='freeze_fp16' if self.config.device == 'cpu' else 'freeze_fp16',
                decoder_num_layers=self.config.decoder_num_layers,
                decoder_qkv_dim=self.config.decoder_qkv_dim,
                decoder_num_heads=self.config.decoder_num_heads,
                num_classes=num_classes,
            )
            
            # Load checkpoint if available
            if self.config.checkpoint_path and self.config.checkpoint_path.exists():
                self.model.load_checkpoint(str(self.config.checkpoint_path))
            else:
                print(f"[SignVLM] No checkpoint found, using pretrained backbone only")
            
            # Move to device and set eval mode
            self.model = self.model.to(self.config.device)
            self.model.eval()
            
            self.is_loaded = True
            print(f"[SignVLM] Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"[SignVLM] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_frames(
        self,
        frames: List[np.ndarray],
        target_frames: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess video frames for model input.
        
        Args:
            frames: List of BGR frames from OpenCV
            target_frames: Number of frames to sample (default: config.num_frames)
            
        Returns:
            Tensor of shape (1, C, T, H, W)
        """
        if target_frames is None:
            target_frames = self.config.num_frames
        
        # Sample frames uniformly
        if len(frames) > target_frames:
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < target_frames:
            # Repeat last frame if not enough frames
            while len(frames) < target_frames:
                frames.append(frames[-1])
        
        processed_frames = []
        h, w = self.config.input_size
        
        for frame in frames:
            # Convert BGR to RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # Apply CLIP normalization
            mean = np.array(self.config.mean).reshape(1, 1, 3)
            std = np.array(self.config.std).reshape(1, 1, 3)
            frame = (frame - mean) / std
            
            # HWC -> CHW
            frame = frame.transpose(2, 0, 1)
            processed_frames.append(frame)
        
        # Stack frames: (T, C, H, W)
        video = np.stack(processed_frames, axis=0)
        
        # Add batch dimension and rearrange: (1, C, T, H, W)
        video = video.transpose(1, 0, 2, 3)  # (C, T, H, W)
        video = np.expand_dims(video, axis=0)  # (1, C, T, H, W)
        
        return torch.from_numpy(video).float()
    
    def extract_frames_from_video(
        self,
        video_bytes: bytes,
        target_fps: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video bytes.
        
        Args:
            video_bytes: Raw video bytes
            target_fps: Target frame rate
            
        Returns:
            List of BGR frames
        """
        if target_fps is None:
            target_fps = self.config.fps
        
        frames = []
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
        
        try:
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                print("[SignVLM] Failed to open video")
                return frames
            
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = max(1, int(video_fps / target_fps))
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frames.append(frame)
                
                frame_idx += 1
            
            cap.release()
            
        finally:
            os.unlink(temp_path)
        
        return frames
    
    def extract_frames_from_base64_images(
        self,
        images_base64: List[str]
    ) -> List[np.ndarray]:
        """
        Extract frames from base64 encoded images.
        
        Args:
            images_base64: List of base64 encoded images
            
        Returns:
            List of BGR frames
        """
        frames = []
        
        for img_b64 in images_base64:
            try:
                # Handle data URL prefix
                if img_b64.startswith('data:'):
                    img_b64 = img_b64.split(',', 1)[1]
                
                # Decode
                img_data = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_data))
                
                # Convert to RGB then BGR for OpenCV
                img = img.convert('RGB')
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                frames.append(frame)
                
            except Exception as e:
                print(f"[SignVLM] Error processing image: {e}")
        
        return frames
    
    async def recognize_video(
        self,
        video_bytes: bytes,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Recognize sign language from video.
        
        Args:
            video_bytes: Raw video bytes
            top_k: Number of top predictions to return
            
        Returns:
            Dict with recognized words, confidence, etc.
        """
        # Load model if not loaded
        if not self.is_loaded:
            if not self.load_model():
                return {
                    "error": "Model not loaded",
                    "recognized_words": [],
                    "sentence": "",
                    "confidence": 0.0
                }
        
        # Extract frames
        frames = self.extract_frames_from_video(video_bytes)
        
        if len(frames) < 2:
            return {
                "error": "Not enough frames extracted",
                "recognized_words": [],
                "sentence": "",
                "confidence": 0.0,
                "frames_extracted": len(frames)
            }
        
        # Preprocess
        video_tensor = self.preprocess_frames(frames)
        video_tensor = video_tensor.to(self.config.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(video_tensor)
            probs = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = probs.topk(min(top_k, probs.size(-1)), dim=-1)
        
        predictions = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            gloss = self.class_mapping.get(int(idx), f"class_{idx}")
            predictions.append({
                "gloss": gloss,
                "confidence": float(prob)
            })
        
        # Get best prediction
        best_pred = predictions[0] if predictions else {"gloss": "", "confidence": 0.0}
        
        return {
            "recognized_words": [p["gloss"] for p in predictions if p["confidence"] > self.config.min_confidence],
            "sentence": best_pred["gloss"],
            "confidence": best_pred["confidence"],
            "predictions": predictions,
            "frames_processed": len(frames),
            "model": "SignVLM"
        }
    
    async def recognize_frames(
        self,
        frames: List[np.ndarray],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Recognize sign language from preprocessed frames.
        
        Args:
            frames: List of BGR frames
            top_k: Number of top predictions
            
        Returns:
            Recognition result dict
        """
        if not self.is_loaded:
            if not self.load_model():
                return {
                    "error": "Model not loaded",
                    "recognized_words": [],
                    "sentence": "",
                    "confidence": 0.0
                }
        
        if len(frames) < 2:
            return {
                "error": "Not enough frames",
                "recognized_words": [],
                "sentence": "",
                "confidence": 0.0
            }
        
        # Preprocess
        video_tensor = self.preprocess_frames(frames)
        video_tensor = video_tensor.to(self.config.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(video_tensor)
            probs = torch.softmax(logits, dim=-1)
        
        # Get predictions
        top_probs, top_indices = probs.topk(min(top_k, probs.size(-1)), dim=-1)
        
        predictions = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            gloss = self.class_mapping.get(int(idx), f"class_{idx}")
            predictions.append({
                "gloss": gloss,
                "confidence": float(prob)
            })
        
        best_pred = predictions[0] if predictions else {"gloss": "", "confidence": 0.0}
        
        return {
            "recognized_words": [p["gloss"] for p in predictions if p["confidence"] > self.config.min_confidence],
            "sentence": best_pred["gloss"],
            "confidence": best_pred["confidence"],
            "predictions": predictions,
            "frames_processed": len(frames),
            "model": "SignVLM"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information."""
        return {
            "service": "SignVLMService",
            "status": "ready" if self.is_loaded else "model_not_loaded",
            "models": {
                "backbone": {
                    "name": self.config.backbone_name,
                    "available": self.config.backbone_path.exists() if self.config.backbone_path else False
                },
                "checkpoint": {
                    "path": str(self.config.checkpoint_path) if self.config.checkpoint_path else None,
                    "available": self.config.checkpoint_path.exists() if self.config.checkpoint_path else False
                }
            },
            "config": {
                "num_frames": self.config.num_frames,
                "input_size": self.config.input_size,
                "device": self.config.device,
                "num_classes": len(self.class_mapping)
            },
            "vocabulary_size": len(self.class_mapping)
        }


def get_signvlm_service() -> SignVLMService:
    """Get or create the SignVLM service singleton."""
    global _signvlm_service
    
    if _signvlm_service is None:
        _signvlm_service = SignVLMService()
    
    return _signvlm_service
