"""
I3D Inference Service
=====================
Video-based sign language recognition using Inflated 3D ConvNet (I3D).
Pretrained on Kinetics, fine-tuned for WLASL (Word-Level ASL).

Reference: https://github.com/sumedhsp/Sign-Language-Recognition
"""

import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# PyTorch imports - required for nn.Module class definitions
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import get_config, load_vocabulary, MODELS_DIR

# Lazy import for OpenCV (not needed at class definition time)
cv2 = None

I3D_AVAILABLE = False
_i3d_model = None
_i3d_vocab = None


def _load_cv2():
    """Lazily load OpenCV."""
    global cv2
    
    if cv2 is None:
        try:
            import cv2 as cv2_import
            cv2 = cv2_import
        except ImportError:
            raise ImportError("OpenCV is required: pip install opencv-python-headless")


class MaxPool3dSamePadding(object):
    """3D Max Pooling with same padding (compatible with TensorFlow)."""
    
    def __init__(self, kernel_size, stride, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._pool = nn.MaxPool3d(kernel_size, stride, padding)
    
    def __call__(self, x):
        return self._pool(x)


class Unit3D(object):
    """Basic 3D convolutional unit for I3D."""
    
    def __init__(self, in_channels, output_channels, kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1), padding=0, activation_fn=None,
                 use_batch_norm=True, use_bias=False, name='unit_3d'):
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        self.activation_fn = activation_fn if activation_fn else F.relu
        self.use_bias = use_bias
        self.name = name
        self.padding = padding
        
        # Will be initialized when building
        self.conv3d = None
        self.bn = None
    
    def build(self, in_channels):
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self.output_channels,
            kernel_size=self.kernel_shape,
            stride=self.stride,
            padding=self.padding,
            bias=self.use_bias
        )
        if self.use_batch_norm:
            self.bn = nn.BatchNorm3d(self.output_channels)
        return self


class InceptionI3d(nn.Module):
    """
    Inception-v1 I3D architecture.
    
    Inflated 3D ConvNet based on 2D Inception architecture.
    Pretrained on Kinetics-400, can be fine-tuned for sign language.
    
    Reference: "Quo Vadis, Action Recognition?" - Carreira & Zisserman
    """
    
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3',
        'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f',
        'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'Logits', 'Predictions',
    )
    
    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', in_channels=3, dropout_keep_prob=0.5):
        super(InceptionI3d, self).__init__()
        
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        
        self.end_points = {}
        self._build_network(in_channels, dropout_keep_prob)
    
    def _build_network(self, in_channels, dropout_keep_prob):
        """Build the I3D network architecture."""
        # Conv3d_1a_7x7
        self.end_points['Conv3d_1a_7x7'] = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        # MaxPool3d_2a_3x3
        self.end_points['MaxPool3d_2a_3x3'] = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Conv3d_2b_1x1
        self.end_points['Conv3d_2b_1x1'] = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        # Conv3d_2c_3x3
        self.end_points['Conv3d_2c_3x3'] = nn.Sequential(
            nn.Conv3d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True)
        )
        
        # MaxPool3d_3a_3x3
        self.end_points['MaxPool3d_3a_3x3'] = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Mixed_3b - Inception module
        self.end_points['Mixed_3b'] = self._build_inception_module(192, [64, 96, 128, 16, 32, 32])
        
        # Mixed_3c
        self.end_points['Mixed_3c'] = self._build_inception_module(256, [128, 128, 192, 32, 96, 64])
        
        # MaxPool3d_4a_3x3
        self.end_points['MaxPool3d_4a_3x3'] = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        
        # Mixed_4b through Mixed_4f
        self.end_points['Mixed_4b'] = self._build_inception_module(480, [192, 96, 208, 16, 48, 64])
        self.end_points['Mixed_4c'] = self._build_inception_module(512, [160, 112, 224, 24, 64, 64])
        self.end_points['Mixed_4d'] = self._build_inception_module(512, [128, 128, 256, 24, 64, 64])
        self.end_points['Mixed_4e'] = self._build_inception_module(512, [112, 144, 288, 32, 64, 64])
        self.end_points['Mixed_4f'] = self._build_inception_module(528, [256, 160, 320, 32, 128, 128])
        
        # MaxPool3d_5a_2x2
        self.end_points['MaxPool3d_5a_2x2'] = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # Mixed_5b, Mixed_5c
        self.end_points['Mixed_5b'] = self._build_inception_module(832, [256, 160, 320, 32, 128, 128])
        self.end_points['Mixed_5c'] = self._build_inception_module(832, [384, 192, 384, 48, 128, 128])
        
        # Logits
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = nn.Conv3d(1024, self._num_classes, kernel_size=1)
        
        # Register all modules
        for k, v in self.end_points.items():
            self.add_module(k, v)
    
    def _build_inception_module(self, in_channels, out_channels):
        """Build an Inception module with 4 branches."""
        class InceptionModule(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.b0 = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch[0], kernel_size=1),
                    nn.BatchNorm3d(out_ch[0]),
                    nn.ReLU(inplace=True)
                )
                self.b1 = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch[1], kernel_size=1),
                    nn.BatchNorm3d(out_ch[1]),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch[1], out_ch[2], kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch[2]),
                    nn.ReLU(inplace=True)
                )
                self.b2 = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch[3], kernel_size=1),
                    nn.BatchNorm3d(out_ch[3]),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_ch[3], out_ch[4], kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch[4]),
                    nn.ReLU(inplace=True)
                )
                self.b3 = nn.Sequential(
                    nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                    nn.Conv3d(in_ch, out_ch[5], kernel_size=1),
                    nn.BatchNorm3d(out_ch[5]),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                return torch.cat([self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1)
        
        return InceptionModule(in_channels, out_channels)
    
    def forward(self, x):
        """
        Forward pass through I3D network.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
               B: batch size, C: channels (3), T: temporal frames,
               H: height (224), W: width (224)
        
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        for endpoint in self.VALID_ENDPOINTS[:-2]:  # Exclude Logits and Predictions
            if endpoint in self.end_points:
                x = self._modules[endpoint](x)
        
        # Final pooling and classification
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.logits(x)
        
        if self._spatial_squeeze:
            x = x.squeeze(3).squeeze(3).squeeze(2)
        
        return x
    
    def extract_features(self, x):
        """Extract features before final classification."""
        for endpoint in self.VALID_ENDPOINTS[:-2]:
            if endpoint in self.end_points:
                x = self._modules[endpoint](x)
        return self.avg_pool(x).flatten(1)


class I3DInferenceService:
    """
    I3D-based video sign language recognition service.
    
    Uses pretrained I3D model for video-to-word classification.
    Supports WLASL (Word-Level ASL) dataset vocabulary.
    """
    
    def __init__(self):
        _load_cv2()
        
        self.config = get_config()
        self.model_config = self.config.i3d
        self.device = self.config.device
        self.vocab = load_vocabulary(self.model_config.vocab_path)
        
        self.model = None
        self.is_loaded = False
        
        # Video preprocessing settings
        self.input_size = self.model_config.input_size
        self.num_frames = self.model_config.num_frames
        self.fps = self.model_config.fps
        
        # Normalization (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        print(f"I3D Service initialized (device: {self.device})")
    
    def load_model(self) -> bool:
        """
        Load the pretrained I3D model.
        
        Returns:
            True if model loaded successfully
        """
        if self.is_loaded:
            return True
        
        model_path = self.model_config.model_path
        
        if not model_path.exists():
            print(f"I3D model not found at {model_path}")
            print("Please run: python -m app.services.sign_language.download_models")
            return False
        
        try:
            # Create model
            self.model = InceptionI3d(
                num_classes=self.model_config.num_classes,
                in_channels=3,
                dropout_keep_prob=0.5
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Handle different checkpoint formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print(f"I3D model loaded successfully ({self.model_config.num_classes} classes)")
            return True
            
        except Exception as e:
            print(f"Error loading I3D model: {e}")
            return False
    
    def preprocess_video(self, video_bytes: bytes) -> Optional[np.ndarray]:
        """
        Preprocess video for I3D inference.
        
        Args:
            video_bytes: Raw video bytes
            
        Returns:
            Numpy array of shape (1, C, T, H, W) or None if failed
        """
        try:
            # Save to temp file for OpenCV
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_bytes)
                temp_path = f.name
            
            try:
                cap = cv2.VideoCapture(temp_path)
                
                if not cap.isOpened():
                    print("Failed to open video")
                    return None
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                
                # Calculate frame indices to sample
                if total_frames <= self.num_frames:
                    # Repeat frames if video is too short
                    frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                else:
                    # Uniformly sample frames
                    frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                
                frames = []
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Resize and convert to RGB
                        frame = cv2.resize(frame, self.input_size)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    else:
                        # Repeat last frame if read fails
                        if frames:
                            frames.append(frames[-1])
                
                cap.release()
                
                if len(frames) < self.num_frames:
                    # Pad with last frame
                    while len(frames) < self.num_frames:
                        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
                
                # Convert to numpy array: (T, H, W, C)
                frames = np.stack(frames[:self.num_frames])
                
                # Normalize: (T, H, W, C) -> float32, 0-1 range
                frames = frames.astype(np.float32) / 255.0
                frames = (frames - self.mean) / self.std
                
                # Transpose to (C, T, H, W) then add batch dim
                frames = frames.transpose(3, 0, 1, 2)  # (C, T, H, W)
                frames = np.expand_dims(frames, 0)  # (1, C, T, H, W)
                
                return frames
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"Error preprocessing video: {e}")
            return None
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Preprocess a list of frames for I3D inference.
        
        Args:
            frames: List of BGR frames (H, W, C)
            
        Returns:
            Numpy array of shape (1, C, T, H, W) or None if failed
        """
        try:
            if not frames:
                return None
            
            processed = []
            for frame in frames:
                # Resize and convert to RGB
                frame = cv2.resize(frame, self.input_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed.append(frame)
            
            # Pad or sample to num_frames
            if len(processed) < self.num_frames:
                # Repeat frames
                while len(processed) < self.num_frames:
                    processed.append(processed[-1])
            elif len(processed) > self.num_frames:
                # Uniformly sample
                indices = np.linspace(0, len(processed) - 1, self.num_frames, dtype=int)
                processed = [processed[i] for i in indices]
            
            # Stack and normalize
            frames_array = np.stack(processed)
            frames_array = frames_array.astype(np.float32) / 255.0
            frames_array = (frames_array - self.mean) / self.std
            
            # Transpose: (T, H, W, C) -> (C, T, H, W) -> (1, C, T, H, W)
            frames_array = frames_array.transpose(3, 0, 1, 2)
            frames_array = np.expand_dims(frames_array, 0)
            
            return frames_array
            
        except Exception as e:
            print(f"Error preprocessing frames: {e}")
            return None
    
    async def recognize(self, video_bytes: bytes) -> Dict[str, Any]:
        """
        Recognize sign language from video.
        
        Args:
            video_bytes: Raw video bytes
            
        Returns:
            Recognition result with words and confidence
        """
        if not self.load_model():
            return {
                "error": "I3D model not available",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        # Preprocess video
        input_tensor = self.preprocess_video(video_bytes)
        
        if input_tensor is None:
            return {
                "error": "Failed to preprocess video",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        try:
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=-1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
                
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
                
                # Map to words
                predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    word = self.vocab.get(int(idx), f"SIGN_{idx}")
                    predictions.append({
                        "word": word.upper(),
                        "confidence": float(prob)
                    })
                
                # Primary prediction
                best_word = predictions[0]["word"]
                best_conf = predictions[0]["confidence"]
                
                return {
                    "recognized_words": [best_word],
                    "confidence": best_conf,
                    "all_predictions": predictions,
                    "model": "I3D"
                }
                
        except Exception as e:
            print(f"I3D inference error: {e}")
            return {
                "error": str(e),
                "recognized_words": [],
                "confidence": 0.0
            }
    
    async def recognize_from_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Recognize sign language from a list of frames.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            Recognition result
        """
        if not self.load_model():
            return {
                "error": "I3D model not available",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        input_tensor = self.preprocess_frames(frames)
        
        if input_tensor is None:
            return {
                "error": "Failed to preprocess frames",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        try:
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=-1)
                
                top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
                
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
                
                predictions = []
                for prob, idx in zip(top_probs, top_indices):
                    word = self.vocab.get(int(idx), f"SIGN_{idx}")
                    predictions.append({
                        "word": word.upper(),
                        "confidence": float(prob)
                    })
                
                return {
                    "recognized_words": [predictions[0]["word"]],
                    "confidence": predictions[0]["confidence"],
                    "all_predictions": predictions,
                    "model": "I3D"
                }
                
        except Exception as e:
            print(f"I3D inference error: {e}")
            return {
                "error": str(e),
                "recognized_words": [],
                "confidence": 0.0
            }


# Singleton instance
_i3d_service: Optional[I3DInferenceService] = None


def get_i3d_service() -> I3DInferenceService:
    """Get the I3D inference service singleton."""
    global _i3d_service
    if _i3d_service is None:
        _i3d_service = I3DInferenceService()
    return _i3d_service
