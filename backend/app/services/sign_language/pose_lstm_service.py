"""
Pose-based LSTM Service
=======================
Low-latency sign language recognition using MediaPipe landmarks + LSTM.

Uses MediaPipe Hand and Pose landmarks as input features,
processed through an LSTM network for temporal modeling.

This is the fallback model when video-based models have low confidence.
"""

import os
import io
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import deque
from datetime import datetime

# PyTorch imports - required for nn.Module class definitions
import torch
import torch.nn as nn

from .config import get_config, load_vocabulary, MODELS_DIR

# Lazy imports for MediaPipe and OpenCV
cv2 = None
mp = None
mp_hands = None
mp_pose = None

POSE_LSTM_AVAILABLE = False
_pose_lstm_model = None


def _load_media_dependencies():
    """Lazily load MediaPipe and OpenCV dependencies."""
    global cv2, mp, mp_hands, mp_pose
    
    if cv2 is None:
        try:
            import cv2 as cv2_import
            cv2 = cv2_import
        except ImportError:
            raise ImportError("OpenCV is required")
    
    if mp is None:
        try:
            import mediapipe as mp_import
            mp = mp_import
            mp_hands = mp.solutions.hands
            mp_pose = mp.solutions.pose
        except ImportError:
            raise ImportError("MediaPipe is required")


class SignLanguageLSTM(nn.Module):
    """
    LSTM-based sign language classifier using pose landmarks.
    
    Architecture:
    - Input: Sequence of landmark features (per frame)
    - LSTM layers for temporal modeling
    - Fully connected classifier
    
    Features per frame:
    - 21 hand landmarks x 3 coords x 2 hands = 126
    - 33 pose landmarks x 3 coords = 99
    - Total: 225 features (or subset)
    """
    
    def __init__(
        self,
        input_size: int = 126,  # 21 landmarks x 3 coords x 2 hands
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(SignLanguageLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention layer (optional)
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            lengths: Optional sequence lengths for packing
            
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        logits = self.classifier(attended)
        
        return logits


class PoseLSTMService:
    """
    Pose-based sign language recognition service.
    
    Uses MediaPipe to extract hand and pose landmarks,
    then classifies using an LSTM model.
    
    This is the low-latency fallback when video models fail.
    """
    
    def __init__(self):
        _load_media_dependencies()
        
        self.config = get_config()
        self.model_config = self.config.pose_lstm
        self.device = self.config.device
        self.vocab = load_vocabulary(self.model_config.vocab_path)
        
        self.model = None
        self.is_loaded = False
        
        # MediaPipe solutions
        self.hands = None
        self.pose = None
        
        # Landmark buffer for real-time processing
        self.landmark_buffer = deque(maxlen=60)  # ~2 seconds at 30fps
        self.last_update = None
        
        # Feature dimensions
        self.num_hand_landmarks = 21
        self.num_pose_landmarks = 33
        self.feature_dim = self.num_hand_landmarks * 3 * 2  # Both hands
        
        print(f"Pose-LSTM Service initialized (device: {self.device})")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe solutions."""
        if self.hands is None:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        if self.pose is None:
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def load_model(self) -> bool:
        """
        Load the pretrained pose-LSTM model.
        
        Returns:
            True if model loaded successfully
        """
        if self.is_loaded:
            return True
        
        model_path = self.model_config.model_path
        
        # Create model architecture
        self.model = SignLanguageLSTM(
            input_size=self.feature_dim,
            hidden_size=256,
            num_layers=2,
            num_classes=self.model_config.num_classes,
            dropout=0.3,
            bidirectional=True
        )
        
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                
                # Handle different checkpoint formats
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Pose-LSTM model loaded from {model_path}")
                
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
                print("Using randomly initialized model (for demo purposes)")
        else:
            print(f"Pose-LSTM model not found at {model_path}")
            print("Using randomly initialized model (for demo purposes)")
        
        self.model.to(self.device)
        self.model.eval()
        self._init_mediapipe()
        self.is_loaded = True
        
        return True
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from a single frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Landmark features array or None if no hands detected
        """
        self._init_mediapipe()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Initialize feature array (2 hands x 21 landmarks x 3 coords)
        features = np.zeros(self.feature_dim)
        
        # Extract landmarks for each hand
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            hand_offset = idx * self.num_hand_landmarks * 3
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                features[hand_offset + i * 3] = landmark.x
                features[hand_offset + i * 3 + 1] = landmark.y
                features[hand_offset + i * 3 + 2] = landmark.z
        
        return features
    
    def extract_landmarks_from_video(self, video_bytes: bytes) -> List[np.ndarray]:
        """
        Extract landmarks from all frames in a video.
        
        Args:
            video_bytes: Raw video bytes
            
        Returns:
            List of landmark feature arrays
        """
        landmarks_list = []
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_bytes)
                temp_path = f.name
            
            try:
                cap = cv2.VideoCapture(temp_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                
                # Sample at target FPS
                target_fps = self.config.frame_extraction_fps
                frame_interval = max(1, int(fps / target_fps))
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        landmarks = self.extract_landmarks(frame)
                        if landmarks is not None:
                            landmarks_list.append(landmarks)
                    
                    frame_count += 1
                
                cap.release()
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"Error extracting landmarks from video: {e}")
        
        return landmarks_list
    
    def extract_landmarks_from_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract landmarks from a list of frames.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            List of landmark feature arrays
        """
        landmarks_list = []
        
        for frame in frames:
            landmarks = self.extract_landmarks(frame)
            if landmarks is not None:
                landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def add_frame(self, frame: np.ndarray) -> None:
        """
        Add a frame to the landmark buffer for real-time processing.
        
        Args:
            frame: BGR image frame
        """
        landmarks = self.extract_landmarks(frame)
        if landmarks is not None:
            self.landmark_buffer.append(landmarks)
            self.last_update = datetime.now()
    
    def preprocess_landmarks(self, landmarks: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Preprocess landmark sequence for model input.
        
        Args:
            landmarks: List of landmark arrays
            
        Returns:
            Tensor of shape (1, seq_len, features)
        """
        if not landmarks:
            return None
        
        # Target sequence length
        target_len = self.model_config.num_frames
        
        # Pad or sample to target length
        if len(landmarks) < target_len:
            # Pad by repeating last frame
            while len(landmarks) < target_len:
                landmarks.append(landmarks[-1])
        elif len(landmarks) > target_len:
            # Uniformly sample
            indices = np.linspace(0, len(landmarks) - 1, target_len, dtype=int)
            landmarks = [landmarks[i] for i in indices]
        
        # Stack into array
        sequence = np.stack(landmarks)  # (seq_len, features)
        
        # Normalize landmarks (center and scale)
        # Simple normalization: subtract mean and divide by std
        sequence = (sequence - sequence.mean()) / (sequence.std() + 1e-8)
        
        # Add batch dimension
        sequence = np.expand_dims(sequence, 0)  # (1, seq_len, features)
        
        return sequence
    
    async def recognize(self, video_bytes: bytes) -> Dict[str, Any]:
        """
        Recognize sign language from video using pose landmarks.
        
        Args:
            video_bytes: Raw video bytes
            
        Returns:
            Recognition result
        """
        if not self.load_model():
            return {
                "error": "Pose-LSTM model not available",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        # Extract landmarks
        landmarks = self.extract_landmarks_from_video(video_bytes)
        
        if not landmarks:
            return {
                "error": "No hands detected in video",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        # Preprocess
        input_tensor = self.preprocess_landmarks(landmarks)
        
        if input_tensor is None:
            return {
                "error": "Failed to preprocess landmarks",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        try:
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1)
                
                # Get top predictions
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
                    "model": "Pose-LSTM",
                    "frames_processed": len(landmarks)
                }
                
        except Exception as e:
            print(f"Pose-LSTM inference error: {e}")
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
                "error": "Pose-LSTM model not available",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        # Extract landmarks
        landmarks = self.extract_landmarks_from_frames(frames)
        
        if not landmarks:
            return {
                "error": "No hands detected in frames",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        # Preprocess
        input_tensor = self.preprocess_landmarks(landmarks)
        
        if input_tensor is None:
            return {
                "error": "Failed to preprocess landmarks",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        try:
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1)
                
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
                    "model": "Pose-LSTM",
                    "frames_processed": len(landmarks)
                }
                
        except Exception as e:
            print(f"Pose-LSTM inference error: {e}")
            return {
                "error": str(e),
                "recognized_words": [],
                "confidence": 0.0
            }
    
    async def recognize_realtime(self) -> Dict[str, Any]:
        """
        Recognize from the landmark buffer (real-time processing).
        
        Returns:
            Recognition result from buffered frames
        """
        if not self.load_model():
            return {
                "error": "Pose-LSTM model not available",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        if len(self.landmark_buffer) < 10:  # Minimum frames needed
            return {
                "error": "Not enough frames in buffer",
                "recognized_words": [],
                "confidence": 0.0,
                "frames_buffered": len(self.landmark_buffer)
            }
        
        # Get landmarks from buffer
        landmarks = list(self.landmark_buffer)
        
        # Preprocess
        input_tensor = self.preprocess_landmarks(landmarks)
        
        if input_tensor is None:
            return {
                "error": "Failed to preprocess landmarks",
                "recognized_words": [],
                "confidence": 0.0
            }
        
        try:
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor).float().to(self.device)
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=-1)
                
                top_probs, top_indices = torch.topk(probs, k=3, dim=-1)
                
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
                    "model": "Pose-LSTM-Realtime",
                    "frames_processed": len(landmarks)
                }
                
        except Exception as e:
            print(f"Pose-LSTM realtime inference error: {e}")
            return {
                "error": str(e),
                "recognized_words": [],
                "confidence": 0.0
            }
    
    def clear_buffer(self):
        """Clear the landmark buffer."""
        self.landmark_buffer.clear()
        self.last_update = None


# Singleton instance
_pose_lstm_service: Optional[PoseLSTMService] = None


def get_pose_lstm_service() -> PoseLSTMService:
    """Get the Pose-LSTM service singleton."""
    global _pose_lstm_service
    if _pose_lstm_service is None:
        _pose_lstm_service = PoseLSTMService()
    return _pose_lstm_service
