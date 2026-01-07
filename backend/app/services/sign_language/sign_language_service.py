"""
Unified Sign Language Recognition Service
==========================================
Orchestrates multiple pretrained models for robust sign language recognition.

Pipeline:
1. Receive video stream / frames from frontend
2. Extract frames using OpenCV (15 FPS)
3. Extract hand & pose landmarks using MediaPipe
4. Run I3D video model (primary)
5. Fall back to Pose-LSTM if confidence is low
6. Output sign words / glosses
7. Send to Groq LLM for sentence correction
8. Return final sentence as JSON

Models:
- I3D (Primary): Video-based Inflated 3D ConvNet
- Pose-LSTM (Fallback): MediaPipe landmarks + LSTM
"""

import os
import io
import base64
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import deque
from datetime import datetime

from .config import get_config, load_vocabulary, MODELS_DIR
from .i3d_service import get_i3d_service, I3DInferenceService
from .pose_lstm_service import get_pose_lstm_service, PoseLSTMService

# Lazy imports
cv2 = None
torch = None


def _load_dependencies():
    """Lazily load heavy dependencies."""
    global cv2, torch
    
    if cv2 is None:
        try:
            import cv2 as cv2_import
            cv2 = cv2_import
        except ImportError:
            raise ImportError("OpenCV is required")
    
    if torch is None:
        try:
            import torch as torch_import
            torch = torch_import
        except ImportError:
            raise ImportError("PyTorch is required")


class SignLanguageRecognitionService:
    """
    Unified service for video-based sign language recognition.
    
    Integrates multiple pretrained models with confidence-based fallback:
    1. I3D (Primary): Best accuracy for video sequences
    2. Pose-LSTM (Fallback): Low-latency, works when I3D confidence is low
    
    Processing flow:
    1. Extract frames from video at 15 FPS
    2. Run I3D model on frame sequence
    3. If confidence < threshold, run Pose-LSTM as fallback
    4. Combine results and send to Groq for sentence refinement
    """
    
    def __init__(self):
        _load_dependencies()
        
        self.config = get_config()
        self.vocab = load_vocabulary()
        
        # Model services (lazy loaded)
        self._i3d_service: Optional[I3DInferenceService] = None
        self._pose_lstm_service: Optional[PoseLSTMService] = None
        
        # Groq service for sentence correction (injected)
        self._groq_service = None
        
        # Processing settings
        self.target_fps = self.config.frame_extraction_fps
        self.sliding_window_seconds = self.config.sliding_window_seconds
        self.fallback_threshold = self.config.fallback_confidence_threshold
        self.min_confidence = self.config.min_confidence_threshold
        
        # Frame buffer for streaming
        self.frame_buffer: deque = deque(maxlen=int(self.target_fps * 3))  # 3 seconds
        self.recognized_words: List[str] = []
        
        print("SignLanguageRecognitionService initialized")
    
    @property
    def i3d_service(self) -> I3DInferenceService:
        """Lazy load I3D service."""
        if self._i3d_service is None:
            self._i3d_service = get_i3d_service()
        return self._i3d_service
    
    @property
    def pose_lstm_service(self) -> PoseLSTMService:
        """Lazy load Pose-LSTM service."""
        if self._pose_lstm_service is None:
            self._pose_lstm_service = get_pose_lstm_service()
        return self._pose_lstm_service
    
    def set_groq_service(self, groq_service):
        """Set the Groq service for sentence correction."""
        self._groq_service = groq_service
    
    def extract_frames_from_video(
        self,
        video_bytes: bytes,
        target_fps: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from video at target FPS.
        
        Args:
            video_bytes: Raw video bytes
            target_fps: Target frame rate (default: 15)
            
        Returns:
            List of BGR frames
        """
        if target_fps is None:
            target_fps = self.target_fps
        
        frames = []
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(video_bytes)
                temp_path = f.name
            
            try:
                cap = cv2.VideoCapture(temp_path)
                
                if not cap.isOpened():
                    print("Failed to open video file")
                    return frames
                
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                frame_interval = max(1, int(video_fps / target_fps))
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        # Resize to 224x224
                        frame = cv2.resize(frame, (224, 224))
                        frames.append(frame)
                    
                    frame_count += 1
                
                cap.release()
                print(f"Extracted {len(frames)} frames from video")
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            print(f"Error extracting frames: {e}")
        
        return frames
    
    def extract_frames_from_base64_video(
        self,
        video_base64: str,
        target_fps: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from base64-encoded video.
        
        Args:
            video_base64: Base64-encoded video
            target_fps: Target frame rate
            
        Returns:
            List of BGR frames
        """
        try:
            video_bytes = base64.b64decode(video_base64)
            return self.extract_frames_from_video(video_bytes, target_fps)
        except Exception as e:
            print(f"Error decoding video: {e}")
            return []
    
    async def recognize_from_video(
        self,
        video_bytes: bytes,
        use_groq_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Recognize sign language from video.
        
        This is the main entry point for video-based recognition.
        
        Args:
            video_bytes: Raw video bytes
            use_groq_correction: Whether to use Groq for sentence correction
            
        Returns:
            Recognition result with words, sentence, and confidence
        """
        # Extract frames
        frames = self.extract_frames_from_video(video_bytes)
        
        if not frames:
            return {
                "recognized_words": [],
                "sentence": "No frames could be extracted from video",
                "confidence": 0.0,
                "error": "Failed to extract frames"
            }
        
        return await self.recognize_from_frames(frames, use_groq_correction)
    
    async def recognize_from_base64_video(
        self,
        video_base64: str,
        use_groq_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Recognize sign language from base64-encoded video.
        
        Args:
            video_base64: Base64-encoded video
            use_groq_correction: Whether to use Groq for sentence correction
            
        Returns:
            Recognition result
        """
        try:
            video_bytes = base64.b64decode(video_base64)
            return await self.recognize_from_video(video_bytes, use_groq_correction)
        except Exception as e:
            return {
                "recognized_words": [],
                "sentence": f"Error decoding video: {str(e)}",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def recognize_from_frames(
        self,
        frames: List[np.ndarray],
        use_groq_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Recognize sign language from a list of frames.
        
        Uses sliding window approach to recognize multiple signs.
        
        Args:
            frames: List of BGR frames (224x224)
            use_groq_correction: Whether to use Groq for sentence correction
            
        Returns:
            Recognition result
        """
        if not frames:
            return {
                "recognized_words": [],
                "sentence": "No frames provided",
                "confidence": 0.0
            }
        
        all_words = []
        all_confidences = []
        models_used = []
        
        # Calculate window size (2 seconds of frames)
        window_size = int(self.target_fps * self.sliding_window_seconds)
        step_size = window_size // 2  # 50% overlap
        
        # Process video in sliding windows
        for start_idx in range(0, len(frames), step_size):
            end_idx = min(start_idx + window_size, len(frames))
            window_frames = frames[start_idx:end_idx]
            
            if len(window_frames) < 10:  # Skip very short windows
                continue
            
            # Try I3D first (primary model)
            result = await self.i3d_service.recognize_from_frames(window_frames)
            
            model_used = "I3D"
            confidence = result.get("confidence", 0.0)
            words = result.get("recognized_words", [])
            
            # Fall back to Pose-LSTM if I3D confidence is low
            if confidence < self.fallback_threshold or "error" in result:
                pose_result = await self.pose_lstm_service.recognize_from_frames(window_frames)
                
                pose_confidence = pose_result.get("confidence", 0.0)
                
                if pose_confidence > confidence:
                    result = pose_result
                    model_used = "Pose-LSTM"
                    confidence = pose_confidence
                    words = pose_result.get("recognized_words", [])
            
            # Add to results if confidence is above minimum
            if confidence >= self.min_confidence and words:
                # Avoid consecutive duplicates
                if not all_words or all_words[-1] != words[0]:
                    all_words.extend(words)
                    all_confidences.append(confidence)
                    models_used.append(model_used)
        
        if not all_words:
            return {
                "recognized_words": [],
                "sentence": "No signs detected in video. Please ensure hands are clearly visible.",
                "confidence": 0.0,
                "frames_processed": len(frames)
            }
        
        # Calculate average confidence
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        
        # Form initial sentence from words
        raw_sentence = " ".join(all_words)
        
        # Use Groq to correct/refine the sentence
        final_sentence = raw_sentence
        if use_groq_correction and self._groq_service:
            try:
                corrected = await self._correct_sentence_with_groq(all_words)
                if corrected:
                    final_sentence = corrected
            except Exception as e:
                print(f"Groq sentence correction failed: {e}")
        
        return {
            "recognized_words": all_words,
            "sentence": final_sentence,
            "confidence": avg_confidence,
            "frames_processed": len(frames),
            "windows_processed": len(all_confidences),
            "models_used": list(set(models_used)),
            "raw_sentence": raw_sentence
        }
    
    async def _correct_sentence_with_groq(self, words: List[str]) -> Optional[str]:
        """
        Use Groq LLM to correct and refine the recognized words into a sentence.
        
        Args:
            words: List of recognized sign words
            
        Returns:
            Corrected sentence or None if failed
        """
        if not self._groq_service:
            return None
        
        try:
            prompt = f"""You are helping convert sign language glosses to natural English.
Given these recognized ASL signs: {', '.join(words)}

Convert them into a natural, grammatically correct English sentence.
ASL has different grammar than English (topic-comment structure, no articles, etc.).

Rules:
1. Keep the meaning intact
2. Add appropriate articles (a, an, the) if needed
3. Adjust verb tenses appropriately
4. Make it sound natural
5. Keep it concise (max 15 words)

Output ONLY the corrected sentence, nothing else."""

            # Use Groq's chat completion
            result = await self._groq_service.chat_completion(
                messages=[
                    {"role": "system", "content": "You convert ASL glosses to natural English sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            if result and "text" in result:
                return result["text"].strip()
            elif result and "choices" in result:
                return result["choices"][0]["message"]["content"].strip()
            
            return None
            
        except Exception as e:
            print(f"Groq correction error: {e}")
            return None
    
    async def recognize_from_base64_frame(
        self,
        frame_base64: str
    ) -> Dict[str, Any]:
        """
        Add a single frame and try to recognize (for streaming).
        
        Args:
            frame_base64: Base64-encoded frame image
            
        Returns:
            Recognition result (partial if not enough frames)
        """
        try:
            # Decode frame
            frame_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {
                    "recognized_words": [],
                    "sentence": "",
                    "confidence": 0.0,
                    "status": "frame_decode_error"
                }
            
            # Resize and add to buffer
            frame = cv2.resize(frame, (224, 224))
            self.frame_buffer.append(frame)
            
            # Add to Pose-LSTM buffer for real-time
            self.pose_lstm_service.add_frame(frame)
            
            # Check if we have enough frames for recognition
            buffer_len = len(self.frame_buffer)
            min_frames = int(self.target_fps * 1.0)  # 1 second minimum
            
            if buffer_len < min_frames:
                return {
                    "recognized_words": [],
                    "sentence": "",
                    "confidence": 0.0,
                    "status": "buffering",
                    "frames_buffered": buffer_len,
                    "frames_needed": min_frames
                }
            
            # Try real-time recognition with Pose-LSTM
            result = await self.pose_lstm_service.recognize_realtime()
            
            if result.get("confidence", 0) > self.min_confidence:
                words = result.get("recognized_words", [])
                
                # Add to recognized words if new
                if words and (not self.recognized_words or self.recognized_words[-1] != words[0]):
                    self.recognized_words.extend(words)
                
                return {
                    "recognized_words": words,
                    "sentence": " ".join(self.recognized_words[-5:]),  # Last 5 words
                    "confidence": result.get("confidence", 0.0),
                    "status": "recognized",
                    "total_words": len(self.recognized_words),
                    "model": "Pose-LSTM-Realtime"
                }
            
            return {
                "recognized_words": [],
                "sentence": " ".join(self.recognized_words[-5:]) if self.recognized_words else "",
                "confidence": 0.0,
                "status": "no_sign_detected",
                "frames_buffered": buffer_len
            }
            
        except Exception as e:
            print(f"Streaming recognition error: {e}")
            return {
                "recognized_words": [],
                "sentence": "",
                "confidence": 0.0,
                "status": "error",
                "error": str(e)
            }
    
    def clear_buffers(self):
        """Clear all buffers for a fresh start."""
        self.frame_buffer.clear()
        self.recognized_words.clear()
        self.pose_lstm_service.clear_buffer()
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status and model availability."""
        i3d_loaded = self.i3d_service.is_loaded if self._i3d_service else False
        pose_loaded = self.pose_lstm_service.is_loaded if self._pose_lstm_service else False
        
        return {
            "service": "SignLanguageRecognitionService",
            "status": "ready" if (i3d_loaded or pose_loaded) else "models_not_loaded",
            "models": {
                "i3d": {
                    "name": "I3D-WLASL",
                    "loaded": i3d_loaded,
                    "type": "video",
                    "primary": True
                },
                "pose_lstm": {
                    "name": "Pose-LSTM",
                    "loaded": pose_loaded,
                    "type": "landmarks",
                    "fallback": True
                }
            },
            "config": {
                "target_fps": self.target_fps,
                "sliding_window_seconds": self.sliding_window_seconds,
                "fallback_threshold": self.fallback_threshold,
                "min_confidence": self.min_confidence,
                "device": self.config.device
            },
            "vocabulary_size": len(self.vocab),
            "frame_buffer_size": len(self.frame_buffer),
            "recognized_words_count": len(self.recognized_words)
        }


# Singleton instance
_sign_language_service: Optional[SignLanguageRecognitionService] = None


def get_sign_language_service() -> SignLanguageRecognitionService:
    """Get the sign language recognition service singleton."""
    global _sign_language_service
    if _sign_language_service is None:
        _sign_language_service = SignLanguageRecognitionService()
    return _sign_language_service
