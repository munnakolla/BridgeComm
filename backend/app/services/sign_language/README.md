# Sign Language Recognition Module

This module provides video-based American Sign Language (ASL) recognition using pretrained deep learning models.

## Architecture

The module uses a **multi-model fallback architecture**:

1. **I3D (Inflated 3D ConvNet)** - Primary model for video-based recognition
   - Processes sequences of 64 frames
   - Uses 3D convolutions to capture temporal dynamics
   - Pretrained on WLASL-100 dataset (100 common ASL words)

2. **Pose-LSTM** - Low-latency fallback model
   - Uses MediaPipe hand landmarks (126-dimensional features)
   - Bidirectional LSTM with attention mechanism
   - Faster inference, suitable for real-time applications

## Processing Flow

```
Video Input
    ↓
Frame Extraction (15 FPS, 224×224)
    ↓
┌─────────────────────────────────────┐
│         I3D Recognition             │
│   (64 frames → 3D ConvNet)          │
└─────────────────────────────────────┘
    ↓ (if confidence < 0.6)
┌─────────────────────────────────────┐
│       Pose-LSTM Fallback            │
│   (MediaPipe → LSTM → Attention)    │
└─────────────────────────────────────┘
    ↓
Sliding Window → Word Sequence
    ↓
┌─────────────────────────────────────┐
│     Groq LLM Sentence Correction    │
│   (ASL Gloss → Natural English)     │
└─────────────────────────────────────┘
    ↓
JSON Response
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Models

**Option A: Using the batch script (Windows)**
```cmd
cd backend
setup_sign_models.bat
```

**Option B: Using Python directly**
```bash
cd backend
python -m app.services.sign_language.download_models
```

**Option C: Manual download**
Download model weights to `backend/models/sign_language/`:
- `i3d_wlasl100_rgb.pth` - I3D model weights (~150MB)
- `pose_lstm_wlasl100.pth` - Pose-LSTM weights (~10MB)

## API Endpoints

### Check Service Status
```
GET /sign-video/status
```
Returns model loading status and availability.

### Recognize from Video URL
```
POST /sign-video/recognize
Content-Type: application/json

{
    "video_url": "https://example.com/video.mp4",
    "use_sentence_correction": true
}
```

### Recognize from Uploaded Video
```
POST /sign-video/recognize/upload
Content-Type: multipart/form-data

file: <video file>
use_sentence_correction: true
```

### Stream Individual Frames (Real-time)
```
POST /sign-video/stream/frame
Content-Type: multipart/form-data

frame: <image file>
session_id: "user-session-123"
```

### Clear Frame Buffer
```
POST /sign-video/stream/clear?session_id=user-session-123
```

### Get Vocabulary
```
GET /sign-video/vocabulary
```
Returns all 100 supported ASL words.

## Response Format

```json
{
    "recognized_words": [
        {"word": "hello", "confidence": 0.92, "timestamp": 0.0},
        {"word": "how", "confidence": 0.88, "timestamp": 2.1},
        {"word": "you", "confidence": 0.91, "timestamp": 4.2}
    ],
    "sentence": "Hello, how are you?",
    "confidence": 0.90,
    "models_used": ["i3d"],
    "processing_time": 1.234
}
```

## Supported Vocabulary (WLASL-100)

The model recognizes 100 common ASL words including:

**Greetings:** hello, goodbye, please, thank_you, sorry, yes, no

**People:** father, mother, brother, sister, friend, teacher, student, baby

**Actions:** help, want, need, like, love, eat, drink, go, come, work, play, learn

**Questions:** what, where, when, who, why, how

**Time:** now, later, tomorrow, yesterday, morning, night

**And many more...** Use `GET /sign-video/vocabulary` for the complete list.

## Configuration

Settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `frame_extraction_fps` | 15 | Frames per second to extract |
| `sliding_window_duration` | 2.0 | Seconds per recognition window |
| `fallback_threshold` | 0.6 | Min confidence before fallback |
| `input_size` | 224 | Frame size (224×224 pixels) |

## Model Training Sources

- **I3D Architecture**: Based on [Kinetics I3D](https://github.com/deepmind/kinetics-i3d)
- **WLASL Dataset**: [World-Level ASL](https://dxli94.github.io/WLASL/)
- **Pose-LSTM**: Custom bidirectional LSTM with attention

## Troubleshooting

### Models not loading
1. Check that model files exist in `backend/models/sign_language/`
2. Run `python -m app.services.sign_language.download_models` to download
3. Check disk space (need ~200MB for both models)

### Low recognition accuracy
1. Ensure good lighting in the video
2. Keep hands clearly visible
3. Sign at a moderate pace
4. Use the supported vocabulary

### Memory issues
1. The I3D model requires ~1GB RAM for inference
2. For low-memory systems, set `i3d_config.enabled = False` in config

## License

Models are released under Apache 2.0 license. See individual model repositories for details.
