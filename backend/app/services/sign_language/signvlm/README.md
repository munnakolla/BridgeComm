# SignVLM Integration for BridgeComm

This module integrates the **SignVLM** (Sign Vision Language Model) for video-based sign language recognition.

Based on: ["SignVLM: A Pre-trained Large Vision Model for Sign Language Recognition"](https://github.com/Hamzah-Luqman/signVLM) by Hamzah Luqman.

## Overview

SignVLM uses a CLIP-based Vision Transformer backbone with a temporal decoder to recognize sign language from video. It achieves state-of-the-art results on multiple sign language datasets.

### Architecture

```
Video Input (N, C, T, H, W)
    │
    ▼
┌─────────────────────────────┐
│  CLIP ViT-L/14 Backbone     │  ◄── Frozen, FP16
│  (24 Transformer Layers)    │
└─────────────────────────────┘
    │
    ▼ Features from last 4 layers
┌─────────────────────────────┐
│  EVL Temporal Decoder       │
│  • Temporal Conv1D          │
│  • Temporal Position Embed  │
│  • Cross-Attention          │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Classification Head        │
│  LayerNorm → Dropout → FC   │
└─────────────────────────────┘
    │
    ▼
Output: Class Probabilities
```

## Quick Start

### 1. Download Models

First, download the CLIP backbone weights:

```bash
cd backend
python -m app.services.sign_language.signvlm.download_models
```

This will:
- Create the `models/signvlm/` directory
- Download CLIP ViT-L/14 backbone (~890 MB)
- Create WLASL-100 class mapping

### 2. Test the Integration

```bash
# Run quick check
python -m tests.test_signvlm

# Or run full tests with pytest
pytest tests/test_signvlm.py -v
```

### 3. Start the Server

```bash
cd backend
uvicorn app.main:app --reload
```

### 4. Use the API

```bash
# Check status
curl http://localhost:8000/sign-video/signvlm/status

# Load model
curl -X POST http://localhost:8000/sign-video/signvlm/load

# Recognize from video
curl -X POST http://localhost:8000/sign-video/signvlm/recognize \
  -H "Content-Type: application/json" \
  -d '{"video_base64": "<base64-video>", "top_k": 5}'
```

## API Endpoints

### GET `/sign-video/signvlm/status`
Get model status and configuration.

### POST `/sign-video/signvlm/load`
Explicitly load the model (optional, auto-loads on first request).

### POST `/sign-video/signvlm/recognize`
Recognize sign language from base64 video.

**Request:**
```json
{
  "video_base64": "base64-encoded-video",
  "top_k": 5,
  "use_groq_correction": true
}
```

**Response:**
```json
{
  "recognized_words": ["hello", "thank"],
  "sentence": "Hello, thank you!",
  "confidence": 0.87,
  "predictions": [
    {"gloss": "hello", "confidence": 0.87},
    {"gloss": "help", "confidence": 0.05}
  ],
  "model": "SignVLM"
}
```

### POST `/sign-video/signvlm/recognize/upload`
Recognize from uploaded video file.

### POST `/sign-video/signvlm/recognize/frames`
Recognize from a sequence of base64 images.

### GET `/sign-video/signvlm/vocabulary`
Get the list of recognizable sign glosses.

### GET `/sign-video/signvlm/health`
Health check endpoint.

## Configuration

The service can be configured via `SignVLMConfig`:

```python
from app.services.sign_language.signvlm.signvlm_service import SignVLMConfig

config = SignVLMConfig(
    backbone_name="ViT-L/14-lnpre",  # CLIP backbone
    num_frames=24,                    # Frames per video clip
    input_size=(224, 224),            # Input resolution
    fps=15,                           # Target FPS for extraction
    min_confidence=0.3,               # Minimum confidence threshold
    use_cuda=False,                   # Use GPU if available
)
```

## Supported Datasets

SignVLM can be trained/fine-tuned on:

- **WLASL** (Word-Level American Sign Language) - 100/300/1000/2000 classes
- **KArSL** (King Abdullah Arabic Sign Language) - 502 classes
- **AUTSL** (Ankara University Turkish Sign Language) - 226 classes
- **LSA64** (Argentinian Sign Language) - 64 classes

## File Structure

```
backend/app/services/sign_language/signvlm/
├── __init__.py              # Module exports
├── model.py                 # EVLTransformer model
├── vision_transformer.py    # ViT backbone components
├── weight_loaders.py        # CLIP/MAE weight loading
├── signvlm_service.py       # Inference service
├── download_models.py       # Model download script
└── README.md                # This file

backend/models/signvlm/
├── ViT-L-14.pt             # CLIP backbone weights
├── wlasl100_classes.json   # Class mapping
└── signvlm_wlasl100.pth    # Fine-tuned checkpoint (optional)
```

## Training Your Own Model

To fine-tune on your own sign language dataset:

1. Clone the original SignVLM repo:
   ```bash
   git clone https://github.com/Hamzah-Luqman/signVLM.git
   ```

2. Prepare your dataset in the format:
   ```
   video_path_1    label_1
   video_path_2    label_2
   ```

3. Train using the provided scripts:
   ```bash
   python -m torch.distributed.run --nproc_per_node 1 main.py \
     --backbone "ViT-L/14-lnpre" \
     --backbone_path /path/to/ViT-L-14.pt \
     --num_classes 100 \
     --train_list_path train.txt \
     --val_list_path val.txt \
     --num_frames 24 \
     --num_steps 10000
   ```

4. Copy the trained checkpoint to `backend/models/signvlm/signvlm_custom.pth`

5. Update the class mapping JSON file

## Troubleshooting

### Model not loading
- Ensure CLIP backbone is downloaded
- Check GPU memory (model requires ~4GB)
- Try CPU mode: set `use_cuda=False`

### Poor recognition accuracy
- Ensure video is 1.5-3 seconds long
- Center the signer in frame
- Good lighting helps
- Try different `num_frames` values

### Out of memory
- Reduce batch size
- Use CPU instead of GPU
- Reduce video resolution

## References

- [SignVLM Paper](https://github.com/Hamzah-Luqman/signVLM)
- [CLIP (OpenAI)](https://github.com/openai/CLIP)
- [WLASL Dataset](https://dxli94.github.io/WLASL/)

## License

This integration follows the original SignVLM repository license.
CLIP weights are provided by OpenAI under their license.
