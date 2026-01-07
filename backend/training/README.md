# BridgeComm Model Training Pipeline

Train custom ML models for better sign language and emotion detection accuracy.

## 🎯 Overview

Custom training provides:
- **ASL Gesture Recognition**: 87,000+ images for 29 classes (A-Z + space/delete/nothing)
- **Emotion Detection**: 35,000+ images for 7 emotions (replaces DeepFace for 10x faster inference)

## 📋 Prerequisites

1. **Python 3.10+** with TensorFlow
2. **Kaggle Account** for dataset downloads
3. **GPU (Optional)**: CUDA-enabled GPU speeds up training significantly

## 🚀 Quick Start (Windows)

1. **Download datasets from Kaggle** (see below)
2. **Extract to `training/datasets/`**
3. **Run**: `train_all.bat`

---

## 📥 Datasets Required

### 1. ASL Alphabet Dataset
| Info | Details |
|------|---------|
| **URL** | https://www.kaggle.com/datasets/grassknoted/asl-alphabet |
| **Size** | ~1GB (87,000 images) |
| **Classes** | 29 (A-Z + space, delete, nothing) |

**Extract to:**
```
training/datasets/asl-alphabet/asl_alphabet_train/
  ├── A/
  ├── B/
  ├── C/
  └── ... (29 folders)
```

### 2. FER-2013 Emotion Dataset
| Info | Details |
|------|---------|
| **URL** | https://www.kaggle.com/datasets/msambare/fer2013 |
| **Size** | ~60MB (35,000 images) |
| **Classes** | 7 (angry, disgust, fear, happy, neutral, sad, surprise) |

**Extract to:**
```
training/datasets/fer2013/
  ├── train/
  │   ├── angry/
  │   ├── happy/
  │   └── ... (7 folders)
  └── test/
      └── ... (same structure)
```

---

## 🔧 Manual Training Steps

### Step 1: Install Dependencies
```bash
cd backend/training
pip install -r requirements.txt
```

### Step 2: Train ASL Model (~15-30 min)
```bash
python train_asl_gestures.py
```
**Output**: `outputs/asl_gesture_recognizer.task`

### Step 3: Train Emotion Model (~30-60 min)
```bash
python train_emotion_model.py
```
**Output**: `outputs/emotion_model.h5`

### Step 4: Deploy to Backend
```bash
python deploy_models.py
```

### Step 5: Restart Backend
```bash
cd ..
python -m uvicorn app.main:app --reload
```

---

## 📊 Expected Results

| Model | Accuracy | Size | Speed |
|-------|----------|------|-------|
| ASL Gestures | 85-95% | 5-10MB | ~50ms/image |
| Emotion (CNN) | 60-70% | 10-20MB | ~10ms/image |
| Emotion (DeepFace) | 65-70% | 500MB+ | ~100ms/image |

---

## 🔍 Troubleshooting

### Dataset Not Found
Ensure correct folder structure:
```
training/datasets/asl-alphabet/asl_alphabet_train/A/  ← Check this exists
training/datasets/fer2013/train/angry/  ← And this
```

### Out of Memory
Reduce batch size in training scripts:
- `train_asl_gestures.py`: `MAX_SAMPLES_PER_CLASS = 200`
- `train_emotion_model.py`: `BATCH_SIZE = 32`

### TensorFlow Not Found
```bash
pip install tensorflow>=2.15.0
```

### GPU Not Detected
```bash
pip install tensorflow[and-cuda]
```

---

## 📁 File Structure

```
training/
├── README.md              # This file
├── requirements.txt       # Dependencies
├── train_all.bat          # One-click Windows training
├── train_asl_gestures.py  # ASL model training
├── train_emotion_model.py # Emotion model training
├── deploy_models.py       # Copy models to backend
├── datasets/              # ← Put Kaggle data here
│   ├── asl-alphabet/
│   └── fer2013/
└── outputs/               # Trained models (auto-created)
```

---

## ✅ After Training

The health endpoint (`/health`) will show:
```json
{
  "custom_emotion_model": "available",
  "custom_asl_model": "available"
}
```
