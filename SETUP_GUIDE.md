# BridgeComm Setup & Configuration Guide

## Prerequisites

### Required Azure Services

You need the following Azure API keys configured in `backend/.env`:

#### ✅ Currently Configured:
- **Azure Speech Services** - Speech-to-Text and Text-to-Speech
  - Key: `AZURE_SPEECH_KEY`
  - Region: `AZURE_SPEECH_REGION` (centralindia)

- **Azure Vision Services** - Gesture and face analysis
  - Key: `AZURE_VISION_KEY`
  - Endpoint: `AZURE_VISION_ENDPOINT`

- **Azure Language Services** - Text analysis and simplification
  - Key: `AZURE_LANGUAGE_KEY`
  - Endpoint: `AZURE_LANGUAGE_ENDPOINT`

- **Azure Translator** - Multi-language support
  - Key: `AZURE_TRANSLATOR_KEY`
  - Endpoint: `AZURE_TRANSLATOR_ENDPOINT`
  - Region: `AZURE_TRANSLATOR_REGION`

#### ⚠️ Optional (Not Currently Set):
- **Azure OpenAI** - Enhanced NLP (can use Azure Language instead)
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_DEPLOYMENT_NAME`

- **Azure Blob Storage** - Media file storage
  - `AZURE_STORAGE_CONNECTION_STRING`

- **Azure Cosmos DB** - User data persistence
  - `AZURE_COSMOS_ENDPOINT`
  - `AZURE_COSMOS_KEY`

---

## Backend Setup

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install FFmpeg (Required for Audio Processing)

**Windows:**
1. Download from: https://www.gyan.dev/ffmpeg/builds/
2. Extract and add `bin` folder to PATH
3. Verify: `ffmpeg -version`

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

### 3. Verify MediaPipe Gesture Model

The gesture recognizer model should be at:
```
backend/models/gesture_recognizer.task
```

If missing, download from MediaPipe:
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task

### 4. Run Startup Validation

```bash
cd backend
python -m app.core.startup_checks
```

This will verify:
- ✓ Azure credentials configured
- ✓ ffmpeg available
- ✓ pydub installed
- ✓ MediaPipe installed
- ✓ OpenCV installed
- ✓ Gesture model exists
- ⚠ DeepFace (optional emotion detection)

### 5. Start Backend Server

**Windows:**
```bash
cd backend
run_server.bat
```

**macOS/Linux:**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## Frontend Setup

### 1. Update Environment Variables

Edit `.env` in project root:

```env
# Use your machine's IP for mobile device testing
EXPO_PUBLIC_API_BASE_URL=http://192.168.29.2:8000

# Or for localhost testing (web/emulator only):
# EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
```

**Finding your IP:**
- Windows: `ipconfig` (look for IPv4 Address)
- macOS/Linux: `ifconfig` or `ip addr`

### 2. Install Dependencies

```bash
npm install
```

### 3. Start Expo Development Server

```bash
npm start
# or
npx expo start
```

Then:
- Press `w` for web browser
- Press `a` for Android emulator
- Press `i` for iOS simulator
- Scan QR code with Expo Go app on mobile

---

## Key Features & Usage

### 1. **Speech to Symbols** (Speak Tab)
- Tap mic button to record speech
- Or type text directly
- Converts to simplified text + visual symbols
- Works offline with local fallback

### 2. **Sign Language Detection** (Communicate Tab)
- **Continuous Video Detection**: Gestures are accumulated over 3-second window
- Multiple gestures form sentences automatically
- Examples:
  - Wave + Open Palm = "Hello, how are you?"
  - Thumbs Up + Thumbs Up = "Very good"
  - Prayer Hands + Wave = "Thank you, goodbye"
- Clear session to start new gesture sequence

### 3. **Eye Tracking / Emotion Detection**
- Detects facial expressions
- Converts emotions to natural phrases
- Multiple sentence variations

### 4. **Touch Input**
- Tap symbols to build message
- Symbols combine into sentence
- Speak output with TTS

---

## Troubleshooting

### Backend Issues

**"ffmpeg NOT found"**
- Install ffmpeg and add to PATH
- Restart terminal after installation

**"Gesture model NOT found"**
- Download gesture_recognizer.task
- Place in `backend/models/` folder

**"pydub NOT installed"**
```bash
pip install pydub
```

**"Azure credentials missing"**
- Check `backend/.env` file exists
- Verify all required keys are set
- Keys should not have quotes or extra spaces

### Frontend Issues

**"Backend unreachable" banner**
- Check backend is running: http://localhost:8000/health
- Verify `EXPO_PUBLIC_API_BASE_URL` matches backend URL
- Use machine IP (not localhost) for mobile testing
- Restart Expo server after changing .env

**Speech recognition not working**
- Grant microphone permissions
- Check backend `AZURE_SPEECH_KEY` is set
- Verify audio format (backend logs will show conversion attempts)

**Sign language not detecting**
- Grant camera permissions
- Ensure MediaPipe model exists
- Check backend logs for vision service errors
- Good lighting and clear hand visibility improve accuracy

**Dark mode text not visible**
- Theme support is built-in
- Check Settings tab for theme options
- High contrast mode available

---

## API Key Requirements Summary

**To ask for these if not working:**

1. **Azure Speech Services** (REQUIRED)
   - Service: Cognitive Services - Speech
   - Needed for: Speech-to-text, Text-to-speech
   - Get from: Azure Portal > Create Speech Service

2. **Azure Vision Services** (REQUIRED)
   - Service: Cognitive Services - Computer Vision
   - Needed for: Gesture recognition, Face analysis
   - Get from: Azure Portal > Create Computer Vision

3. **Azure Language Services** (OPTIONAL but recommended)
   - Service: Cognitive Services - Language
   - Needed for: Text simplification, Key phrase extraction
   - Get from: Azure Portal > Create Language Service

4. **Azure OpenAI** (OPTIONAL alternative to Language)
   - Service: Azure OpenAI
   - Needed for: Advanced NLP, Text generation
   - Requires approval: https://aka.ms/oai/access

---

## Development Commands

```bash
# Backend
cd backend
python -m app.core.startup_checks  # Validate setup
python -m uvicorn app.main:app --reload  # Start server
python -m pytest  # Run tests

# Frontend
npm start  # Start Expo
npm run android  # Android emulator
npm run ios  # iOS simulator
npm run web  # Web browser

# Check for errors
npx expo doctor  # Diagnose Expo issues
```

---

## Architecture Notes

### Gesture Sequence Detection
- Gestures are tracked in 3-second sliding window
- Session-based: each user/device has own gesture history
- Automatic sentence formation from gesture patterns
- Duplicate consecutive gestures filtered
- Confidence threshold: 0.6 (adjustable)

### Speech Processing
- Audio converted to 16kHz mono WAV
- Supports: WAV, M4A, AAC, MP4, WebM, OGG, MP3
- Falls back to local symbols if backend unavailable
- Demo mode for offline testing

### Model Accuracy Improvements
- **Speech**: Uses Azure Speech with confidence scores
- **Gestures**: MediaPipe Gesture Recognizer (pre-trained)
- **Emotions**: DeepFace (optional, more accurate than basic CV)
- All services report confidence levels to frontend

---

For issues or questions, check:
- Backend logs: Terminal running uvicorn
- Frontend logs: Expo DevTools
- API documentation: http://localhost:8000/docs
