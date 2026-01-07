# BridgeComm 🌉

**Bridging Communication Gaps** - An AI-powered accessibility app for bi-directional communication between verbal and non-verbal users.

![React Native](https://img.shields.io/badge/React_Native-Expo-blue)
![Python](https://img.shields.io/badge/Python-FastAPI-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Overview

BridgeComm enables seamless communication between:
- **Verbal users** → Speech converted to simplified text, symbols (AAC)
- **Non-verbal users** → Sign language & gestures converted to speech/text

### Key Features
- 🎤 **Speech-to-Text** with visual symbols (ARASAAC)
- ✋ **Sign Language Recognition** (ASL alphabet + gestures)
- 😊 **Emotion Detection** from facial expressions
- 🔊 **Text-to-Speech** for responses
- 🌍 **Multi-language Support**

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+
- **FFmpeg** (for audio processing)
- **Azure Account** with Speech/Language services (see [SETUP_GUIDE.md](SETUP_GUIDE.md))

### 1. Clone the Repository

```bash
git clone https://github.com/munnakolla/BridgeComm.git
cd BridgeComm
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download ML models (IMPORTANT!)
python setup_models.py

# Create .env file with your Azure credentials
# Copy from .env.example or see SETUP_GUIDE.md
```

### 3. Frontend Setup

```bash
# From project root
cd ..

# Install Node dependencies
npm install

# Start Expo development server
npx expo start
```

### 4. Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
npx expo start
```

Then scan the QR code with Expo Go app, or press:
- `a` for Android emulator
- `i` for iOS simulator
- `w` for web browser

---

## 📁 Project Structure

```
BridgeComm/
├── app/                    # React Native screens (Expo Router)
│   ├── index.tsx          # Entry/splash screen
│   ├── home.tsx           # Main dashboard
│   ├── speak.tsx          # Speech-to-symbols
│   ├── communicate.tsx    # Two-way communication
│   ├── symbols.tsx        # Symbol library
│   └── settings.tsx       # App settings
├── components/            # Reusable React components
├── services/              # API client services
├── hooks/                 # Custom React hooks
├── backend/               # FastAPI Python backend
│   ├── app/
│   │   ├── api/routes/    # API endpoints
│   │   ├── services/      # Business logic
│   │   └── models/        # Data schemas
│   ├── models/            # ML model files (not in repo)
│   └── training/          # Model training scripts
└── assets/                # Images, fonts, etc.
```

---

## 🔧 Environment Configuration

### Backend (.env)

Create `backend/.env` with your Azure credentials:

```env
# Azure Speech Services
AZURE_SPEECH_KEY=your_speech_key
AZURE_SPEECH_REGION=centralindia

# Azure Language Services
AZURE_LANGUAGE_KEY=your_language_key
AZURE_LANGUAGE_ENDPOINT=https://your-resource.cognitiveservices.azure.com

# Azure Translator
AZURE_TRANSLATOR_KEY=your_translator_key
AZURE_TRANSLATOR_ENDPOINT=https://api.cognitive.microsofttranslator.com
AZURE_TRANSLATOR_REGION=centralindia

# Optional: Groq for fast LLM inference
GROQ_API_KEY=your_groq_key
```

### Frontend (.env)

Create `.env` in project root:

```env
EXPO_PUBLIC_API_URL=http://localhost:8000
```

---

## 🤖 ML Models

Models are **NOT included** in the repository (too large). After cloning:

```bash
cd backend
python setup_models.py
```

This downloads:
- ✅ **gesture_recognizer.task** - MediaPipe hand gestures (auto-downloaded)
- ○ **asl_cnn_model.h5** - ASL alphabet recognition (optional, train locally)
- ○ **emotion_model.h5** - Emotion detection (optional, train locally)

### Training Custom Models

See [backend/training/README.md](backend/training/README.md) for:
- Downloading training datasets (Kaggle)
- Training ASL and emotion models
- Deploying trained models

---

## 📖 API Documentation

Once the backend is running:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

---

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/

# Verify startup
python -m app.core.startup_checks
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [ARASAAC](https://arasaac.org/) - AAC Symbols
- [MediaPipe](https://mediapipe.dev/) - Gesture Recognition
- [Azure Cognitive Services](https://azure.microsoft.com/en-us/products/cognitive-services/)
- [Expo](https://expo.dev/) - React Native framework

