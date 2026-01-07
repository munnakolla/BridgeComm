# BridgeComm AI Backend

Azure-powered backend for bi-directional communication between normal users and communication-disabled users.

## 🎯 Overview

This backend provides AI-powered APIs to:
- Convert speech to simplified text and visual symbols
- Recognize sign language gestures and convert to natural language
- Interpret behavioral patterns (touch, eye tracking, facial expressions)
- Personalize AI models for each user
- Enable natural bi-directional communication

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BridgeComm Frontend                              │
│                    (React Native / Expo App)                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    BridgeComm Backend (FastAPI)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ Speech API   │  │ Symbols API  │  │ Sign Lang API│  │ Behavior API│ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                 │                 │         │
│  ┌──────┴─────────────────┴─────────────────┴─────────────────┴──────┐ │
│  │                         Service Layer                              │ │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐  │ │
│  │  │Speech Svc  │ │OpenAI Svc  │ │Vision Svc  │ │ Symbol Svc     │  │ │
│  │  │(Azure STT) │ │(Azure GPT) │ │(MediaPipe) │ │ (ARASAAC)      │  │ │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Azure Services                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │Azure Speech  │  │Azure OpenAI  │  │ Azure Blob   │  │Azure Cosmos │ │
│  │ Services     │  │   (GPT-4)    │  │  Storage     │  │    DB       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Azure account with the following services:
  - Azure Speech Services
  - Azure OpenAI
  - Azure Blob Storage
  - Azure Cosmos DB

### Local Development

1. **Clone and navigate to backend:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Azure credentials
   ```

5. **Run the server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access API documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker Development

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d
```

## 📡 API Endpoints

### Speech APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/azure/speech-to-text` | POST | Convert speech audio to text |
| `/azure/speech-to-text/upload` | POST | Upload audio file for conversion |
| `/azure/text-to-speech` | POST | Convert text to speech audio |

### Symbol APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/azure/text-to-symbols` | POST | Convert text to simplified text + symbols |
| `/azure/symbols/categories` | GET | Get all symbol categories |
| `/azure/symbols/category/{category}` | GET | Get symbols by category |
| `/azure/symbols/search/{query}` | GET | Search for symbols |

### Sign Language APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/azure/sign-to-intent` | POST | Recognize sign language gestures |
| `/azure/sign-to-intent/upload` | POST | Upload image for recognition |
| `/azure/analyze-gesture` | POST | Low-level gesture analysis |

### Behavior APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/azure/behavior-to-intent` | POST | Interpret behavioral patterns |

### Text Generation APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/azure/generate-text` | POST | Generate natural text from intent |
| `/azure/intent-to-full-message` | POST | Convert intent to complete message |

### Feedback APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/azure/feedback` | POST | Submit user feedback |
| `/azure/feedback/stats/{user_id}` | GET | Get feedback statistics |

### User APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/users/` | POST | Create new user |
| `/users/{user_id}` | GET | Get user profile |
| `/users/{user_id}` | PATCH | Update user profile |
| `/users/{user_id}` | DELETE | Delete user |
| `/users/{user_id}/personalization` | GET | Get personalization data |
| `/users/{user_id}/stats` | GET | Get user statistics |

## 🔧 Configuration

### Required Environment Variables

```env
# Azure Speech Services
AZURE_SPEECH_KEY=your_key
AZURE_SPEECH_REGION=eastus

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_STORAGE_CONTAINER_NAME=bridgecomm-media

# Azure Cosmos DB
AZURE_COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
AZURE_COSMOS_KEY=your_key
AZURE_COSMOS_DATABASE_NAME=bridgecomm
AZURE_COSMOS_CONTAINER_NAME=users

# Application
APP_SECRET_KEY=your_random_secret_key
APP_DEBUG=false
```

## ☁️ Azure Deployment

### Option 1: Azure App Service (Recommended)

1. **Using Azure CLI:**
   ```bash
   cd azure
   ./deploy.ps1  # Windows
   # or
   ./deploy.sh   # Linux/Mac
   ```

2. **Using ARM Template:**
   ```bash
   az deployment group create \
     --resource-group bridgecomm-rg \
     --template-file azure/arm-template.json \
     --parameters @azure/parameters.json
   ```

### Option 2: Azure Container Instances

```bash
# Build and push Docker image
az acr build --registry yourregistry --image bridgecomm-api:latest .

# Deploy container
az container create \
  --resource-group bridgecomm-rg \
  --name bridgecomm-api \
  --image yourregistry.azurecr.io/bridgecomm-api:latest \
  --dns-name-label bridgecomm-api \
  --ports 8000
```

### Option 3: Azure Kubernetes Service

See `azure/k8s/` for Kubernetes manifests (coming soon).

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_speech.py
```

## 📊 Communication Workflows

### Normal User → Disabled User

```
1. Frontend sends audio → POST /azure/speech-to-text
   Response: { "text": "Please drink water now", "confidence": 0.95 }

2. Frontend sends text → POST /azure/text-to-symbols
   Response: {
     "simplified_text": "Drink water",
     "symbols": [
       { "id": "2408", "name": "drink", "url": "..." },
       { "id": "2415", "name": "water", "url": "..." }
     ],
     "confidence": 0.97
   }

3. Frontend displays simplified text + symbols to disabled user
```

### Disabled User → Normal User

```
1. Frontend captures gesture image → POST /azure/sign-to-intent
   Response: {
     "intent": "request_water",
     "text": "I want water",
     "confidence": 0.94
   }

2. (Optional) Frontend requests natural phrasing → POST /azure/generate-text
   Response: {
     "text": "Excuse me, could I please have some water?",
     "alternatives": ["I'd like some water", "Water, please"]
   }

3. Frontend speaks or displays the text to normal user
```

## 🔒 Security Best Practices

1. **Use Azure Key Vault** for storing secrets in production
2. **Enable HTTPS** on all endpoints
3. **Implement rate limiting** for public APIs
4. **Use Azure AD** for authentication (optional)
5. **Enable Azure Monitor** for logging and alerts
6. **Isolate user data** using container-level access in Cosmos DB

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── speech.py
│   │       ├── symbols.py
│   │       ├── sign_language.py
│   │       ├── behavior.py
│   │       ├── text_generation.py
│   │       ├── feedback.py
│   │       └── users.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Configuration
│   │   └── azure_clients.py # Azure SDK clients
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   └── services/
│       ├── __init__.py
│       ├── speech_service.py
│       ├── openai_service.py
│       ├── symbol_service.py
│       ├── vision_service.py
│       ├── user_service.py
│       └── storage_service.py
├── azure/
│   ├── arm-template.json
│   ├── deploy.sh
│   └── deploy.ps1
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
- Open a GitHub issue
- Contact the development team
