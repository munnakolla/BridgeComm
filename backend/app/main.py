"""
BridgeComm AI Backend - Main Application
Azure-powered bi-directional communication system
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.api import (
    speech_router,
    symbols_router,
    sign_language_router,
    sign_video_router,
    behavior_router,
    text_generation_router,
    feedback_router,
    users_router,
    translator_router,
)
from app.api.routes.groq import router as groq_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    print("🚀 BridgeComm AI Backend starting up...")
    print("📡 Connecting to Azure services...")
    
    # Run startup validation checks
    from app.core.startup_checks import run_startup_checks
    checks_passed = run_startup_checks(verbose=True)
    
    if not checks_passed:
        print("\n⚠️  WARNING: Some critical checks failed. Review the messages above.")
        print("⚠️  The server will start but some features may not work correctly.\n")
    
    # Initialize services (lazy loading will handle actual connections)
    settings = get_settings()
    print(f"✅ Configuration loaded (Debug: {settings.app_debug})")
    
    yield
    
    # Shutdown
    print("👋 BridgeComm AI Backend shutting down...")


# Create FastAPI application
app = FastAPI(
    title="BridgeComm AI Backend",
    description="""
    Azure-powered backend for bi-directional communication between 
    normal users and communication-disabled users.
    
    ## Features
    
    - **Speech-to-Text**: Convert spoken words to text using Azure Cognitive Services
    - **Text-to-Symbols**: Simplify text and convert to visual symbols (ARASAAC)
    - **Sign Language Recognition**: Detect and interpret sign language gestures
    - **Behavior Recognition**: Interpret touch, eye tracking, and behavioral patterns
    - **Natural Language Generation**: Convert intents to natural speech
    - **Personalization**: Per-user AI model adaptation
    
    ## Workflows
    
    ### Normal User → Disabled User
    1. Speech → Text (Azure Speech-to-Text)
    2. Text → Simplified Text (Azure OpenAI)
    3. Simplified Text → Symbols (ARASAAC mapping)
    
    ### Disabled User → Normal User
    1. Gesture/Behavior → Features (MediaPipe/OpenCV)
    2. Features → Intent (Azure ML)
    3. Intent → Natural Text (Azure OpenAI)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "An internal error occurred",
            "status_code": 500
        }
    )


# Register routers
app.include_router(speech_router)
app.include_router(symbols_router)
app.include_router(sign_language_router)
app.include_router(sign_video_router)  # Video-based sign language with pretrained models
app.include_router(behavior_router)
app.include_router(text_generation_router)
app.include_router(feedback_router)
app.include_router(users_router)
app.include_router(translator_router)
app.include_router(groq_router)  # Groq AI routes


# Root endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "BridgeComm AI Backend",
        "version": "1.0.0",
        "description": "Azure-powered bi-directional communication API",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all dependent services.
    """
    services = {}
    
    # Check services configuration
    try:
        settings = get_settings()
        
        # Groq AI (primary AI service)
        services["groq_ai"] = "configured" if settings.groq_api_key else "not_configured"
        
        # Local ML models (MediaPipe, DeepFace)
        from app.services.vision_service import MEDIAPIPE_AVAILABLE, DEEPFACE_AVAILABLE, CUSTOM_EMOTION_AVAILABLE
        services["mediapipe_gestures"] = "available" if MEDIAPIPE_AVAILABLE else "not_available"
        services["deepface_emotion"] = "available" if DEEPFACE_AVAILABLE else "not_available"
        
        # Custom trained models (FER-2013, ASL)
        services["custom_emotion_model"] = "available" if CUSTOM_EMOTION_AVAILABLE else "not_trained"
        
        # Check for ASL model
        import os
        asl_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       "models", "asl_gesture_recognizer.task")
        services["custom_asl_model"] = "available" if os.path.exists(asl_model_path) else "not_trained"
        
        # Optional Azure services (only for backup/fallback)
        services["azure_speech"] = "configured" if settings.azure_speech_key else "optional_not_set"
        services["azure_translator"] = "configured" if settings.azure_translator_key else "optional_not_set"
        
    except Exception as e:
        services["configuration"] = f"error: {str(e)}"
    
    # Core services: Groq AI + MediaPipe for gestures
    groq_ok = services.get("groq_ai") == "configured"
    mediapipe_ok = services.get("mediapipe_gestures") == "available"
    status = "healthy" if (groq_ok and mediapipe_ok) else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        services=services
    )


@app.get("/api/endpoints", tags=["Health"])
async def list_endpoints():
    """List all available API endpoints."""
    return {
        "endpoints": {
            "speech": {
                "POST /azure/speech-to-text": "Convert speech to text",
                "POST /azure/speech-to-text/upload": "Upload audio file for conversion",
                "POST /azure/text-to-speech": "Convert text to speech audio"
            },
            "symbols": {
                "POST /azure/text-to-symbols": "Convert text to visual symbols",
                "GET /azure/symbols/categories": "Get symbol categories",
                "GET /azure/symbols/category/{category}": "Get symbols by category",
                "GET /azure/symbols/search/{query}": "Search for symbols"
            },
            "sign_language": {
                "POST /azure/sign-to-intent": "Recognize sign language gestures",
                "POST /azure/sign-to-intent/upload": "Upload image for sign recognition",
                "POST /azure/analyze-gesture": "Low-level gesture analysis"
            },
            "behavior": {
                "POST /azure/behavior-to-intent": "Interpret behavioral patterns"
            },
            "text_generation": {
                "POST /azure/generate-text": "Generate natural text from intent",
                "POST /azure/intent-to-full-message": "Convert intent to complete message"
            },
            "feedback": {
                "POST /azure/feedback": "Submit user feedback",
                "GET /azure/feedback/stats/{user_id}": "Get feedback statistics"
            },
            "users": {
                "POST /users/": "Create new user",
                "GET /users/{user_id}": "Get user profile",
                "PATCH /users/{user_id}": "Update user profile",
                "DELETE /users/{user_id}": "Delete user",
                "GET /users/{user_id}/personalization": "Get personalization data",
                "GET /users/{user_id}/stats": "Get user statistics"
            }
        }
    }
