"""
Configuration settings for BridgeComm Backend.
Loads environment variables and provides typed configuration.
"""

from functools import lru_cache
from typing import List
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import json

# Get the backend directory (where .env should be)
BACKEND_DIR = Path(__file__).parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra fields from other .env files
    )
    
    # Azure Speech Services
    azure_speech_key: str = Field(..., description="Azure Speech API Key")
    azure_speech_region: str = Field(default="eastus", description="Azure region")
    
    # Azure Vision Services
    azure_vision_key: str | None = Field(default=None, description="Azure Vision API Key")
    azure_vision_endpoint: str | None = Field(default=None, description="Azure Vision endpoint URL")
    
    # Azure Language Services (for text analysis, key phrase extraction, summarization)
    azure_language_key: str | None = Field(default=None, description="Azure Language API Key")
    azure_language_endpoint: str | None = Field(default=None, description="Azure Language endpoint URL")
    
    # Azure Translator Services
    azure_translator_key: str | None = Field(default=None, description="Azure Translator API Key")
    azure_translator_endpoint: str = Field(default="https://api.cognitive.microsofttranslator.com/", description="Azure Translator endpoint")
    azure_translator_region: str = Field(default="centralindia", description="Azure Translator region")
    
    # Azure OpenAI (optional - can use Azure Language instead)
    azure_openai_endpoint: str | None = Field(default=None, description="Azure OpenAI endpoint URL")
    azure_openai_api_key: str | None = Field(default=None, description="Azure OpenAI API key")
    azure_openai_deployment_name: str = Field(default="gpt-4", description="Deployment name")
    azure_openai_api_version: str = Field(default="2024-02-01", description="API version")
    
    # OpenAI API (for Whisper STT/TTS - more reliable than Azure Speech)
    openai_api_key: str | None = Field(default=None, description="OpenAI API key for Whisper")
    
    # Groq API (Fast LLM and Whisper - recommended)
    groq_api_key: str | None = Field(default=None, description="Groq API key for fast LLM and Whisper")
    
    # Azure Blob Storage (optional - can use local storage)
    azure_storage_connection_string: str | None = Field(default=None, description="Storage connection string")
    azure_storage_container_name: str = Field(default="bridgecomm-media")
    
    # Azure Cosmos DB (optional - can use local storage)
    azure_cosmos_endpoint: str | None = Field(default=None, description="Cosmos DB endpoint")
    azure_cosmos_key: str | None = Field(default=None, description="Cosmos DB key")
    azure_cosmos_database_name: str = Field(default="bridgecomm")
    azure_cosmos_container_name: str = Field(default="users")
    
    # Azure Key Vault (optional)
    azure_key_vault_url: str | None = Field(default=None)
    
    # Application Settings
    app_secret_key: str = Field(default="dev-secret-key-change-in-production", description="Application secret key")
    app_debug: bool = Field(default=False)
    app_cors_origins: str = Field(
        default='["http://localhost:8081","http://localhost:19006"]'
    )
    
    # Redis (optional)
    redis_url: str | None = Field(default=None)
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from JSON string."""
        try:
            return json.loads(self.app_cors_origins)
        except (json.JSONDecodeError, TypeError):
            return ["http://localhost:8081", "http://localhost:19006"]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
