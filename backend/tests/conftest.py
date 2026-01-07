"""
Tests configuration and fixtures.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import os

# Set test environment variables before importing app
os.environ["AZURE_SPEECH_KEY"] = "test-key"
os.environ["AZURE_SPEECH_REGION"] = "eastus"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net"
os.environ["AZURE_COSMOS_ENDPOINT"] = "https://test.documents.azure.com:443/"
os.environ["AZURE_COSMOS_KEY"] = "test-key"
os.environ["APP_SECRET_KEY"] = "test-secret-key-12345678901234567890"


@pytest.fixture
def client():
    """Create test client."""
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_azure_speech():
    """Mock Azure Speech SDK."""
    with patch("app.services.speech_service.speechsdk") as mock:
        yield mock


@pytest.fixture
def mock_azure_openai():
    """Mock Azure OpenAI client."""
    with patch("app.core.azure_clients.AzureOpenAI") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_cosmos():
    """Mock Azure Cosmos DB client."""
    with patch("app.core.azure_clients.CosmosClient") as mock:
        mock_client = MagicMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_blob_storage():
    """Mock Azure Blob Storage client."""
    with patch("app.core.azure_clients.BlobServiceClient") as mock:
        mock_client = MagicMock()
        mock.from_connection_string.return_value = mock_client
        yield mock_client


@pytest.fixture
def sample_audio_base64():
    """Sample base64 encoded audio for testing."""
    # This is just a minimal WAV header for testing
    import base64
    # Minimal WAV file (44 bytes header + 0 data bytes)
    wav_bytes = bytes([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x00, 0x00, 0x00,  # File size
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # Chunk size
        0x01, 0x00,              # Audio format (PCM)
        0x01, 0x00,              # Number of channels
        0x44, 0xAC, 0x00, 0x00,  # Sample rate
        0x88, 0x58, 0x01, 0x00,  # Byte rate
        0x02, 0x00,              # Block align
        0x10, 0x00,              # Bits per sample
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x00, 0x00, 0x00,  # Data size
    ])
    return base64.b64encode(wav_bytes).decode('utf-8')


@pytest.fixture
def sample_image_base64():
    """Sample base64 encoded image for testing."""
    import base64
    # Minimal 1x1 PNG
    png_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
        0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
        0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,
        0x44, 0xAE, 0x42, 0x60, 0x82,
    ])
    return base64.b64encode(png_bytes).decode('utf-8')
