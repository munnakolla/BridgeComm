"""
Azure Service Clients Manager
Provides singleton instances of Azure SDK clients.
Handles optional services gracefully.
"""

from functools import lru_cache
from typing import Optional
import azure.cognitiveservices.speech as speechsdk
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
from openai import AzureOpenAI

from app.core.config import get_settings


class AzureClients:
    """Manager for Azure service clients."""
    
    def __init__(self):
        self.settings = get_settings()
        self._speech_config: Optional[speechsdk.SpeechConfig] = None
        self._blob_client: Optional[BlobServiceClient] = None
        self._cosmos_client: Optional[CosmosClient] = None
        self._openai_client: Optional[AzureOpenAI] = None
    
    @property
    def speech_config(self) -> speechsdk.SpeechConfig:
        """Get Azure Speech SDK configuration."""
        if self._speech_config is None:
            self._speech_config = speechsdk.SpeechConfig(
                subscription=self.settings.azure_speech_key,
                region=self.settings.azure_speech_region
            )
            self._speech_config.speech_recognition_language = "en-US"
        return self._speech_config
    
    @property
    def blob_client(self) -> Optional[BlobServiceClient]:
        """Get Azure Blob Storage client (optional)."""
        if self._blob_client is None and self.settings.azure_storage_connection_string:
            self._blob_client = BlobServiceClient.from_connection_string(
                self.settings.azure_storage_connection_string
            )
        return self._blob_client
    
    @property
    def cosmos_client(self) -> Optional[CosmosClient]:
        """Get Azure Cosmos DB client (optional)."""
        if self._cosmos_client is None and self.settings.azure_cosmos_endpoint and self.settings.azure_cosmos_key:
            self._cosmos_client = CosmosClient(
                url=self.settings.azure_cosmos_endpoint,
                credential=self.settings.azure_cosmos_key
            )
        return self._cosmos_client
    
    @property
    def cosmos_database(self):
        """Get Cosmos DB database (returns None if not configured)."""
        if self.cosmos_client is None:
            return None
        return self.cosmos_client.get_database_client(
            self.settings.azure_cosmos_database_name
        )
    
    @property
    def cosmos_container(self):
        """Get Cosmos DB container (returns None if not configured)."""
        if self.cosmos_database is None:
            return None
        return self.cosmos_database.get_container_client(
            self.settings.azure_cosmos_container_name
        )
    
    @property
    def openai_client(self) -> Optional[AzureOpenAI]:
        """Get Azure OpenAI client (optional)."""
        if self._openai_client is None and self.settings.azure_openai_endpoint and self.settings.azure_openai_api_key:
            self._openai_client = AzureOpenAI(
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version
            )
        return self._openai_client
    
    def get_blob_container(self, container_name: Optional[str] = None):
        """Get a blob container client."""
        name = container_name or self.settings.azure_storage_container_name
        return self.blob_client.get_container_client(name)


@lru_cache()
def get_azure_clients() -> AzureClients:
    """Get cached Azure clients instance."""
    return AzureClients()
