"""
Storage Service
Handles file storage using Azure Blob Storage.
"""

import base64
import io
from datetime import datetime, timedelta
from typing import Optional
import uuid

from azure.storage.blob import BlobSasPermissions, generate_blob_sas

from app.core.azure_clients import get_azure_clients
from app.core.config import get_settings


class StorageService:
    """Service for Azure Blob Storage operations."""
    
    def __init__(self):
        self.clients = get_azure_clients()
        self.settings = get_settings()
    
    def _get_container_client(self, container_name: Optional[str] = None):
        """Get a blob container client."""
        name = container_name or self.settings.azure_storage_container_name
        return self.clients.blob_client.get_container_client(name)
    
    async def upload_audio(
        self,
        audio_data: bytes,
        user_id: Optional[str] = None,
        file_format: str = "wav"
    ) -> str:
        """
        Upload audio data to blob storage.
        
        Args:
            audio_data: Raw audio bytes
            user_id: Optional user ID for organization
            file_format: Audio file format
            
        Returns:
            URL to the uploaded audio file
        """
        # Generate unique blob name
        blob_name = self._generate_blob_name("audio", user_id, file_format)
        
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload
        blob_client.upload_blob(
            audio_data,
            blob_type="BlockBlob",
            content_settings={
                "content_type": f"audio/{file_format}"
            }
        )
        
        # Generate SAS URL
        return self._generate_sas_url(blob_name)
    
    async def upload_image(
        self,
        image_data: bytes,
        user_id: Optional[str] = None,
        file_format: str = "png"
    ) -> str:
        """
        Upload image data to blob storage.
        
        Args:
            image_data: Raw image bytes
            user_id: Optional user ID
            file_format: Image format
            
        Returns:
            URL to the uploaded image
        """
        blob_name = self._generate_blob_name("image", user_id, file_format)
        
        container_client = self._get_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        
        blob_client.upload_blob(
            image_data,
            blob_type="BlockBlob",
            content_settings={
                "content_type": f"image/{file_format}"
            }
        )
        
        return self._generate_sas_url(blob_name)
    
    async def upload_base64(
        self,
        base64_data: str,
        content_type: str = "audio/wav",
        user_id: Optional[str] = None
    ) -> str:
        """
        Upload base64 encoded data to blob storage.
        
        Args:
            base64_data: Base64 encoded data
            content_type: MIME content type
            user_id: Optional user ID
            
        Returns:
            URL to the uploaded file
        """
        # Decode base64
        data = base64.b64decode(base64_data)
        
        # Determine format from content type
        main_type, sub_type = content_type.split("/")
        
        if main_type == "audio":
            return await self.upload_audio(data, user_id, sub_type)
        elif main_type == "image":
            return await self.upload_image(data, user_id, sub_type)
        else:
            # Generic upload
            blob_name = self._generate_blob_name("file", user_id, sub_type)
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(blob_name)
            
            blob_client.upload_blob(
                data,
                blob_type="BlockBlob",
                content_settings={"content_type": content_type}
            )
            
            return self._generate_sas_url(blob_name)
    
    async def get_blob_url(self, blob_name: str) -> str:
        """
        Get a SAS URL for an existing blob.
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            SAS URL
        """
        return self._generate_sas_url(blob_name)
    
    async def delete_blob(self, blob_name: str) -> bool:
        """
        Delete a blob from storage.
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            True if deleted
        """
        try:
            container_client = self._get_container_client()
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            return True
        except Exception:
            return False
    
    def _generate_blob_name(
        self,
        prefix: str,
        user_id: Optional[str],
        extension: str
    ) -> str:
        """Generate a unique blob name."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        if user_id:
            return f"{prefix}/{user_id}/{timestamp}_{unique_id}.{extension}"
        return f"{prefix}/{timestamp}_{unique_id}.{extension}"
    
    def _generate_sas_url(
        self,
        blob_name: str,
        expiry_hours: int = 24
    ) -> str:
        """Generate a SAS URL for blob access."""
        # Parse connection string to get account info
        conn_str = self.settings.azure_storage_connection_string
        account_name = None
        account_key = None
        
        for part in conn_str.split(";"):
            if part.startswith("AccountName="):
                account_name = part.split("=", 1)[1]
            elif part.startswith("AccountKey="):
                account_key = part.split("=", 1)[1]
        
        if not account_name or not account_key:
            # Return direct URL without SAS if we can't parse connection string
            container_name = self.settings.azure_storage_container_name
            return f"https://{account_name or 'storage'}.blob.core.windows.net/{container_name}/{blob_name}"
        
        # Generate SAS token
        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=self.settings.azure_storage_container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
        )
        
        container_name = self.settings.azure_storage_container_name
        return f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"


# Singleton instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get the storage service singleton."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
