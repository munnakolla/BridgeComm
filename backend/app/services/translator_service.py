"""
Azure Translator Service
Handles text translation using Azure AI Translator.
"""

from typing import Optional, List, Dict, Any
import httpx

from app.core.config import get_settings


class TranslatorService:
    """Service for Azure AI Translator operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._has_translator = self.settings.azure_translator_key is not None
    
    async def translate(
        self,
        text: str,
        to_language: str = "en",
        from_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            to_language: Target language code (e.g., 'en', 'es', 'hi')
            from_language: Source language code (auto-detect if None)
            
        Returns:
            Translation result with detected language and translated text
        """
        if not self._has_translator:
            return {
                "original": text,
                "translated": text,
                "from_language": "en",
                "to_language": to_language,
                "error": "Translator not configured"
            }
        
        url = f"{self.settings.azure_translator_endpoint.rstrip('/')}/translate"
        
        params = {
            "api-version": "3.0",
            "to": to_language
        }
        if from_language:
            params["from"] = from_language
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.settings.azure_translator_key,
            "Ocp-Apim-Subscription-Region": self.settings.azure_translator_region,
            "Content-Type": "application/json"
        }
        
        payload = [{"text": text}]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, params=params, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if result and len(result) > 0:
                    translation = result[0]
                    detected = translation.get("detectedLanguage", {})
                    translations = translation.get("translations", [])
                    
                    return {
                        "original": text,
                        "translated": translations[0]["text"] if translations else text,
                        "from_language": detected.get("language", from_language or "unknown"),
                        "from_confidence": detected.get("score", 0),
                        "to_language": to_language
                    }
                
                return {"original": text, "translated": text, "error": "No translation result"}
            except Exception as e:
                return {"original": text, "translated": text, "error": str(e)}
    
    async def translate_batch(
        self,
        texts: List[str],
        to_language: str = "en",
        from_language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Translate multiple texts at once.
        
        Args:
            texts: List of texts to translate
            to_language: Target language code
            from_language: Source language code (auto-detect if None)
            
        Returns:
            List of translation results
        """
        if not self._has_translator:
            return [{"original": t, "translated": t, "error": "Translator not configured"} for t in texts]
        
        url = f"{self.settings.azure_translator_endpoint.rstrip('/')}/translate"
        
        params = {
            "api-version": "3.0",
            "to": to_language
        }
        if from_language:
            params["from"] = from_language
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.settings.azure_translator_key,
            "Ocp-Apim-Subscription-Region": self.settings.azure_translator_region,
            "Content-Type": "application/json"
        }
        
        payload = [{"text": t} for t in texts]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, params=params, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                translations = []
                for i, item in enumerate(result):
                    trans = item.get("translations", [])
                    detected = item.get("detectedLanguage", {})
                    translations.append({
                        "original": texts[i],
                        "translated": trans[0]["text"] if trans else texts[i],
                        "from_language": detected.get("language", from_language or "unknown"),
                        "to_language": to_language
                    })
                
                return translations
            except Exception as e:
                return [{"original": t, "translated": t, "error": str(e)} for t in texts]
    
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language info
        """
        if not self._has_translator:
            return {"language": "en", "confidence": 0, "error": "Translator not configured"}
        
        url = f"{self.settings.azure_translator_endpoint.rstrip('/')}/detect"
        
        params = {"api-version": "3.0"}
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.settings.azure_translator_key,
            "Ocp-Apim-Subscription-Region": self.settings.azure_translator_region,
            "Content-Type": "application/json"
        }
        
        payload = [{"text": text}]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, params=params, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if result and len(result) > 0:
                    detection = result[0]
                    return {
                        "language": detection.get("language", "unknown"),
                        "confidence": detection.get("score", 0),
                        "is_translation_supported": detection.get("isTranslationSupported", False)
                    }
                
                return {"language": "unknown", "confidence": 0}
            except Exception as e:
                return {"language": "unknown", "confidence": 0, "error": str(e)}
    
    async def get_supported_languages(self) -> Dict[str, Any]:
        """Get list of supported languages."""
        url = f"{self.settings.azure_translator_endpoint.rstrip('/')}/languages"
        
        params = {"api-version": "3.0", "scope": "translation"}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                result = response.json()
                
                return result.get("translation", {})
            except Exception as e:
                return {"error": str(e)}


# Singleton instance
_translator_service: Optional[TranslatorService] = None


def get_translator_service() -> TranslatorService:
    """Get the Translator service singleton."""
    global _translator_service
    if _translator_service is None:
        _translator_service = TranslatorService()
    return _translator_service
