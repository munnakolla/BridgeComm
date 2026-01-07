"""
Azure Language Service
Handles text analysis, key phrase extraction, and summarization using Azure AI Language.
Replaces Azure OpenAI for text processing tasks.
"""

from typing import Optional, List, Dict, Any
import httpx

from app.core.config import get_settings


class LanguageService:
    """Service for Azure AI Language operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self._has_language = (
            self.settings.azure_language_key is not None and 
            self.settings.azure_language_endpoint is not None
        )
    
    async def extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from text using Azure Language.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key phrases
        """
        if not self._has_language:
            # Fallback to simple extraction
            return self._simple_keyword_extraction(text)
        
        endpoint = self.settings.azure_language_endpoint.rstrip('/')
        url = f"{endpoint}/language/:analyze-text?api-version=2023-04-01"
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.settings.azure_language_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "kind": "KeyPhraseExtraction",
            "parameters": {
                "modelVersion": "latest"
            },
            "analysisInput": {
                "documents": [
                    {"id": "1", "language": "en", "text": text}
                ]
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                documents = result.get("results", {}).get("documents", [])
                if documents:
                    key_phrases = documents[0].get("keyPhrases", [])
                    # If Azure returns empty, use simple extraction as fallback
                    if key_phrases:
                        return key_phrases
                    return self._simple_keyword_extraction(text)
                return self._simple_keyword_extraction(text)
            except Exception as e:
                print(f"Language API error: {e}")
                return self._simple_keyword_extraction(text)
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis result
        """
        if not self._has_language:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        endpoint = self.settings.azure_language_endpoint.rstrip('/')
        url = f"{endpoint}/language/:analyze-text?api-version=2023-04-01"
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.settings.azure_language_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "kind": "SentimentAnalysis",
            "parameters": {
                "modelVersion": "latest",
                "opinionMining": False
            },
            "analysisInput": {
                "documents": [
                    {"id": "1", "language": "en", "text": text}
                ]
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                documents = result.get("results", {}).get("documents", [])
                if documents:
                    doc = documents[0]
                    return {
                        "sentiment": doc.get("sentiment", "neutral"),
                        "confidence": doc.get("confidenceScores", {})
                    }
                return {"sentiment": "neutral", "confidence": {}}
            except Exception as e:
                print(f"Sentiment API error: {e}")
                return {"sentiment": "neutral", "confidence": {}}
    
    async def abstractive_summarize(self, text: str, sentence_count: int = 3) -> str:
        """
        Generate an abstractive summary of text.
        
        Args:
            text: Text to summarize
            sentence_count: Number of sentences in summary
            
        Returns:
            Summarized text
        """
        if not self._has_language or len(text) < 100:
            # Too short for summarization or service not available
            return text
        
        endpoint = self.settings.azure_language_endpoint.rstrip('/')
        url = f"{endpoint}/language/analyze-text/jobs?api-version=2023-04-01"
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.settings.azure_language_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "displayName": "Text Simplification",
            "analysisInput": {
                "documents": [
                    {"id": "1", "language": "en", "text": text}
                ]
            },
            "tasks": [
                {
                    "kind": "AbstractiveSummarization",
                    "taskName": "simplify",
                    "parameters": {
                        "sentenceCount": sentence_count
                    }
                }
            ]
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Start the job
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                # Get operation location
                operation_url = response.headers.get("operation-location")
                if not operation_url:
                    return self._simple_simplify(text)
                
                # Poll for results (max 30 seconds)
                import asyncio
                for _ in range(15):
                    await asyncio.sleep(2)
                    result_response = await client.get(operation_url, headers=headers)
                    result = result_response.json()
                    
                    status = result.get("status", "")
                    if status == "succeeded":
                        tasks = result.get("tasks", {}).get("items", [])
                        if tasks:
                            docs = tasks[0].get("results", {}).get("documents", [])
                            if docs and docs[0].get("summaries"):
                                return docs[0]["summaries"][0].get("text", text)
                        return text
                    elif status in ["failed", "cancelled"]:
                        return self._simple_simplify(text)
                
                return self._simple_simplify(text)
            except Exception as e:
                print(f"Summarization API error: {e}")
                return self._simple_simplify(text)
    
    async def simplify_text(self, text: str, level: str = "simple") -> str:
        """
        Simplify text for communication-disabled users.
        Uses key phrase extraction + summarization.
        
        Args:
            text: Text to simplify
            level: Simplification level (simple, very_simple, keywords_only)
            
        Returns:
            Simplified text
        """
        if level == "keywords_only":
            keywords = await self.extract_key_phrases(text)
            return " ".join(keywords[:7])
        
        # For simple/very_simple, try summarization first
        if len(text) > 100:
            sentence_count = 2 if level == "very_simple" else 3
            simplified = await self.abstractive_summarize(text, sentence_count)
            if simplified != text:
                return simplified
        
        # Fallback to simple processing
        return self._simple_simplify(text)
    
    def _simple_keyword_extraction(self, text: str) -> List[str]:
        """Simple keyword extraction fallback."""
        import re
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'into', 'through', 'during', 'before', 'after',
                     'above', 'below', 'between', 'under', 'and', 'but', 'or', 'if',
                     'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she',
                     'it', 'they', 'them', 'his', 'her', 'its', 'their', 'this', 'that'}
        
        # Important AAC keywords to always include
        aac_keywords = {'hungry', 'thirsty', 'tired', 'happy', 'sad', 'angry', 'scared',
                        'pain', 'hurt', 'help', 'want', 'need', 'go', 'stop', 'yes', 'no',
                        'please', 'thanks', 'hello', 'goodbye', 'water', 'food', 'drink',
                        'eat', 'sleep', 'bathroom', 'home', 'mom', 'dad', 'doctor', 'love',
                        'hot', 'cold', 'more', 'done', 'play', 'read', 'talk', 'walk'}
        
        keywords = []
        for w in words:
            # Always include AAC keywords
            if w in aac_keywords:
                keywords.append(w)
            # Include non-stop words that are substantial
            elif w not in stop_words and len(w) > 2:
                keywords.append(w)
        seen = set()
        unique = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique[:7]
    
    def _simple_simplify(self, text: str) -> str:
        """Simple text simplification fallback."""
        import re
        text = re.sub(r'[;:\-—]', ' ', text)
        sentences = re.split(r'[.!?]+', text)
        
        fillers = ['actually', 'basically', 'literally', 'honestly', 'obviously',
                  'definitely', 'absolutely', 'certainly', 'probably', 'perhaps',
                  'however', 'therefore', 'furthermore', 'moreover', 'nevertheless']
        
        simplified = []
        for sentence in sentences[:3]:  # Keep max 3 sentences
            sentence = sentence.strip()
            if not sentence:
                continue
            words = sentence.split()
            words = [w for w in words if w.lower() not in fillers]
            if words:
                simplified.append(' '.join(words))
        
        return '. '.join(simplified) + '.' if simplified else text


# Singleton instance
_language_service: Optional[LanguageService] = None


def get_language_service() -> LanguageService:
    """Get the Language service singleton."""
    global _language_service
    if _language_service is None:
        _language_service = LanguageService()
    return _language_service
