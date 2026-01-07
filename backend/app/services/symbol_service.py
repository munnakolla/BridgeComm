"""
Symbol Mapping Service
Maps text keywords to visual symbols using ARASAAC and custom mappings.
"""

from typing import List, Dict, Optional
import httpx
from app.models.schemas import Symbol


# Core symbol mapping database
# Maps keywords to ARASAAC symbol IDs or custom icons
SYMBOL_MAPPING: Dict[str, Dict[str, str]] = {
    # Basic needs
    "water": {"id": "2415", "name": "water", "category": "food_drink"},
    "drink": {"id": "2408", "name": "drink", "category": "actions"},
    "food": {"id": "2501", "name": "food", "category": "food_drink"},
    "eat": {"id": "2485", "name": "eat", "category": "actions"},
    "hungry": {"id": "6633", "name": "hungry", "category": "feelings"},
    "thirsty": {"id": "6634", "name": "thirsty", "category": "feelings"},
    "tired": {"id": "6635", "name": "tired", "category": "feelings"},
    "sleep": {"id": "4501", "name": "sleep", "category": "actions"},
    "bathroom": {"id": "2440", "name": "bathroom", "category": "places"},
    "toilet": {"id": "2441", "name": "toilet", "category": "objects"},
    
    # Emotions
    "happy": {"id": "2301", "name": "happy", "category": "feelings"},
    "sad": {"id": "2302", "name": "sad", "category": "feelings"},
    "angry": {"id": "2303", "name": "angry", "category": "feelings"},
    "scared": {"id": "2304", "name": "scared", "category": "feelings"},
    "pain": {"id": "2305", "name": "pain", "category": "feelings"},
    "hurt": {"id": "2306", "name": "hurt", "category": "feelings"},
    "love": {"id": "2307", "name": "love", "category": "feelings"},
    "like": {"id": "2308", "name": "like", "category": "feelings"},
    
    # Actions
    "help": {"id": "3101", "name": "help", "category": "actions"},
    "want": {"id": "3102", "name": "want", "category": "actions"},
    "need": {"id": "3103", "name": "need", "category": "actions"},
    "go": {"id": "3104", "name": "go", "category": "actions"},
    "stop": {"id": "3105", "name": "stop", "category": "actions"},
    "wait": {"id": "3106", "name": "wait", "category": "actions"},
    "look": {"id": "3107", "name": "look", "category": "actions"},
    "listen": {"id": "3108", "name": "listen", "category": "actions"},
    "play": {"id": "3109", "name": "play", "category": "actions"},
    "read": {"id": "3110", "name": "read", "category": "actions"},
    "write": {"id": "3111", "name": "write", "category": "actions"},
    "talk": {"id": "3112", "name": "talk", "category": "actions"},
    "sit": {"id": "3113", "name": "sit", "category": "actions"},
    "stand": {"id": "3114", "name": "stand", "category": "actions"},
    "walk": {"id": "3115", "name": "walk", "category": "actions"},
    "run": {"id": "3116", "name": "run", "category": "actions"},
    
    # People
    "i": {"id": "4001", "name": "I/me", "category": "people"},
    "me": {"id": "4001", "name": "I/me", "category": "people"},
    "you": {"id": "4002", "name": "you", "category": "people"},
    "mom": {"id": "4003", "name": "mom", "category": "people"},
    "mother": {"id": "4003", "name": "mom", "category": "people"},
    "dad": {"id": "4004", "name": "dad", "category": "people"},
    "father": {"id": "4004", "name": "dad", "category": "people"},
    "friend": {"id": "4005", "name": "friend", "category": "people"},
    "doctor": {"id": "4006", "name": "doctor", "category": "people"},
    "teacher": {"id": "4007", "name": "teacher", "category": "people"},
    
    # Places
    "home": {"id": "5001", "name": "home", "category": "places"},
    "school": {"id": "5002", "name": "school", "category": "places"},
    "hospital": {"id": "5003", "name": "hospital", "category": "places"},
    "outside": {"id": "5004", "name": "outside", "category": "places"},
    "inside": {"id": "5005", "name": "inside", "category": "places"},
    
    # Time
    "now": {"id": "6001", "name": "now", "category": "time"},
    "later": {"id": "6002", "name": "later", "category": "time"},
    "today": {"id": "6003", "name": "today", "category": "time"},
    "tomorrow": {"id": "6004", "name": "tomorrow", "category": "time"},
    "morning": {"id": "6005", "name": "morning", "category": "time"},
    "night": {"id": "6006", "name": "night", "category": "time"},
    
    # Common phrases
    "yes": {"id": "7001", "name": "yes", "category": "responses"},
    "no": {"id": "7002", "name": "no", "category": "responses"},
    "please": {"id": "7003", "name": "please", "category": "responses"},
    "thank": {"id": "7004", "name": "thank you", "category": "responses"},
    "thanks": {"id": "7004", "name": "thank you", "category": "responses"},
    "sorry": {"id": "7005", "name": "sorry", "category": "responses"},
    "hello": {"id": "7006", "name": "hello", "category": "responses"},
    "goodbye": {"id": "7007", "name": "goodbye", "category": "responses"},
    "bye": {"id": "7007", "name": "goodbye", "category": "responses"},
}

# ARASAAC API base URL
ARASAAC_API_URL = "https://api.arasaac.org/api/pictograms"
ARASAAC_STATIC_URL = "https://static.arasaac.org/pictograms"


class SymbolService:
    """Service for symbol mapping operations."""
    
    def __init__(self):
        self.mapping = SYMBOL_MAPPING
        self.arasaac_url = ARASAAC_API_URL
        self.arasaac_static_url = ARASAAC_STATIC_URL
    
    def get_symbol_url(self, symbol_id: str, size: int = 300) -> str:
        """Generate URL for ARASAAC symbol image."""
        return f"{self.arasaac_static_url}/{symbol_id}/{symbol_id}_{size}.png"
    
    async def map_keywords_to_symbols(
        self,
        keywords: List[str],
        max_symbols: int = 10
    ) -> List[Symbol]:
        """
        Map a list of keywords to visual symbols.
        
        Args:
            keywords: List of keywords to map
            max_symbols: Maximum number of symbols to return
            
        Returns:
            List of Symbol objects
        """
        symbols = []
        seen_ids = set()
        
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            
            # Skip empty keywords
            if not keyword_lower:
                continue
            
            if keyword_lower in self.mapping:
                symbol_data = self.mapping[keyword_lower]
                symbol_id = symbol_data["id"]
                
                # Avoid duplicates
                if symbol_id not in seen_ids:
                    seen_ids.add(symbol_id)
                    symbols.append(Symbol(
                        id=symbol_id,
                        name=symbol_data["name"],
                        url=self.get_symbol_url(symbol_id),
                        category=symbol_data.get("category")
                    ))
            else:
                # Try searching ARASAAC API for unknown keywords
                try:
                    arasaac_results = await self.search_arasaac_symbols(keyword_lower, limit=1)
                    if arasaac_results and arasaac_results[0].id not in seen_ids:
                        seen_ids.add(arasaac_results[0].id)
                        symbols.append(arasaac_results[0])
                except Exception:
                    pass  # Skip this keyword if API fails
            
            if len(symbols) >= max_symbols:
                break
        
        return symbols
    
    async def search_arasaac_symbols(
        self,
        query: str,
        language: str = "en",
        limit: int = 5
    ) -> List[Symbol]:
        """
        Search ARASAAC API for symbols matching a query.
        
        Args:
            query: Search query
            language: Language code
            limit: Maximum results
            
        Returns:
            List of Symbol objects
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.arasaac_url}/{language}/search/{query}",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    symbols = []
                    
                    for item in data[:limit]:
                        symbols.append(Symbol(
                            id=str(item.get("_id", "")),
                            name=item.get("keywords", [{}])[0].get("keyword", query),
                            url=self.get_symbol_url(str(item.get("_id", ""))),
                            category=item.get("categories", ["other"])[0] if item.get("categories") else None
                        ))
                    
                    return symbols
        except Exception:
            # Fallback to local mapping on API failure
            pass
        
        return []
    
    def get_all_categories(self) -> List[str]:
        """Get all available symbol categories."""
        categories = set()
        for data in self.mapping.values():
            if "category" in data:
                categories.add(data["category"])
        return sorted(list(categories))
    
    def get_symbols_by_category(self, category: str) -> List[Symbol]:
        """Get all symbols in a specific category."""
        symbols = []
        for keyword, data in self.mapping.items():
            if data.get("category") == category:
                symbols.append(Symbol(
                    id=data["id"],
                    name=data["name"],
                    url=self.get_symbol_url(data["id"]),
                    category=category
                ))
        return symbols


# Singleton instance
_symbol_service: Optional[SymbolService] = None


def get_symbol_service() -> SymbolService:
    """Get the symbol service singleton."""
    global _symbol_service
    if _symbol_service is None:
        _symbol_service = SymbolService()
    return _symbol_service
