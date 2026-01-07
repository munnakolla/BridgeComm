"""
Symbols API Routes
Endpoints for text-to-symbols conversion.
Uses Groq AI for text simplification.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from app.models.schemas import (
    TextToSymbolsRequest,
    TextToSymbolsResponse,
    Symbol,
)
from app.services import get_openai_service, get_symbol_service
from app.services.groq_service import get_groq_service

# Demo symbol mappings for fallback
DEMO_SYMBOL_MAP = {
    "hello": Symbol(id="hello", name="Hello", url="https://static.arasaac.org/pictograms/6020/6020_300.png", category="greetings"),
    "yes": Symbol(id="yes", name="Yes", url="https://static.arasaac.org/pictograms/5584/5584_300.png", category="responses"),
    "no": Symbol(id="no", name="No", url="https://static.arasaac.org/pictograms/5586/5586_300.png", category="responses"),
    "help": Symbol(id="help", name="Help", url="https://static.arasaac.org/pictograms/6555/6555_300.png", category="actions"),
    "thanks": Symbol(id="thanks", name="Thanks", url="https://static.arasaac.org/pictograms/6051/6051_300.png", category="greetings"),
    "please": Symbol(id="please", name="Please", url="https://static.arasaac.org/pictograms/6050/6050_300.png", category="greetings"),
    "happy": Symbol(id="happy", name="Happy", url="https://static.arasaac.org/pictograms/2356/2356_300.png", category="emotions"),
    "sad": Symbol(id="sad", name="Sad", url="https://static.arasaac.org/pictograms/2357/2357_300.png", category="emotions"),
    "hungry": Symbol(id="hungry", name="Hungry", url="https://static.arasaac.org/pictograms/6556/6556_300.png", category="needs"),
    "thirsty": Symbol(id="thirsty", name="Thirsty", url="https://static.arasaac.org/pictograms/6552/6552_300.png", category="needs"),
    "water": Symbol(id="water", name="Water", url="https://static.arasaac.org/pictograms/2410/2410_300.png", category="food"),
    "food": Symbol(id="food", name="Food", url="https://static.arasaac.org/pictograms/2447/2447_300.png", category="food"),
    "home": Symbol(id="home", name="Home", url="https://static.arasaac.org/pictograms/2266/2266_300.png", category="places"),
    "want": Symbol(id="want", name="Want", url="https://static.arasaac.org/pictograms/6554/6554_300.png", category="actions"),
    "need": Symbol(id="need", name="Need", url="https://static.arasaac.org/pictograms/6553/6553_300.png", category="actions"),
    "go": Symbol(id="go", name="Go", url="https://static.arasaac.org/pictograms/2449/2449_300.png", category="actions"),
    "stop": Symbol(id="stop", name="Stop", url="https://static.arasaac.org/pictograms/6057/6057_300.png", category="actions"),
    "good": Symbol(id="good", name="Good", url="https://static.arasaac.org/pictograms/2359/2359_300.png", category="emotions"),
    "bad": Symbol(id="bad", name="Bad", url="https://static.arasaac.org/pictograms/2360/2360_300.png", category="emotions"),
    "i": Symbol(id="i", name="I", url="https://static.arasaac.org/pictograms/6017/6017_300.png", category="pronouns"),
    "you": Symbol(id="you", name="You", url="https://static.arasaac.org/pictograms/6018/6018_300.png", category="pronouns"),
    "love": Symbol(id="love", name="Love", url="https://static.arasaac.org/pictograms/2355/2355_300.png", category="emotions"),
}

def fallback_text_to_symbols(text: str, max_symbols: int = 10):
    """Fallback function to convert text to symbols without AI."""
    words = text.lower().split()
    symbols = []
    keywords = []
    
    for word in words:
        clean_word = ''.join(c for c in word if c.isalpha())
        if clean_word in DEMO_SYMBOL_MAP and len(symbols) < max_symbols:
            symbols.append(DEMO_SYMBOL_MAP[clean_word])
            keywords.append(clean_word)
    
    # Simplified text is just cleaned up original
    simplified = ' '.join(words).capitalize()
    
    return simplified, symbols, keywords

router = APIRouter(prefix="/azure", tags=["Symbols"])


@router.post("/text-to-symbols", response_model=TextToSymbolsResponse)
async def text_to_symbols(request: TextToSymbolsRequest):
    """
    Convert text to simplified text and visual symbols.
    
    This endpoint:
    1. Simplifies the input text for easier understanding
    2. Extracts key words from the simplified text
    3. Maps keywords to ARASAAC visual symbols
    
    Returns simplified text and array of symbol objects with URLs.
    """
    try:
        # Try Groq first for text processing
        groq_service = get_groq_service()
        use_groq = groq_service.api_key is not None
        
        symbol_service = get_symbol_service()
    except Exception as e:
        use_groq = False
        # AI services not available, use fallback
        simplified, symbols, keywords = fallback_text_to_symbols(
            request.text, 
            request.max_symbols
        )
        return TextToSymbolsResponse(
            original_text=request.text,
            simplified_text=simplified or request.text,
            symbols=symbols,
            keywords=keywords,
            confidence=0.75
        )
    
    try:
        # Step 1: Simplify text using Groq if available
        if request.simplify and use_groq:
            try:
                result = await groq_service.process_text(request.text, task="simplify")
                simplified_text = result.get("simplified_text", request.text)
            except Exception as e:
                print(f"Groq simplification failed: {e}")
                simplified_text = request.text
        else:
            simplified_text = request.text
        
        # Step 2: Extract keywords using Groq or fallback
        # Use a dedicated keyword extraction prompt for better results with full sentences
        if use_groq:
            try:
                # Use Groq LLM to extract meaningful keywords from the full sentence
                keyword_result = await groq_service.process_text(
                    request.text,
                    task="keywords",
                    context=f"Extract up to {request.max_symbols} most important words that can be represented as symbols/pictograms"
                )
                extracted_keywords = keyword_result.get("keywords", [])
                
                if not extracted_keywords or not isinstance(extracted_keywords, list):
                    # Fallback: get keywords from simplified text
                    result = await groq_service.text_to_symbols(request.text, max_symbols=request.max_symbols)
                    # Extract all meaningful words (nouns, verbs, adjectives), not just first 5
                    simplified = result.get("simplified_text", request.text)
                    stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they', 
                                  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                                  'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                                  'and', 'or', 'but', 'if', 'then', 'so', 'than', 'that', 'this', 'these'}
                    keywords = [w.lower().strip('.,!?;:') for w in simplified.split() 
                               if w.lower().strip('.,!?;:') not in stop_words and len(w) > 1][:request.max_symbols]
                else:
                    keywords = extracted_keywords[:request.max_symbols]
                    
            except Exception as e:
                print(f"Groq keyword extraction failed: {e}")
                # Enhanced fallback: filter stop words
                stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
                              'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                              'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                              'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                              'and', 'or', 'but', 'if', 'then', 'so', 'than', 'that', 'this', 'these'}
                keywords = [w.lower().strip('.,!?;:') for w in request.text.split() 
                           if w.lower().strip('.,!?;:') not in stop_words and len(w) > 1][:request.max_symbols]
        else:
            # Enhanced fallback without Groq: filter stop words
            stop_words = {'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
                          'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                          'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                          'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                          'and', 'or', 'but', 'if', 'then', 'so', 'than', 'that', 'this', 'these'}
            keywords = [w.lower().strip('.,!?;:') for w in request.text.split() 
                       if w.lower().strip('.,!?;:') not in stop_words and len(w) > 1][:request.max_symbols]
        
        # Step 3: Map to symbols
        try:
            symbols = await symbol_service.map_keywords_to_symbols(
                keywords=keywords,
                max_symbols=request.max_symbols
            )
        except Exception:
            # Use fallback symbols
            symbols = [DEMO_SYMBOL_MAP[k] for k in keywords if k in DEMO_SYMBOL_MAP][:request.max_symbols]
        
        # Calculate confidence based on symbol coverage
        coverage = len(symbols) / max(len(keywords), 1)
        confidence = min(0.95, coverage * 0.9 + 0.1)
        
        return TextToSymbolsResponse(
            original_text=request.text,
            simplified_text=simplified_text,
            symbols=symbols,
            keywords=keywords,
            confidence=confidence
        )
        
    except Exception as e:
        # Complete fallback
        simplified, symbols, keywords = fallback_text_to_symbols(
            request.text,
            request.max_symbols
        )
        return TextToSymbolsResponse(
            original_text=request.text,
            simplified_text=simplified or request.text,
            symbols=symbols,
            keywords=keywords,
            confidence=0.70
        )


@router.get("/symbols/categories")
async def get_symbol_categories():
    """
    Get all available symbol categories.
    """
    symbol_service = get_symbol_service()
    
    return {
        "categories": symbol_service.get_all_categories()
    }


@router.get("/symbols/category/{category}")
async def get_symbols_by_category(category: str):
    """
    Get all symbols in a specific category.
    """
    symbol_service = get_symbol_service()
    
    symbols = symbol_service.get_symbols_by_category(category)
    
    return {
        "category": category,
        "symbols": symbols,
        "count": len(symbols)
    }


@router.get("/symbols/search/{query}")
async def search_symbols(
    query: str,
    language: str = "en",
    limit: int = 10
):
    """
    Search for symbols matching a query using ARASAAC API.
    """
    symbol_service = get_symbol_service()
    
    try:
        symbols = await symbol_service.search_arasaac_symbols(
            query=query,
            language=language,
            limit=limit
        )
        
        return {
            "query": query,
            "symbols": symbols,
            "count": len(symbols)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
