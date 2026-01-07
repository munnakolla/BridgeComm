"""Full API Integration Test for BridgeComm Backend."""

import asyncio
import httpx
import base64
import json

BASE_URL = "http://localhost:8000"


async def test_health():
    """Test the health endpoint."""
    print("\n[1/8] Testing Health Endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Health check passed: {data}")
                return True
            else:
                print(f"  ❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


async def test_text_to_symbols():
    """Test text to symbols conversion."""
    print("\n[2/8] Testing Text-to-Symbols API...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/azure/text-to-symbols",
                json={
                    "text": "I am hungry and thirsty. I want to go home.",
                    "simplify": True,
                    "max_symbols": 5
                }
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Text-to-Symbols working!")
                print(f"     Original: {data.get('original_text', 'N/A')}")
                print(f"     Simplified: {data.get('simplified_text', 'N/A')}")
                print(f"     Keywords: {data.get('keywords', [])}")
                print(f"     Symbols: {len(data.get('symbols', []))} symbols generated")
                return True
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


async def test_generate_text():
    """Test intent-to-text generation."""
    print("\n[3/8] Testing Text Generation API...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/azure/generate-text",
                json={
                    "intent": "hungry",
                    "context": "at home",
                    "style": "simple"
                }
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Text Generation working!")
                print(f"     Intent: {data.get('intent', 'N/A')}")
                print(f"     Generated Text: {data.get('text', 'N/A')}")
                return True
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


async def test_symbol_search():
    """Test symbol search."""
    print("\n[4/8] Testing Symbol Search API...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{BASE_URL}/azure/symbols/search/water")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Symbol Search working!")
                print(f"     Query: water")
                print(f"     Found: {data.get('count', 0)} symbols")
                if data.get('symbols'):
                    symbol = data['symbols'][0]
                    print(f"     First symbol: {symbol.get('name', 'N/A')} - {symbol.get('url', 'N/A')[:50]}...")
                return True
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


async def test_symbol_categories():
    """Test symbol categories."""
    print("\n[5/8] Testing Symbol Categories API...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(f"{BASE_URL}/azure/symbols/categories")
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Symbol Categories working!")
                print(f"     Categories: {data.get('categories', [])}")
                return True
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


async def test_translator():
    """Test translator service."""
    print("\n[6/8] Testing Translator API...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/azure/translate",
                json={
                    "text": "I need help please",
                    "to_language": "hi"
                }
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Translator working!")
                print(f"     Original (en): {data.get('original', 'N/A')}")
                print(f"     Translated (hi): {data.get('translated', 'N/A')}")
                return True
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


async def test_behavior_to_intent():
    """Test behavior to intent conversion."""
    print("\n[7/8] Testing Behavior-to-Intent API...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/azure/behavior-to-intent",
                json={
                    "behavior_data": {
                        "touch_patterns": [
                            {"type": "tap", "count": 2, "x": 100, "y": 200}
                        ],
                        "interaction_sequence": ["tap", "tap", "swipe_up"]
                    },
                    "context": "home screen"
                }
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Behavior-to-Intent working!")
                print(f"     Detected Intent: {data.get('intent', 'N/A')}")
                print(f"     Generated Text: {data.get('text', 'N/A')}")
                print(f"     Confidence: {data.get('confidence', 0):.2f}")
                return True
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


async def test_sign_to_intent():
    """Test sign language to intent (without real image)."""
    print("\n[8/8] Testing Sign-to-Intent API (mock)...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Create a tiny 1x1 white pixel PNG as mock image
            mock_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            response = await client.post(
                f"{BASE_URL}/azure/sign-to-intent",
                json={"image_base64": mock_image}
            )
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Sign-to-Intent API working!")
                print(f"     Intent: {data.get('intent', 'N/A')}")
                print(f"     Text: {data.get('text', 'N/A')}")
                return True
            else:
                print(f"  ❌ Failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False


async def main():
    """Run all API tests."""
    print("=" * 60)
    print("BRIDGECOMM API INTEGRATION TEST")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    
    results = {}
    results["health"] = await test_health()
    results["text_to_symbols"] = await test_text_to_symbols()
    results["generate_text"] = await test_generate_text()
    results["symbol_search"] = await test_symbol_search()
    results["symbol_categories"] = await test_symbol_categories()
    results["translator"] = await test_translator()
    results["behavior_to_intent"] = await test_behavior_to_intent()
    results["sign_to_intent"] = await test_sign_to_intent()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name.replace('_', ' ').title()}")
        if status:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed}/{len(results)} tests passed")
    
    if failed == 0:
        print("\n🎉 All API endpoints are working correctly!")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    asyncio.run(main())
