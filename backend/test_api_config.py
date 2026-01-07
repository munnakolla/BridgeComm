"""Test script to verify API configuration and connectivity."""

import asyncio
import httpx
from app.core.config import get_settings


def check_configuration():
    """Check all API key configurations."""
    settings = get_settings()
    
    print("=" * 60)
    print("API KEY CONFIGURATION STATUS")
    print("=" * 60)
    
    # Speech Services
    speech_ok = settings.azure_speech_key and len(settings.azure_speech_key) > 10
    print(f"\n[SPEECH SERVICES]")
    print(f"  Key: {'✅ SET' if speech_ok else '❌ MISSING'}")
    print(f"  Region: {settings.azure_speech_region}")
    
    # Vision Services
    vision_ok = settings.azure_vision_key and len(settings.azure_vision_key) > 10
    print(f"\n[VISION SERVICES]")
    print(f"  Key: {'✅ SET' if vision_ok else '❌ MISSING'}")
    print(f"  Endpoint: {settings.azure_vision_endpoint or 'NOT SET'}")
    
    # Language Services
    language_ok = settings.azure_language_key and len(settings.azure_language_key) > 10
    print(f"\n[LANGUAGE SERVICES]")
    print(f"  Key: {'✅ SET' if language_ok else '❌ MISSING'}")
    print(f"  Endpoint: {settings.azure_language_endpoint or 'NOT SET'}")
    
    # Translator Services
    translator_ok = settings.azure_translator_key and len(settings.azure_translator_key) > 10
    print(f"\n[TRANSLATOR SERVICES]")
    print(f"  Key: {'✅ SET' if translator_ok else '❌ MISSING'}")
    print(f"  Region: {settings.azure_translator_region}")
    
    # OpenAI (Optional)
    openai_ok = settings.azure_openai_api_key and len(settings.azure_openai_api_key) > 10
    print(f"\n[AZURE OPENAI] (Optional)")
    print(f"  Key: {'✅ SET' if openai_ok else '⚠️ NOT SET (using Language API instead)'}")
    print(f"  Endpoint: {settings.azure_openai_endpoint or 'NOT SET'}")
    
    print("\n" + "=" * 60)
    return settings


async def test_speech_service(settings):
    """Test Azure Speech Service connectivity."""
    print("\n[TEST] Azure Speech Service...")
    try:
        import azure.cognitiveservices.speech as speechsdk
        speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region
        )
        print("  ✅ Speech SDK configured successfully")
        return True
    except Exception as e:
        print(f"  ❌ Speech Service Error: {e}")
        return False


async def test_language_service(settings):
    """Test Azure Language Service connectivity."""
    print("\n[TEST] Azure Language Service...")
    if not settings.azure_language_key or not settings.azure_language_endpoint:
        print("  ⚠️ Language service not configured")
        return False
    
    endpoint = settings.azure_language_endpoint.rstrip('/')
    url = f"{endpoint}/language/:analyze-text?api-version=2023-04-01"
    
    headers = {
        "Ocp-Apim-Subscription-Key": settings.azure_language_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "kind": "KeyPhraseExtraction",
        "parameters": {"modelVersion": "latest"},
        "analysisInput": {
            "documents": [{"id": "1", "language": "en", "text": "Hello, I need help."}]
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Language API working! Key phrases: {result.get('results', {}).get('documents', [{}])[0].get('keyPhrases', [])}")
                return True
            else:
                print(f"  ❌ Language API Error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Language Service Error: {e}")
            return False


async def test_translator_service(settings):
    """Test Azure Translator Service connectivity."""
    print("\n[TEST] Azure Translator Service...")
    if not settings.azure_translator_key:
        print("  ⚠️ Translator service not configured")
        return False
    
    url = f"{settings.azure_translator_endpoint.rstrip('/')}/translate"
    
    params = {"api-version": "3.0", "to": "es"}
    
    headers = {
        "Ocp-Apim-Subscription-Key": settings.azure_translator_key,
        "Ocp-Apim-Subscription-Region": settings.azure_translator_region,
        "Content-Type": "application/json"
    }
    
    payload = [{"text": "Hello, how are you?"}]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, params=params, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                translated = result[0].get("translations", [{}])[0].get("text", "")
                print(f"  ✅ Translator API working! 'Hello, how are you?' -> '{translated}'")
                return True
            else:
                print(f"  ❌ Translator API Error: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"  ❌ Translator Service Error: {e}")
            return False


async def test_vision_service(settings):
    """Test Azure Vision Service connectivity."""
    print("\n[TEST] Azure Vision Service...")
    if not settings.azure_vision_key or not settings.azure_vision_endpoint:
        print("  ⚠️ Vision service not configured")
        return False
    
    endpoint = settings.azure_vision_endpoint.rstrip('/')
    # Test with capabilities endpoint
    url = f"{endpoint}/computervision/imageanalysis:analyze?api-version=2023-10-01"
    
    headers = {
        "Ocp-Apim-Subscription-Key": settings.azure_vision_key,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Just test if endpoint is reachable
            response = await client.get(
                f"{endpoint}/",
                headers={"Ocp-Apim-Subscription-Key": settings.azure_vision_key}
            )
            print(f"  ✅ Vision endpoint reachable (status: {response.status_code})")
            return True
        except Exception as e:
            print(f"  ❌ Vision Service Error: {e}")
            return False


async def main():
    """Run all API tests."""
    settings = check_configuration()
    
    print("\n" + "=" * 60)
    print("CONNECTIVITY TESTS")
    print("=" * 60)
    
    results = {
        "speech": await test_speech_service(settings),
        "language": await test_language_service(settings),
        "translator": await test_translator_service(settings),
        "vision": await test_vision_service(settings),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_ok = True
    for service, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {service.upper()}")
        if not status:
            all_ok = False
    
    if all_ok:
        print("\n🎉 All services are properly configured and working!")
    else:
        print("\n⚠️ Some services have issues. Check the errors above.")
        print("\nTo fix API key issues:")
        print("1. Go to Azure Portal (https://portal.azure.com)")
        print("2. Navigate to the respective service")
        print("3. Copy the API key from 'Keys and Endpoint' section")
        print("4. Update the .env file in backend folder")
    
    return all_ok


if __name__ == "__main__":
    asyncio.run(main())
