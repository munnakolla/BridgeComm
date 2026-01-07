"""
SignVLM Integration Test
========================

Tests for the SignVLM sign language recognition model integration.

Usage:
    # Run all tests
    pytest backend/tests/test_signvlm.py -v
    
    # Run specific test
    pytest backend/tests/test_signvlm.py::test_service_initialization -v
"""

import os
import sys
import json
import pytest
import asyncio
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSignVLMIntegration:
    """Test SignVLM model integration."""
    
    def test_module_imports(self):
        """Test that SignVLM modules can be imported."""
        from app.services.sign_language.signvlm import SignVLMService, get_signvlm_service
        from app.services.sign_language.signvlm.vision_transformer import (
            VisionTransformer2D, vit_presets, QuickGELU, LayerNorm
        )
        from app.services.sign_language.signvlm.model import EVLTransformer, EVLDecoder
        from app.services.sign_language.signvlm.weight_loaders import weight_loader_fn_dict
        
        assert SignVLMService is not None
        assert EVLTransformer is not None
        assert 'ViT-L/14-lnpre' in vit_presets
        assert 'clip' in weight_loader_fn_dict
    
    def test_service_initialization(self):
        """Test SignVLM service can be initialized."""
        from app.services.sign_language.signvlm import get_signvlm_service
        
        service = get_signvlm_service()
        assert service is not None
        assert hasattr(service, 'config')
        assert hasattr(service, 'class_mapping')
        assert len(service.class_mapping) > 0
        
        print(f"Service initialized with {len(service.class_mapping)} classes")
    
    def test_config_defaults(self):
        """Test SignVLM configuration defaults."""
        from app.services.sign_language.signvlm.signvlm_service import SignVLMConfig
        
        config = SignVLMConfig()
        
        assert config.backbone_name == "ViT-L/14-lnpre"
        assert config.num_frames == 24
        assert config.input_size == (224, 224)
        assert config.fps == 15
        assert config.device == "cpu"  # Default to CPU
    
    def test_class_mapping(self):
        """Test class mapping is loaded correctly."""
        from app.services.sign_language.signvlm import get_signvlm_service
        
        service = get_signvlm_service()
        
        # Should have WLASL-100 classes by default
        assert len(service.class_mapping) == 100
        
        # Check some common classes exist
        classes = list(service.class_mapping.values())
        assert any(c.lower() in ['book', 'hello', 'help', 'thank'] for c in classes[:20])
    
    def test_service_status(self):
        """Test service status reporting."""
        from app.services.sign_language.signvlm import get_signvlm_service
        
        service = get_signvlm_service()
        status = service.get_status()
        
        assert 'service' in status
        assert 'status' in status
        assert 'models' in status
        assert 'config' in status
        assert 'vocabulary_size' in status
        
        print(f"Service status: {status['status']}")
        print(f"Backbone available: {status['models'].get('backbone', {}).get('available', False)}")
    
    def test_vit_presets(self):
        """Test ViT preset configurations."""
        from app.services.sign_language.signvlm.vision_transformer import vit_presets
        
        # Check ViT-L/14-lnpre (used by SignVLM)
        assert 'ViT-L/14-lnpre' in vit_presets
        config = vit_presets['ViT-L/14-lnpre']
        
        assert config['feature_dim'] == 1024
        assert config['input_size'] == (224, 224)
        assert config['patch_size'] == (14, 14)
        assert config['num_heads'] == 16
        assert config['num_layers'] == 24
        assert config['ln_pre'] == True
    
    def test_vision_transformer_creation(self):
        """Test VisionTransformer2D can be created."""
        import torch
        from app.services.sign_language.signvlm.vision_transformer import (
            VisionTransformer2D, vit_presets
        )
        
        # Create a smaller model for testing
        model = VisionTransformer2D(
            feature_dim=256,
            input_size=(224, 224),
            patch_size=(16, 16),
            num_heads=4,
            num_layers=2,
            return_all_features=True
        )
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        features = model(x)
        
        assert len(features) == 2  # num_layers
        assert 'out' in features[0]
        assert features[0]['out'].shape[0] == 1  # batch size
        
        print(f"Feature shape: {features[0]['out'].shape}")
    
    def test_evl_decoder_creation(self):
        """Test EVLDecoder can be created."""
        import torch
        from app.services.sign_language.signvlm.model import EVLDecoder
        
        decoder = EVLDecoder(
            num_frames=8,
            spatial_size=(14, 14),
            num_layers=2,
            in_feature_dim=256,
            qkv_dim=256,
            num_heads=4
        )
        
        # Create dummy features
        N, T, L, C = 1, 8, 196, 256  # batch, frames, patches, dim
        features = [
            {
                'out': torch.randn(N, T, L, C),
                'q': torch.randn(N, T, C, L),
                'k': torch.randn(N, T, C, L)
            }
            for _ in range(2)
        ]
        
        output = decoder(features)
        assert output.shape == (1, 1, 256)  # N, 1, C
        
        print(f"Decoder output shape: {output.shape}")
    
    @pytest.mark.skipif(
        not Path(__file__).parent.parent.joinpath("models", "signvlm", "ViT-L-14.pt").exists(),
        reason="CLIP backbone not downloaded"
    )
    def test_model_loading(self):
        """Test full model loading (requires backbone)."""
        from app.services.sign_language.signvlm import get_signvlm_service
        
        service = get_signvlm_service()
        success = service.load_model()
        
        assert success, "Model loading should succeed when backbone is available"
        assert service.is_loaded
        assert service.model is not None
    
    def test_frame_preprocessing(self):
        """Test frame preprocessing pipeline."""
        import numpy as np
        import torch
        from app.services.sign_language.signvlm import get_signvlm_service
        
        service = get_signvlm_service()
        
        # Create dummy frames (BGR format like OpenCV)
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(24)]
        
        # Preprocess
        tensor = service.preprocess_frames(frames)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 24, 224, 224)  # N, C, T, H, W
        assert tensor.dtype == torch.float32
        
        print(f"Preprocessed tensor shape: {tensor.shape}")


class TestSignVLMAPI:
    """Test SignVLM API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_signvlm_status_endpoint(self, client):
        """Test /sign-video/signvlm/status endpoint."""
        response = client.get("/sign-video/signvlm/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'service' in data
        assert data['service'] == 'SignVLMService'
        assert 'status' in data
        assert 'vocabulary_size' in data
        
        print(f"SignVLM Status: {data['status']}")
        print(f"Vocabulary size: {data['vocabulary_size']}")
    
    def test_signvlm_vocabulary_endpoint(self, client):
        """Test /sign-video/signvlm/vocabulary endpoint."""
        response = client.get("/sign-video/signvlm/vocabulary")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'vocabulary_size' in data
        assert 'words' in data
        
        if data['vocabulary_size'] > 0:
            assert len(data['words']) == data['vocabulary_size']
            print(f"Vocabulary: {data['words'][:10]}...")
    
    def test_signvlm_health_endpoint(self, client):
        """Test /sign-video/signvlm/health endpoint."""
        response = client.get("/sign-video/signvlm/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'healthy' in data
        assert 'service' in data
        assert data['service'] == 'signvlm'
        
        print(f"Health check: {data}")
    
    def test_signvlm_load_endpoint(self, client):
        """Test /sign-video/signvlm/load endpoint."""
        response = client.post("/sign-video/signvlm/load")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'success' in data
        assert 'message' in data
        
        print(f"Load result: {data['message']}")


class TestDownloadScript:
    """Test the model download script."""
    
    def test_download_script_imports(self):
        """Test download script can be imported."""
        from app.services.sign_language.signvlm.download_models import (
            CLIP_MODELS,
            WLASL100_VOCABULARY,
            get_models_dir,
            create_class_mapping
        )
        
        assert 'ViT-L-14' in CLIP_MODELS
        assert len(WLASL100_VOCABULARY) == 100
    
    def test_models_dir_creation(self):
        """Test models directory is created correctly."""
        from app.services.sign_language.signvlm.download_models import get_models_dir
        
        models_dir = get_models_dir()
        assert models_dir.exists()
        assert models_dir.is_dir()
        
        print(f"Models directory: {models_dir}")
    
    def test_class_mapping_creation(self):
        """Test class mapping file creation."""
        from app.services.sign_language.signvlm.download_models import (
            create_class_mapping, get_models_dir
        )
        
        models_dir = get_models_dir()
        mapping_path = create_class_mapping(models_dir, "wlasl100")
        
        assert mapping_path.exists()
        
        with open(mapping_path) as f:
            mapping = json.load(f)
        
        assert len(mapping) == 100
        print(f"Created mapping with {len(mapping)} classes")


def test_run_quick_check():
    """Quick integration check that runs without pytest."""
    print("\n" + "="*60)
    print("SignVLM Quick Integration Check")
    print("="*60)
    
    # 1. Check imports
    print("\n[1] Checking imports...")
    try:
        from app.services.sign_language.signvlm import SignVLMService, get_signvlm_service
        from app.services.sign_language.signvlm.model import EVLTransformer
        print("    ✓ All imports successful")
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        return False
    
    # 2. Check service initialization
    print("\n[2] Initializing service...")
    try:
        service = get_signvlm_service()
        print(f"    ✓ Service initialized")
        print(f"    ✓ Classes loaded: {len(service.class_mapping)}")
    except Exception as e:
        print(f"    ✗ Service initialization failed: {e}")
        return False
    
    # 3. Check status
    print("\n[3] Getting service status...")
    try:
        status = service.get_status()
        print(f"    ✓ Status: {status['status']}")
        print(f"    ✓ Backbone available: {status['models'].get('backbone', {}).get('available', False)}")
    except Exception as e:
        print(f"    ✗ Status check failed: {e}")
        return False
    
    # 4. Check model components
    print("\n[4] Testing model components...")
    try:
        import torch
        from app.services.sign_language.signvlm.vision_transformer import VisionTransformer2D
        from app.services.sign_language.signvlm.model import EVLDecoder
        
        # Quick tensor test
        x = torch.randn(1, 3, 224, 224)
        vit = VisionTransformer2D(
            feature_dim=256, num_layers=2, num_heads=4, return_all_features=True
        )
        features = vit(x)
        print(f"    ✓ VisionTransformer2D works (output layers: {len(features)})")
    except Exception as e:
        print(f"    ✗ Model components test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All quick checks passed!")
    print("="*60)
    
    # Check if backbone needs download
    backbone_path = service.config.backbone_path
    if not backbone_path.exists():
        print(f"\n⚠️  CLIP backbone not found at: {backbone_path}")
        print("   Run this command to download:")
        print("   python backend/app/services/sign_language/signvlm/download_models.py")
    else:
        print(f"\n✓ CLIP backbone found at: {backbone_path}")
    
    return True


if __name__ == "__main__":
    # Run quick check when executed directly
    success = test_run_quick_check()
    sys.exit(0 if success else 1)
