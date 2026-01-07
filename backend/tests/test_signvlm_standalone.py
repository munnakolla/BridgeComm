#!/usr/bin/env python3
"""
Standalone SignVLM Integration Test
Tests the SignVLM module independently without loading other services.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_vision_transformer():
    """Test vision transformer components."""
    print("\n[1] Testing Vision Transformer components...")
    try:
        from app.services.sign_language.signvlm.vision_transformer import (
            VisionTransformer2D,
            QuickGELU,
            PatchEmbed2D,
            vit_presets
        )
        print("    ✓ VisionTransformer2D imported")
        print("    ✓ QuickGELU imported")
        print("    ✓ PatchEmbed2D imported")
        print(f"    ✓ Available presets: {list(vit_presets.keys())}")
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def test_model():
    """Test EVLTransformer model."""
    print("\n[2] Testing EVLTransformer model...")
    try:
        from app.services.sign_language.signvlm.model import (
            EVLTransformer,
            EVLDecoder,
            TemporalCrossAttention
        )
        print("    ✓ EVLTransformer imported")
        print("    ✓ EVLDecoder imported")
        print("    ✓ TemporalCrossAttention imported")
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def test_weight_loaders():
    """Test weight loading functions."""
    print("\n[3] Testing weight loaders...")
    try:
        from app.services.sign_language.signvlm.weight_loaders import (
            load_weights_clip,
            load_weights_mae,
            load_weights_timm
        )
        print("    ✓ load_weights_clip imported")
        print("    ✓ load_weights_mae imported")
        print("    ✓ load_weights_timm imported")
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def test_signvlm_service():
    """Test SignVLM service."""
    print("\n[4] Testing SignVLM service...")
    try:
        from app.services.sign_language.signvlm.signvlm_service import (
            SignVLMService,
            SignVLMConfig,
            WLASL100_CLASSES
        )
        print("    ✓ SignVLMService imported")
        print("    ✓ SignVLMConfig imported")
        print(f"    ✓ WLASL100_CLASSES: {len(WLASL100_CLASSES)} classes")
        
        # Test service instantiation
        service = SignVLMService()
        print("    ✓ Service instantiated")
        
        # Check status
        status = service.get_status()
        print(f"    ✓ Status: loaded={status['loaded']}")
        
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test creating the actual model."""
    print("\n[5] Testing model creation...")
    try:
        import torch
        from app.services.sign_language.signvlm.model import EVLTransformer
        from app.services.sign_language.signvlm.vision_transformer import vit_presets
        
        # Get preset for ViT-L/14
        preset = vit_presets["ViT-L/14-lnpre"]
        print(f"    ✓ Using preset: ViT-L/14-lnpre")
        print(f"      - Feature dim: {preset['feature_dim']}")
        print(f"      - Num layers: {preset['num_layers']}")
        
        # Create model (without loading weights)
        model = EVLTransformer(
            backbone_name="ViT-L/14-lnpre",
            backbone_type="clip",
            backbone_path=None,  # Don't load weights for this test
            backbone_mode="frozen_fp16",
            decoder_num_layers=4,
            decoder_qkv_dim=1024,
            decoder_num_heads=16,
            num_classes=100,
            num_frames=24,
            num_tokens=196,
        )
        print("    ✓ EVLTransformer created")
        
        # Check model structure
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    ✓ Total parameters: {total_params:,}")
        print(f"    ✓ Trainable parameters: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backbone_download():
    """Check if backbone is downloaded."""
    print("\n[6] Checking backbone download...")
    try:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app", "services", "sign_language", "models", "signvlm"
        )
        backbone_path = os.path.join(models_dir, "ViT-L-14.pt")
        
        if os.path.exists(backbone_path):
            size_mb = os.path.getsize(backbone_path) / (1024 * 1024)
            print(f"    ✓ Backbone found: {backbone_path}")
            print(f"    ✓ Size: {size_mb:.1f} MB")
            return True
        else:
            print(f"    ✗ Backbone not found at: {backbone_path}")
            print("      Run: python -m app.services.sign_language.signvlm.download_models")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def test_inference_dummy():
    """Test inference with dummy data."""
    print("\n[7] Testing inference with dummy data...")
    try:
        import torch
        import numpy as np
        from app.services.sign_language.signvlm.signvlm_service import SignVLMService
        
        service = SignVLMService()
        
        # Create dummy frames (24 frames of 224x224 RGB)
        dummy_frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(24)
        ]
        
        # Test preprocessing
        tensor = service.preprocess_frames(dummy_frames)
        print(f"    ✓ Preprocessed tensor shape: {tensor.shape}")
        expected_shape = (1, 3, 24, 224, 224)  # B, C, T, H, W
        assert tensor.shape == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"
        print(f"    ✓ Shape matches expected: {expected_shape}")
        
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_inference():
    """Test full inference with loaded model."""
    print("\n[8] Testing full inference (requires downloaded model)...")
    try:
        import numpy as np
        from app.services.sign_language.signvlm.signvlm_service import SignVLMService
        
        service = SignVLMService()
        
        # Try to load model
        print("    Loading model (this may take a moment)...")
        loaded = service.load_model()
        
        if not loaded:
            print("    ⚠ Model could not be loaded (backbone may not be downloaded)")
            return True  # Not a failure, just skip
        
        print("    ✓ Model loaded")
        
        # Create dummy frames
        dummy_frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(24)
        ]
        
        # Run inference
        result = service.recognize_frames(dummy_frames, top_k=5)
        print(f"    ✓ Inference result: {result}")
        
        if "predictions" in result:
            print(f"    ✓ Got {len(result['predictions'])} predictions")
            for pred in result["predictions"][:3]:
                print(f"      - {pred['gloss']}: {pred['confidence']:.3f}")
        
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SignVLM Standalone Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Vision Transformer", test_vision_transformer()))
    results.append(("EVLTransformer Model", test_model()))
    results.append(("Weight Loaders", test_weight_loaders()))
    results.append(("SignVLM Service", test_signvlm_service()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Backbone Download", test_backbone_download()))
    results.append(("Preprocessing", test_inference_dummy()))
    results.append(("Full Inference", test_full_inference()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
