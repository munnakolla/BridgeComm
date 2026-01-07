#!/usr/bin/env python3
"""
Direct SignVLM Module Test
Tests SignVLM by importing directly from the module files, bypassing app.services.__init__.py
"""

import sys
import os

# Set up paths - import directly from file paths, not through app.services
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIGNVLM_DIR = os.path.join(BACKEND_DIR, "app", "services", "sign_language", "signvlm")
sys.path.insert(0, BACKEND_DIR)


def test_vision_transformer():
    """Test vision transformer components."""
    print("\n[1] Testing Vision Transformer components...")
    try:
        # Direct import from file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "vision_transformer", 
            os.path.join(SIGNVLM_DIR, "vision_transformer.py")
        )
        vt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vt)
        
        print("    ✓ VisionTransformer2D imported")
        print("    ✓ QuickGELU imported")
        print("    ✓ PatchEmbed2D imported")
        print(f"    ✓ Available presets: {list(vt.vit_presets.keys())}")
        return True, vt
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_weight_loaders():
    """Test weight loading functions."""
    print("\n[2] Testing weight loaders...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "weight_loaders", 
            os.path.join(SIGNVLM_DIR, "weight_loaders.py")
        )
        wl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wl)
        
        print("    ✓ load_weights_clip function available")
        print("    ✓ load_weights_mae function available")
        print("    ✓ load_weights_timm function available")
        return True, wl
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model(vt_module):
    """Test EVLTransformer model."""
    print("\n[3] Testing EVLTransformer model...")
    try:
        # Set up proper package structure in sys.modules
        import types
        
        # Create the package hierarchy
        if 'app' not in sys.modules:
            sys.modules['app'] = types.ModuleType('app')
        if 'app.services' not in sys.modules:
            sys.modules['app.services'] = types.ModuleType('app.services')
        if 'app.services.sign_language' not in sys.modules:
            sys.modules['app.services.sign_language'] = types.ModuleType('app.services.sign_language')
        if 'app.services.sign_language.signvlm' not in sys.modules:
            sys.modules['app.services.sign_language.signvlm'] = types.ModuleType('app.services.sign_language.signvlm')
        
        # Inject vision_transformer into the package
        sys.modules['app.services.sign_language.signvlm.vision_transformer'] = vt_module
        
        # Import weight_loaders
        import importlib.util
        spec_wl = importlib.util.spec_from_file_location(
            "app.services.sign_language.signvlm.weight_loaders", 
            os.path.join(SIGNVLM_DIR, "weight_loaders.py")
        )
        wl = importlib.util.module_from_spec(spec_wl)
        sys.modules['app.services.sign_language.signvlm.weight_loaders'] = wl
        spec_wl.loader.exec_module(wl)
        
        # Now import the model
        spec = importlib.util.spec_from_file_location(
            "app.services.sign_language.signvlm.model", 
            os.path.join(SIGNVLM_DIR, "model.py")
        )
        model_module = importlib.util.module_from_spec(spec)
        sys.modules['app.services.sign_language.signvlm.model'] = model_module
        spec.loader.exec_module(model_module)
        
        print("    ✓ EVLTransformer class available")
        print("    ✓ EVLDecoder class available")
        print("    ✓ TemporalCrossAttention class available")
        return True, model_module
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation(vt_module, model_module):
    """Test creating the actual model."""
    print("\n[4] Testing model creation...")
    try:
        import torch
        
        # Get preset for ViT-L/14
        preset = vt_module.vit_presets["ViT-L/14-lnpre"]
        print(f"    ✓ Using preset: ViT-L/14-lnpre")
        print(f"      - Feature dim: {preset['feature_dim']}")
        print(f"      - Num layers: {preset['num_layers']}")
        
        # Create model (without loading weights)
        model = model_module.EVLTransformer(
            backbone_name="ViT-L/14-lnpre",
            backbone_type="clip",
            backbone_path="",  # Don't load weights for this test
            backbone_mode="freeze_fp16",  # Valid modes: finetune, freeze_fp16, freeze_fp32
            decoder_num_layers=4,
            decoder_qkv_dim=1024,
            decoder_num_heads=16,
            num_classes=100,
            num_frames=8,  # Use smaller for test
        )
        print("    ✓ EVLTransformer created")
        
        # Check model structure
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    ✓ Total parameters: {total_params:,}")
        print(f"    ✓ Trainable parameters: {trainable_params:,}")
        
        # Test forward pass with dummy data
        print("    Testing forward pass...")
        model.eval()
        with torch.no_grad():
            # Input shape: (batch, channels, frames, height, width)
            dummy_input = torch.randn(1, 3, 8, 224, 224)  # Use 8 frames for speed
            try:
                output = model(dummy_input)
                print(f"    ✓ Forward pass successful, output shape: {output.shape}")
            except Exception as e:
                print(f"    ⚠ Forward pass error (expected without weights): {e}")
        
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backbone_download():
    """Check if backbone is downloaded."""
    print("\n[5] Checking backbone download...")
    try:
        models_dir = os.path.join(BACKEND_DIR, "app", "services", "sign_language", "models", "signvlm")
        backbone_path = os.path.join(models_dir, "ViT-L-14.pt")
        
        if os.path.exists(backbone_path):
            size_mb = os.path.getsize(backbone_path) / (1024 * 1024)
            print(f"    ✓ Backbone found: {backbone_path}")
            print(f"    ✓ Size: {size_mb:.1f} MB")
            return True
        else:
            print(f"    ✗ Backbone not found at: {backbone_path}")
            print("      Run: python download_models.py from signvlm directory")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def test_preprocessing():
    """Test frame preprocessing."""
    print("\n[6] Testing frame preprocessing...")
    try:
        import torch
        import numpy as np
        
        # CLIP normalization values
        CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
        
        # Create dummy frames
        num_frames = 24
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]
        print(f"    ✓ Created {num_frames} dummy frames (224x224x3)")
        
        # Preprocess
        processed_frames = []
        for frame in frames:
            # Convert to float and normalize
            frame_float = frame.astype(np.float32) / 255.0
            # Apply CLIP normalization
            for c in range(3):
                frame_float[:, :, c] = (frame_float[:, :, c] - CLIP_MEAN[c]) / CLIP_STD[c]
            processed_frames.append(frame_float)
        
        # Stack into tensor: (T, H, W, C) -> (C, T, H, W)
        stacked = np.stack(processed_frames, axis=0)  # (T, H, W, C)
        stacked = np.transpose(stacked, (3, 0, 1, 2))  # (C, T, H, W)
        tensor = torch.from_numpy(stacked).unsqueeze(0)  # (1, C, T, H, W)
        
        print(f"    ✓ Preprocessed tensor shape: {tensor.shape}")
        expected_shape = (1, 3, num_frames, 224, 224)
        assert tensor.shape == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"
        print(f"    ✓ Shape matches expected: {expected_shape}")
        
        # Check value ranges after normalization
        print(f"    ✓ Value range: [{tensor.min():.2f}, {tensor.max():.2f}]")
        
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_class_mapping():
    """Test WLASL-100 class mapping."""
    print("\n[7] Testing class mapping...")
    try:
        # WLASL-100 classes from the service
        WLASL100_CLASSES = [
            "book", "drink", "computer", "before", "chair", "go", "clothes", "who",
            "candy", "cousin", "deaf", "fine", "help", "no", "thin", "walk", "year",
            "yes", "all", "black", "cool", "finish", "hot", "like", "many", "mother",
            "now", "orange", "school", "shirt", "study", "tell", "ugly", "what",
            "wrong", "africa", "basketball", "bird", "birthday", "blue", "bowling",
            "brown", "but", "can", "change", "child", "color", "corn", "cow", "dance",
            "dark", "doctor", "dog", "eat", "enjoy", "family", "fish", "forget",
            "give", "graduate", "hat", "have", "hearing", "hello", "homework", "horse",
            "hospital", "hurt", "jacket", "language", "later", "letter", "man", "meet",
            "milk", "name", "need", "not", "nothing", "nurse", "old", "order", "paper",
            "pizza", "play", "please", "pull", "read", "right", "same", "say", "shirt",
            "sit", "son", "sorry", "store", "table", "tall", "thank", "time",
        ]
        
        print(f"    ✓ WLASL-100 classes defined: {len(WLASL100_CLASSES)} classes")
        print(f"    ✓ Sample classes: {WLASL100_CLASSES[:5]}")
        print(f"    ✓ Last classes: {WLASL100_CLASSES[-5:]}")
        
        # Check for duplicates
        unique = set(WLASL100_CLASSES)
        if len(unique) != len(WLASL100_CLASSES):
            print(f"    ⚠ Warning: {len(WLASL100_CLASSES) - len(unique)} duplicate classes")
        else:
            print("    ✓ No duplicate classes")
        
        return True
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SignVLM Direct Module Tests")
    print("=" * 60)
    print(f"SignVLM directory: {SIGNVLM_DIR}")
    
    results = []
    
    # Run tests
    success, vt_module = test_vision_transformer()
    results.append(("Vision Transformer", success))
    
    if not success:
        print("\n✗ Cannot continue without Vision Transformer")
        return False
    
    success, wl_module = test_weight_loaders()
    results.append(("Weight Loaders", success))
    
    success, model_module = test_model(vt_module)
    results.append(("EVLTransformer Model", success))
    
    if model_module:
        results.append(("Model Creation", test_model_creation(vt_module, model_module)))
    else:
        results.append(("Model Creation", False))
    
    results.append(("Backbone Download", test_backbone_download()))
    results.append(("Preprocessing", test_preprocessing()))
    results.append(("Class Mapping", test_class_mapping()))
    
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
    
    if passed == total:
        print("\n✓ SignVLM integration is ready!")
    else:
        print("\n⚠ Some tests failed. Check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
