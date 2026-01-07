"""
ASL Gesture Recognition Training Script
Uses MediaPipe Model Maker to train a custom gesture recognizer
for American Sign Language alphabet (A-Z + space, delete, nothing)
"""

import os
import shutil
import json
from pathlib import Path

# Check for MediaPipe Model Maker
try:
    from mediapipe_model_maker import gesture_recognizer
    from mediapipe_model_maker.gesture_recognizer import GestureRecognizer
    from mediapipe_model_maker.gesture_recognizer import HParams
    from mediapipe_model_maker.gesture_recognizer import ModelOptions
    from mediapipe_model_maker.gesture_recognizer import Dataset
    MAKER_AVAILABLE = True
except ImportError:
    print("MediaPipe Model Maker not installed. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "mediapipe-model-maker"], check=True)
    from mediapipe_model_maker import gesture_recognizer
    from mediapipe_model_maker.gesture_recognizer import GestureRecognizer
    from mediapipe_model_maker.gesture_recognizer import HParams
    from mediapipe_model_maker.gesture_recognizer import ModelOptions
    from mediapipe_model_maker.gesture_recognizer import Dataset
    MAKER_AVAILABLE = True


# Configuration
DATASET_PATH = Path("datasets/asl-alphabet/asl_alphabet_train/asl_alphabet_train")
OUTPUT_PATH = Path("outputs")
MODEL_NAME = "asl_gesture_recognizer"

# ASL Alphabet classes (what we're training)
ASL_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

# Mapping from ASL letters to meaningful text for the app
ASL_TO_TEXT = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
    "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
    "K": "K", "L": "L", "M": "M", "N": "N", "O": "O",
    "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
    "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y",
    "Z": "Z", "del": "[DELETE]", "nothing": "", "space": " "
}


def check_dataset():
    """Verify the dataset exists and has correct structure."""
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("\nPlease download the ASL Alphabet dataset from Kaggle:")
        print("  1. Go to: https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        print("  2. Download and extract to: training/datasets/asl-alphabet/")
        print("\nExpected structure:")
        print("  datasets/asl-alphabet/asl_alphabet_train/A/")
        print("  datasets/asl-alphabet/asl_alphabet_train/B/")
        print("  ... etc")
        return False
    
    # Check for class folders
    found_classes = []
    for class_name in ASL_CLASSES:
        class_path = DATASET_PATH / class_name
        if class_path.exists():
            num_images = len(list(class_path.glob("*.jpg"))) + len(list(class_path.glob("*.png")))
            found_classes.append((class_name, num_images))
            print(f"  Found class '{class_name}': {num_images} images")
        else:
            print(f"  WARNING: Class '{class_name}' not found")
    
    if len(found_classes) < 10:
        print(f"\nERROR: Only found {len(found_classes)} classes. Need at least 10.")
        return False
    
    print(f"\n✓ Found {len(found_classes)} classes for training")
    return True


def prepare_data_for_mediapipe():
    """
    Prepare the ASL dataset for MediaPipe Model Maker.
    MediaPipe expects a specific folder structure.
    """
    print("\n=== Preparing Data for MediaPipe ===")
    
    # Create prepared data folder
    prepared_path = Path("datasets/asl-prepared")
    if prepared_path.exists():
        shutil.rmtree(prepared_path)
    prepared_path.mkdir(parents=True)
    
    # Limit samples per class (for faster training, can increase for better accuracy)
    MAX_SAMPLES_PER_CLASS = 500  # Reduce for faster training, increase for better accuracy
    
    for class_name in ASL_CLASSES:
        src_path = DATASET_PATH / class_name
        if not src_path.exists():
            continue
        
        dst_path = prepared_path / class_name
        dst_path.mkdir(parents=True, exist_ok=True)
        
        # Copy images (limit number for faster training)
        images = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png"))
        for i, img_path in enumerate(images[:MAX_SAMPLES_PER_CLASS]):
            shutil.copy(img_path, dst_path / f"{class_name}_{i}.jpg")
        
        print(f"  Prepared {min(len(images), MAX_SAMPLES_PER_CLASS)} images for class '{class_name}'")
    
    return prepared_path


def train_gesture_model(data_path: Path):
    """Train the gesture recognition model using MediaPipe Model Maker."""
    print("\n=== Training ASL Gesture Model ===")
    
    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    data = Dataset.from_folder(
        dirname=str(data_path),
        hparams=HParams(export_dir=str(OUTPUT_PATH))
    )
    
    # Split into train and validation
    train_data, validation_data = data.split(0.8)
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(validation_data)}")
    
    # Configure model
    print("\nConfiguring model...")
    model_options = ModelOptions(
        dropout_rate=0.2,
        layer_widths=[128, 64]  # Adjust for model complexity
    )
    
    hparams = HParams(
        export_dir=str(OUTPUT_PATH),
        epochs=30,  # Increase for better accuracy
        batch_size=32,
        learning_rate=0.001
    )
    
    # Train model
    print("\nTraining model (this may take 15-30 minutes)...")
    model = GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        model_options=model_options,
        hparams=hparams
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(validation_data)
    print(f"  Validation Loss: {loss:.4f}")
    print(f"  Validation Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Export model
    print("\nExporting model...")
    model_path = OUTPUT_PATH / f"{MODEL_NAME}.task"
    model.export_model(model_bundle_name=f"{MODEL_NAME}.task")
    print(f"  Model saved to: {model_path}")
    
    # Save class mapping
    mapping_path = OUTPUT_PATH / "asl_class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(ASL_TO_TEXT, f, indent=2)
    print(f"  Class mapping saved to: {mapping_path}")
    
    return model_path


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("ASL Gesture Recognition Model Training")
    print("=" * 60)
    
    # Step 1: Check dataset
    print("\n=== Checking Dataset ===")
    if not check_dataset():
        return
    
    # Step 2: Prepare data
    prepared_path = prepare_data_for_mediapipe()
    
    # Step 3: Train model
    try:
        model_path = train_gesture_model(prepared_path)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\nModel saved to: {model_path}")
        print("\nTo use this model in the backend:")
        print(f"  1. Copy {model_path} to backend/models/gesture_recognizer.task")
        print("  2. Restart the backend server")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
