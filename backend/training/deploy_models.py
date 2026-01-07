"""
Deploy Trained Models
Copies trained models to the backend and updates configuration.
"""

import shutil
import json
from pathlib import Path


# Paths
TRAINING_OUTPUT = Path("outputs")
BACKEND_MODELS = Path("../models")


def deploy_gesture_model():
    """Deploy the trained ASL gesture model (MediaPipe or CNN)."""
    # Try MediaPipe model first
    source = TRAINING_OUTPUT / "asl_gesture_recognizer.task"
    dest = BACKEND_MODELS / "gesture_recognizer.task"
    
    if source.exists():
        # Backup existing model
        if dest.exists():
            backup = BACKEND_MODELS / "gesture_recognizer.task.backup"
            shutil.copy(dest, backup)
            print(f"Backed up existing model to {backup}")
        
        # Copy new model
        shutil.copy(source, dest)
        print(f"✓ Deployed gesture model to {dest}")
        
        # Copy class mapping
        mapping_src = TRAINING_OUTPUT / "asl_class_mapping.json"
        if mapping_src.exists():
            shutil.copy(mapping_src, BACKEND_MODELS / "asl_class_mapping.json")
            print(f"✓ Deployed class mapping")
        
        return True
    
    # Try CNN model as fallback
    cnn_source = TRAINING_OUTPUT / "asl_cnn_model.h5"
    cnn_dest = BACKEND_MODELS / "asl_cnn_model.h5"
    
    if cnn_source.exists():
        # Backup existing model
        if cnn_dest.exists():
            backup = BACKEND_MODELS / "asl_cnn_model.h5.backup"
            shutil.copy(cnn_dest, backup)
            print(f"Backed up existing model to {backup}")
        
        # Copy new model
        shutil.copy(cnn_source, cnn_dest)
        print(f"✓ Deployed ASL CNN model to {cnn_dest}")
        
        # Copy class mapping
        mapping_src = TRAINING_OUTPUT / "asl_class_mapping.json"
        if mapping_src.exists():
            shutil.copy(mapping_src, BACKEND_MODELS / "asl_class_mapping.json")
            print(f"✓ Deployed class mapping")
        
        # Copy TFLite if available
        tflite_src = TRAINING_OUTPUT / "asl_cnn_model.tflite"
        if tflite_src.exists():
            shutil.copy(tflite_src, BACKEND_MODELS / "asl_cnn_model.tflite")
            print(f"✓ Deployed TFLite model")
        
        return True
    
    print(f"ERROR: No ASL model found")
    print("Run train_asl_gestures.py or train_asl_cnn.py first!")
    return False


def deploy_emotion_model():
    """Deploy the trained emotion detection model."""
    source = TRAINING_OUTPUT / "emotion_model.h5"
    dest = BACKEND_MODELS / "emotion_model.h5"
    
    if not source.exists():
        print(f"ERROR: Emotion model not found at {source}")
        print("Run train_emotion_model.py first!")
        return False
    
    # Backup existing model
    if dest.exists():
        backup = BACKEND_MODELS / "emotion_model.h5.backup"
        shutil.copy(dest, backup)
        print(f"Backed up existing model to {backup}")
    
    # Copy new model
    shutil.copy(source, dest)
    print(f"✓ Deployed emotion model to {dest}")
    
    # Copy class mapping
    mapping_src = TRAINING_OUTPUT / "emotion_class_mapping.json"
    if mapping_src.exists():
        shutil.copy(mapping_src, BACKEND_MODELS / "emotion_class_mapping.json")
        print(f"✓ Deployed class mapping")
    
    # Also copy TFLite if available
    tflite_src = TRAINING_OUTPUT / "emotion_model.tflite"
    if tflite_src.exists():
        shutil.copy(tflite_src, BACKEND_MODELS / "emotion_model.tflite")
        print(f"✓ Deployed TFLite model")
    
    return True


def main():
    """Deploy all trained models."""
    print("=" * 60)
    print("Deploying Trained Models to Backend")
    print("=" * 60)
    
    # Ensure backend models directory exists
    BACKEND_MODELS.mkdir(parents=True, exist_ok=True)
    
    gesture_ok = deploy_gesture_model()
    emotion_ok = deploy_emotion_model()
    
    print("\n" + "=" * 60)
    if gesture_ok and emotion_ok:
        print("All models deployed successfully!")
        print("\nRestart the backend server to use the new models.")
    else:
        print("Some models could not be deployed.")
        print("Make sure to train the models first.")
    print("=" * 60)


if __name__ == "__main__":
    main()
