"""
ASL Alphabet CNN Training Script
Trains a CNN model for ASL alphabet recognition (A-Z + space, delete, nothing).
Alternative to MediaPipe Model Maker - works on all Python versions.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
DATASET_PATH = Path("datasets/asl-alphabet/asl_alphabet_train/asl_alphabet_train")
OUTPUT_PATH = Path("outputs")
MODEL_NAME = "asl_cnn_model"

# ASL Alphabet classes
ASL_CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

# Mapping from ASL letters to text
ASL_TO_TEXT = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
    "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
    "K": "K", "L": "L", "M": "M", "N": "N", "O": "O",
    "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
    "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y",
    "Z": "Z", "del": "[DELETE]", "nothing": "", "space": " "
}

# Training parameters
IMG_SIZE = 64  # Resize images to 64x64
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
MAX_SAMPLES_PER_CLASS = 1000  # Limit samples per class for faster training


def check_dataset():
    """Verify the ASL dataset exists."""
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        print("\nPlease download from Kaggle:")
        print("  https://www.kaggle.com/datasets/grassknoted/asl-alphabet")
        return False
    
    print("Found ASL classes:")
    total = 0
    for cls in ASL_CLASSES:
        cls_path = DATASET_PATH / cls
        if cls_path.exists():
            count = len(list(cls_path.glob("*.jpg"))) + len(list(cls_path.glob("*.png")))
            total += min(count, MAX_SAMPLES_PER_CLASS)
            print(f"  {cls}: {count} images (using {min(count, MAX_SAMPLES_PER_CLASS)})")
        else:
            print(f"  WARNING: {cls} not found")
    
    print(f"\nTotal samples to use: {total}")
    return total > 1000


def load_data():
    """Load and preprocess the ASL dataset."""
    print("\n=== Loading Dataset ===")
    
    images = []
    labels = []
    class_indices = {cls: i for i, cls in enumerate(ASL_CLASSES)}
    
    for cls in ASL_CLASSES:
        cls_path = DATASET_PATH / cls
        if not cls_path.exists():
            continue
        
        # Get image files
        img_files = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
        img_files = img_files[:MAX_SAMPLES_PER_CLASS]  # Limit samples
        
        for img_path in img_files:
            try:
                # Load and resize image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                images.append(img_array)
                labels.append(class_indices[cls])
            except Exception as e:
                continue
        
        print(f"  Loaded {len(img_files)} images for class '{cls}'")
    
    X = np.array(images)
    y = keras.utils.to_categorical(labels, num_classes=len(ASL_CLASSES))
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split: 80% train, 10% val, 10% test
    n = len(X)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_indices


def build_asl_model():
    """Build a CNN model for ASL recognition."""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # First Conv Block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Conv Block
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(len(ASL_CLASSES), activation='softmax')
    ])
    
    return model


def train_model():
    """Train the ASL recognition model."""
    print("\n=== Training ASL Recognition Model ===")
    
    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_indices = load_data()
    
    # Create augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip - ASL signs are not symmetric
        fill_mode='nearest'
    )
    train_datagen.fit(X_train)
    
    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    
    # Build model
    print("\nBuilding model...")
    model = build_asl_model()
    model.summary()
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    model_callbacks = [
        callbacks.ModelCheckpoint(
            OUTPUT_PATH / f"{MODEL_NAME}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\nTraining model...")
    steps_per_epoch = len(X_train) // BATCH_SIZE
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        callbacks=model_callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n=== Evaluating on Test Set ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    
    # Save final model
    final_model_path = OUTPUT_PATH / f"{MODEL_NAME}.h5"
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    # Save TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = OUTPUT_PATH / f"{MODEL_NAME}.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to: {tflite_path}")
    except Exception as e:
        print(f"Warning: Could not save TFLite model: {e}")
    
    # Save class mapping
    index_to_class = {str(i): cls for cls, i in class_indices.items()}
    mapping_path = OUTPUT_PATH / "asl_class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_indices': class_indices,
            'index_to_class': index_to_class,
            'asl_to_text': ASL_TO_TEXT,
            'img_size': IMG_SIZE
        }, f, indent=2)
    print(f"Class mapping saved to: {mapping_path}")
    
    return model, history


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("ASL Alphabet CNN Model Training")
    print("=" * 60)
    
    # Check GPU
    print("\n=== Hardware Check ===")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found. Training will use CPU.")
    
    # Check dataset
    print("\n=== Checking Dataset ===")
    if not check_dataset():
        return
    
    # Train
    try:
        model, history = train_model()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\nModel saved to: {OUTPUT_PATH / f'{MODEL_NAME}.h5'}")
        print("\nTo deploy this model:")
        print("  1. Run: python deploy_models.py")
        print("  2. Restart the backend server")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
