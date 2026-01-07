"""
Emotion Detection Model Training Script
Trains a CNN model on FER-2013 dataset for facial emotion recognition.
This will replace DeepFace with a custom, faster model.

Supports both:
- CSV format (fer2013.csv) - original Kaggle format
- Folder format (train/emotion/*.jpg) - preprocessed format
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import io

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Configuration
DATASET_PATH = Path("datasets/fer2013")
OUTPUT_PATH = Path("outputs")
MODEL_NAME = "emotion_model"

# FER-2013 emotion classes (indexed 0-6 in CSV)
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Emotion to text mapping for the app
EMOTION_RESPONSES = {
    "angry": "I sense frustration. How can I help?",
    "disgust": "Something seems unpleasant. Let me assist.",
    "fear": "You seem worried. I'm here to help.",
    "happy": "Great to see you in good spirits!",
    "neutral": "How can I assist you today?",
    "sad": "I'm sorry you're feeling down. How can I help?",
    "surprise": "Something unexpected? Let me know what you need."
}

# Training parameters
IMG_SIZE = 48  # FER-2013 images are 48x48
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001


def check_dataset():
    """Verify the FER-2013 dataset exists (CSV or folder format)."""
    csv_path = DATASET_PATH / "fer2013.csv"
    train_path = DATASET_PATH / "train"
    
    # Check for CSV format first
    if csv_path.exists():
        print(f"Found FER-2013 CSV at {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Total samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Count by emotion
        if 'emotion' in df.columns:
            for i, emotion in enumerate(EMOTIONS):
                count = len(df[df['emotion'] == i])
                print(f"  {emotion}: {count} samples")
        return "csv"
    
    # Check for folder format
    if train_path.exists():
        print("Found folder format dataset")
        total_train = 0
        for emotion in EMOTIONS:
            emotion_path = train_path / emotion
            if emotion_path.exists():
                count = len(list(emotion_path.glob("*.jpg"))) + len(list(emotion_path.glob("*.png")))
                total_train += count
                print(f"  {emotion}: {count} images")
        
        if total_train < 1000:
            print("ERROR: Not enough training images")
            return None
        return "folder"
    
    print(f"ERROR: Dataset not found at {DATASET_PATH}")
    print("\nPlease download the FER-2013 dataset from Kaggle:")
    print("  https://www.kaggle.com/datasets/msambare/fer2013")
    return None


def load_data_from_csv():
    """Load and preprocess data from fer2013.csv."""
    print("\n=== Loading Data from CSV ===")
    
    csv_path = DATASET_PATH / "fer2013.csv"
    df = pd.read_csv(csv_path)
    
    # Parse pixel strings to numpy arrays
    print("Parsing pixel data...")
    pixels = df['pixels'].apply(lambda x: np.fromstring(x, sep=' ').reshape(48, 48).astype('float32'))
    X = np.stack(pixels.values)
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
    
    # Get labels
    y = df['emotion'].values
    y = keras.utils.to_categorical(y, num_classes=len(EMOTIONS))
    
    # Split based on 'Usage' column if available, else manual split
    if 'Usage' in df.columns:
        train_mask = df['Usage'] == 'Training'
        test_mask = df['Usage'] == 'PublicTest'
        val_mask = df['Usage'] == 'PrivateTest'
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
    else:
        # Manual split: 80% train, 10% val, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_emotion_model():
    """
    Build a CNN model for emotion detection.
    Architecture optimized for 48x48 grayscale facial images.
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        # First Conv Block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Conv Block
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
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
        layers.Dense(len(EMOTIONS), activation='softmax')
    ])
    
    return model


def create_data_generators():
    """Create training and validation data generators with augmentation (folder format)."""
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1  # 10% for validation
    )
    
    # Test data - only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH / "train",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH / "train",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Test generator
    test_generator = test_datagen.flow_from_directory(
        DATASET_PATH / "test",
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator


def train_model(dataset_format="csv"):
    """Train the emotion detection model."""
    print("\n=== Training Emotion Detection Model ===")
    
    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    if dataset_format == "csv":
        # Load data from CSV
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data_from_csv()
        
        # Create augmentation for training
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_datagen.fit(X_train)
        
        train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)
        steps_per_epoch = len(X_train) // BATCH_SIZE
        class_indices = {emotion: i for i, emotion in enumerate(EMOTIONS)}
        
    else:
        # Load from folder structure
        print("Creating data generators...")
        train_gen, val_gen, test_gen = create_data_generators()
        val_data = val_gen
        test_data = test_gen
        steps_per_epoch = None
        class_indices = train_gen.class_indices
        
        print(f"  Training samples: {train_gen.samples}")
        print(f"  Validation samples: {val_gen.samples}")
        print(f"  Test samples: {test_gen.samples}")
    
    print(f"  Classes: {list(class_indices.keys())}")
    
    # Build model
    print("\nBuilding model...")
    model = build_emotion_model()
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
        # Save best model
        callbacks.ModelCheckpoint(
            OUTPUT_PATH / f"{MODEL_NAME}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # TensorBoard logging
        callbacks.TensorBoard(
            log_dir=OUTPUT_PATH / "logs",
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nTraining model (this may take 30-60 minutes)...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        callbacks=model_callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n=== Evaluating on Test Set ===")
    if dataset_format == "csv":
        test_loss, test_accuracy = model.evaluate(test_data[0], test_data[1], verbose=1)
    else:
        test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    
    # Save final model
    final_model_path = OUTPUT_PATH / f"{MODEL_NAME}.h5"
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    # Save as TFLite for mobile deployment (optional)
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
    
    # Save training history
    import json
    history_path = OUTPUT_PATH / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
        }, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save class mapping
    class_mapping = {str(v): k for k, v in class_indices.items()}
    mapping_path = OUTPUT_PATH / "emotion_class_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump({
            'class_indices': class_indices,
            'index_to_class': class_mapping,
            'emotion_responses': EMOTION_RESPONSES
        }, f, indent=2)
    print(f"Class mapping saved to: {mapping_path}")
    
    return model, history


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Emotion Detection Model Training (FER-2013)")
    print("=" * 60)
    
    # Check GPU availability
    print("\n=== Hardware Check ===")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {gpus}")
        # Enable memory growth to prevent OOM
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("No GPU found. Training will use CPU (slower).")
    
    # Check dataset
    print("\n=== Checking Dataset ===")
    dataset_format = check_dataset()
    if not dataset_format:
        return
    
    # Train model
    try:
        model, history = train_model(dataset_format)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\nModel saved to: {OUTPUT_PATH / f'{MODEL_NAME}.h5'}")
        print("\nTo use this model in the backend:")
        print(f"  1. Copy {OUTPUT_PATH / f'{MODEL_NAME}.h5'} to backend/models/")
        print("  2. Update vision_service.py to load the custom model")
        print("  3. Restart the backend server")
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
