import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


def create_emotion_model(input_shape=(48, 48, 1), num_classes=1):
    """Create a CNN model that adapts to your dataset"""
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Second Conv Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Third Conv Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')  # Use sigmoid for binary classification
    ])

    return model


# ---------- 1️⃣ Check and fix dataset structure ----------
dataset_dir = "dataset/train"

if not os.path.exists(dataset_dir):
    print("❌ Dataset folder not found! Creating sample structure...")
    os.makedirs(dataset_dir, exist_ok=True)

    # Create proper emotion folder structure
    emotions = ['angry', 'happy', 'sad', 'neutral', 'surprise', 'fear', 'disgust']
    for emotion in emotions:
        emotion_dir = os.path.join(dataset_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        print(f"📁 Created: {emotion_dir}")

    print("\n📝 Please add your emotion images to these folders!")
    print("💡 You can download emotion datasets from:")
    print("   - FER2013: https://www.kaggle.com/datasets/msambare/fer2013")
    print("   - CK+: https://www.kaggle.com/datasets/shawon10/ckplus")
    exit()

# Check what folders actually exist
print("📁 Checking dataset structure...")
subfolders = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
print(f"📂 Found subfolders: {subfolders}")

# Count images in each folder
print("\n📊 Image count per folder:")
for folder in subfolders:
    folder_path = os.path.join(dataset_dir, folder)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_count = len(image_files)
    print(f"  {folder}: {image_count} images")

    # Show first few files
    if image_files:
        print(f"    Sample files: {image_files[:3]}")

# Determine number of classes and appropriate configuration
num_classes = len(subfolders)
print(f"\n🎯 Number of classes detected: {num_classes}")

if num_classes == 1:
    print("⚠️  Only 1 class found! This will be treated as binary classification.")
    class_mode = "binary"
    final_activation = "sigmoid"
    loss_function = "binary_crossentropy"
elif num_classes == 2:
    print("⚠️  2 classes found! Using binary classification.")
    class_mode = "binary"
    final_activation = "sigmoid"
    loss_function = "binary_crossentropy"
else:
    print("✅ Multiple classes found! Using categorical classification.")
    class_mode = "categorical"
    final_activation = "softmax"
    loss_function = "categorical_crossentropy"

# ---------- 2️⃣ Create model with correct configuration ----------
print("🔄 Creating emotion recognition model...")
model = create_emotion_model(num_classes=num_classes)

# Update the final layer based on number of classes
model.pop()  # Remove the last layer
if num_classes <= 2:
    model.add(Dense(1, activation='sigmoid'))
else:
    model.add(Dense(num_classes, activation='softmax'))

model.summary()

# ---------- 3️⃣ Prepare data generators ----------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

print(f"\n📁 Loading images from: {dataset_dir}")

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode=class_mode,
    subset="training",
    batch_size=32,
    shuffle=True,
)

val_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    class_mode=class_mode,
    subset="validation",
    batch_size=32,
    shuffle=False,
)

# Print class indices to verify
print("📊 Class indices:", train_gen.class_indices)
print(f"📊 Class mode: {class_mode}")

# ---------- 4️⃣ Compile the model ----------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=loss_function,
    metrics=["accuracy"]
)
print("✅ Model compiled and ready to train!")

# ---------- 5️⃣ Callbacks ----------
checkpoint = ModelCheckpoint(
    "emotion_model_best.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode='max',
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# ---------- 6️⃣ Train the model ----------
print("🚀 Starting training...")
print(f"📊 Training samples: {train_gen.samples}")
print(f"📊 Validation samples: {val_gen.samples}")

# Adjust epochs based on dataset size
epochs = 30 if train_gen.samples > 1000 else 50

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint, earlystop, reduce_lr],
    verbose=1
)

# ---------- 7️⃣ Save the final model ----------
model.save("emotion_model_final.h5")
print("✅ Training complete! Model saved as emotion_model_final.h5")

# Print final accuracy
if 'accuracy' in history.history:
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"📊 Final Training Accuracy: {final_train_acc:.4f}")
    print(f"📊 Final Validation Accuracy: {final_val_acc:.4f}")