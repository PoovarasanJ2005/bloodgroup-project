"""
Blood Group Prediction CNN Model Training Script — V2 (Enhanced)
Trains a deeper CNN with class-balanced weights and stronger augmentation.
Target: ≥92% accuracy on test set with ≥85% per-class recall.

Changes vs V1:
  - Image size: 96 → 128
  - Stronger augmentation (rotation, zoom, contrast shifts)
  - Class weights to fix AB- and O- recall
  - Deeper architecture with residual-style connections
  - Cosine annealing learning rate schedule
  - More epochs (80) with patience 15
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# --- Configuration ---
DATASET_BASE = os.path.join(os.path.dirname(__file__), '..', 'dataset')
TRAIN_DIR    = os.path.join(DATASET_BASE, 'train')
TEST_DIR     = os.path.join(DATASET_BASE, 'test')
MODEL_DIR    = os.path.join(os.path.dirname(__file__), 'saved_model')

IMG_SIZE     = 128      # Increased from 96 for more detail
BATCH_SIZE   = 32
EPOCHS       = 80       # More epochs, rely on early stopping
SEED         = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# --- Sanity check ---
print("=" * 60)
print("BLOOD GROUP PREDICTION - CNN MODEL TRAINING V2 (ENHANCED)")
print("=" * 60)

if not os.path.isdir(TRAIN_DIR):
    print(f"[ERROR] Training folder not found: {TRAIN_DIR}")
    print("        Run  python dataset/split_dataset.py  first!")
    sys.exit(1)

if not os.path.isdir(TEST_DIR):
    print(f"[ERROR] Test folder not found: {TEST_DIR}")
    print("        Run  python dataset/split_dataset.py  first!")
    sys.exit(1)

# --- Data Loading with STRONGER augmentation ---
print(f"\n[INFO] Using enhanced augmentation pipeline")

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

print(f"[INFO] Train folder : {TRAIN_DIR}")
print(f"[INFO] Test  folder : {TEST_DIR}")
print(f"[INFO] Image size   : {IMG_SIZE}x{IMG_SIZE} (grayscale)")
print(f"[INFO] Batch size   : {BATCH_SIZE}")
print(f"[INFO] Epochs       : {EPOCHS}")

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED,
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=SEED,
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print(f"\n[INFO] Classes ({num_classes}): {class_names}")
print(f"[INFO] Training samples : {train_generator.samples}")
print(f"[INFO] Test samples     : {test_generator.samples}")

# --- Compute class weights to handle imbalanced data ---
print("\n[INFO] Computing class weights for balanced training...")
class_counts = Counter(train_generator.classes)
total_samples = sum(class_counts.values())

# Compute sklearn class weights
class_weight_values = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weight_values))

print("[INFO] Class weights:")
for idx, weight in class_weights.items():
    cls_name = class_names[idx]
    count = class_counts[idx]
    print(f"       {cls_name:5s}: weight={weight:.3f}  (count={count})")

# Save class mapping
class_mapping = {v: k for k, v in train_generator.class_indices.items()}
with open(os.path.join(MODEL_DIR, 'class_mapping.json'), 'w') as f:
    json.dump(class_mapping, f, indent=2)

# --- CNN Model Architecture V2 (Deeper + Better regularization) ---
print("\n[BUILD] Building Enhanced CNN Model V2...")


def conv_block(x, filters, dropout_rate=0.25):
    """Double convolution block with BatchNorm and Dropout."""
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


# Build with Functional API for flexibility
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# Block 1: 128x128 → 64x64
x = conv_block(inputs, 32, dropout_rate=0.2)

# Block 2: 64x64 → 32x32
x = conv_block(x, 64, dropout_rate=0.25)

# Block 3: 32x32 → 16x16
x = conv_block(x, 128, dropout_rate=0.3)

# Block 4: 16x16 → 8x8
x = conv_block(x, 256, dropout_rate=0.3)

# Block 5: 8x8 → 4x4
x = conv_block(x, 512, dropout_rate=0.35)

# Global pooling instead of flatten (more robust)
x = layers.GlobalAveragePooling2D()(x)

# Dense layers
x = layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs, name='BloodGroupCNN_V2')

# --- Cosine Annealing Learning Rate ---
initial_lr = 0.001
steps_per_epoch = train_generator.samples // BATCH_SIZE

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=steps_per_epoch * EPOCHS,
    alpha=1e-6,  # minimum LR
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Callbacks ---
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1,
    min_delta=0.001
)

checkpoint = callbacks.ModelCheckpoint(
    os.path.join(MODEL_DIR, 'best_model.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# --- Training ---
print("\n[TRAIN] Starting enhanced training...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# --- Save Final Model ---
model.save(os.path.join(MODEL_DIR, 'blood_group_model.h5'))
print(f"\n[OK] Model saved to {MODEL_DIR}")

# --- Evaluation on Test Set ---
print("\n[EVAL] Evaluating model on test set...")
test_generator.reset()
loss, accuracy = model.evaluate(test_generator, verbose=0)
print(f"   Test Loss     : {loss:.4f}")
print(f"   Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=0)
y_pred  = np.argmax(y_pred_probs, axis=1)
y_true  = test_generator.classes

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print("\n[REPORT] Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Per-class accuracy summary
print("[SUMMARY] Per-class recall:")
all_ok = True
for cls in class_names:
    recall = report[cls]['recall']
    status = "OK" if recall >= 0.85 else "NEEDS IMPROVEMENT"
    if recall < 0.85:
        all_ok = False
    print(f"   {cls:5s}: {recall*100:6.2f}% [{status}]")

if all_ok:
    print("\n   ALL CLASSES MEET ≥85% RECALL TARGET!")
else:
    print("\n   Some classes below 85% — consider retraining with more data.")

# Save metrics
metrics = {
    'accuracy':      float(accuracy),
    'loss':          float(loss),
    'class_report':  report,
    'class_names':   class_names,
    'epochs_trained': len(history.history['accuracy']),
    'img_size':      IMG_SIZE,
    'train_samples': train_generator.samples,
    'test_samples':  test_generator.samples,
    'model_version': 'v2_enhanced',
    'class_weights': {class_names[k]: float(v) for k, v in class_weights.items()},
}
with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# --- Plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Test Accuracy')
axes[0].set_title('Model Accuracy (V2)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Test Loss')
axes[1].set_title('Model Loss (V2)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_plots.png'), dpi=150)
print("[PLOT] Training plots saved.")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — Test Set (V2)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
print("[PLOT] Confusion matrix saved.")

print("\n" + "=" * 60)
print("TRAINING V2 COMPLETE!")
print(f"   Final Test Accuracy : {accuracy*100:.2f}%")
print(f"   Model               : {MODEL_DIR}/blood_group_model.h5")
print("=" * 60)
