"""
Train the blood-group model on normalized fingertip crops.

Training now matches inference more closely:
1. Each training image is cropped to the main fingertip region.
2. The crop is converted to grayscale and contrast-enhanced.
3. Gabor filtering enhances ridge/valley texture in the normalized crop.
4. The normalized grayscale crop is replicated to 3 channels for EfficientNet.
5. Non-fingerprint rejection still happens before inference in the Flask API.
"""

import json
import os
import shutil
import sys
from collections import Counter

import cv2
import matplotlib
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from image_validator import normalize_fingerprint_image

matplotlib.use('Agg')
import matplotlib.pyplot as plt


if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


DATASET_BASE = os.path.join(os.path.dirname(__file__), '..', 'dataset')
TRAIN_DIR = os.path.join(DATASET_BASE, 'train')
TEST_DIR = os.path.join(DATASET_BASE, 'test')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, 'blood_group_model.h5')
CANDIDATE_MODEL_PATH = os.path.join(MODEL_DIR, 'candidate_blood_group_model.h5')
CURRENT_METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
CANDIDATE_METRICS_PATH = os.path.join(MODEL_DIR, 'candidate_metrics.json')

RAW_SIZE = 256
IMG_SIZE = 128
BATCH_SIZE = 24
SEED = 42
LABEL_SMOOTH = 0.1
IMAGE_EXTENSIONS = {'.bmp', '.dib', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}

os.makedirs(MODEL_DIR, exist_ok=True)


class FingerprintDirectorySequence(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, augmenter=None, shuffle=False, seed=SEED):
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        self.class_names = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        self.class_indices = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        filepaths = []
        classes = []
        for class_name in self.class_names:
            class_dir = os.path.join(directory, class_name)
            for root, _, filenames in os.walk(class_dir):
                for filename in sorted(filenames):
                    if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                        filepaths.append(os.path.join(root, filename))
                        classes.append(self.class_indices[class_name])

        self.filepaths = np.array(filepaths)
        self.classes = np.array(classes, dtype=np.int32)
        self.samples = len(self.filepaths)
        self.index_array = np.arange(self.samples)
        self._processed_cache = {}
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.samples / self.batch_size))

    def __getitem__(self, batch_index):
        batch_ids = self.index_array[
            batch_index * self.batch_size:(batch_index + 1) * self.batch_size
        ]
        batch_x = np.zeros((len(batch_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_ids), len(self.class_indices)), dtype=np.float32)

        for out_idx, sample_idx in enumerate(batch_ids):
            cache_key = int(sample_idx)
            if cache_key not in self._processed_cache:
                image = np.array(Image.open(self.filepaths[sample_idx]).convert('RGB'))
                self._processed_cache[cache_key] = preprocess_training_gray(image)
            processed = cv2.cvtColor(self._processed_cache[cache_key], cv2.COLOR_GRAY2RGB).astype(np.float32)
            if self.augmenter is not None:
                processed = self.augmenter.random_transform(processed)
            batch_x[out_idx] = np.clip(processed, 0.0, 255.0)
            batch_y[out_idx, self.classes[sample_idx]] = 1.0

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.index_array)

    def reset(self):
        self.index_array = np.arange(self.samples)


def preprocess_training_gray(img_array):
    array = np.clip(img_array, 0, 255).astype(np.uint8)
    return normalize_fingerprint_image(array, target_size=IMG_SIZE, enhance_ridges=True)


def preprocess_training_image(img_array):
    normalized = preprocess_training_gray(img_array)
    rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
    return rgb.astype(np.float32)


def build_generators():
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest',
    )

    train_generator = FingerprintDirectorySequence(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        augmenter=train_datagen,
        seed=SEED,
        shuffle=True,
    )

    test_generator = FingerprintDirectorySequence(
        TEST_DIR,
        batch_size=BATCH_SIZE,
        seed=SEED,
        shuffle=False,
    )

    return train_generator, test_generator


def compute_class_weights(train_generator):
    class_counts = Counter(train_generator.classes)
    total_samples = sum(class_counts.values())
    num_classes = len(train_generator.class_indices)
    class_weights = {}
    for cls_idx, count in class_counts.items():
        class_weights[int(cls_idx)] = total_samples / (num_classes * count)
    return class_weights


def build_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='BloodGroupCNN_fingertip_v2')
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=['accuracy'],
    )


def progressive_train(model, base_model, train_generator, test_generator, class_weights):
    histories = []

    if os.environ.get('TRAINING_PROFILE') == 'quick':
        phases = [
            ('head_only', 1e-3, 3, 1.0),
            ('unfreeze_30', 5e-5, 2, 0.70),
        ]
    else:
        phases = [
            ('head_only', 1e-3, 12, 1.0),
            ('unfreeze_30', 5e-5, 10, 0.70),
            ('unfreeze_60', 1e-5, 10, 0.40),
            ('full_finetune', 5e-6, 10, 0.0),
        ]

    total_layers = len(base_model.layers)

    for phase_name, learning_rate, epochs, freeze_fraction in phases:
        freeze_until = int(total_layers * freeze_fraction)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in base_model.layers[freeze_until:]:
            layer.trainable = True

        compile_model(model, learning_rate)

        phase_history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=test_generator,
            class_weight=class_weights,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=6,
                    restore_best_weights=True,
                    verbose=1,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1,
                ),
                callbacks.ModelCheckpoint(
                    CANDIDATE_MODEL_PATH,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1,
                ),
            ],
            verbose=1,
        )
        histories.append((phase_name, phase_history))

    return histories


def save_artifacts(model, histories, train_generator, test_generator, class_weights):
    model.save(CANDIDATE_MODEL_PATH)

    class_names = list(train_generator.class_indices.keys())
    class_mapping = {v: k for k, v in train_generator.class_indices.items()}
    with open(os.path.join(MODEL_DIR, 'candidate_class_mapping.json'), 'w') as handle:
        json.dump(class_mapping, handle, indent=2)

    test_generator.reset()
    loss, accuracy = model.evaluate(test_generator, verbose=0)

    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': float(accuracy),
        'loss': float(loss),
        'class_report': report,
        'class_names': class_names,
        'epochs_trained': sum(len(history.history['accuracy']) for _, history in histories),
        'img_size': IMG_SIZE,
        'raw_input_size': RAW_SIZE,
        'model_version': 'fingertip_crop_gabor_v2',
        'architecture': 'EfficientNetB0 + fingertip crop + CLAHE + Gabor ridge enhancement',
        'input_value_range': '0_255',
        'preprocessing': {
            'crop': 'foreground/skin/texture fingertip crop',
            'contrast': 'CLAHE',
            'ridge_enhancement': 'Gabor filter bank, 12 orientations',
        },
        'label_smoothing': LABEL_SMOOTH,
        'train_samples': train_generator.samples,
        'test_samples': test_generator.samples,
        'class_weights': {class_mapping[k]: round(v, 3) for k, v in class_weights.items()},
    }

    with open(CANDIDATE_METRICS_PATH, 'w') as handle:
        json.dump(metrics, handle, indent=2)

    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    boundaries = []
    running = 0
    for _, history in histories:
        train_acc.extend(history.history['accuracy'])
        val_acc.extend(history.history['val_accuracy'])
        train_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        running += len(history.history['accuracy'])
        boundaries.append(running)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_acc, label='Train Accuracy')
    axes[0].plot(val_acc, label='Val Accuracy')
    for boundary in boundaries[:-1]:
        axes[0].axvline(x=boundary, color='red', linestyle='--', alpha=0.35)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_loss, label='Train Loss')
    axes[1].plot(val_loss, label='Val Loss')
    for boundary in boundaries[:-1]:
        axes[1].axvline(x=boundary, color='red', linestyle='--', alpha=0.35)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'candidate_training_plots.png'), dpi=150)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({accuracy * 100:.2f}% accuracy)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'candidate_confusion_matrix.png'), dpi=150)

    return accuracy, loss


def maybe_promote_candidate(candidate_accuracy):
    promote = os.environ.get('PROMOTE_CANDIDATE_MODEL') == '1'
    if not promote:
        print('[INFO] Candidate model was not promoted. Set PROMOTE_CANDIDATE_MODEL=1 to promote after comparison.')
        return

    current_accuracy = -1.0
    if os.path.exists(CURRENT_METRICS_PATH):
        with open(CURRENT_METRICS_PATH, 'r') as handle:
            current_accuracy = float(json.load(handle).get('accuracy', -1.0))

    if candidate_accuracy < current_accuracy:
        print(f'[INFO] Candidate accuracy {candidate_accuracy:.4f} is below current best {current_accuracy:.4f}; keeping current model.')
        return

    shutil.copy2(CANDIDATE_MODEL_PATH, CURRENT_MODEL_PATH)
    shutil.copy2(CANDIDATE_METRICS_PATH, CURRENT_METRICS_PATH)
    shutil.copy2(
        os.path.join(MODEL_DIR, 'candidate_class_mapping.json'),
        os.path.join(MODEL_DIR, 'class_mapping.json'),
    )
    print(f'[INFO] Candidate promoted: {candidate_accuracy:.4f} >= {current_accuracy:.4f}')


def main():
    print('=' * 60)
    print('BLOOD GROUP TRAINING - FINGERTIP NORMALIZATION PIPELINE')
    print('=' * 60)

    if not os.path.isdir(TRAIN_DIR):
        print(f'[ERROR] Training folder not found: {TRAIN_DIR}')
        sys.exit(1)
    if not os.path.isdir(TEST_DIR):
        print(f'[ERROR] Test folder not found: {TEST_DIR}')
        sys.exit(1)

    train_generator, test_generator = build_generators()
    class_weights = compute_class_weights(train_generator)

    print(f'[INFO] Classes: {list(train_generator.class_indices.keys())}')
    print(f'[INFO] Train samples: {train_generator.samples}')
    print(f'[INFO] Test samples: {test_generator.samples}')
    print(f'[INFO] Model input: {IMG_SIZE}x{IMG_SIZE}x3')
    print('[INFO] Preprocessing: fingertip crop -> grayscale -> CLAHE -> Gabor ridge enhancement -> RGB replication')

    model, base_model = build_model(len(train_generator.class_indices))
    histories = progressive_train(model, base_model, train_generator, test_generator, class_weights)
    accuracy, loss = save_artifacts(model, histories, train_generator, test_generator, class_weights)
    maybe_promote_candidate(accuracy)

    print('=' * 60)
    print('TRAINING COMPLETE')
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Loss: {loss:.4f}')
    print(f'Candidate model saved to: {CANDIDATE_MODEL_PATH}')
    print('=' * 60)


if __name__ == '__main__':
    main()
