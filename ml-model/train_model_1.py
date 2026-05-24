"""
Train the blood group prediction model.

This script is offline-friendly: it does not download pretrained weights, and
it fails early if the dataset still contains Git LFS pointer files instead of
real BMP images.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_BASE = ROOT_DIR / "dataset"
TRAIN_DIR = DATASET_BASE / "train"
TEST_DIR = DATASET_BASE / "test"
MODEL_DIR = Path(__file__).resolve().parent / "saved_model"

IMG_SIZE = 128
BATCH_SIZE = 32
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def is_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:160]
        return "git-lfs" in head and "oid sha256:" in head
    except UnicodeDecodeError:
        return False
    except OSError:
        return False


def fail_if_dataset_is_not_ready() -> None:
    missing = [str(p) for p in (TRAIN_DIR, TEST_DIR) if not p.is_dir()]
    if missing:
        raise SystemExit(
            "[ERROR] Missing dataset folders: "
            + ", ".join(missing)
            + "\nRun dataset/download_lfs_images.py, then dataset/split_dataset.py."
        )

    pointers = []
    for folder in (TRAIN_DIR, TEST_DIR):
        pointers.extend([p for p in folder.rglob("*") if p.is_file() and is_lfs_pointer(p)])

    if pointers:
        sample = pointers[0]
        raise SystemExit(
            "[ERROR] Dataset still contains Git LFS pointer files, not real images.\n"
            f"Example: {sample}\n"
            "Run: py dataset/download_lfs_images.py\n"
            "Then: py dataset/split_dataset.py"
        )


def apply_clahe(img_array: np.ndarray) -> np.ndarray:
    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
        lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        enhanced = clahe.apply(img_uint8)

    return enhanced.astype(np.float32) / 255.0


def build_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=apply_clahe,
        rotation_range=18,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.08,
        zoom_range=0.12,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    )

    test_datagen = ImageDataGenerator(preprocessing_function=apply_clahe)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
        shuffle=False,
    )

    return train_generator, test_generator


def build_model(num_classes: int) -> tf.keras.Model:
    model_path = MODEL_DIR / "blood_group_model.h5"
    if model_path.exists():
        print(f"\n[INFO] Found existing model at {model_path}. Resuming training!")
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"],
        )
        return model

    print("\n[INFO] Building new model from scratch.")
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="BloodGroupCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    return model


def class_weights_for(generator):
    counts = Counter(generator.classes)
    total = sum(counts.values())
    num_classes = len(counts)
    return {int(cls): total / (num_classes * count) for cls, count in counts.items()}


def save_plots(histories, class_names, y_true, y_pred, accuracy):
    acc, val_acc, loss, val_loss = [], [], [], []
    for history in histories:
        acc.extend(history.history.get("accuracy", []))
        val_acc.extend(history.history.get("val_accuracy", []))
        loss.extend(history.history.get("loss", []))
        val_loss.extend(history.history.get("val_loss", []))

    if acc:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(acc, label="Train Accuracy")
        axes[0].plot(val_acc, label="Test Accuracy")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(loss, label="Train Loss")
        axes[1].plot(val_loss, label="Test Loss")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(MODEL_DIR / "training_plots.png", dpi=150)
        plt.close(fig)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix ({accuracy * 100:.2f}% accuracy)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def train_round(round_number, max_epochs, train_generator, test_generator, class_weights):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED + round_number)

    model = build_model(len(train_generator.class_indices))
    print(f"\n[ROUND {round_number}] Training up to {max_epochs} epochs")

    history = model.fit(
        train_generator,
        epochs=max_epochs,
        validation_data=test_generator,
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(filepath=str(MODEL_DIR / "blood_group_model.h5"), monitor="val_accuracy", save_best_only=True, verbose=1),
            callbacks.ModelCheckpoint(filepath=str(MODEL_DIR / "blood_group_model.keras"), monitor="val_accuracy", save_best_only=True, verbose=0),
        ],
        verbose=1,
    )

    test_generator.reset()
    loss, accuracy = model.evaluate(test_generator, verbose=0)
    return model, history, float(loss), float(accuracy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--target-accuracy", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    fail_if_dataset_is_not_ready()
    train_generator, test_generator = build_generators()

    class_names = list(train_generator.class_indices.keys())
    class_mapping = {int(v): k for k, v in train_generator.class_indices.items()}
    (MODEL_DIR / "class_mapping.json").write_text(json.dumps(class_mapping, indent=2), encoding="utf-8")

    weights = class_weights_for(train_generator)
    print("\n[INFO] Classes:", class_names)
    print(f"[INFO] Train samples: {train_generator.samples}")
    print(f"[INFO] Test samples : {test_generator.samples}")
    print("[INFO] Class weights:", {class_mapping[k]: round(v, 3) for k, v in weights.items()})

    best = {"accuracy": -1.0, "loss": None, "model": None, "history": None, "round": 0}
    histories = []

    for round_number in range(1, args.max_rounds + 1):
        model, history, loss, accuracy = train_round(
            round_number, args.epochs, train_generator, test_generator, weights
        )
        histories.append(history)
        print(f"[ROUND {round_number}] Test accuracy: {accuracy * 100:.2f}%")

        if accuracy > best["accuracy"]:
            best.update({"accuracy": accuracy, "loss": loss, "model": model, "history": history, "round": round_number})
            model.save(MODEL_DIR / "blood_group_model.h5")
            model.save(MODEL_DIR / "blood_group_model.keras")
            print("[SAVE] New best model saved.")

        if accuracy >= args.target_accuracy:
            print(f"[OK] Target accuracy reached: {accuracy * 100:.2f}%")
            break

    best_model = best["model"]
    if best_model is None:
        raise SystemExit("[ERROR] Training did not produce a model.")

    test_generator.reset()
    y_pred_probs = best_model.predict(test_generator, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    print("\n[REPORT]")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    metrics = {
        "accuracy": best["accuracy"],
        "loss": best["loss"],
        "best_round": best["round"],
        "epochs_requested_per_round": args.epochs,
        "rounds_run": len(histories),
        "class_report": report,
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "model_version": "offline_cnn_clahe_v1",
        "architecture": "Custom CNN + CLAHE preprocessing + class weights",
        "train_samples": int(train_generator.samples),
        "test_samples": int(test_generator.samples),
        "class_weights": {class_mapping[k]: round(v, 3) for k, v in weights.items()},
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_plots(histories, class_names, y_true, y_pred, best["accuracy"])

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best round : {best['round']}")
    print(f"Accuracy   : {best['accuracy'] * 100:.2f}%")
    print(f"Model      : {MODEL_DIR / 'blood_group_model.h5'}")
    if best["accuracy"] < args.target_accuracy:
        print(f"Target     : {args.target_accuracy * 100:.2f}% was not reached.")
    print("=" * 60)


if __name__ == "__main__":
    main()
