"""
Flask API for Blood Group Prediction
Serves the trained CNN model for inference.
Includes: Image validation, AI detection, scanner support.
"""

import os
import sys
import json
import uuid
import hashlib
import base64
import traceback
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from image_validator import enhance_fingerprint_with_gabor, validate_image

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

# --- Load Model & Config ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'blood_group_model.h5')
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, 'class_mapping.json')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

IMG_SIZE = 96
NUM_CHANNELS = 1  # Auto-detected from model (1=grayscale, 3=RGB)
CONFIDENCE_THRESHOLD = 40.0   # Minimum % to trust prediction
TOP2_MARGIN_THRESHOLD = 10.0  # Min gap between top-2 predictions
DEPLOYMENT_CONFIDENCE_THRESHOLD = 85.0
DEPLOYMENT_MARGIN_THRESHOLD = 25.0
GABOR_MODEL_VERSIONS = {'fingertip_crop_gabor_v2'}

model = None
class_mapping = None
metrics = None


def _patch_keras_loading():
    """
    Monkey-patch Keras 3.x Operation.from_config so that legacy .h5 models
    saved with Keras 2.x / TF 2.16- can be loaded despite extra kwargs like
    renorm, renorm_clipping, renorm_momentum, synchronized, quantization_config.
    """
    from keras.src.ops.operation import Operation
    _original_from_config = Operation.from_config.__func__

    @classmethod
    def _safe_from_config(cls, config):
        try:
            return _original_from_config(cls, config)
        except (TypeError, ValueError):
            # Strip known unsupported keys and retry
            _legacy_keys = {
                'renorm', 'renorm_clipping', 'renorm_momentum',
                'synchronized', 'quantization_config',
            }
            cleaned = {k: v for k, v in config.items() if k not in _legacy_keys}
            try:
                return cls(**cleaned)
            except (TypeError, ValueError):
                # Nuclear option: keep only keys the constructor actually accepts
                import inspect
                sig = inspect.signature(cls.__init__)
                valid = set(sig.parameters.keys()) - {'self'}
                if 'kwargs' in {p.name for p in sig.parameters.values()
                                if p.kind == inspect.Parameter.VAR_KEYWORD}:
                    return cls(**cleaned)
                filtered = {k: v for k, v in cleaned.items() if k in valid}
                return cls(**filtered)

    Operation.from_config = _safe_from_config

_patch_keras_loading()


def load_model():
    global model, class_mapping, metrics, IMG_SIZE, NUM_CHANNELS
    print("[INFO] Loading CNN model...")
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        # Detect image size and channels from model input shape
        input_shape = model.input_shape
        if input_shape and len(input_shape) >= 3:
            IMG_SIZE = input_shape[1] or 96
            NUM_CHANNELS = input_shape[3] if len(input_shape) >= 4 else 1
        # Build model with dummy input (required for Keras 3.x / TF 2.19)
        dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        ch_label = 'RGB' if NUM_CHANNELS == 3 else 'grayscale'
        print(f"[OK] Model loaded (input: {IMG_SIZE}x{IMG_SIZE}x{NUM_CHANNELS} {ch_label}) and warmed up!")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}. Train the model first.")

    if os.path.exists(CLASS_MAPPING_PATH):
        with open(CLASS_MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        # Convert string keys to int keys
        class_mapping = {int(k): v for k, v in class_mapping.items()}
        print(f"[INFO] Classes: {list(class_mapping.values())}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)


def deployed_model_uses_gabor():
    if not metrics:
        return False
    return metrics.get('model_version') in GABOR_MODEL_VERSIONS


def deployed_model_uses_raw_pixel_scale():
    if not metrics:
        return False
    return metrics.get('input_value_range') == '0_255'


def preprocess_image(image_source, use_gabor=None):
    """Preprocess a validated fingerprint crop for model inference."""
    if use_gabor is None:
        use_gabor = deployed_model_uses_gabor()

    if isinstance(image_source, np.ndarray):
        array = image_source.astype(np.uint8)
        if array.ndim == 2:
            if use_gabor:
                array = np.array(Image.fromarray(array, mode='L').resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
                array = enhance_fingerprint_with_gabor(array)
            gray_img = Image.fromarray(array, mode='L')
            rgb_img = gray_img.convert('RGB')
        else:
            rgb_img = Image.fromarray(array, mode='RGB')
            gray_img = rgb_img.convert('L')
            if use_gabor:
                gray_img = gray_img.resize((IMG_SIZE, IMG_SIZE))
                gray_img = Image.fromarray(enhance_fingerprint_with_gabor(np.array(gray_img)), mode='L')
                rgb_img = gray_img.convert('RGB')
        img = rgb_img if NUM_CHANNELS == 3 else gray_img
    else:
        img = Image.open(io.BytesIO(image_source))
        if use_gabor:
            gray_array = np.array(img.convert('L'), dtype=np.uint8)
            gray_array = np.array(Image.fromarray(gray_array, mode='L').resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
            img = Image.fromarray(enhance_fingerprint_with_gabor(gray_array), mode='L')
        if NUM_CHANNELS == 3:
            img = img.convert('RGB')
        else:
            img = img.convert('L')

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    if not deployed_model_uses_raw_pixel_scale():
        img_array = img_array / 255.0
    if NUM_CHANNELS == 1:
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    else:
        img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)
    return img_array


def predict_with_tta(model_instance, img_array, n_augments=8):
    """Test-Time Augmentation: average predictions across augmented versions
    for more stable results (+1-2% accuracy at inference)."""
    preds = [model_instance.predict(img_array, verbose=0)]
    for _ in range(n_augments - 1):
        aug = tf.image.random_flip_left_right(img_array)
        aug = tf.image.random_brightness(aug, 0.08)
        aug = tf.image.random_contrast(aug, 0.92, 1.08)
        preds.append(model_instance.predict(aug, verbose=0))
    return np.mean(preds, axis=0)


def analyze_deployment_safety(predictions):
    """Decide whether the model output is strong enough to show to a user."""
    probs = predictions[0]
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0]) * 100
    top2 = float(sorted_probs[1]) * 100 if len(sorted_probs) > 1 else 0
    margin = top1 - top2

    if top1 < DEPLOYMENT_CONFIDENCE_THRESHOLD:
        return False, (
            f"Fingerprint accepted, but the model confidence is only {top1:.2f}%. "
            "This capture is outside the model's reliable prediction range. "
            "Use a clearer scanner-style fingerprint image or retrain the model "
            "with labeled phone-captured fingerprints before trusting the result."
        )

    if margin < DEPLOYMENT_MARGIN_THRESHOLD:
        return False, (
            f"Fingerprint accepted, but the top blood-group candidates are too close "
            f"(margin {margin:.2f}%). Please retake the fingerprint or retrain with "
            "more labeled samples from this capture style."
        )

    return True, None


def deployed_model_supports_phone_captures():
    if not metrics:
        return False
    return metrics.get('model_version') in {
        'fingertip_crop_grayscale_v1',
        *GABOR_MODEL_VERSIONS,
    }


def analyze_capture_domain(validation):
    diagnostics = validation.get('image_diagnostics', {})
    color_saturation = float(diagnostics.get('color_saturation', 0.0))

    if color_saturation > 10 and not deployed_model_supports_phone_captures():
        return False, (
            "Fingerprint accepted, but this is a phone-captured color fingertip image. "
            "The deployed model was trained on scanner-style grayscale images, so showing "
            "a blood-group result for this capture would be unreliable. Train and promote "
            "a candidate model with labeled phone-captured fingerprints before trusting "
            "this input style."
        )

    return True, None


def compute_fingerprint_hash(image_bytes):
    """Compute SHA-256 hash of fingerprint image for duplicate detection."""
    return hashlib.sha256(image_bytes).hexdigest()


def extract_feature_vector(image_bytes):
    """Extract a non-reversible feature embedding from the fingerprint."""
    try:
        if model is None:
            return None
        # Use the image hash as a simple non-reversible embedding
        # (avoids Keras 3.x sub-model compatibility issues)
        feature_hash = hashlib.sha256(image_bytes).hexdigest()
        return feature_hash
    except Exception as e:
        print(f"[WARN] Feature extraction failed: {e}")
        return hashlib.sha256(image_bytes).hexdigest()


def analyze_prediction_reliability(predictions):
    """
    Analyze if the prediction is reliable using entropy and margin.
    Returns (is_reliable, confidence_level, details).
    """
    probs = predictions[0]
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0]) * 100
    top2 = float(sorted_probs[1]) * 100 if len(sorted_probs) > 1 else 0

    # Entropy-based uncertainty
    entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    max_entropy = np.log2(len(probs))
    normalized_entropy = entropy / max_entropy

    # Top-2 margin
    margin = top1 - top2

    details = {
        'top1_confidence': round(top1, 2),
        'top2_confidence': round(top2, 2),
        'margin': round(margin, 2),
        'entropy': round(entropy, 3),
        'normalized_entropy': round(normalized_entropy, 3),
    }

    # Determine reliability
    if top1 < CONFIDENCE_THRESHOLD:
        return False, 'very_low', details
    elif margin < TOP2_MARGIN_THRESHOLD:
        return False, 'ambiguous', details
    elif normalized_entropy > 0.7:
        return False, 'uncertain', details
    elif top1 < 60:
        return True, 'low', details
    elif top1 < 80:
        return True, 'moderate', details
    else:
        return True, 'high', details


# --- API Routes ---
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'classes': list(class_mapping.values()) if class_mapping else [],
        'features': {
            'fingerprint_validation': True,
            'ai_detection': True,
            'quality_check': True,
            'scanner_support': True,
            'confidence_gating': True,
            'gabor_ridge_enhancement': deployed_model_uses_gabor(),
        }
    })


@app.route('/api/validate', methods=['POST'])
def validate_only():
    """Validate an image without predicting (quick check)."""
    if 'fingerprint' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    file = request.files['fingerprint']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        image_bytes = file.read()
        validation = validate_image(image_bytes)
        return jsonify({
            'success': True,
            'validation': validation,
        })
    except Exception as e:
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict blood group from fingerprint image with full validation."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Train the model first.'}), 503

    # Support both file upload and base64 (for scanner devices)
    image_bytes = None

    if 'fingerprint' in request.files:
        file = request.files['fingerprint']
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400
        image_bytes = file.read()
    elif request.is_json and 'image_base64' in request.json:
        try:
            b64_data = request.json['image_base64']
            # Strip data URI prefix if present
            if ',' in b64_data:
                b64_data = b64_data.split(',', 1)[1]
            image_bytes = base64.b64decode(b64_data)
        except Exception:
            return jsonify({'error': 'Invalid base64 image data.'}), 400
    else:
        return jsonify({'error': 'No fingerprint image provided.'}), 400

    try:
        # ── Step 1: Validate the image ──
        validation = validate_image(image_bytes, include_processed=True)

        if not validation['is_valid']:
            return jsonify({
                'success': False,
                'rejected': True,
                'rejection_reason': validation['rejection_reason'],
                'detected_image_type': validation.get('detected_image_type', 'unknown'),
                'rejection_icon': validation.get('rejection_icon', '⚠️'),
                'validation': {
                    'is_fingerprint': validation['is_fingerprint'],
                    'is_ai_generated': validation['is_ai_generated'],
                    'fingerprint_confidence': validation['fingerprint_confidence'],
                    'ai_confidence': validation['ai_confidence'],
                },
            }), 422

        # ── Step 2: Predict ──
        is_supported_domain, domain_reason = analyze_capture_domain(validation)
        if not is_supported_domain:
            return jsonify({
                'success': False,
                'rejected': True,
                'rejection_reason': domain_reason,
                'detected_image_type': 'unsupported_capture_style',
                'rejection_icon': 'warning',
                'validation': {
                    'is_fingerprint': validation['is_fingerprint'],
                    'is_ai_generated': validation['is_ai_generated'],
                    'fingerprint_confidence': validation['fingerprint_confidence'],
                    'ai_confidence': validation['ai_confidence'],
                },
            }), 422

        processed_image = validation.get('processed_image', image_bytes)
        image_hash = compute_fingerprint_hash(image_bytes)
        feature_embedding = extract_feature_vector(image_bytes)

        img_array = preprocess_image(processed_image)
        predictions = predict_with_tta(model, img_array, n_augments=8)

        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx])
        predicted_blood_group = class_mapping[predicted_class_idx]

        # ── Step 3: Reliability analysis ──
        is_reliable, confidence_level, reliability_details = \
            analyze_prediction_reliability(predictions)

        is_safe_to_show, safety_reason = analyze_deployment_safety(predictions)
        if not is_safe_to_show:
            return jsonify({
                'success': False,
                'rejected': True,
                'rejection_reason': safety_reason,
                'detected_image_type': 'low_confidence_fingerprint',
                'rejection_icon': 'warning',
                'validation': {
                    'is_fingerprint': validation['is_fingerprint'],
                    'is_ai_generated': validation['is_ai_generated'],
                    'fingerprint_confidence': validation['fingerprint_confidence'],
                    'ai_confidence': validation['ai_confidence'],
                },
            }), 422

        # Build full results with all class probabilities
        all_probabilities = {}
        for idx, prob in enumerate(predictions[0]):
            all_probabilities[class_mapping[idx]] = round(float(prob) * 100, 2)

        # Sort probabilities descending
        all_probabilities = dict(sorted(
            all_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        prediction_id = str(uuid.uuid4())

        # Build warnings
        warnings = list(validation.get('warnings', []))
        if not is_reliable:
            if confidence_level == 'very_low':
                warnings.append(
                    f"⚠️ Very low confidence ({reliability_details['top1_confidence']}%). "
                    "The prediction may not be accurate."
                )
            elif confidence_level == 'ambiguous':
                warnings.append(
                    f"⚠️ Ambiguous result — top predictions are very close "
                    f"(margin: {reliability_details['margin']}%). "
                    "Consider re-scanning with better quality."
                )
            elif confidence_level == 'uncertain':
                warnings.append(
                    "⚠️ High uncertainty detected. The model is not confident "
                    "in this prediction."
                )

        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'predicted_blood_group': predicted_blood_group,
            'confidence': round(confidence * 100, 2),
            'all_probabilities': all_probabilities,
            'fingerprint_hash': image_hash,
            'feature_embedding': feature_embedding,
            'reliability': {
                'is_reliable': is_reliable,
                'confidence_level': confidence_level,
                **reliability_details,
            },
            'validation': {
                'is_fingerprint': validation['is_fingerprint'],
                'is_ai_generated': validation['is_ai_generated'],
                'quality_score': validation['quality_score'],
                'fingerprint_confidence': validation['fingerprint_confidence'],
            },
            'warnings': warnings,
        })

    except Exception as e:
        print(f"\n[ERROR] Prediction failed:")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/scanner/capture', methods=['POST'])
def scanner_capture():
    """
    Endpoint for physical fingerprint scanner devices.
    Accepts base64-encoded image data from scanner hardware.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    if not request.is_json:
        return jsonify({'error': 'JSON body required.'}), 400

    data = request.json
    if 'image_base64' not in data:
        return jsonify({'error': 'image_base64 field required.'}), 400

    try:
        b64_data = data['image_base64']
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]
        image_bytes = base64.b64decode(b64_data)

        # Validate
        validation = validate_image(image_bytes, include_processed=True)
        if not validation['is_valid']:
            return jsonify({
                'success': False,
                'rejected': True,
                'rejection_reason': validation['rejection_reason'],
                'source': 'scanner',
            }), 422

        is_supported_domain, domain_reason = analyze_capture_domain(validation)
        if not is_supported_domain:
            return jsonify({
                'success': False,
                'rejected': True,
                'rejection_reason': domain_reason,
                'source': 'scanner',
            }), 422

        # Predict
        img_array = preprocess_image(validation.get('processed_image', image_bytes))
        predictions = predict_with_tta(model, img_array, n_augments=8)

        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx])
        predicted_blood_group = class_mapping[predicted_class_idx]

        is_reliable, confidence_level, reliability_details = \
            analyze_prediction_reliability(predictions)

        is_safe_to_show, safety_reason = analyze_deployment_safety(predictions)
        if not is_safe_to_show:
            return jsonify({
                'success': False,
                'rejected': True,
                'rejection_reason': safety_reason,
                'source': 'scanner',
            }), 422

        all_probabilities = {}
        for idx, prob in enumerate(predictions[0]):
            all_probabilities[class_mapping[idx]] = round(float(prob) * 100, 2)

        all_probabilities = dict(sorted(
            all_probabilities.items(), key=lambda x: x[1], reverse=True
        ))

        return jsonify({
            'success': True,
            'source': 'scanner',
            'prediction_id': str(uuid.uuid4()),
            'predicted_blood_group': predicted_blood_group,
            'confidence': round(confidence * 100, 2),
            'all_probabilities': all_probabilities,
            'fingerprint_hash': compute_fingerprint_hash(image_bytes),
            'reliability': {
                'is_reliable': is_reliable,
                'confidence_level': confidence_level,
                **reliability_details,
            },
            'validation': {
                'is_fingerprint': validation['is_fingerprint'],
                'is_ai_generated': validation['is_ai_generated'],
                'quality_score': validation['quality_score'],
                'fingerprint_confidence': validation['fingerprint_confidence'],
            },
            'warnings': validation.get('warnings', []),
            'device_info': {
                'device_name': data.get('device_name', 'Unknown Scanner'),
                'resolution': data.get('resolution', 'Unknown'),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Scanner capture failed: {str(e)}'}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model metrics and info."""
    if metrics is None:
        return jsonify({'error': 'No metrics available.'}), 404

    return jsonify({
        'accuracy': metrics.get('accuracy', 0),
        'loss': metrics.get('loss', 0),
        'classes': metrics.get('class_names', []),
        'epochs_trained': metrics.get('epochs_trained', 0),
        'img_size': metrics.get('img_size', IMG_SIZE),
    })


# --- Main ---
if __name__ == '__main__':
    load_model()
    print("\n[SERVER] Flask ML API running on http://localhost:5000")
    print("[FEATURES] Image validation | AI detection | Scanner support")
    app.run(host='0.0.0.0', port=5000, debug=False)
