"""
Fingerprint validation and normalization pipeline.

Goal:
1. Accept close-up fingertip photos in color or grayscale.
2. Reject non-fingerprint content such as person photos, documents, or generic objects.
3. Convert accepted inputs into a tight grayscale crop before prediction.
"""

from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_GABOR_BANK: Optional[list[np.ndarray]] = None


def _load_image(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    rgb = np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB'))
    gray = np.array(Image.fromarray(rgb).convert('L'))
    return rgb, gray


def _border_values(mask: np.ndarray) -> np.ndarray:
    return np.concatenate([
        mask[0, :],
        mask[-1, :],
        mask[:, 0],
        mask[:, -1],
    ])


def _build_skin_mask(rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    skin_ycrcb = cv2.inRange(ycrcb, (0, 133, 77), (255, 183, 135))
    skin_hsv = cv2.inRange(hsv, (0, 15, 40), (30, 170, 255))
    mask = cv2.bitwise_and(skin_ycrcb, skin_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _build_foreground_mask(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border_mean = float(np.mean(_border_values(thresh)))
    if border_mean > 127:
        mask = cv2.bitwise_not(thresh)
    else:
        mask = thresh

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _texture_mask(gray: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = cv2.convertScaleAbs(lap)
    _, mask = cv2.threshold(texture, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _get_gabor_bank() -> list[np.ndarray]:
    global _GABOR_BANK
    if _GABOR_BANK is not None:
        return _GABOR_BANK

    kernels = []
    for theta_deg in range(0, 180, 15):
        theta = np.deg2rad(theta_deg)
        kernel = cv2.getGaborKernel(
            (21, 21),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F,
        )
        kernel -= np.mean(kernel)
        kernel /= np.sum(np.abs(kernel)) + 1e-6
        kernels.append(kernel)

    _GABOR_BANK = kernels
    return kernels


def enhance_fingerprint_with_gabor(gray: np.ndarray) -> np.ndarray:
    """Enhance ridge/valley structure with a directional Gabor filter bank."""
    source = np.clip(gray, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(source)
    denoised = cv2.bilateralFilter(clahe, 5, 35, 35)

    responses = []
    for kernel in _get_gabor_bank():
        filtered = cv2.filter2D(denoised, cv2.CV_32F, kernel)
        responses.append(np.abs(filtered))

    ridge_response = np.max(np.stack(responses, axis=0), axis=0)
    ridge_response = cv2.normalize(ridge_response, None, 0, 255, cv2.NORM_MINMAX)
    ridge_response = ridge_response.astype(np.uint8)

    enhanced = cv2.addWeighted(clahe, 0.62, ridge_response, 0.38, 0)
    return cv2.medianBlur(enhanced, 3)


def _expand_square(x: int, y: int, w: int, h: int, width: int, height: int, pad_ratio: float = 0.12) -> Tuple[int, int, int, int]:
    side = int(max(w, h) * (1.0 + pad_ratio))
    cx = x + w / 2.0
    cy = y + h / 2.0

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        shift = x2 - width
        x1 = max(0, x1 - shift)
        x2 = width
    if y2 > height:
        shift = y2 - height
        y1 = max(0, y1 - shift)
        y2 = height

    return x1, y1, x2, y2


def _prepare_crop(rgb: np.ndarray, gray: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    h, w = gray.shape
    fg_mask = _build_foreground_mask(gray)
    skin_mask = _build_skin_mask(rgb)
    texture_mask = _texture_mask(gray)

    combined = cv2.bitwise_or(fg_mask, skin_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_or(texture_mask, combined))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    contour = _largest_contour(combined)
    if contour is None or cv2.contourArea(contour) < 0.06 * (h * w):
        contour = _largest_contour(fg_mask)

    if contour is None or cv2.contourArea(contour) < 0.04 * (h * w):
        return None, {
            'crop_ratio': 0.0,
            'skin_ratio': float(np.mean(skin_mask > 0)),
            'foreground_ratio': float(np.mean(fg_mask > 0)),
        }

    x, y, cw, ch = cv2.boundingRect(contour)
    x1, y1, x2, y2 = _expand_square(x, y, cw, ch, w, h)
    crop = gray[y1:y2, x1:x2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(crop)

    return normalized, {
        'crop_ratio': float((x2 - x1) * (y2 - y1) / (w * h)),
        'skin_ratio': float(np.mean(skin_mask > 0)),
        'foreground_ratio': float(np.mean(fg_mask > 0)),
    }


def normalize_fingerprint_image(
    image_source,
    target_size: Optional[int] = None,
    return_metadata: bool = False,
    enhance_ridges: bool = False,
):
    if isinstance(image_source, bytes):
        rgb, gray = _load_image(image_source)
    elif isinstance(image_source, np.ndarray):
        array = image_source.astype(np.uint8)
        if array.ndim == 2:
            gray = array
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            rgb = array
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise TypeError('Unsupported image source type.')

    processed, crop_meta = _prepare_crop(rgb, gray)
    if processed is None:
        processed = gray

    if target_size is not None:
        interpolation = cv2.INTER_AREA if processed.shape[0] > target_size else cv2.INTER_CUBIC
        processed = cv2.resize(processed, (target_size, target_size), interpolation=interpolation)

    if enhance_ridges:
        processed = enhance_fingerprint_with_gabor(processed)

    if return_metadata:
        return processed, crop_meta
    return processed

def _check_person_photo(rgb: np.ndarray, gray: np.ndarray) -> Tuple[bool, str]:
    cascade_names = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_profileface.xml',
        'haarcascade_upperbody.xml',
    ]
    for cascade_name in cascade_names:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)
        detections = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(24, 24))
        if len(detections) > 0:
            return True, 'Detected a face or upper-body pattern.'

    skin_ratio = float(np.mean(_build_skin_mask(rgb) > 0))
    edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
    if skin_ratio > 0.55 and edge_density < 0.03:
        return True, 'Large smooth skin region without fingerprint texture.'

    return False, ''


def _check_document_or_graphics(gray: np.ndarray) -> Tuple[bool, str]:
    edges = cv2.Canny(gray, 50, 150)
    min_len = max(32, min(gray.shape) // 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=min_len, maxLineGap=8)
    if lines is not None and len(lines) > 14:
        return True, 'Too many long straight lines for a fingertip.'

    if float(np.std(gray)) < 10:
        return True, 'Image is nearly blank or flat.'

    return False, ''


def _check_edge_distribution(gray: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    edges = cv2.Canny(gray, 35, 120)
    h, w = edges.shape
    grid = 4
    cell_h, cell_w = h // grid, w // grid
    densities = []

    for gy in range(grid):
        for gx in range(grid):
            cell = edges[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
            densities.append(float(np.mean(cell > 0)))

    overall_density = float(np.mean(edges > 0))
    active_cells = sum(1 for d in densities if d > 0.02)
    density_std = float(np.std(densities))
    density_mean = float(np.mean(densities))
    uniformity = 1.0 - (density_std / (density_mean + 1e-6))

    passed = active_cells >= 8 and 0.02 < overall_density < 0.45 and uniformity > 0.08
    return passed, {
        'active_cells': float(active_cells),
        'edge_density': round(overall_density, 4),
        'uniformity': round(uniformity, 3),
    }


def _verify_ridge_pattern(gray: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    scores: Dict[str, float] = {}

    gabor_responses = []
    for theta_deg in range(0, 180, 15):
        theta = np.deg2rad(theta_deg)
        kernel = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta, lambd=10.0, gamma=0.5, psi=0)
        filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
        gabor_responses.append(float(np.mean(np.abs(filtered))))

    gabor_std = float(np.std(gabor_responses))
    gabor_mean = float(np.mean(gabor_responses))
    directionality = gabor_std / (gabor_mean + 1e-10)
    scores['directionality'] = directionality

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gxx = cv2.GaussianBlur(gx * gx, (15, 15), 3)
    gyy = cv2.GaussianBlur(gy * gy, (15, 15), 3)
    gxy = cv2.GaussianBlur(gx * gy, (15, 15), 3)
    coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / (gxx + gyy + 1e-10)
    ridge_density = float(np.mean(coherence))
    scores['ridge_density'] = ridge_density

    f = np.fft.fft2(gray.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    magnitude[cy - 3:cy + 3, cx - 3:cx + 3] = 0

    max_r = min(cx, cy)
    yy, xx = np.ogrid[:h, :w]
    radial = np.zeros(max_r)
    radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
    for radius in range(max_r):
        mask = radii == radius
        if np.any(mask):
            radial[radius] = np.mean(magnitude[mask])

    low = max_r // 8
    mid_start = max_r // 6
    mid_end = max_r // 2
    high_start = int(max_r * 0.6)

    mid_energy = np.sum(radial[mid_start:mid_end])
    total_energy = np.sum(radial) + 1e-10
    freq_ratio = float(mid_energy / total_energy)
    scores['freq_ratio'] = freq_ratio

    peak_prominence = 0.0
    if mid_end > mid_start:
        peak_val = float(np.max(radial[mid_start:mid_end]))
        mean_val = float(np.mean(radial[1:])) + 1e-10
        peak_prominence = peak_val / mean_val
    scores['peak_prominence'] = peak_prominence

    block_size = 16
    orientations = np.full((h // block_size, w // block_size), np.nan)
    for by in range(h // block_size):
        for bx in range(w // block_size):
            block = gray[by * block_size:(by + 1) * block_size, bx * block_size:(bx + 1) * block_size]
            block_gx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
            block_gy = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
            angle = 0.5 * np.arctan2(2 * np.sum(block_gx * block_gy), np.sum(block_gx * block_gx - block_gy * block_gy))
            block_coh = np.sqrt(np.sum(block_gx * block_gx - block_gy * block_gy) ** 2 + 4 * np.sum(block_gx * block_gy) ** 2)
            block_coh /= (np.sum(block_gx * block_gx) + np.sum(block_gy * block_gy) + 1e-10)
            if block_coh > 0.20:
                orientations[by, bx] = angle

    valid = ~np.isnan(orientations)
    orientation_diffs = []
    rows, cols = orientations.shape
    for row in range(rows - 1):
        for col in range(cols - 1):
            if valid[row, col] and valid[row, col + 1]:
                diff = abs(orientations[row, col] - orientations[row, col + 1])
                orientation_diffs.append(min(diff, np.pi - diff))
            if valid[row, col] and valid[row + 1, col]:
                diff = abs(orientations[row, col] - orientations[row + 1, col])
                orientation_diffs.append(min(diff, np.pi - diff))

    orient_smoothness = 0.0
    if orientation_diffs:
        orient_smoothness = 1.0 - float(np.mean(orientation_diffs)) / (np.pi / 2)
    orient_coverage = float(np.mean(valid))
    scores['orient_smoothness'] = orient_smoothness
    scores['orient_coverage'] = orient_coverage

    score = 0.0
    if directionality > 0.12:
        score += 0.16
    elif directionality > 0.07:
        score += 0.08
    if ridge_density > 0.28:
        score += 0.18
    elif ridge_density > 0.20:
        score += 0.09
    if 0.20 < freq_ratio < 0.75:
        score += 0.16
    elif 0.15 < freq_ratio < 0.82:
        score += 0.08
    if peak_prominence > 1.35:
        score += 0.14
    elif peak_prominence > 1.15:
        score += 0.07
    if orient_smoothness > 0.32:
        score += 0.18
    elif orient_smoothness > 0.22:
        score += 0.09
    if orient_coverage > 0.22:
        score += 0.18
    elif orient_coverage > 0.15:
        score += 0.09

    return score >= 0.52, round(score * 100, 1), scores


def _to_gray_array(image_source) -> np.ndarray:
    if isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            return image_source.astype(np.uint8)
        return cv2.cvtColor(image_source.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return _load_image(image_source)[1]


def detect_ai_generated(image_source) -> Tuple[bool, float, str]:
    gray = cv2.resize(_to_gray_array(image_source), (256, 256), interpolation=cv2.INTER_AREA)
    ai_score = 0.0

    blurred = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 1.5)
    noise = gray.astype(np.float64) - blurred
    noise_std = float(np.std(noise))
    if noise_std < 2.0:
        ai_score += 0.25

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
    if entropy > 7.5:
        ai_score += 0.20

    block_size = 32
    spacings = []
    h, w = gray.shape
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y + block_size, x:x + block_size]
            f = np.fft.fft2(block.astype(np.float64))
            mag = np.abs(np.fft.fftshift(f))
            mag[block_size // 2, block_size // 2] = 0
            idx = np.unravel_index(np.argmax(mag), mag.shape)
            dist = np.sqrt((idx[0] - block_size // 2) ** 2 + (idx[1] - block_size // 2) ** 2)
            if dist > 1:
                spacings.append(dist)

    if len(spacings) > 3:
        cv_spacing = float(np.std(spacings) / (np.mean(spacings) + 1e-10))
        if cv_spacing < 0.11:
            ai_score += 0.35

    kurtosis = float(np.mean((noise - np.mean(noise)) ** 4) / (np.std(noise) ** 4 + 1e-10) - 3)
    if abs(kurtosis) > 10:
        ai_score += 0.15

    is_ai = ai_score >= 0.60
    reason = 'AI-generated fingerprint detected.' if is_ai else 'Passes authenticity check.'
    return is_ai, round(ai_score * 100, 1), reason


def assess_quality(image_source) -> Tuple[bool, float, list[str]]:
    gray = _to_gray_array(image_source)
    issues = []
    h, w = gray.shape

    if h < 64 or w < 64:
        issues.append('Image too small (minimum 64x64).')
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 45:
        issues.append('Image is too blurry.')
    if float(np.std(gray)) < 14:
        issues.append('Very low contrast fingerprint.')

    mean_brightness = float(np.mean(gray))
    if mean_brightness < 30:
        issues.append('Image is too dark.')
    elif mean_brightness > 240:
        issues.append('Image is overexposed.')

    score = max(0.0, 1.0 - 0.22 * len(issues))
    return len(issues) == 0, round(score * 100, 1), issues


def classify_image_type(rgb: np.ndarray, gray: np.ndarray, ridge_confidence: float = 0.0) -> Tuple[str, str, str]:
    is_person, reason = _check_person_photo(rgb, gray)
    if is_person:
        return (
            'person_photo',
            'You uploaded a person or skin photo instead of a close-up fingertip fingerprint image.',
            '🚫',
        )

    is_document, _ = _check_document_or_graphics(gray)
    if is_document:
        return (
            'document_or_graphic',
            'You uploaded a poster, design, screenshot, document, blank image, or other graphic instead of a fingertip fingerprint image.',
            '🚫',
        )

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    quantized = (rgb // 32).reshape(-1, 3)
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant_ratio = float(np.max(counts) / max(1, quantized.shape[0]))

    if mean_sat > 18 and dominant_ratio > 0.18 and ridge_confidence < 40:
        return (
            'poster_or_design',
            'You uploaded a poster, design, logo, or other graphic image instead of a close-up fingertip fingerprint.',
            '🚫',
        )

    if mean_sat > 25 and ridge_confidence < 35:
        return (
            'generic_photo',
            'You uploaded a normal photo such as a car, object, scene, or other non-fingerprint image. Please upload only a close-up picture of a fingertip fingerprint.',
            '🚫',
        )

    return (
        'unknown_image',
        'The uploaded image does not look like a close-up fingertip fingerprint.',
        '🚫',
    )


def validate_image(image_bytes: bytes, include_processed: bool = False) -> Dict[str, object]:
    result: Dict[str, object] = {
        'is_valid': False,
        'is_fingerprint': False,
        'is_ai_generated': False,
        'quality_ok': False,
        'fingerprint_confidence': 0.0,
        'ai_confidence': 0.0,
        'quality_score': 0.0,
        'rejection_reason': None,
        'detected_image_type': None,
        'rejection_icon': None,
        'warnings': [],
        'image_diagnostics': {},
    }

    try:
        rgb, gray = _load_image(image_bytes)
    except Exception as error:
        result['rejection_reason'] = f'Cannot read image: {error}'
        result['detected_image_type'] = 'unreadable'
        result['rejection_icon'] = '❌'
        return result

    color_saturation = float(np.mean(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1]))

    crop, crop_meta = _prepare_crop(rgb, gray)
    result['image_diagnostics'] = {
        'color_saturation': round(color_saturation, 2),
        'crop_ratio': round(float(crop_meta.get('crop_ratio', 0.0)), 4),
        'skin_ratio': round(float(crop_meta.get('skin_ratio', 0.0)), 4),
        'foreground_ratio': round(float(crop_meta.get('foreground_ratio', 0.0)), 4),
    }
    if crop is None:
        scanner_edges_ok, scanner_edge_meta = _check_edge_distribution(gray)
        scanner_is_fp, scanner_fp_confidence, scanner_ridge_meta = _verify_ridge_pattern(gray)

        if color_saturation <= 5 and scanner_edges_ok and scanner_is_fp:
            crop = gray
            result['image_diagnostics'].update({
                'scanner_full_frame_fallback': True,
                **{
                    f'ridge_{key}': round(float(value), 4)
                    for key, value in scanner_ridge_meta.items()
                },
            })
            result['warnings'].append(
                'Scanner fingerprint accepted as a full-frame capture because foreground crop was sparse.'
            )
        else:
            image_type, message, icon = classify_image_type(rgb, gray)
            result['rejection_reason'] = message
            result['detected_image_type'] = image_type
            result['rejection_icon'] = icon
            return result

    edges_ok, edge_meta = _check_edge_distribution(crop)
    is_fp, fp_confidence, ridge_meta = _verify_ridge_pattern(crop)
    result['is_fingerprint'] = is_fp
    result['fingerprint_confidence'] = fp_confidence
    result['image_diagnostics'].update({
        f'ridge_{key}': round(float(value), 4)
        for key, value in ridge_meta.items()
    })

    if not edges_ok or not is_fp:
        image_type, message, icon = classify_image_type(rgb, gray, fp_confidence)
        result['rejection_reason'] = message
        result['detected_image_type'] = image_type
        result['rejection_icon'] = icon
        return result

    is_ai, ai_confidence, _ = detect_ai_generated(crop)
    result['is_ai_generated'] = is_ai
    result['ai_confidence'] = ai_confidence
    if is_ai:
        result['rejection_reason'] = 'This fingerprint looks synthetic or AI-generated. Please upload a real fingertip image.'
        result['detected_image_type'] = 'ai_generated'
        result['rejection_icon'] = '🤖'
        return result

    quality_ok, quality_score, issues = assess_quality(crop)
    result['quality_ok'] = quality_ok
    result['quality_score'] = quality_score

    warnings = list(result.get('warnings', [])) + list(issues)
    if color_saturation > 10:
        warnings.append('Color fingertip image accepted and converted to grayscale automatically.')
    if crop_meta['crop_ratio'] < 0.92 and not result['image_diagnostics'].get('scanner_full_frame_fallback'):
        warnings.append('Fingerprint area was cropped automatically for close-up analysis.')
    if edge_meta['edge_density'] < 0.04:
        warnings.append('Fingerprint ridges are faint; better lighting or a closer capture may improve accuracy.')

    result['warnings'] = warnings
    result['is_valid'] = True

    if include_processed:
        result['processed_image'] = crop

    return result
