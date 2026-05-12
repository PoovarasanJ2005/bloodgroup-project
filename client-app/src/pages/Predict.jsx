import { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { predictionService } from '../services/api';
import toast from 'react-hot-toast';
import {
  HiOutlineFingerPrint, HiOutlineUpload, HiOutlineX,
  HiOutlineCheckCircle, HiOutlineExclamation,
  HiOutlineShieldCheck, HiOutlineDesktopComputer
} from 'react-icons/hi';
import './Predict.css';

// Scanner quality thresholds
const MFS100_CAPTURE_QUALITY    = 70;
const MFS100_MIN_ACCEPTED_QUALITY = 60;
const MFS100_MAX_ACCEPTED_NFIQ    = 3;

const stripDataUrlPrefix = (value) => {
  if (!value || typeof value !== 'string') return '';
  return value.includes(',') ? value.split(',', 2)[1] : value;
};

const getBase64Mime = (base64) => {
  if (base64.startsWith('Qk')) return 'image/bmp';
  if (base64.startsWith('iVBOR')) return 'image/png';
  if (base64.startsWith('/9j/')) return 'image/jpeg';
  return 'image/bmp';
};

const base64ToFile = (base64, filename, mimeType) => {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new File([bytes], filename, { type: mimeType });
};

const getScannerField = (payload, fieldNames) => {
  for (const name of fieldNames) {
    if (payload?.[name] !== undefined && payload?.[name] !== null) {
      return payload[name];
    }
    if (payload?.data?.[name] !== undefined && payload?.data?.[name] !== null) {
      return payload.data[name];
    }
  }
  return undefined;
};

const parseScannerNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const captureFromMantraMfs100 = async (options = {}) => {
  // ── Route through the Node server to avoid browser CORS block ────────────
  // The Mantra RD Service (port 8004/8003) does NOT send CORS headers,
  // so the browser blocks any direct fetch().  The /mfs100-capture proxy
  // on our Node server makes the call server-side where CORS does not apply.
  let payload;
  try {
    const res = await predictionService.mfs100Capture(MFS100_CAPTURE_QUALITY, {
      allowSavedFileFallback: options.allowSavedFileFallback !== false,
    });
    if (!res.data.success) {
      const err = new Error(res.data.error || 'MFS100 service not reachable.');
      err.isServiceUnavailable = true;
      throw err;
    }
    payload = res.data.data;
  } catch (axiosErr) {
    // 503 = server confirmed that MFS100 RD Service is not running
    if (axiosErr.response?.status === 503) {
      const svcErr = new Error(
        axiosErr.response.data?.error ||
        'MFS100 local service not running. Start MantraRDService.exe first.'
      );
      svcErr.isServiceUnavailable = true;
      throw svcErr;
    }
    if (axiosErr.isServiceUnavailable) throw axiosErr;
    throw axiosErr;
  }

  // ── Parse scanner payload (same field-name aliases as before) ────────────
  const errorCode        = getScannerField(payload, ['ErrorCode', 'errorCode']);
  const errorDescription = getScannerField(payload, ['ErrorDescription', 'errorDescription', 'Status', 'status']);
  const rawBitmap        = getScannerField(payload, ['BitmapData', 'bitmapData', 'ImageData', 'imageData']);
  const bitmapData       = stripDataUrlPrefix(rawBitmap);

  if (!bitmapData) {
    throw new Error(errorDescription || 'MFS100 did not return image data.');
  }

  if (errorCode !== undefined && String(errorCode) !== '0') {
    throw new Error(errorDescription || `MFS100 capture failed (code ${errorCode}).`);
  }

  const quality = parseScannerNumber(getScannerField(payload, ['Quality', 'quality']));
  const nfiq    = parseScannerNumber(getScannerField(payload, ['NFIQ', 'Nfiq', 'nfiq']));

  if (quality !== null && quality < MFS100_MIN_ACCEPTED_QUALITY) {
    const qualityError = new Error(
      `Fingerprint quality is ${quality}. Rescan with the finger flat on the MFS100 sensor.`
    );
    qualityError.isCaptureQualityError = true;
    throw qualityError;
  }

  if (nfiq !== null && nfiq > MFS100_MAX_ACCEPTED_NFIQ) {
    const nfiqError = new Error(
      `Fingerprint NFIQ is ${nfiq}. Rescan before prediction to avoid a wrong result.`
    );
    nfiqError.isCaptureQualityError = true;
    throw nfiqError;
  }

  const mimeType = getBase64Mime(bitmapData);
  return {
    base64:     bitmapData,
    mimeType,
    previewUrl: `data:${mimeType};base64,${bitmapData}`,
    deviceName: 'Mantra MFS100',
    resolution: getScannerField(payload, ['Resolution', 'resolution']) || '500dpi',
  };
};


const Predict = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [duplicateWarning, setDuplicateWarning] = useState(null);
  const [rejection, setRejection] = useState(null);
  const [reliability, setReliability] = useState(null);
  const [warnings, setWarnings] = useState([]);
  const [scannerMode, setScannerMode] = useState(true);
  const [scannerStatus, setScannerStatus] = useState('idle');
  const canvasRef = useRef(null);
  const autoScannerStartedRef = useRef(false);
  const scannerCaptureInFlightRef = useRef(false);

  const onDrop = useCallback((acceptedFiles) => {
    const f = acceptedFiles[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setResult(null);
      setDuplicateWarning(null);
      setRejection(null);
      setReliability(null);
      setWarnings([]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.bmp', '.tiff'] },
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024,
  });

  const handlePredict = async () => {
    if (!file) return toast.error('Please upload a fingerprint image first.');

    setLoading(true);
    setResult(null);
    setRejection(null);
    setReliability(null);
    setWarnings([]);
    try {
      const formData = new FormData();
      formData.append('fingerprint', file);

      const res = await predictionService.predict(formData);
      setResult(res.data.prediction);
      setDuplicateWarning(res.data.duplicateWarning);
      setReliability(res.data.reliability);
      setWarnings(res.data.warnings || []);
      toast.success('Prediction complete!');
    } catch (error) {
      const status = error.response?.status;
      const data = error.response?.data;

      if (status === 422 && data?.rejected) {
        setRejection(data);
        toast.error('Image rejected by validation.');
      } else {
        toast.error(data?.error || 'Prediction failed. Is the ML server running?');
      }
    } finally {
      setLoading(false);
    }
  };

  // ─── Scanner Support ──────────────────────────────────────────────────────
  const handleScannerCapture = useCallback(async (options = {}) => {
    if (scannerCaptureInFlightRef.current) return;

    scannerCaptureInFlightRef.current = true;
    setScannerStatus('scanning');
    setLoading(true);
    setRejection(null);
    setResult(null);
    setReliability(null);
    setWarnings([]);
    setDuplicateWarning(null);

    try {
      const capture = await captureFromMantraMfs100({
        allowSavedFileFallback: !options.auto,
      });
      setScannerStatus('connected');
      setScannerMode(true);
      setPreview(capture.previewUrl);
      setFile(base64ToFile(capture.base64, 'mfs100-live-capture.bmp', capture.mimeType));

      const res = await predictionService.scannerPredict(
        capture.base64,
        capture.deviceName,
        capture.resolution
      );

      if (res.data.success) {
        setResult({
          predictedBloodGroup: res.data.prediction.predictedBloodGroup,
          confidence: res.data.prediction.confidence,
          allProbabilities: res.data.prediction.allProbabilities,
          predictionId: res.data.prediction.predictionId,
          createdAt: res.data.prediction.createdAt,
        });
        setDuplicateWarning(res.data.duplicateWarning);
        setReliability(res.data.reliability);
        setWarnings(res.data.warnings || []);
        toast.success('MFS100 live scan prediction complete!');
      }
    } catch (error) {
      const status = error.response?.status;
      const data = error.response?.data;

      if (status === 422 && data?.rejected) {
        setRejection(data);
        toast.error('Scanner capture rejected. Rescan clearly.');
      } else if (!options.silentNoService || !error.isServiceUnavailable) {
        toast.error(error.message || data?.error || 'MFS100 scanner capture failed.');
      }

      if (!options.auto && error.isServiceUnavailable) {
        fallbackScannerInput();
      }
    } finally {
      scannerCaptureInFlightRef.current = false;
      setScannerStatus('idle');
      setLoading(false);
    }
  }, []);

  const fallbackScannerInput = () => {
    // Fallback: open file dialog for scanner-saved images
    setScannerStatus('idle');
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.capture = 'environment';
    input.onchange = (e) => {
      const f = e.target.files[0];
      if (f) {
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setScannerMode(false);
        toast('Scanner image loaded — click Predict to analyze.', { icon: '📷' });
      }
    };
    input.click();
  };

  useEffect(() => {
    if (!scannerMode || autoScannerStartedRef.current) return;

    autoScannerStartedRef.current = true;
    const timer = window.setTimeout(() => {
      handleScannerCapture({ auto: true, silentNoService: true });
    }, 500);

    return () => window.clearTimeout(timer);
  }, [scannerMode, handleScannerCapture]);

  const resetForm = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setDuplicateWarning(null);
    setRejection(null);
    setReliability(null);
    setWarnings([]);
  };

  const getConfidenceColor = (conf) => {
    if (conf >= 90) return '#22c55e';
    if (conf >= 70) return '#f59e0b';
    return '#ef4444';
  };

  const getReliabilityBadge = (level) => {
    const badges = {
      'high':      { label: 'High Confidence',  color: '#22c55e', icon: '✅' },
      'moderate':  { label: 'Moderate',          color: '#f59e0b', icon: '⚠️' },
      'low':       { label: 'Low Confidence',    color: '#f97316', icon: '⚠️' },
      'very_low':  { label: 'Very Low',          color: '#ef4444', icon: '❌' },
      'ambiguous': { label: 'Ambiguous Result',  color: '#ef4444', icon: '❓' },
      'uncertain': { label: 'Uncertain',         color: '#ef4444', icon: '❓' },
    };
    return badges[level] || badges['low'];
  };

  return (
    <div className="page-container">
      <motion.div
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>🩸 Blood Group Prediction</h1>
        <p>Upload a fingerprint image to predict the blood group using our AI model</p>

        {/* Scanner toggle */}
        <div className="input-mode-toggle">
          <button
            className={`mode-btn ${!scannerMode ? 'active' : ''}`}
            onClick={() => setScannerMode(false)}
          >
            <HiOutlineUpload /> File Upload
          </button>
          <button
            className={`mode-btn ${scannerMode ? 'active' : ''}`}
            onClick={() => setScannerMode(true)}
          >
            <HiOutlineDesktopComputer /> Scanner Device
          </button>
        </div>
      </motion.div>

      <div className="predict-layout">
        {/* Upload Section */}
        <motion.div
          className="glass-card predict-upload-section"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h3 className="section-title">
            {scannerMode ? (
              <><HiOutlineDesktopComputer /> Scanner Capture</>
            ) : (
              <><HiOutlineUpload /> Upload Fingerprint</>
            )}
          </h3>

          {scannerMode ? (
            <div className="scanner-section">
              <div className="scanner-visual">
                <div className="scanner-icon-wrapper">
                  <HiOutlineFingerPrint className={`scanner-fp-icon ${scannerStatus === 'scanning' ? 'scanning' : ''}`} />
                  {scannerStatus === 'scanning' && <div className="scanner-laser" />}
                </div>
                <p className="scanner-instruction">
                  {scannerStatus === 'idle' && 'Connect Mantra MFS100 and place your finger to capture'}
                  {scannerStatus === 'scanning' && 'Place your finger on the scanner...'}
                  {scannerStatus === 'connected' && 'MFS100 captured image. Sending to prediction...'}
                </p>
              </div>
              <motion.button
                className="btn btn-primary predict-btn"
                onClick={handleScannerCapture}
                disabled={scannerStatus === 'scanning'}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {scannerStatus === 'scanning' ? (
                  <>
                    <span className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
                    Capturing...
                  </>
                ) : (
                  <><HiOutlineDesktopComputer /> Capture from MFS100</>
                )}
              </motion.button>
              <div className="scanner-note">
                <p>🔌 Requires <strong>MantraRDService.exe</strong> running on this PC (port 8004 / 8003).</p>
                <p>Capture is routed through the server — CORS is bypassed automatically.</p>
                <p>BMP image → quality check → AI validation → blood group prediction.</p>
              </div>
            </div>
          ) : (
            <>
              {!preview ? (
                <div
                  {...getRootProps()}
                  className={`dropzone ${isDragActive ? 'dropzone-active' : ''}`}
                >
                  <input {...getInputProps()} />
                  <div className="dropzone-icon">🖐️</div>
                  <h3>Drop fingerprint image here</h3>
                  <p>or click to browse • JPEG, PNG, BMP • Max 5MB</p>
                  <div className="dropzone-validation-note">
                    <HiOutlineShieldCheck />
                    <span>AI validation will check for authentic fingerprints</span>
                  </div>
                </div>
              ) : (
                <div className="preview-container">
                  <div className="preview-image-wrapper">
                    <img src={preview} alt="Fingerprint preview" className="preview-image" />
                    <button className="preview-remove" onClick={resetForm}>
                      <HiOutlineX />
                    </button>
                  </div>
                  <div className="preview-info">
                    <p className="preview-filename">{file?.name}</p>
                    <p className="preview-size">
                      {(file?.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
              )}

              <motion.button
                className="btn btn-primary predict-btn"
                onClick={handlePredict}
                disabled={!file || loading}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {loading ? (
                  <>
                    <span className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
                    Analyzing...
                  </>
                ) : (
                  <><HiOutlineFingerPrint /> Predict Blood Group</>
                )}
              </motion.button>
            </>
          )}

          {/* Security info */}
          <div className="security-note">
            <p>🔒 Your fingerprint is encrypted with AES-256 before storage. Raw images are never stored.</p>
            <p>🛡️ AI-generated and non-fingerprint images are automatically rejected.</p>
          </div>
        </motion.div>

        {/* Result Section */}
        <motion.div
          className="glass-card predict-result-section"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h3 className="section-title">
            <HiOutlineCheckCircle /> Prediction Result
          </h3>

          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div
                key="loading"
                className="result-loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="analysis-animation">
                  <div className="scan-ring" />
                  <HiOutlineFingerPrint className="scan-icon" />
                </div>
                <p>Analyzing fingerprint patterns...</p>
                <p className="text-muted">Validating image → AI check → CNN prediction</p>
              </motion.div>

            ) : rejection ? (
              /* ── Smart Rejection Display ── */
              <motion.div
                key="rejection"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="result-content"
              >
                <div className={`rejection-card ${
                  rejection.detected_image_type === 'ai_generated' ? 'ai-reject' : 'type-reject'
                }`}>
                  <div className="rejection-icon">
                    {rejection.rejection_icon || '🚫'}
                  </div>
                  <h3 className="rejection-title">
                    {rejection.detected_image_type === 'ai_generated'
                      ? 'AI-Generated Image Detected'
                      : rejection.detected_image_type === 'person_photo'
                      ? 'Person Photo Detected'
                      : rejection.detected_image_type === 'color_photo'
                      ? 'Color Photo — Not a Fingerprint'
                      : rejection.detected_image_type === 'document'
                      ? 'Document / Screenshot Detected'
                      : rejection.detected_image_type === 'qr_barcode'
                      ? 'QR Code / Barcode Detected'
                      : rejection.detected_image_type === 'blank_or_solid'
                      ? 'Blank or Solid Image'
                      : rejection.detected_image_type === 'low_confidence_fingerprint'
                      ? 'Low Confidence Fingerprint'
                      : 'Not a Fingerprint Image'
                    }
                  </h3>
                  <p className="rejection-message">
                    {rejection.rejection_reason}
                  </p>

                  {rejection.validation && (
                    <div className="rejection-details">
                      {!rejection.validation.is_fingerprint && (
                        <div className="rejection-detail-item">
                          <span>Fingerprint Confidence</span>
                          <span className="rejection-score low">
                            {rejection.validation.fingerprint_confidence}%
                          </span>
                        </div>
                      )}
                      {rejection.validation.is_ai_generated && (
                        <div className="rejection-detail-item">
                          <span>AI Detection Score</span>
                          <span className="rejection-score high">
                            {rejection.validation.ai_confidence}%
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="rejection-tip">
                    <span>💡</span>
                    <span>Tip: Use a fingerprint scanner app or inkpad + camera to capture your fingertip in grayscale.</span>
                  </div>

                  <button
                    className="btn btn-primary"
                    onClick={resetForm}
                    style={{ marginTop: '16px', width: '100%' }}
                  >
                    Upload a Different Image
                  </button>
                </div>
              </motion.div>

            ) : result ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                className="result-content"
              >
                {/* Warnings */}
                {warnings.length > 0 && (
                  <div className="warnings-list">
                    {warnings.map((w, i) => (
                      <div key={i} className="warning-item">
                        <HiOutlineExclamation />
                        <span>{w}</span>
                      </div>
                    ))}
                  </div>
                )}

                {/* Duplicate warning */}
                {duplicateWarning && (
                  <div className="duplicate-warning">
                    <HiOutlineExclamation />
                    <span>{duplicateWarning}</span>
                  </div>
                )}

                {/* Reliability badge */}
                {reliability && (
                  <div className="reliability-badge-wrapper">
                    {(() => {
                      const badge = getReliabilityBadge(reliability.confidence_level);
                      return (
                        <div className="reliability-badge" style={{ borderColor: badge.color }}>
                          <span className="reliability-icon">{badge.icon}</span>
                          <span style={{ color: badge.color }}>{badge.label}</span>
                          {reliability.margin !== undefined && (
                            <span className="reliability-margin">
                              Margin: {reliability.margin}%
                            </span>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                )}

                {/* Blood group display */}
                <div className="result-blood-group">
                  <div className="blood-group-display">{result.predictedBloodGroup}</div>
                  <p className="result-label">Predicted Blood Group</p>
                </div>

                {/* Confidence */}
                <div className="result-confidence">
                  <div className="confidence-header">
                    <span>Confidence Score</span>
                    <span style={{ color: getConfidenceColor(result.confidence), fontWeight: 700 }}>
                      {result.confidence}%
                    </span>
                  </div>
                  <div className="confidence-bar">
                    <motion.div
                      className="confidence-bar-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${result.confidence}%` }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                      style={{
                        background: `linear-gradient(90deg, ${getConfidenceColor(result.confidence)}, ${getConfidenceColor(result.confidence)}88)`
                      }}
                    />
                  </div>
                </div>

                {/* All Probabilities */}
                <div className="result-probabilities">
                  <h4>All Probabilities</h4>
                  <div className="prob-list">
                    {result.allProbabilities && Object.entries(result.allProbabilities).map(([group, prob]) => (
                      <div key={group} className="prob-item">
                        <span className="prob-group">{group}</span>
                        <div className="prob-bar-container">
                          <motion.div
                            className="prob-bar"
                            initial={{ width: 0 }}
                            animate={{ width: `${prob}%` }}
                            transition={{ duration: 0.8, delay: 0.3 }}
                          />
                        </div>
                        <span className="prob-value">{prob}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Prediction ID */}
                <div className="result-meta">
                  <p>Prediction ID: <span>{result.predictionId}</span></p>
                  <p>Date: <span>{new Date(result.createdAt).toLocaleString()}</span></p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                className="result-empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <div className="empty-fingerprint">
                  <HiOutlineFingerPrint />
                </div>
                <h3>No Prediction Yet</h3>
                <p>Upload a fingerprint image and click <strong>Predict</strong> to see results</p>

                <div className="validation-features">
                  <div className="vf-item"><span>🛡️</span> Auto-rejects non-fingerprint images</div>
                  <div className="vf-item"><span>🤖</span> Detects AI-generated fingerprints</div>
                  <div className="vf-item"><span>📊</span> Confidence & reliability scoring</div>
                  <div className="vf-item"><span>🔌</span> Physical scanner support</div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>

      {/* Hidden canvas for scanner image processing */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
};

export default Predict;
