import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Attach token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 responses (expired token)
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ─── Auth Services ────────────────────────────────────────────────────────────
export const authService = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  verifyOtp: (otp) => api.post('/auth/verify-otp', { otp }),
  getProfile: () => api.get('/auth/profile'),
  updateProfile: (data) => api.put('/auth/profile', data),
};

// ─── Prediction Services ─────────────────────────────────────────────────────
export const predictionService = {
  predict: (formData) =>
    api.post('/predictions/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 30000,
    }),

  mfs100Capture: (quality = 70, options = {}) =>
    api.post('/predictions/mfs100-capture', {
      quality,
      timeout: 15,
      allowSavedFileFallback: options.allowSavedFileFallback !== false,
    }, { timeout: 25000 }),

  scannerPredict: (imageBase64, deviceName, resolution) =>
    api.post('/predictions/scanner-predict', {
      image_base64: imageBase64,
      device_name: deviceName,
      resolution: resolution,
    }, { timeout: 30000 }),
  getHistory: (page = 1, limit = 10) =>
    api.get(`/predictions/history?page=${page}&limit=${limit}`),
  getPrediction: (id) => api.get(`/predictions/prediction/${id}`),
  getStats: () => api.get('/predictions/stats'),
};

// ─── Admin Services ───────────────────────────────────────────────────────────
export const adminService = {
  getDashboard: () => api.get('/admin/dashboard'),
  getUsers: (page = 1, limit = 20, search = '') =>
    api.get(`/admin/users?page=${page}&limit=${limit}&search=${search}`),
  getPredictions: (page = 1, limit = 20) =>
    api.get(`/admin/predictions?page=${page}&limit=${limit}`),
  deleteUser: (id) => api.delete(`/admin/users/${id}`),
};

export default api;
