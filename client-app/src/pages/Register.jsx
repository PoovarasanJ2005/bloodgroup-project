import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../context/useAuth';
import { useTheme } from '../context/ThemeContext';
import toast from 'react-hot-toast';
import {
  HiOutlineUser, HiOutlineMail, HiOutlineLockClosed,
  HiOutlineCalendar, HiOutlinePhone, HiOutlineFingerPrint,
  HiOutlineShieldCheck, HiOutlineSun, HiOutlineMoon
} from 'react-icons/hi';
import './Auth.css';

const Register = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    dateOfBirth: '',
    phone: '',
  });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setErrors({ ...errors, [e.target.name]: '' });
  };

  const validateForm = () => {
    const newErrors = {};

    if (formData.name.length < 2) newErrors.name = 'Name must be at least 2 characters';
    if (!/\S+@\S+\.\S+/.test(formData.email)) newErrors.email = 'Valid email required';
    if (formData.password.length < 8) newErrors.password = 'Minimum 8 characters';
    if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])/.test(formData.password)) {
      newErrors.password = 'Must include uppercase, lowercase, number & special character';
    }
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    if (!formData.dateOfBirth) {
      newErrors.dateOfBirth = 'Date of birth required';
    } else {
      const dob = new Date(formData.dateOfBirth);
      const today = new Date();
      let age = today.getFullYear() - dob.getFullYear();
      const m = today.getMonth() - dob.getMonth();
      if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) age--;
      if (age < 18) newErrors.dateOfBirth = 'You must be 18 or older to register';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    setLoading(true);
    try {
      await register({
        name: formData.name,
        email: formData.email,
        password: formData.password,
        dateOfBirth: formData.dateOfBirth,
        phone: formData.phone,
      });
      toast.success('Account created successfully!');
      navigate('/dashboard');
    } catch (error) {
      const msg = error.response?.data?.error || error.response?.data?.errors?.[0]?.msg || 'Registration failed';
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  // Calculate max date for 18+
  const maxDate = new Date();
  maxDate.setFullYear(maxDate.getFullYear() - 18);
  const maxDateStr = maxDate.toISOString().split('T')[0];

  return (
    <div className="auth-page">
      {/* Theme toggle */}
      <div className="auth-theme-toggle">
        <motion.button
          className="theme-toggle"
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          whileHover={{ scale: 1.15, rotate: 20 }}
          whileTap={{ scale: 0.9 }}
        >
          <motion.span className="icon" key={theme} initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} transition={{ duration: 0.3 }}>
            {theme === 'dark' ? <HiOutlineSun /> : <HiOutlineMoon />}
          </motion.span>
        </motion.button>
      </div>

      <div className="auth-bg-orbs">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      <motion.div
        className="auth-container auth-container-register"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        {/* Hero */}
        <div className="auth-hero">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            <div className="auth-hero-icon">
              <HiOutlineFingerPrint />
            </div>
            <h1>Create<br />Account</h1>
            <p>Join BloodAI to predict your blood group using advanced fingerprint analysis.</p>

            <div className="auth-features">
              <div className="auth-feature">
                <HiOutlineShieldCheck />
                <span>AES-256 Encrypted Data</span>
              </div>
              <div className="auth-feature">
                <HiOutlineFingerPrint />
                <span>Secure Biometric Processing</span>
              </div>
              <div className="auth-feature">
                <HiOutlineLockClosed />
                <span>JWT Authentication</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Form */}
        <div className="auth-form-panel">
          <motion.form
            className="auth-form"
            onSubmit={handleSubmit}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="auth-form-header">
              <h2>Get Started</h2>
              <p>Fill in your details to create an account</p>
            </div>

            <div className="auth-form-grid">
              <div className="input-group">
                <label htmlFor="name">Full Name</label>
                <div className="input-with-icon">
                  <HiOutlineUser className="input-icon" />
                  <input
                    id="name" name="name" type="text"
                    className={`input-field ${errors.name ? 'input-error' : ''}`}
                    placeholder="John Doe"
                    value={formData.name}
                    onChange={handleChange}
                    required
                  />
                </div>
                {errors.name && <span className="error-text">{errors.name}</span>}
              </div>

              <div className="input-group">
                <label htmlFor="email">Email Address</label>
                <div className="input-with-icon">
                  <HiOutlineMail className="input-icon" />
                  <input
                    id="email" name="email" type="email"
                    className={`input-field ${errors.email ? 'input-error' : ''}`}
                    placeholder="you@example.com"
                    value={formData.email}
                    onChange={handleChange}
                    required
                  />
                </div>
                {errors.email && <span className="error-text">{errors.email}</span>}
              </div>

              <div className="input-group">
                <label htmlFor="password">Password</label>
                <div className="input-with-icon">
                  <HiOutlineLockClosed className="input-icon" />
                  <input
                    id="password" name="password" type="password"
                    className={`input-field ${errors.password ? 'input-error' : ''}`}
                    placeholder="Min 8 chars, mixed case"
                    value={formData.password}
                    onChange={handleChange}
                    required
                  />
                </div>
                {errors.password && <span className="error-text">{errors.password}</span>}
              </div>

              <div className="input-group">
                <label htmlFor="confirmPassword">Confirm Password</label>
                <div className="input-with-icon">
                  <HiOutlineLockClosed className="input-icon" />
                  <input
                    id="confirmPassword" name="confirmPassword" type="password"
                    className={`input-field ${errors.confirmPassword ? 'input-error' : ''}`}
                    placeholder="Re-enter password"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    required
                  />
                </div>
                {errors.confirmPassword && <span className="error-text">{errors.confirmPassword}</span>}
              </div>

              <div className="input-group">
                <label htmlFor="dateOfBirth">Date of Birth (18+ only)</label>
                <div className="input-with-icon">
                  <HiOutlineCalendar className="input-icon" />
                  <input
                    id="dateOfBirth" name="dateOfBirth" type="date"
                    className={`input-field ${errors.dateOfBirth ? 'input-error' : ''}`}
                    max={maxDateStr}
                    value={formData.dateOfBirth}
                    onChange={handleChange}
                    required
                  />
                </div>
                {errors.dateOfBirth && <span className="error-text">{errors.dateOfBirth}</span>}
              </div>

              <div className="input-group">
                <label htmlFor="phone">Phone (Optional)</label>
                <div className="input-with-icon">
                  <HiOutlinePhone className="input-icon" />
                  <input
                    id="phone" name="phone" type="tel"
                    className="input-field"
                    placeholder="+91 9876543210"
                    value={formData.phone}
                    onChange={handleChange}
                  />
                </div>
              </div>
            </div>

            <motion.button
              type="submit"
              className="btn btn-primary auth-submit-btn"
              disabled={loading}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              {loading ? (
                <span className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }} />
              ) : (
                'Create Account'
              )}
            </motion.button>

            <p className="auth-switch">
              Already have an account? <Link to="/login">Sign In</Link>
            </p>
          </motion.form>
        </div>
      </motion.div>
    </div>
  );
};

export default Register;
