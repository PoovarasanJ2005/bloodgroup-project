import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../context/useAuth';
import { useTheme } from '../context/ThemeContext';
import toast from 'react-hot-toast';
import { HiOutlineMail, HiOutlineLockClosed, HiOutlineFingerPrint, HiOutlineSun, HiOutlineMoon } from 'react-icons/hi';
import './Auth.css';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const user = await login(email, password);
      toast.success(`Welcome back, ${user.name}!`);
      navigate(user.role === 'admin' ? '/admin' : '/dashboard');
    } catch (error) {
      toast.error(error.response?.data?.error || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

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

      {/* Animated background orbs */}
      <div className="auth-bg-orbs">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
      </div>

      <motion.div
        className="auth-container"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
      >
        {/* Left Panel */}
        <div className="auth-hero">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <div className="auth-hero-icon">
              <HiOutlineFingerPrint />
            </div>
            <h1>Blood Group<br />Predictor</h1>
            <p>AI-powered blood group prediction using fingerprint analysis with advanced CNN deep learning.</p>
            <div className="auth-hero-stats">
              <div className="hero-stat">
                <span className="hero-stat-value">8</span>
                <span className="hero-stat-label">Blood Groups</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-value">6K+</span>
                <span className="hero-stat-label">Samples</span>
              </div>
              <div className="hero-stat">
                <span className="hero-stat-value">CNN</span>
                <span className="hero-stat-label">Deep Learning</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Right Panel - Form */}
        <div className="auth-form-panel">
          <motion.form
            className="auth-form"
            onSubmit={handleSubmit}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            <div className="auth-form-header">
              <h2>Welcome Back</h2>
              <p>Sign in to continue to your dashboard</p>
            </div>

            <div className="input-group">
              <label htmlFor="email">Email Address</label>
              <div className="input-with-icon">
                <HiOutlineMail className="input-icon" />
                <input
                  id="email"
                  type="email"
                  className="input-field"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div className="input-group">
              <label htmlFor="password">Password</label>
              <div className="input-with-icon">
                <HiOutlineLockClosed className="input-icon" />
                <input
                  id="password"
                  type="password"
                  className="input-field"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
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
                'Sign In'
              )}
            </motion.button>

            <p className="auth-switch">
              Don't have an account?{' '}
              <Link to="/register">Create Account</Link>
            </p>
          </motion.form>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;
