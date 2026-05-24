import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/useAuth';
import { useTheme } from '../../context/ThemeContext';
import {
  HiOutlineHome, HiOutlineFingerPrint, HiOutlineClock,
  HiOutlineChartBar, HiOutlineUsers, HiOutlineCog,
  HiOutlineLogout, HiOutlineMenu, HiOutlineX,
  HiOutlineShieldCheck, HiOutlineUser,
  HiOutlineSun, HiOutlineMoon
} from 'react-icons/hi';
import './Layout.css';

const Sidebar = () => {
  const { user, isAdmin, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const userLinks = [
    { path: '/dashboard', icon: <HiOutlineHome />, label: 'Dashboard' },
    { path: '/predict', icon: <HiOutlineFingerPrint />, label: 'Predict' },
    { path: '/history', icon: <HiOutlineClock />, label: 'History' },
    { path: '/profile', icon: <HiOutlineUser />, label: 'Profile' },
  ];

  const adminLinks = [
    { path: '/admin', icon: <HiOutlineChartBar />, label: 'Analytics' },
    { path: '/admin/users', icon: <HiOutlineUsers />, label: 'Users' },
    { path: '/admin/predictions', icon: <HiOutlineFingerPrint />, label: 'Predictions' },
  ];

  return (
    <>
      {/* Mobile toggle */}
      <button className="sidebar-toggle" onClick={() => setCollapsed(!collapsed)}>
        {collapsed ? <HiOutlineMenu /> : <HiOutlineX />}
      </button>

      <motion.aside
        className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''}`}
        initial={{ x: -280 }}
        animate={{ x: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      >
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="brand-icon">🩸</div>
          {!collapsed && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <h2>BloodAI</h2>
              <span>Predictor</span>
            </motion.div>
          )}
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          <div className="nav-section">
            {!collapsed && <p className="nav-label">MAIN MENU</p>}
            {userLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`nav-link ${location.pathname === link.path ? 'active' : ''}`}
              >
                <span className="nav-icon">{link.icon}</span>
                {!collapsed && <span>{link.label}</span>}
                {location.pathname === link.path && (
                  <motion.div className="nav-active-indicator" layoutId="activeTab" />
                )}
              </Link>
            ))}
          </div>

          {isAdmin && (
            <div className="nav-section">
              {!collapsed && <p className="nav-label">ADMIN</p>}
              {adminLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`nav-link ${location.pathname === link.path ? 'active' : ''}`}
                >
                  <span className="nav-icon">{link.icon}</span>
                  {!collapsed && <span>{link.label}</span>}
                  {location.pathname === link.path && (
                    <motion.div className="nav-active-indicator" layoutId="activeTab" />
                  )}
                </Link>
              ))}
            </div>
          )}
        </nav>

        {/* User info + Theme Toggle */}
        <div className="sidebar-footer">
          {!collapsed && (
            <div className="user-info">
              <div className="user-avatar">
                {user?.name?.charAt(0).toUpperCase()}
              </div>
              <div className="user-details">
                <p className="user-name">{user?.name}</p>
                <p className="user-role">
                  {isAdmin ? (
                    <><HiOutlineShieldCheck /> Admin</>
                  ) : (
                    'User'
                  )}
                </p>
              </div>
              {/* Theme Toggle */}
              <motion.button
                className="theme-toggle"
                onClick={toggleTheme}
                title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
                whileHover={{ scale: 1.15, rotate: 20 }}
                whileTap={{ scale: 0.9 }}
              >
                <motion.span
                  className="icon"
                  key={theme}
                  initial={{ rotate: -90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  {theme === 'dark' ? <HiOutlineSun /> : <HiOutlineMoon />}
                </motion.span>
              </motion.button>
            </div>
          )}
          {collapsed && (
            <motion.button
              className="theme-toggle"
              onClick={toggleTheme}
              title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
              whileHover={{ scale: 1.15, rotate: 20 }}
              whileTap={{ scale: 0.9 }}
              style={{ margin: '0 auto 12px', display: 'flex' }}
            >
              <motion.span
                className="icon"
                key={theme}
                initial={{ rotate: -90, opacity: 0 }}
                animate={{ rotate: 0, opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                {theme === 'dark' ? <HiOutlineSun /> : <HiOutlineMoon />}
              </motion.span>
            </motion.button>
          )}
          <button className="btn-logout" onClick={handleLogout} title="Logout">
            <HiOutlineLogout />
            {!collapsed && <span>Logout</span>}
          </button>
        </div>
      </motion.aside>
    </>
  );
};

export default Sidebar;
