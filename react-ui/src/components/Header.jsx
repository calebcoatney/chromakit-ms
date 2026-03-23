/**
 * Header Component — slim app bar with title, API status, and theme toggle.
 */
import React, { useState, useEffect } from 'react';

const Header = ({ apiStatus }) => {
  const [dark, setDark] = useState(() => {
    return localStorage.getItem('chromakit-theme') === 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
    localStorage.setItem('chromakit-theme', dark ? 'dark' : 'light');
  }, [dark]);

  return (
    <header className="header">
      <span className="header-title">ChromaKit-MS</span>
      <div className="header-actions">
        <div className={`status-indicator ${apiStatus?.connected ? 'success' : 'error'}`}>
          <span className={`status-dot ${apiStatus?.connected ? 'success' : 'error'}`} />
          {apiStatus?.connected ? 'Connected' : 'Disconnected'}
        </div>
        <button
          className="theme-toggle"
          onClick={() => setDark(!dark)}
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {dark ? 'Light' : 'Dark'}
        </button>
      </div>
    </header>
  );
};

export default Header;
