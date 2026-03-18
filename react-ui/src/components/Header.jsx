/**
 * Header Component
 *
 * App title, API status, and theme toggle.
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>🧪 ChromaKit-MS</h1>
          <p>GC-MS Data Analysis Platform</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button
            onClick={() => setDark(!dark)}
            style={{
              background: 'rgba(255,255,255,0.2)',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '6px',
              padding: '0.4rem 0.75rem',
              color: 'white',
              cursor: 'pointer',
              fontSize: '0.85rem',
            }}
            title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {dark ? '☀️ Light' : '🌙 Dark'}
          </button>
          {apiStatus && (
            <div className={`status-indicator ${apiStatus.connected ? 'success' : 'error'}`}>
              <span className={`status-dot ${apiStatus.connected ? 'success' : 'error'}`}></span>
              {apiStatus.connected ? 'API Connected' : 'API Disconnected'}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;
