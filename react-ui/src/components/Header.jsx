/**
 * Header Component
 * 
 * Displays the application title and optional status indicator.
 */
import React from 'react';

const Header = ({ apiStatus }) => {
  return (
    <header className="header">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>🧪 ChromaKit-MS</h1>
          <p>GC-MS Data Analysis Platform</p>
        </div>
        {apiStatus && (
          <div className={`status-indicator ${apiStatus.connected ? 'success' : 'error'}`}>
            <span className={`status-dot ${apiStatus.connected ? 'success' : 'error'}`}></span>
            {apiStatus.connected ? 'API Connected' : 'API Disconnected'}
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
