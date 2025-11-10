/**
 * FileBrowser Component
 * 
 * Allows users to browse directories and select .D files from the server.
 */
import React, { useState, useEffect } from 'react';
import { browseDirectory } from '../services/api';

const FileBrowser = ({ onFileSelect }) => {
  const [currentPath, setCurrentPath] = useState('.');
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [parentPath, setParentPath] = useState(null);

  useEffect(() => {
    loadDirectory(currentPath);
  }, []);

  const loadDirectory = async (path) => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await browseDirectory(path);
      setEntries(data.entries || []);
      setCurrentPath(data.current_path);
      setParentPath(data.parent_path);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load directory');
    } finally {
      setLoading(false);
    }
  };

  const handleEntryClick = (entry) => {
    if (entry.type === 'directory') {
      loadDirectory(entry.path);
    } else if (entry.format === 'agilent_d') {
      onFileSelect(entry);
    }
  };

  const handleGoUp = () => {
    if (parentPath) {
      loadDirectory(parentPath);
    }
  };

  return (
    <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="card-header">
        <h2>📁 File Browser</h2>
      </div>
      
      <div className="card-body" style={{ flex: 1, overflow: 'auto' }}>
        {/* Current Path */}
        <div className="mb-2">
          <div className="form-label">Current Path:</div>
          <div style={{ 
            fontSize: '0.875rem', 
            color: 'var(--text-secondary)',
            wordBreak: 'break-all'
          }}>
            {currentPath}
          </div>
        </div>

        {/* Parent Directory Button */}
        {parentPath && (
          <button 
            className="btn btn-secondary btn-sm mb-2" 
            onClick={handleGoUp}
            disabled={loading}
          >
            ⬆️ Parent Directory
          </button>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-container">
            <div className="spinner"></div>
            <div className="text-muted">Loading directory...</div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="status-indicator error mb-2">
            ⚠️ {error}
          </div>
        )}

        {/* Directory Entries */}
        {!loading && !error && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {entries.length === 0 ? (
              <div className="text-center text-muted">
                No files or directories found
              </div>
            ) : (
              entries.map((entry, index) => (
                <div
                  key={index}
                  onClick={() => handleEntryClick(entry)}
                  style={{
                    padding: '0.75rem',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    transition: 'all 0.2s',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'var(--primary-color)';
                    e.currentTarget.style.backgroundColor = 'rgba(102, 126, 234, 0.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'var(--border-color)';
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  <span style={{ fontSize: '1.25rem' }}>
                    {entry.type === 'directory' ? '📂' : '📄'}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 500 }}>
                      {entry.name}
                    </div>
                    {entry.format && (
                      <div style={{ 
                        fontSize: '0.75rem', 
                        color: 'var(--text-secondary)' 
                      }}>
                        {entry.format}
                      </div>
                    )}
                  </div>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {entry.type === 'directory' ? '→' : ''}
                  </span>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default FileBrowser;
