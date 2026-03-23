/**
 * FileBrowser Component — browse directories and select .D files from the server.
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
    if (parentPath) loadDirectory(parentPath);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="card-header">
        <h2>Files</h2>
      </div>

      <div className="card-body" style={{ flex: 1, overflow: 'auto' }}>
        <div className="file-browser-path">{currentPath}</div>

        {parentPath && (
          <button className="btn btn-secondary btn-sm mb-2 full-width" onClick={handleGoUp} disabled={loading}>
            Parent Directory
          </button>
        )}

        {loading && (
          <div className="loading-container">
            <div className="spinner" />
            <span>Loading...</span>
          </div>
        )}

        {error && (
          <div className="status-indicator-banner error mb-2">{error}</div>
        )}

        {!loading && !error && (
          <div>
            {entries.length === 0 ? (
              <div className="text-center text-muted" style={{ padding: '1rem' }}>
                No files found
              </div>
            ) : (
              entries.map((entry, index) => (
                <div key={index} className="file-entry" onClick={() => handleEntryClick(entry)}>
                  <span className="file-entry-icon">
                    {entry.type === 'directory' ? '\u25B8' : '\u25AB'}
                  </span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="file-entry-name">{entry.name}</div>
                    {entry.format && (
                      <div className="file-entry-format">{entry.format}</div>
                    )}
                  </div>
                  {entry.type === 'directory' && (
                    <span className="file-entry-arrow">&rsaquo;</span>
                  )}
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
