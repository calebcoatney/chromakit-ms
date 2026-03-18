/**
 * Main App Component
 *
 * Orchestrates all components and manages application state.
 * Processes chromatogram in real-time as parameters change.
 */
import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import FileBrowser from './components/FileBrowser';
import ProcessingControls from './components/ProcessingControls';
import ChromatogramPlot from './components/ChromatogramPlot';
import PeakTable from './components/PeakTable';
import * as api from './services/api.ts';
import './styles/App.css';

function App() {
  // State
  const [apiStatus, setApiStatus] = useState({ connected: false });
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [integrationResults, setIntegrationResults] = useState(null);
  const [currentParams, setCurrentParams] = useState(null);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [availableDetectors, setAvailableDetectors] = useState([]);
  const [currentDetector, setCurrentDetector] = useState(null);

  // Health check on mount + interval
  useEffect(() => {
    const check = async () => {
      try {
        await api.checkHealth();
        setApiStatus({ connected: true });
      } catch {
        setApiStatus({ connected: false });
      }
    };
    check();
    const id = setInterval(check, 30000);
    return () => clearInterval(id);
  }, []);

  // File selection
  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    setFileData(null);
    setProcessedData(null);
    setIntegrationResults(null);
    setError(null);
    setLoading(true);

    try {
      const data = await api.loadFile(file.path);
      setFileData(data);
      setAvailableDetectors(data.available_detectors || []);
      setCurrentDetector(data.current_detector || null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load file');
    } finally {
      setLoading(false);
    }
  };

  // Detector change
  const handleDetectorChange = async (detector) => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      const data = await api.loadFile(selectedFile.path, detector);
      setFileData(data);
      setCurrentDetector(detector);
      setProcessedData(null);
      setIntegrationResults(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to switch detector');
    } finally {
      setLoading(false);
    }
  };

  // Real-time parameter changes → process
  const handleParametersChange = useCallback(async (params) => {
    if (!fileData || processing) return;

    setCurrentParams(params);
    setIntegrationResults(null);
    setProcessing(true);
    setError(null);

    try {
      const msRange = fileData.has_ms && fileData.tic.x.length > 0
        ? [Math.min(...fileData.tic.x), Math.max(...fileData.tic.x)]
        : undefined;

      const data = await api.processChromato(
        fileData.chromatogram.x,
        fileData.chromatogram.y,
        params,
        msRange,
      );
      setProcessedData(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process chromatogram');
    } finally {
      setProcessing(false);
    }
  }, [fileData, processing]);

  // Integration
  const handleIntegrate = async () => {
    if (!processedData) return;
    setIntegrationResults(null);
    setError(null);
    setLoading(true);

    try {
      const peakGroups = currentParams?.integration?.peak_groups;
      const data = await api.integratePeaks(
        processedData,
        0.0784,
        undefined,
        peakGroups?.length ? peakGroups : undefined,
      );
      setIntegrationResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to integrate peaks');
    } finally {
      setLoading(false);
    }
  };

  // Navigation
  const handleNavigate = async (direction) => {
    setError(null);
    try {
      const resp = direction === 'next'
        ? await fetch('/api/navigate/next').then(r => r.json())
        : await fetch('/api/navigate/previous').then(r => r.json());

      if (resp.file_path) {
        handleFileSelect({ name: resp.file_path.split('/').pop(), path: resp.file_path });
      }
    } catch (err) {
      setError('Navigation failed');
    }
  };

  return (
    <div className="app">
      <Header apiStatus={apiStatus} />

      <div className="main-content">
        {/* Sidebar */}
        <aside className="sidebar">
          <FileBrowser onFileSelect={handleFileSelect} />
        </aside>

        {/* Main Panel */}
        <main className="main-panel">
          {/* Error */}
          {error && (
            <div className="status-indicator error">⚠️ {error}</div>
          )}

          {/* File info bar */}
          {selectedFile && (
            <div className="card">
              <div className="card-body">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                  <div>
                    <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                      📄 {selectedFile.name}
                    </div>
                    <div className="text-muted" style={{ fontSize: '0.875rem' }}>
                      {fileData ? (
                        <>
                          {fileData.chromatogram.x.length} points
                          {currentDetector && ` | ${currentDetector}`}
                          {fileData.has_ms && ` | MS (${fileData.tic.x.length} scans)`}
                        </>
                      ) : 'Loading...'}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    {/* Detector selector */}
                    {availableDetectors.length > 1 && (
                      <select
                        className="form-control"
                        style={{ width: 'auto', minWidth: '100px' }}
                        value={currentDetector || ''}
                        onChange={(e) => handleDetectorChange(e.target.value)}
                      >
                        {availableDetectors.map((d) => (
                          <option key={d} value={d}>{d}</option>
                        ))}
                      </select>
                    )}
                    {/* Navigation buttons */}
                    <button className="btn btn-sm btn-secondary" onClick={() => handleNavigate('previous')} title="Previous sample">◀</button>
                    <button className="btn btn-sm btn-secondary" onClick={() => handleNavigate('next')} title="Next sample">▶</button>
                    {loading && <div className="spinner"></div>}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Processing Controls */}
          {fileData && (
            <ProcessingControls
              onParametersChange={handleParametersChange}
              disabled={processing}
            />
          )}

          {/* Plot */}
          <ChromatogramPlot
            data={fileData?.chromatogram}
            processedData={processedData}
            integrationResults={integrationResults}
            showCorrectedSignal={currentParams?.baseline?.show_corrected || false}
            showBaseline={true}
          />

          {/* Integrate button */}
          {processedData && processedData.peaks_x?.length > 0 && !integrationResults && (
            <div className="card">
              <div className="card-body">
                <button
                  className="btn btn-success"
                  onClick={handleIntegrate}
                  disabled={loading}
                  style={{ width: '100%', fontSize: '1rem', padding: '1rem' }}
                >
                  {loading ? '⏳ Integrating...' : '📈 Integrate Peaks'}
                </button>
              </div>
            </div>
          )}

          {/* Results table */}
          {integrationResults && (
            <PeakTable
              integrationResults={integrationResults}
              onIntegrate={handleIntegrate}
              disabled={loading}
            />
          )}

          {/* Welcome */}
          {!selectedFile && !loading && (
            <div className="card">
              <div className="card-body text-center" style={{ padding: '3rem' }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📁</div>
                <h3 style={{ marginBottom: '0.5rem' }}>Welcome to ChromaKit-MS</h3>
                <p className="text-muted">
                  Select a .D file from the browser to begin analysis
                </p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
