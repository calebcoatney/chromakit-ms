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
import { checkHealth, loadFile, processChromato, integratePeaks } from './services/api';
import './styles/App.css';

function App() {
  // State management
  const [apiStatus, setApiStatus] = useState({ connected: false });
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [integrationResults, setIntegrationResults] = useState(null);
  const [currentParams, setCurrentParams] = useState(null);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  // Check API health on mount
  useEffect(() => {
    const checkAPIHealth = async () => {
      try {
        await checkHealth();
        setApiStatus({ connected: true });
      } catch (err) {
        setApiStatus({ connected: false });
      }
    };

    checkAPIHealth();
    // Check every 30 seconds
    const interval = setInterval(checkAPIHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle file selection
  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    setFileData(null);
    setProcessedData(null);
    setIntegrationResults(null);
    setError(null);
    setLoading(true);

    try {
      const data = await loadFile(file.path);
      setFileData(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load file');
    } finally {
      setLoading(false);
    }
  };

  // Handle real-time parameter changes
  const handleParametersChange = useCallback(async (params) => {
    if (!fileData || processing) return;

    setCurrentParams(params);
    setIntegrationResults(null); // Clear integration when parameters change
    setProcessing(true);
    setError(null);

    try {
      // Prepare MS range if TIC data is available
      const msRange = fileData.has_ms && fileData.tic.x.length > 0
        ? [Math.min(...fileData.tic.x), Math.max(...fileData.tic.x)]
        : null;

      const data = await processChromato({
        x: fileData.chromatogram.x,
        y: fileData.chromatogram.y,
        params,
        ms_range: msRange
      });

      setProcessedData(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process chromatogram');
    } finally {
      setProcessing(false);
    }
  }, [fileData, processing]);

  // Handle peak integration
  const handleIntegrate = async () => {
    if (!processedData) return;

    setIntegrationResults(null);
    setError(null);
    setLoading(true);

    try {
      const data = await integratePeaks({
        processed_data: processedData,
        chemstation_area_factor: 0.0784
      });

      setIntegrationResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to integrate peaks');
    } finally {
      setLoading(false);
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
          {/* Error Display */}
          {error && (
            <div className="status-indicator error">
              ⚠️ {error}
            </div>
          )}

          {/* Selected File Info */}
          {selectedFile && (
            <div className="card">
              <div className="card-body">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                      📄 {selectedFile.name}
                    </div>
                    <div className="text-muted" style={{ fontSize: '0.875rem' }}>
                      {fileData ? (
                        <>
                          {fileData.chromatogram.x.length} points
                          {fileData.has_ms && ` | MS data available (${fileData.tic.x.length} scans)`}
                        </>
                      ) : (
                        'Loading...'
                      )}
                    </div>
                  </div>
                  {loading && <div className="spinner"></div>}
                </div>
              </div>
            </div>
          )}

          {/* Main visualization area with plot and controls side-by-side */}
          <div style={{ display: 'flex', gap: '1rem' }}>
            {/* Chromatogram Plot and Results - takes up most space */}
            <div style={{ flex: 1 }}>
              <ChromatogramPlot
                data={fileData?.chromatogram}
                processedData={processedData}
                integrationResults={integrationResults}
                showCorrectedSignal={currentParams?.baseline?.show_corrected || false}
                showBaseline={true}
              />

              {/* Integrate Peaks Button */}
              {processedData && processedData.peaks_x && processedData.peaks_x.length > 0 && !integrationResults && (
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

              {/* Peak Integration Table */}
              {integrationResults && (
                <PeakTable
                  integrationResults={integrationResults}
                  onIntegrate={handleIntegrate}
                  disabled={loading}
                />
              )}
            </div>

            {/* Processing Controls - on the right */}
            {fileData && (
              <div style={{ width: '450px', flexShrink: 0 }}>
                <ProcessingControls
                  onParametersChange={handleParametersChange}
                  disabled={processing}
                />
              </div>
            )}
          </div>

          {/* No Data Message */}
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
