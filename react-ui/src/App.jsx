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
import MSSpectrumViewer from './components/MSSpectrumViewer';
import ScalingFactorsDialog from './components/ScalingFactorsDialog';
import ExportSettingsDialog from './components/ExportSettingsDialog';
import EditAssignmentDialog from './components/EditAssignmentDialog';
import MSOptionsDialog from './components/MSOptionsDialog';
import QuantitationPanel from './components/QuantitationPanel';
import RTTableManager from './components/RTTableManager';
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
  const [selectedPeakIndex, setSelectedPeakIndex] = useState(null);
  const [spectrum, setSpectrum] = useState(null);
  const [msSearchResults, setMsSearchResults] = useState(null);
  const [msSearching, setMsSearching] = useState(false);

  // Dialog visibility
  const [showScaling, setShowScaling] = useState(false);
  const [showExportSettings, setShowExportSettings] = useState(false);
  const [showEditAssign, setShowEditAssign] = useState(false);
  const [showMSOptions, setShowMSOptions] = useState(false);

  // Feature state
  const [scalingFactors, setScalingFactors] = useState({ signal: 1.0, area: 1.0 });
  const [msOptions, setMsOptions] = useState({});
  const [quantEnabled, setQuantEnabled] = useState(false);
  const [quantSettings, setQuantSettings] = useState({});
  const [quantCalc, setQuantCalc] = useState({});
  const [rtSettings, setRtSettings] = useState({});

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

  // Spectrum extraction
  const handleExtractSpectrum = async (rt) => {
    if (!selectedFile) return;
    try {
      const data = await api.extractSpectrum(selectedFile.path, rt);
      setSpectrum(data);
      setMsSearchResults(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to extract spectrum');
    }
  };

  // MS library search
  const handleSearchSpectrum = async () => {
    if (!spectrum) return;
    setMsSearching(true);
    try {
      const results = await api.searchSpectrum(spectrum);
      setMsSearchResults(results);
    } catch (err) {
      if (err.response?.status === 501) {
        setError('MS library search requires ms-toolkit-nrel on the server');
      } else {
        setError(err.response?.data?.detail || 'MS search failed');
      }
    } finally {
      setMsSearching(false);
    }
  };

  // Peak selection → extract spectrum
  const handlePeakSelect = (idx) => {
    setSelectedPeakIndex(idx);
    if (fileData?.has_ms && integrationResults?.peaks?.[idx]) {
      const rt = integrationResults.peaks[idx].retention_time;
      handleExtractSpectrum(rt);
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

  // Scaling factors
  const handleScalingApply = (signal, area) => {
    setScalingFactors({ signal, area });
    api.setScalingFactors(signal, area).catch(() => {});
  };

  // Edit assignment
  const handleAssign = (compound) => {
    if (selectedPeakIndex == null || !integrationResults) return;
    const peaks = [...integrationResults.peaks];
    peaks[selectedPeakIndex] = { ...peaks[selectedPeakIndex], compound_name: compound.name };
    setIntegrationResults({ ...integrationResults, peaks });
  };

  return (
    <div className="app">
      <Header apiStatus={apiStatus} />

      <div className="main-content">
        {/* Sidebar */}
        <aside className="sidebar">
          <FileBrowser onFileSelect={handleFileSelect} />

          {/* RT Table Manager */}
          <RTTableManager settings={rtSettings} onSettingsChange={setRtSettings} />

          {/* Quantitation Panel */}
          <QuantitationPanel
            enabled={quantEnabled}
            onToggle={setQuantEnabled}
            settings={quantSettings}
            onSettingsChange={setQuantSettings}
            onRequantitate={() => { /* TODO: call /api/quantitate */ }}
            peaks={integrationResults?.peaks}
            calculatedValues={quantCalc}
          />
        </aside>

        {/* Main Panel */}
        <main className="main-panel">
          {/* Error */}
          {error && (
            <div className="status-indicator error" style={{ cursor: 'pointer' }} onClick={() => setError(null)}>
              ⚠️ {error} <span style={{ marginLeft: '0.5rem', opacity: 0.6 }}>✕</span>
            </div>
          )}

          {/* Toolbar */}
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
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
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
                    {/* Navigation */}
                    <button className="btn btn-sm btn-secondary" onClick={() => handleNavigate('previous')} title="Previous sample">◀</button>
                    <button className="btn btn-sm btn-secondary" onClick={() => handleNavigate('next')} title="Next sample">▶</button>
                    {/* Toolbar buttons */}
                    <button className="btn btn-sm btn-secondary" onClick={() => setShowScaling(true)} title="Scaling Factors">⚖️</button>
                    <button className="btn btn-sm btn-secondary" onClick={() => setShowExportSettings(true)} title="Export Settings">📁</button>
                    {fileData?.has_ms && (
                      <button className="btn btn-sm btn-secondary" onClick={() => setShowMSOptions(true)} title="MS Options">🔬</button>
                    )}
                    {selectedPeakIndex != null && (
                      <button className="btn btn-sm btn-secondary" onClick={() => setShowEditAssign(true)} title="Edit Assignment">🏷️</button>
                    )}
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
            ticData={fileData?.tic}
            showCorrectedSignal={currentParams?.baseline?.show_corrected || false}
            showBaseline={true}
            onPeakClick={(idx) => handlePeakSelect(idx)}
            selectedPeakIndex={selectedPeakIndex}
          />

          {/* MS Spectrum Viewer */}
          {fileData?.has_ms && (
            <MSSpectrumViewer
              spectrum={spectrum}
              searchResults={msSearchResults}
              selectedPeakRT={integrationResults?.peaks?.[selectedPeakIndex]?.retention_time}
              hasMS={fileData?.has_ms}
              onExtractSpectrum={handleExtractSpectrum}
              onSearchSpectrum={handleSearchSpectrum}
              searching={msSearching}
              disabled={loading}
            />
          )}

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
              onPeakClick={(idx) => handlePeakSelect(idx)}
              selectedPeakIndex={selectedPeakIndex}
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

      {/* ── Dialogs ── */}
      <ScalingFactorsDialog
        open={showScaling}
        onClose={() => setShowScaling(false)}
        onApply={handleScalingApply}
        initialSignal={scalingFactors.signal}
        initialArea={scalingFactors.area}
      />
      <ExportSettingsDialog
        open={showExportSettings}
        onClose={() => setShowExportSettings(false)}
      />
      <EditAssignmentDialog
        open={showEditAssign}
        onClose={() => setShowEditAssign(false)}
        onAssign={handleAssign}
        peak={integrationResults?.peaks?.[selectedPeakIndex]}
        compoundList={[]}
      />
      <MSOptionsDialog
        open={showMSOptions}
        onClose={() => setShowMSOptions(false)}
        onApply={(opts) => setMsOptions(opts)}
        initialOptions={msOptions}
      />
    </div>
  );
}

export default App;
