/**
 * Main App Component
 *
 * 3-column layout mirroring the desktop app:
 *   Left: File browser
 *   Center: Plot + button bar + results table
 *   Right: Tabbed panels (Parameters, MS, RT Table, Quantitation)
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

const RIGHT_TABS = [
  { id: 'params', label: '⚙️ Parameters' },
  { id: 'ms', label: '🔬 MS' },
  { id: 'rt', label: '📋 RT Table' },
  { id: 'quant', label: '⚗️ Quantitation' },
];

function App() {
  // Core state
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

  // Right panel tab
  const [activeTab, setActiveTab] = useState('params');

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

  // ── Health check ──
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

  // ── File selection ──
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

  // ── Detector change ──
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

  // ── Process ──
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
        fileData.chromatogram.x, fileData.chromatogram.y, params, msRange,
      );
      setProcessedData(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to process chromatogram');
    } finally {
      setProcessing(false);
    }
  }, [fileData, processing]);

  // ── Integrate ──
  const handleIntegrate = async () => {
    if (!processedData) return;
    setIntegrationResults(null);
    setError(null);
    setLoading(true);
    try {
      const peakGroups = currentParams?.integration?.peak_groups;
      const data = await api.integratePeaks(
        processedData, 0.0784, undefined,
        peakGroups?.length ? peakGroups : undefined,
      );
      setIntegrationResults(data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to integrate peaks');
    } finally {
      setLoading(false);
    }
  };

  // ── Spectrum ──
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

  // ── Peak selection ──
  const handlePeakSelect = (idx) => {
    setSelectedPeakIndex(idx);
    if (fileData?.has_ms && integrationResults?.peaks?.[idx]) {
      const rt = integrationResults.peaks[idx].retention_time;
      handleExtractSpectrum(rt);
    }
  };

  // ── Navigation ──
  const handleNavigate = async (direction) => {
    setError(null);
    try {
      const resp = direction === 'next'
        ? await fetch('/api/navigate/next').then(r => r.json())
        : await fetch('/api/navigate/previous').then(r => r.json());
      if (resp.file_path) {
        handleFileSelect({ name: resp.file_path.split('/').pop(), path: resp.file_path });
      }
    } catch {
      setError('Navigation failed');
    }
  };

  // ── Scaling ──
  const handleScalingApply = (signal, area) => {
    setScalingFactors({ signal, area });
    api.setScalingFactors(signal, area).catch(() => {});
  };

  // ── Assignment ──
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
        {/* ═══ LEFT: File Browser ═══ */}
        <aside className="sidebar-left">
          <FileBrowser onFileSelect={handleFileSelect} />
        </aside>

        {/* ═══ CENTER: Plot + Buttons + Results ═══ */}
        <main className="center-panel">
          {/* Error banner */}
          {error && (
            <div className="status-indicator error" style={{ cursor: 'pointer' }} onClick={() => setError(null)}>
              ⚠️ {error} <span style={{ marginLeft: '0.5rem', opacity: 0.6 }}>✕</span>
            </div>
          )}

          {/* File info */}
          {selectedFile && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.25rem 0', fontSize: '0.875rem', flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 600 }}>📄 {selectedFile.name}</span>
              {fileData && (
                <span className="text-muted">
                  {fileData.chromatogram.x.length} pts
                  {currentDetector && ` · ${currentDetector}`}
                  {fileData.has_ms && ` · MS`}
                </span>
              )}
              {availableDetectors.length > 1 && (
                <select className="form-control" style={{ width: 'auto', minWidth: '90px', padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
                  value={currentDetector || ''} onChange={e => handleDetectorChange(e.target.value)}>
                  {availableDetectors.map(d => <option key={d} value={d}>{d}</option>)}
                </select>
              )}
              {loading && <div className="spinner" style={{ width: '1.2rem', height: '1.2rem', borderWidth: '2px' }}></div>}
            </div>
          )}

          {/* Chromatogram Plot */}
          <ChromatogramPlot
            data={fileData?.chromatogram}
            processedData={processedData}
            integrationResults={integrationResults}
            ticData={fileData?.tic}
            showCorrectedSignal={currentParams?.baseline?.show_corrected || false}
            showBaseline={true}
            onPeakClick={handlePeakSelect}
            selectedPeakIndex={selectedPeakIndex}
          />

          {/* Button bar (mirrors desktop ButtonFrame) */}
          {selectedFile && (
            <div className="button-bar">
              <button className="btn btn-sm btn-secondary" onClick={() => setShowExportSettings(true)} title="Export Settings">📁 Export</button>
              <button className="btn btn-sm btn-secondary" onClick={() => handleNavigate('previous')} title="Previous sample">◀ Back</button>
              <button className="btn btn-sm btn-secondary" onClick={() => handleNavigate('next')} title="Next sample">Next ▶</button>
              <button className="btn btn-sm btn-success" onClick={handleIntegrate}
                disabled={!processedData || loading || !processedData?.peaks_x?.length}
                title="Integrate detected peaks">
                {loading ? '⏳' : '📈'} Integrate
              </button>
              <div className="spacer" />
              <button className="btn btn-sm btn-secondary" onClick={() => setShowScaling(true)} title="Scaling Factors">⚖️</button>
              {fileData?.has_ms && (
                <button className="btn btn-sm btn-secondary" onClick={() => setShowMSOptions(true)} title="MS Options">🔬</button>
              )}
              {selectedPeakIndex != null && (
                <button className="btn btn-sm btn-secondary" onClick={() => setShowEditAssign(true)} title="Edit Assignment">🏷️</button>
              )}
            </div>
          )}

          {/* Results table */}
          {integrationResults && (
            <PeakTable
              integrationResults={integrationResults}
              onIntegrate={handleIntegrate}
              onPeakClick={handlePeakSelect}
              selectedPeakIndex={selectedPeakIndex}
              disabled={loading}
            />
          )}

          {/* Welcome screen */}
          {!selectedFile && !loading && (
            <div className="card" style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div className="card-body text-center" style={{ padding: '3rem' }}>
                <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>📁</div>
                <h3 style={{ marginBottom: '0.5rem' }}>Welcome to ChromaKit-MS</h3>
                <p className="text-muted">Select a .D file from the browser to begin analysis</p>
              </div>
            </div>
          )}
        </main>

        {/* ═══ RIGHT: Tabbed Panels ═══ */}
        <aside className="sidebar-right">
          <div className="tab-bar">
            {RIGHT_TABS.map(t => (
              <button key={t.id}
                className={activeTab === t.id ? 'active' : ''}
                onClick={() => setActiveTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>
          <div className="tab-content">
            {activeTab === 'params' && (
              <ProcessingControls
                onParametersChange={handleParametersChange}
                disabled={processing}
              />
            )}
            {activeTab === 'ms' && (
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
            {activeTab === 'rt' && (
              <RTTableManager settings={rtSettings} onSettingsChange={setRtSettings} />
            )}
            {activeTab === 'quant' && (
              <QuantitationPanel
                enabled={quantEnabled}
                onToggle={setQuantEnabled}
                settings={quantSettings}
                onSettingsChange={setQuantSettings}
                onRequantitate={() => { /* TODO: /api/quantitate */ }}
                peaks={integrationResults?.peaks}
                calculatedValues={quantCalc}
              />
            )}
          </div>
        </aside>
      </div>

      {/* ── Dialogs ── */}
      <ScalingFactorsDialog open={showScaling} onClose={() => setShowScaling(false)}
        onApply={handleScalingApply} initialSignal={scalingFactors.signal} initialArea={scalingFactors.area} />
      <ExportSettingsDialog open={showExportSettings} onClose={() => setShowExportSettings(false)} />
      <EditAssignmentDialog open={showEditAssign} onClose={() => setShowEditAssign(false)}
        onAssign={handleAssign} peak={integrationResults?.peaks?.[selectedPeakIndex]} compoundList={[]} />
      <MSOptionsDialog open={showMSOptions} onClose={() => setShowMSOptions(false)}
        onApply={opts => setMsOptions(opts)} initialOptions={msOptions} />
    </div>
  );
}

export default App;
