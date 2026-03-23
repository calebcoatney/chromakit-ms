/**
 * MSOptionsDialog — multi-tab MS library search configuration.
 */
import React, { useState, useEffect } from 'react';

const STORAGE_KEY = 'chromakit-ms-options';

const DEFAULTS = {
  searchMethod: 'vector',
  hybridMethod: 'auto',
  fullMsBaseline: false,
  extractionMethod: 'apex',
  rangePoints: 5,
  midpointWindow: 50,
  weightByTic: false,
  backgroundEnabled: false,
  subtractionMethod: 'left_bound',
  subtractionWeight: 0.5,
  intensityThreshold: 0.01,
  similarity: 'cosine',
  weighting: 'nist',
  unmatchedPeaks: 'keep_all',
  preselector: 'kmeans',
  clusters: 3,
  intensityPower: 0.5,
  topN: 10,
  qualityEnabled: false,
  checkAsymmetry: true,
  checkCoherence: true,
  skewnessThreshold: 1.0,
  coherenceThreshold: 0.8,
  minHighCorrelations: 0.5,
};

function loadOptions() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') }; }
  catch { return { ...DEFAULTS }; }
}

const MSOptionsDialog = ({ open, onClose, onApply, initialOptions }) => {
  const [tab, setTab] = useState('general');
  const [opts, setOpts] = useState(() => ({ ...DEFAULTS, ...initialOptions }));

  useEffect(() => {
    if (open) setOpts(initialOptions ? { ...DEFAULTS, ...initialOptions } : loadOptions());
  }, [open]);

  if (!open) return null;

  const u = (k, v) => setOpts(p => ({ ...p, [k]: v }));

  const handleApply = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(opts));
    onApply(opts);
    onClose();
  };

  const tabs = [
    ['general', 'General'],
    ['extraction', 'Extraction'],
    ['subtraction', 'Background'],
    ['algorithm', 'Algorithm'],
    ['quality', 'Quality'],
  ];

  const inputRow = (label, key, props = {}) => (
    <div className="form-group" key={key}>
      <label className="form-label">{label}</label>
      <input className="form-control" value={opts[key]}
        onChange={e => u(key, props.type === 'number' ? parseFloat(e.target.value) : e.target.value)}
        {...props} />
    </div>
  );

  const selectRow = (label, key, options) => (
    <div className="form-group" key={key}>
      <label className="form-label">{label}</label>
      <select className="form-control" value={opts[key]} onChange={e => u(key, e.target.value)}>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </div>
  );

  const checkRow = (label, key) => (
    <label key={key} style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', cursor: 'pointer', marginBottom: '0.3rem', fontSize: '0.8rem' }}>
      <input type="checkbox" checked={opts[key]} onChange={e => u(key, e.target.checked)} />
      {label}
    </label>
  );

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '500px' }}>
        <div className="modal-header">
          <h3>MS Search Options</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>

        <div className="modal-tab-bar">
          {tabs.map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)} className={tab === id ? 'active' : ''}>
              {label}
            </button>
          ))}
        </div>

        <div className="modal-body" style={{ minHeight: '250px' }}>
          {tab === 'general' && <>
            {selectRow('Search Method', 'searchMethod', [['vector', 'Vector'], ['word2vec', 'Word2Vec'], ['hybrid', 'Hybrid']])}
            {opts.searchMethod === 'hybrid' && selectRow('Hybrid Method', 'hybridMethod', [['auto', 'Auto'], ['fast', 'Fast'], ['ensemble', 'Ensemble']])}
            {checkRow('Full MS Baseline Correction', 'fullMsBaseline')}
          </>}

          {tab === 'extraction' && <>
            {selectRow('Extraction Method', 'extractionMethod', [
              ['apex', 'Apex'], ['average', 'Average'], ['range', 'Range'], ['midpoint', 'Midpoint'],
            ])}
            {opts.extractionMethod === 'range' && inputRow('Points on Each Side', 'rangePoints', { type: 'number', min: 1, max: 50, step: 1 })}
            {opts.extractionMethod === 'midpoint' && inputRow('Window Width (%)', 'midpointWindow', { type: 'number', min: 1, max: 100, step: 1 })}
            {checkRow('Weight by TIC Intensity', 'weightByTic')}
          </>}

          {tab === 'subtraction' && <>
            {checkRow('Enable Background Subtraction', 'backgroundEnabled')}
            {opts.backgroundEnabled && <>
              {selectRow('Subtraction Method', 'subtractionMethod', [
                ['left_bound', 'Left Bound'], ['right_bound', 'Right Bound'],
                ['min_tic', 'Min TIC'], ['average_bounds', 'Average Bounds'],
              ])}
              {inputRow('Subtraction Weight', 'subtractionWeight', { type: 'number', min: 0.01, max: 1.0, step: 0.01 })}
              {inputRow('Intensity Threshold (%)', 'intensityThreshold', { type: 'number', min: 0.001, max: 0.2, step: 0.001 })}
            </>}
          </>}

          {tab === 'algorithm' && <>
            {selectRow('Similarity Measure', 'similarity', [['cosine', 'Cosine'], ['composite', 'Composite']])}
            {selectRow('Weighting Scheme', 'weighting', [['none', 'None'], ['nist', 'NIST'], ['nist_gc', 'NIST GC']])}
            {selectRow('Unmatched Peaks', 'unmatchedPeaks', [
              ['keep_all', 'Keep All'], ['remove_all', 'Remove All'],
              ['keep_library', 'Keep Library'], ['keep_experimental', 'Keep Experimental'],
            ])}
            {selectRow('Preselector Type', 'preselector', [['kmeans', 'K-means'], ['gmm', 'GMM']])}
            {inputRow('Clusters to Consider', 'clusters', { type: 'number', min: 1, max: 10, step: 1 })}
            {inputRow('Intensity Power', 'intensityPower', { type: 'number', min: 0.1, max: 1.0, step: 0.1 })}
            {inputRow('Top N Results', 'topN', { type: 'number', min: 1, max: 50, step: 1 })}
          </>}

          {tab === 'quality' && <>
            {checkRow('Enable Peak Quality Checks', 'qualityEnabled')}
            {opts.qualityEnabled && <>
              {checkRow('Check Peak Asymmetry', 'checkAsymmetry')}
              {checkRow('Check Spectral Coherence', 'checkCoherence')}
              {inputRow('Skewness Threshold', 'skewnessThreshold', { type: 'number', min: 0.1, max: 2.0, step: 0.1 })}
              {inputRow('Coherence Threshold', 'coherenceThreshold', { type: 'number', min: 0.5, max: 0.95, step: 0.05 })}
              {inputRow('Min % High Correlations', 'minHighCorrelations', { type: 'number', min: 0.1, max: 0.9, step: 0.05 })}
            </>}
          </>}
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={() => setOpts({ ...DEFAULTS })}>Restore Defaults</button>
          <div style={{ display: 'flex', gap: '0.375rem' }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleApply}>Apply</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MSOptionsDialog;
