/**
 * ScalingFactorsDialog — configure signal/area scaling factors with presets.
 */
import React, { useState, useEffect } from 'react';

const PRESET_KEY = 'chromakit-scaling-presets';

function loadPresets() {
  try { return JSON.parse(localStorage.getItem(PRESET_KEY) || '{}'); }
  catch { return {}; }
}

function savePresets(presets) {
  localStorage.setItem(PRESET_KEY, JSON.stringify(presets));
}

const ScalingFactorsDialog = ({ open, onClose, onApply, initialSignal = 1.0, initialArea = 1.0 }) => {
  const [signalFactor, setSignalFactor] = useState(initialSignal);
  const [areaFactor, setAreaFactor] = useState(initialArea);
  const [presets, setPresets] = useState(loadPresets);
  const [selectedPreset, setSelectedPreset] = useState('');
  const [presetName, setPresetName] = useState('');

  useEffect(() => {
    if (open) {
      setSignalFactor(initialSignal);
      setAreaFactor(initialArea);
    }
  }, [open, initialSignal, initialArea]);

  if (!open) return null;

  const handleApply = () => { onApply(signalFactor, areaFactor); onClose(); };

  const handleSelectPreset = (name) => {
    setSelectedPreset(name);
    if (presets[name]) {
      setSignalFactor(presets[name].signal);
      setAreaFactor(presets[name].area);
    }
  };

  const handleSavePreset = () => {
    if (!presetName.trim()) return;
    const updated = { ...presets, [presetName.trim()]: { signal: signalFactor, area: areaFactor } };
    setPresets(updated);
    savePresets(updated);
    setSelectedPreset(presetName.trim());
    setPresetName('');
  };

  const handleDeletePreset = () => {
    if (!selectedPreset) return;
    const updated = { ...presets };
    delete updated[selectedPreset];
    setPresets(updated);
    savePresets(updated);
    setSelectedPreset('');
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Scaling Factors</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="form-group">
            <label className="form-label">Signal Factor</label>
            <input type="number" className="form-control" value={signalFactor}
              onChange={e => setSignalFactor(parseFloat(e.target.value) || 0)}
              step="0.000001" min="0" max="1e9" />
          </div>
          <div className="form-group">
            <label className="form-label">Area Factor</label>
            <input type="number" className="form-control" value={areaFactor}
              onChange={e => setAreaFactor(parseFloat(e.target.value) || 0)}
              step="0.000001" min="0" max="1e9" />
          </div>

          <div className="divider" style={{ margin: '0.75rem 0' }} />
          <h4>Presets</h4>

          <div className="form-group">
            <label className="form-label">Load Preset</label>
            <select className="form-control" value={selectedPreset}
              onChange={e => handleSelectPreset(e.target.value)}>
              <option value="">&mdash; Select &mdash;</option>
              {Object.keys(presets).map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>

          <div style={{ display: 'flex', gap: '0.375rem', alignItems: 'flex-end' }}>
            <div style={{ flex: 1 }}>
              <label className="form-label">Save as</label>
              <input className="form-control" value={presetName}
                onChange={e => setPresetName(e.target.value)} placeholder="Preset name" />
            </div>
            <button className="btn btn-primary btn-sm" onClick={handleSavePreset} disabled={!presetName.trim()}>Save</button>
            <button className="btn btn-secondary btn-sm" onClick={handleDeletePreset} disabled={!selectedPreset}>Delete</button>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={() => { setSignalFactor(1); setAreaFactor(1); }}>
            Restore Defaults
          </button>
          <div style={{ display: 'flex', gap: '0.375rem' }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleApply}>Apply</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScalingFactorsDialog;
