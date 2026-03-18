/**
 * ExportSettingsDialog — configure export triggers and file formats.
 *
 * Props:
 *   open (bool)
 *   onClose ()
 */
import React, { useState, useEffect } from 'react';

const SETTINGS_KEY = 'chromakit-export-settings';

const DEFAULTS = {
  afterIntegration: true,
  afterMsSearch: true,
  afterAssignment: true,
  afterBatch: true,
  jsonEnabled: true,
  csvEnabled: true,
  jsonFilenameFormat: '{filename}_integration_results.json',
  csvFilenameFormat: '{filename}_integration_results.csv',
};

const JSON_FORMATS = [
  '{filename}_integration_results.json',
  '{filename}_{timestamp}.json',
  '{filename}_{detector}.json',
];
const CSV_FORMATS = [
  '{filename}_integration_results.csv',
  '{filename}_{timestamp}.csv',
  '{filename}_{detector}.csv',
];

function loadSettings() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') };
  } catch { return { ...DEFAULTS }; }
}

function persistSettings(s) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}

const ExportSettingsDialog = ({ open, onClose }) => {
  const [settings, setSettings] = useState(loadSettings);

  useEffect(() => {
    if (open) setSettings(loadSettings());
  }, [open]);

  if (!open) return null;

  const update = (key, val) => setSettings(prev => ({ ...prev, [key]: val }));

  const handleSave = () => {
    persistSettings(settings);
    onClose();
  };

  const handleRestore = () => setSettings({ ...DEFAULTS });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3>📁 Export Settings</h3>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <h4>Auto-Export Triggers</h4>
          <div style={{ display: 'grid', gap: '0.4rem', marginBottom: '1rem' }}>
            {[
              ['afterIntegration', 'After Integration'],
              ['afterMsSearch', 'After MS Search'],
              ['afterAssignment', 'After Assignment'],
              ['afterBatch', 'After Batch Processing'],
            ].map(([key, label]) => (
              <label key={key} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                <input type="checkbox" checked={settings[key]} onChange={e => update(key, e.target.checked)} />
                {label}
              </label>
            ))}
          </div>

          <h4>Export Formats</h4>
          <div style={{ display: 'grid', gap: '0.4rem', marginBottom: '1rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input type="checkbox" checked={settings.jsonEnabled} onChange={e => update('jsonEnabled', e.target.checked)} />
              JSON
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
              <input type="checkbox" checked={settings.csvEnabled} onChange={e => update('csvEnabled', e.target.checked)} />
              CSV
            </label>
          </div>

          <h4>Filename Formats</h4>
          <div className="form-group">
            <label>JSON Filename</label>
            <select className="form-control" value={settings.jsonFilenameFormat}
              onChange={e => update('jsonFilenameFormat', e.target.value)}>
              {JSON_FORMATS.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>CSV Filename</label>
            <select className="form-control" value={settings.csvFilenameFormat}
              onChange={e => update('csvFilenameFormat', e.target.value)}>
              {CSV_FORMATS.map(f => <option key={f} value={f}>{f}</option>)}
            </select>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={handleRestore}>Restore Defaults</button>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
            <button className="btn btn-primary" onClick={handleSave}>Save</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ExportSettingsDialog;
