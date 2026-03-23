/**
 * RTTableManager — load, view, and configure RT table matching.
 */
import React, { useState, useCallback } from 'react';

const MATCHING_MODES = [
  ['simple_window', 'Simple Window'],
  ['closest_apex', 'Closest Apex'],
  ['weighted_distance', 'Weighted Distance'],
];

const DEFAULTS = {
  enabled: false,
  highPriority: false,
  matchingMode: 'closest_apex',
  tolerance: 0.5,
  weightStart: 0.2,
  weightApex: 0.6,
  weightEnd: 0.2,
  windowExpansion: 0.0,
};

const RTTableManager = ({ onSettingsChange, settings: externalSettings }) => {
  const [settings, setSettings] = useState({ ...DEFAULTS, ...externalSettings });
  const [rtTable, setRtTable] = useState(null);
  const [fileName, setFileName] = useState('');

  const update = useCallback((key, val) => {
    setSettings(prev => {
      const next = { ...prev, [key]: val };
      onSettingsChange?.({ ...next, rtTable });
      return next;
    });
  }, [onSettingsChange, rtTable]);

  const normalizeWeights = (changed, value) => {
    const keys = ['weightStart', 'weightApex', 'weightEnd'];
    const others = keys.filter(k => k !== changed);
    const remaining = Math.max(0, 1 - value);
    const otherSum = settings[others[0]] + settings[others[1]] || 1;

    setSettings(prev => {
      const next = {
        ...prev,
        [changed]: value,
        [others[0]]: (prev[others[0]] / otherSum) * remaining,
        [others[1]]: (prev[others[1]] / otherSum) * remaining,
      };
      onSettingsChange?.({ ...next, rtTable });
      return next;
    });
  };

  const handleLoadFile = async () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv,.json,.xlsx,.xls';
    input.onchange = async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setFileName(file.name);

      const text = await file.text();
      try {
        let parsed;
        if (file.name.endsWith('.json')) {
          parsed = JSON.parse(text);
          if (Array.isArray(parsed)) {
            const columns = Object.keys(parsed[0] || {});
            setRtTable({ columns, rows: parsed });
          }
        } else {
          const lines = text.trim().split('\n');
          const columns = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
          const rows = lines.slice(1).map(line => {
            const vals = line.split(',').map(v => v.trim().replace(/^"|"$/g, ''));
            const row = {};
            columns.forEach((col, i) => { row[col] = isNaN(vals[i]) ? vals[i] : parseFloat(vals[i]); });
            return row;
          });
          setRtTable({ columns, rows });
        }
        update('enabled', true);
      } catch (err) {
        console.error('Failed to parse RT table:', err);
        alert('Failed to parse RT table file. Ensure it is valid CSV or JSON.');
      }
    };
    input.click();
  };

  const handleClear = () => {
    setRtTable(null);
    setFileName('');
    update('enabled', false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="card-header">
        <h3>RT Table</h3>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', cursor: 'pointer', fontSize: '0.8rem' }}>
          <input type="checkbox" checked={settings.enabled}
            onChange={e => update('enabled', e.target.checked)} disabled={!rtTable} />
          Enable Matching
        </label>
      </div>
      <div className="card-body">
        <div style={{ display: 'flex', gap: '0.375rem', marginBottom: '0.5rem' }}>
          <button className="btn btn-primary" onClick={handleLoadFile} style={{ flex: 1 }}>
            Load RT Table
          </button>
          <button className="btn btn-secondary" onClick={handleClear} disabled={!rtTable}>
            Clear
          </button>
        </div>

        {fileName && (
          <p className="text-sm text-muted mb-2">
            Loaded: {fileName} ({rtTable?.rows?.length || 0} entries)
          </p>
        )}

        {rtTable && (
          <div style={{ maxHeight: '180px', overflowY: 'auto', marginBottom: '0.5rem', border: '1px solid var(--border-color)', borderRadius: '4px' }}>
            <table className="data-table">
              <thead>
                <tr>
                  {rtTable.columns.map(col => <th key={col}>{col}</th>)}
                </tr>
              </thead>
              <tbody>
                {rtTable.rows.map((row, i) => (
                  <tr key={i}>
                    {rtTable.columns.map(col => (
                      <td key={col}>
                        {typeof row[col] === 'number' ? row[col].toFixed(3) : row[col]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="divider" />
        <h4 style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.375rem' }}>Matching Settings</h4>

        <div className={settings.enabled ? '' : 'param-disabled'}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', cursor: 'pointer', marginBottom: '0.375rem', fontSize: '0.8rem' }}>
            <input type="checkbox" checked={settings.highPriority}
              onChange={e => update('highPriority', e.target.checked)} />
            High Priority (override MS assignments)
          </label>

          <div className="form-group">
            <label className="form-label">Matching Mode</label>
            <select className="form-control" value={settings.matchingMode}
              onChange={e => update('matchingMode', e.target.value)}>
              {MATCHING_MODES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Apex Tolerance (min)</label>
            <input type="number" className="form-control" value={settings.tolerance}
              onChange={e => update('tolerance', parseFloat(e.target.value) || 0.5)}
              step="0.01" min="0.01" max="5.0" />
          </div>

          {settings.matchingMode === 'weighted_distance' && (
            <>
              <div className="form-hint mb-1">Weights (sum = 1.0)</div>
              {[['weightStart', 'Start RT'], ['weightApex', 'Apex RT'], ['weightEnd', 'End RT']].map(([key, label]) => (
                <div key={key} style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', marginBottom: '0.25rem' }}>
                  <span style={{ fontSize: '0.75rem', width: '55px' }}>{label}</span>
                  <input type="range" min="0" max="1" step="0.05"
                    value={settings[key]}
                    onChange={e => normalizeWeights(key, parseFloat(e.target.value))}
                    style={{ flex: 1 }} />
                  <span style={{ fontSize: '0.75rem', width: '35px', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                    {settings[key].toFixed(2)}
                  </span>
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default RTTableManager;
