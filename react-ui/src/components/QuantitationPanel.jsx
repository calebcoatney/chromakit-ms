/**
 * QuantitationPanel — Polyarc + internal standard quantitation.
 */
import React, { useState, useEffect, useCallback } from 'react';

const DEFAULTS = {
  compoundName: '',
  formula: '',
  mw: 0,
  density: 0,
  volumeAdded: 0,
  sampleVolume: 0,
  sampleDensity: 0,
  overwriteExisting: false,
};

const QuantitationPanel = ({
  enabled = false, onToggle, settings: externalSettings,
  onSettingsChange, onRequantitate, peaks = [], calculatedValues = {},
}) => {
  const [settings, setSettings] = useState({ ...DEFAULTS, ...externalSettings });

  useEffect(() => {
    if (externalSettings) setSettings(prev => ({ ...prev, ...externalSettings }));
  }, [externalSettings]);

  const update = useCallback((key, val) => {
    setSettings(prev => {
      const next = { ...prev, [key]: val };
      onSettingsChange?.(next);
      return next;
    });
  }, [onSettingsChange]);

  const { molCIS, sampleMass, responseFactor, carbonBalance } = calculatedValues;

  const inputRow = (label, key, props = {}) => (
    <div className="form-group" key={key}>
      <label className="form-label">{label}</label>
      <input className="form-control" value={settings[key]}
        onChange={e => update(key, props.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)}
        disabled={!enabled} {...props} />
    </div>
  );

  const readonlyRow = (label, value) => (
    <div className="form-group" key={label}>
      <label className="form-label">{label}</label>
      <div className="readonly-field">{value ?? '\u2014'}</div>
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="card-header">
        <h3>Quantitation</h3>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', cursor: 'pointer', fontSize: '0.8rem' }}>
          <input type="checkbox" checked={enabled} onChange={e => onToggle?.(e.target.checked)} />
          Enable
        </label>
      </div>
      <div className="card-body" style={{ opacity: enabled ? 1 : 0.5, pointerEvents: enabled ? 'auto' : 'none' }}>
        <h4 className="modal-body h4" style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.375rem' }}>Internal Standard</h4>
        {inputRow('Compound Name', 'compoundName')}
        <div className="grid-2">
          {inputRow('Formula', 'formula')}
          {inputRow('Molecular Weight', 'mw', { type: 'number', step: 0.01, min: 0 })}
          {inputRow('Density (g/mL)', 'density', { type: 'number', step: 0.001, min: 0 })}
          {inputRow('Volume Added (\u00B5L)', 'volumeAdded', { type: 'number', step: 0.1, min: 0 })}
        </div>

        <div className="divider" />
        <h4 style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.375rem' }}>Sample</h4>
        <div className="grid-2">
          {inputRow('Sample Volume (\u00B5L)', 'sampleVolume', { type: 'number', step: 0.1, min: 0 })}
          {inputRow('Sample Density (g/mL)', 'sampleDensity', { type: 'number', step: 0.001, min: 0 })}
        </div>

        <div className="divider" />
        <h4 style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.375rem' }}>Calculated Values</h4>
        <div className="grid-2">
          {readonlyRow('mol C of IS', molCIS != null ? molCIS.toExponential(4) : null)}
          {readonlyRow('Sample Mass (mg)', sampleMass != null ? sampleMass.toFixed(4) : null)}
          {readonlyRow('Response Factor', responseFactor != null ? responseFactor.toExponential(4) : null)}
          {readonlyRow('Carbon Balance (%)', carbonBalance != null ? carbonBalance.toFixed(2) : null)}
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', cursor: 'pointer', margin: '0.5rem 0', fontSize: '0.8rem' }}>
          <input type="checkbox" checked={settings.overwriteExisting}
            onChange={e => update('overwriteExisting', e.target.checked)} />
          Overwrite existing results
        </label>

        <button className="btn btn-primary full-width" onClick={onRequantitate}
          disabled={!enabled || !peaks?.length}>
          Re-Quantitate
        </button>
      </div>
    </div>
  );
};

export default QuantitationPanel;
