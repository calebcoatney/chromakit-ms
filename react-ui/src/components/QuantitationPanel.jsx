/**
 * QuantitationPanel — Polyarc + internal standard quantitation.
 *
 * Props:
 *   enabled (bool)
 *   onToggle (bool)
 *   settings (object)
 *   onSettingsChange (settings)
 *   onRequantitate ()
 *   peaks (array) — current integration results
 *   calculatedValues (object) — { molCIS, sampleMass, responseFactor, carbonBalance }
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

  // Derived calculations
  const molCIS = calculatedValues.molCIS;
  const sampleMass = calculatedValues.sampleMass;
  const responseFactor = calculatedValues.responseFactor;
  const carbonBalance = calculatedValues.carbonBalance;

  const inputRow = (label, key, props = {}) => (
    <div className="form-group" key={key}>
      <label>{label}</label>
      <input className="form-control" value={settings[key]}
        onChange={e => update(key, props.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value)}
        disabled={!enabled} {...props} />
    </div>
  );

  const readonlyRow = (label, value) => (
    <div className="form-group" key={label}>
      <label>{label}</label>
      <div style={{
        padding: '0.5rem 0.75rem', background: 'var(--card-bg)',
        border: '1px solid var(--border-color)', borderRadius: '6px',
        fontSize: '0.875rem', color: 'var(--text-muted)',
      }}>
        {value ?? '—'}
      </div>
    </div>
  );

  return (
    <div className="card">
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ margin: 0, fontSize: '1rem' }}>⚗️ Quantitation</h3>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.85rem' }}>
          <input type="checkbox" checked={enabled} onChange={e => onToggle?.(e.target.checked)} />
          Enable
        </label>
      </div>
      <div className="card-body" style={{ opacity: enabled ? 1 : 0.5, pointerEvents: enabled ? 'auto' : 'none' }}>
        <h4 style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>Internal Standard</h4>
        {inputRow('Compound Name', 'compoundName')}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
          {inputRow('Formula', 'formula')}
          {inputRow('Molecular Weight', 'mw', { type: 'number', step: 0.01, min: 0 })}
          {inputRow('Density (g/mL)', 'density', { type: 'number', step: 0.001, min: 0 })}
          {inputRow('Volume Added (µL)', 'volumeAdded', { type: 'number', step: 0.1, min: 0 })}
        </div>

        <hr style={{ margin: '0.75rem 0', borderColor: 'var(--border-color)' }} />
        <h4 style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>Sample</h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
          {inputRow('Sample Volume (µL)', 'sampleVolume', { type: 'number', step: 0.1, min: 0 })}
          {inputRow('Sample Density (g/mL)', 'sampleDensity', { type: 'number', step: 0.001, min: 0 })}
        </div>

        <hr style={{ margin: '0.75rem 0', borderColor: 'var(--border-color)' }} />
        <h4 style={{ fontSize: '0.9rem', marginBottom: '0.5rem' }}>Calculated Values</h4>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
          {readonlyRow('mol C of IS', molCIS != null ? molCIS.toExponential(4) : null)}
          {readonlyRow('Sample Mass (mg)', sampleMass != null ? sampleMass.toFixed(4) : null)}
          {readonlyRow('Response Factor', responseFactor != null ? responseFactor.toExponential(4) : null)}
          {readonlyRow('Carbon Balance (%)', carbonBalance != null ? carbonBalance.toFixed(2) : null)}
        </div>

        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', margin: '0.75rem 0' }}>
          <input type="checkbox" checked={settings.overwriteExisting}
            onChange={e => update('overwriteExisting', e.target.checked)} />
          Overwrite existing results
        </label>

        <button className="btn btn-primary" onClick={onRequantitate}
          disabled={!enabled || !peaks?.length} style={{ width: '100%' }}>
          Re-Quantitate
        </button>
      </div>
    </div>
  );
};

export default QuantitationPanel;
