/**
 * EditAssignmentDialog — assign a compound identity to a peak.
 */
import React, { useState, useEffect, useMemo } from 'react';

const EditAssignmentDialog = ({
  open, onClose, onAssign, peak,
  compoundList = [], allFilesLoaded = false, onApplyToFiles,
}) => {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(null);
  const [applyToFiles, setApplyToFiles] = useState(false);
  const [tolerance, setTolerance] = useState(0.05);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.7);

  useEffect(() => {
    if (open) {
      setQuery('');
      setSelected(null);
      setApplyToFiles(false);
    }
  }, [open]);

  const filtered = useMemo(() => {
    if (query.length < 3) return [];
    const q = query.toLowerCase();
    return compoundList.filter(c => c.toLowerCase().includes(q)).slice(0, 50);
  }, [query, compoundList]);

  useEffect(() => {
    if (filtered.length === 1) {
      setSelected(filtered[0]);
    } else {
      const exact = filtered.find(c => c.toLowerCase() === query.toLowerCase());
      if (exact) setSelected(exact);
    }
  }, [filtered, query]);

  if (!open) return null;

  const handleApply = () => {
    if (!selected) return;
    onAssign({ name: selected });
    if (applyToFiles && onApplyToFiles && peak) {
      onApplyToFiles(selected, peak.rt, tolerance, similarityThreshold);
    }
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '460px' }}>
        <div className="modal-header">
          <h3>Edit Assignment</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          {peak && (
            <p className="text-muted text-sm mb-2">
              Peak at RT {peak.retention_time?.toFixed(3) ?? peak.rt?.toFixed(3)} min &mdash; Area: {peak.area?.toExponential(2)}
            </p>
          )}

          <div className="form-group">
            <label className="form-label">Search Compound (min 3 chars)</label>
            <input className="form-control" value={query}
              onChange={e => { setQuery(e.target.value); setSelected(null); }}
              placeholder="Type compound name..." autoFocus />
          </div>

          {filtered.length > 0 && !selected && (
            <div className="compound-list mb-2">
              {filtered.map(name => (
                <div key={name}
                  className={`compound-item${selected === name ? ' selected' : ''}`}
                  onClick={() => { setSelected(name); setQuery(name); }}
                >
                  {name}
                </div>
              ))}
            </div>
          )}

          {selected && (
            <div className="selected-indicator">
              Selected: <strong>{selected}</strong>
            </div>
          )}

          {allFilesLoaded && (
            <>
              <div className="divider" />
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', cursor: 'pointer', marginBottom: '0.375rem', fontSize: '0.8rem' }}>
                <input type="checkbox" checked={applyToFiles} onChange={e => setApplyToFiles(e.target.checked)} />
                Apply to other loaded files
              </label>
              {applyToFiles && (
                <div className="grid-2">
                  <div className="form-group">
                    <label className="form-label">RT Tolerance (min)</label>
                    <input type="number" className="form-control" value={tolerance}
                      onChange={e => setTolerance(parseFloat(e.target.value) || 0.05)}
                      step="0.01" min="0.01" max="1.0" />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Similarity Threshold</label>
                    <input type="number" className="form-control" value={similarityThreshold}
                      onChange={e => setSimilarityThreshold(parseFloat(e.target.value) || 0.7)}
                      step="0.05" min="0.1" max="1.0" />
                  </div>
                </div>
              )}
            </>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn btn-primary" onClick={handleApply} disabled={!selected}>
            Assign
          </button>
        </div>
      </div>
    </div>
  );
};

export default EditAssignmentDialog;
