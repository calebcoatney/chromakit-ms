/**
 * EditAssignmentDialog — assign a compound identity to a peak.
 *
 * Props:
 *   open (bool)
 *   onClose ()
 *   onAssign (compound: { name, cas, formula })
 *   peak (object with rt, area, etc.)
 *   compoundList (string[] — all known compound names for autocomplete)
 *   allFilesLoaded (bool — show cross-file options)
 *   onApplyToFiles (compoundName, rt, tolerance, similarityThreshold)
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

  // Auto-select on exact or single match
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
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '500px' }}>
        <div className="modal-header">
          <h3>🏷️ Edit Assignment</h3>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          {peak && (
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.75rem' }}>
              Peak at RT {peak.rt?.toFixed(3)} min — Area: {peak.area?.toExponential(2)}
            </p>
          )}

          <div className="form-group">
            <label>Search Compound (min 3 chars)</label>
            <input className="form-control" value={query}
              onChange={e => { setQuery(e.target.value); setSelected(null); }}
              placeholder="Type compound name…" autoFocus />
          </div>

          {filtered.length > 0 && !selected && (
            <div style={{
              maxHeight: '200px', overflowY: 'auto', border: '1px solid var(--border-color)',
              borderRadius: '6px', marginBottom: '0.75rem',
            }}>
              {filtered.map(name => (
                <div key={name}
                  onClick={() => { setSelected(name); setQuery(name); }}
                  style={{
                    padding: '0.4rem 0.75rem', cursor: 'pointer',
                    background: selected === name ? 'var(--accent-color)' : 'transparent',
                    color: selected === name ? 'white' : 'var(--text-color)',
                  }}
                  onMouseEnter={e => e.target.style.background = selected === name ? 'var(--accent-color)' : 'var(--hover-bg)'}
                  onMouseLeave={e => e.target.style.background = selected === name ? 'var(--accent-color)' : 'transparent'}
                >
                  {name}
                </div>
              ))}
            </div>
          )}

          {selected && (
            <div style={{
              padding: '0.5rem 0.75rem', background: 'var(--success-bg, #f0fff4)',
              border: '1px solid var(--success-color, #48bb78)', borderRadius: '6px',
              marginBottom: '0.75rem', fontSize: '0.85rem',
            }}>
              ✅ Selected: <strong>{selected}</strong>
            </div>
          )}

          {allFilesLoaded && (
            <>
              <hr style={{ margin: '0.75rem 0', borderColor: 'var(--border-color)' }} />
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', marginBottom: '0.5rem' }}>
                <input type="checkbox" checked={applyToFiles} onChange={e => setApplyToFiles(e.target.checked)} />
                Apply to other loaded files
              </label>
              {applyToFiles && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                  <div className="form-group">
                    <label>RT Tolerance (min)</label>
                    <input type="number" className="form-control" value={tolerance}
                      onChange={e => setTolerance(parseFloat(e.target.value) || 0.05)}
                      step="0.01" min="0.01" max="1.0" />
                  </div>
                  <div className="form-group">
                    <label>Similarity Threshold</label>
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
