# Cluster Click-to-Search — Design Spec

**Date:** 2026-03-26
**Status:** Draft

## Problem

The spectral deconvolution inspector shows DBSCAN clusters as colored scatter points, each corresponding to a `DeconvolutedComponent` with a reconstructed mass spectrum. Users want to click a cluster and immediately run an MS library search against that spectrum to identify the compound — without leaving the inspector dialog.

## Architecture

**Hybrid signal-based approach.** The inspector handles the click interaction and visual feedback, emits a signal carrying the spectrum, and the app layer performs the actual library search (using MSFrame's `ms_toolkit` and configured search settings), then sends results back to the inspector for inline display.

This keeps search logic in `ChromaKitApp` (consistent with existing search flows) while giving the inspector self-contained results display.

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `SpectralDeconvInspectorDialog` | Click detection, KDTree lookup, cluster highlighting, results display |
| `ChromaKitApp` | Signal handler, search execution via `ms_toolkit`, result routing |
| `MSFrame` | Provides `ms_toolkit` instance and search settings (read-only) |

## Interaction Design

### Click Handling

- `mpl_connect('button_press_event', _on_scatter_click)` registered in `_build_plot_panel()`.
- Click handler builds a `scipy.spatial.cKDTree` over the scatter plot's `(rt, mz)` coordinates (built once per render, stored as `_scatter_tree`).
- Nearest-point lookup with a pixel-distance tolerance (converted to data coords via axis transform). If the nearest point is too far, the click is ignored.
- The nearest point maps to a cluster index via a parallel `_scatter_cluster_ids` array (built during `_render_plots`).
- Cluster index maps to `_last_result['components'][cluster_idx]` for the spectrum.

### Visual Feedback

On cluster selection:
- **Selected cluster:** markers get black edge color, slightly increased size.
- **Other clusters:** alpha reduced to 0.3 (dimmed).
- **Model peak star** for the selected cluster stays prominent.
- Clicking empty space (no nearby point) or clicking the same cluster again clears the selection.

### Inline Results Panel

- `QTreeWidget` positioned below the matplotlib canvas.
- Three columns: **Rank**, **Compound Name**, **Match Score**.
- Header label: "Search results for component at RT = X.XXX min".
- Shows up to `top_n` results (from MSFrame search settings, typically 5).
- Initially empty with placeholder text: "Click a cluster to search".

## Signal Flow

```
User clicks scatter plot
  → _on_scatter_click(MouseEvent)
  → cKDTree nearest point lookup
  → if too far or noise point: show status message, return
  → cluster_idx from _scatter_cluster_ids[point_idx]
  → component = _last_result['components'][cluster_idx]
  → _highlight_cluster(cluster_idx)  # visual feedback
  → emit cluster_search_requested(spectrum: dict, rt: float)

ChromaKitApp._on_cluster_search_requested(spectrum_dict, rt)
  → check ms_frame.ms_toolkit exists
     → if not: QMessageBox warning "MS library not loaded"
  → query = [(mz, intensity) for mz, intensity in spectrum_dict.items()]
  → read search settings from ms_frame
  → call ms_toolkit.search_vector/w2v/hybrid(query, ...)
  → call _deconv_inspector.show_search_results(results, rt)

SpectralDeconvInspectorDialog.show_search_results(results, rt)
  → clear QTreeWidget
  → set header: "Results for component at RT = {rt:.3f} min"
  → populate rows: [(rank, name, f"{score:.4f}"), ...]
```

## Data Structures

### Stored after each render (`_render_plots`)

```python
self._last_result = result                    # full preview result dict
self._scatter_coords = np.array([[rt, mz], ...])  # all plotted scatter points
self._scatter_cluster_ids = [cluster_idx, ...]     # parallel array: cluster index per point (-1 for noise)
self._scatter_tree = cKDTree(self._scatter_coords) # spatial index for click lookup
```

### Component spectrum (already exists)

```python
component.spectrum  # dict {mz: intensity} — e.g., {28.0: 5432.1, 44.0: 12000.5, ...}
```

### Search results (from ms_toolkit)

```python
results  # list of (compound_name: str, match_score: float) tuples
```

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Click on noise point (gray) | Status message: "Noise point — not assigned to any component" |
| Click far from any point | Ignored (tolerance check) |
| No components in result | Click handler disabled; status: "No components found" |
| MS library not loaded | QMessageBox warning with instructions |
| Click during preview computation | Ignored (controls are already disabled during preview) |
| Inspector closed/reopened | `_last_result` cleared; click handler re-registered on next render |
| Cluster has no matching component | Status message explaining the mismatch (shouldn't happen in practice since components are 1:1 with clusters that have model peaks) |

## New Signal

```python
# SpectralDeconvInspectorDialog
cluster_search_requested = Signal(object, float)  # (spectrum_dict: dict, rt: float)
```

## Files Modified

| File | Changes |
|------|---------|
| `ui/dialogs/spectral_deconv_inspector.py` | Store `_last_result`, build KDTree + cluster ID arrays in `_render_plots`, add `_on_scatter_click`, `_highlight_cluster`, `show_search_results`, results QTreeWidget, new signal |
| `ui/app.py` | Connect `cluster_search_requested` signal, implement `_on_cluster_search_requested` handler |

## Out of Scope

- Batch searching all clusters at once (future feature).
- Displaying the component spectrum as a bar chart in the inspector (could be added later).
- Modifying search settings from within the inspector.
