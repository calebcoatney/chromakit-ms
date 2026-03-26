# Cluster Click-to-Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Click a colored DBSCAN cluster in the inspector scatter plot to run an MS library search and display results inline.

**Architecture:** The inspector stores the preview result after each render and builds a KDTree spatial index over scatter coordinates. A matplotlib click handler maps clicks to clusters, highlights the selection, and emits a `cluster_search_requested` signal. `ChromaKitApp` catches it, runs the search via MSFrame's `ms_toolkit` and settings, then sends results back to the inspector's inline `QTreeWidget`.

**Tech Stack:** PySide6, matplotlib (`mpl_connect`, axis transforms), `scipy.spatial.cKDTree`, numpy

**Spec:** `docs/superpowers/specs/2026-03-26-cluster-click-search-design.md`

---

### Task 1: Store preview result and build spatial index in `_render_plots`

**Files:**
- Modify: `ui/dialogs/spectral_deconv_inspector.py` — `__init__`, `_render_plots`, import

After each preview finishes, the result dict needs to persist so the click handler can look up clusters. During rendering, build parallel arrays mapping each scatter point to its cluster index, and a KDTree for nearest-point lookup.

- [ ] **Step 1: Add scipy import and init fields in `__init__`**

At the top of the file, add the `cKDTree` import:

```python
from scipy.spatial import cKDTree
```

In `__init__`, after `self._preview_worker = None` (line 116), add:

```python
self._last_result: dict | None = None
self._scatter_coords: np.ndarray | None = None    # shape (N, 2): [[rt, mz], ...]
self._scatter_cluster_ids: list[int] = []          # parallel: cluster idx per point (-1 = noise)
self._scatter_tree: cKDTree | None = None
self._selected_cluster_idx: int | None = None
```

- [ ] **Step 2: Store result in `_on_preview_finished`**

In `_on_preview_finished` (line 490), before the `_render_plots` call, add:

```python
self._last_result = result
self._selected_cluster_idx = None  # clear selection on new preview
```

- [ ] **Step 3: Build spatial index at end of scatter rendering in `_render_plots`**

At the end of the `else` branch of the cluster-count guard (after the `self._ax_scatter.set_title(...)` call at line 603, before `self._ax_scatter.set_ylabel("m/z")`), add code to build the spatial index:

```python
# Build spatial index for click-to-search
all_coords = []
all_cluster_ids = []

# Noise points get cluster_id = -1
for peak in noise_peaks:
    all_coords.append([peak.rt_apex, peak.mz])
    all_cluster_ids.append(-1)

# Clustered points
for ci, cluster in enumerate(rt_clusters):
    for peak in cluster:
        all_coords.append([peak.rt_apex, peak.mz])
        all_cluster_ids.append(ci)

if all_coords:
    self._scatter_coords = np.array(all_coords)
    self._scatter_cluster_ids = all_cluster_ids
    self._scatter_tree = cKDTree(self._scatter_coords)
else:
    self._scatter_coords = None
    self._scatter_cluster_ids = []
    self._scatter_tree = None
```

Also, in the `if n_clusters > self._MAX_RENDERABLE_CLUSTERS:` branch (line 543), clear the spatial index since clicks shouldn't work there:

```python
self._scatter_coords = None
self._scatter_cluster_ids = []
self._scatter_tree = None
```

And in the `if result.get('empty'):` branch (line 511), also clear:

```python
self._last_result = result
self._scatter_coords = None
self._scatter_cluster_ids = []
self._scatter_tree = None
```

- [ ] **Step 4: Verify preview still renders correctly**

Run: `conda run -n chromakit-env python -c "from ui.dialogs.spectral_deconv_inspector import SpectralDeconvInspectorDialog; print('import ok')"`
Expected: `import ok`

- [ ] **Step 5: Commit**

```bash
git add ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat(inspector): store preview result and build KDTree spatial index"
```

---

### Task 2: Add click handler with nearest-point lookup

**Files:**
- Modify: `ui/dialogs/spectral_deconv_inspector.py` — `_build_plot_panel`, new `_on_scatter_click` method

Register a matplotlib click event on the canvas. When the user clicks, find the nearest scatter point using the KDTree, convert pixel tolerance to data coords, and determine which cluster was clicked.

- [ ] **Step 1: Register `mpl_connect` in `_build_plot_panel`**

In `_build_plot_panel`, after `self._canvas = FigureCanvas(self._fig)` (line 289), add:

```python
self._canvas.mpl_connect('button_press_event', self._on_scatter_click)
```

- [ ] **Step 2: Implement `_on_scatter_click`**

Add this method to the class, after `_on_preview_error` (after line 498):

```python
def _on_scatter_click(self, event):
    """Handle click on the scatter plot to select a cluster."""
    # Ignore clicks outside the scatter axes, during preview, or with no data
    if event.inaxes is not self._ax_scatter:
        return
    if self._preview_worker is not None:
        return
    if self._scatter_tree is None or self._last_result is None:
        return

    # Convert a pixel tolerance (15 px) to data-coordinate distance
    # Transform two points 15 px apart at the click location
    inv = self._ax_scatter.transData.inverted()
    display_pt = self._ax_scatter.transData.transform([event.xdata, event.ydata])
    offset_pt = inv.transform([display_pt[0] + 15, display_pt[1] + 15])
    data_pt = np.array([event.xdata, event.ydata])
    tol = np.linalg.norm(offset_pt - data_pt)

    dist, idx = self._scatter_tree.query(data_pt)
    if dist > tol:
        # Clicked empty space — clear selection
        self._clear_cluster_selection()
        return

    cluster_id = self._scatter_cluster_ids[idx]

    if cluster_id == -1:
        # Noise point
        self._status_label.setText("Noise point — not assigned to any component")
        self._clear_cluster_selection()
        return

    if cluster_id == self._selected_cluster_idx:
        # Toggle off: clicking same cluster clears selection
        self._clear_cluster_selection()
        return

    self._selected_cluster_idx = cluster_id
    self._highlight_cluster(cluster_id)

    # Find the component for this cluster and emit search signal
    components = self._last_result.get('components', [])
    rt_clusters = self._last_result['intermediates']['rt_clusters']
    model_peaks = self._last_result['intermediates']['model_peaks']

    # Match cluster to component via model peak RT
    # Each component corresponds to a model peak; find which model peak
    # belongs to this cluster
    cluster_peaks = rt_clusters[cluster_id]
    model_peak_ids = {id(mp) for mp in model_peaks}
    cluster_model = None
    for peak in cluster_peaks:
        if id(peak) in model_peak_ids:
            cluster_model = peak
            break

    if cluster_model is None:
        self._status_label.setText(
            f"Cluster {cluster_id} has no model peak — cannot search"
        )
        return

    # Find the component whose RT matches this model peak
    target_component = None
    for comp in components:
        if abs(comp.rt - cluster_model.rt_apex) < 1e-6:
            target_component = comp
            break

    if target_component is None or not target_component.spectrum:
        self._status_label.setText(
            f"Cluster {cluster_id}: no spectrum available"
        )
        return

    self._status_label.setText(
        f"Cluster {cluster_id} selected — RT {target_component.rt:.3f} min, "
        f"{len(target_component.spectrum)} m/z ions — searching…"
    )
    self.cluster_search_requested.emit(target_component.spectrum, target_component.rt)
```

- [ ] **Step 3: Add the new signal to the class**

At the top of `SpectralDeconvInspectorDialog` (line 95), alongside `rerun_requested`, add:

```python
cluster_search_requested = Signal(object, float)  # (spectrum_dict, rt)
```

- [ ] **Step 4: Verify import succeeds**

Run: `conda run -n chromakit-env python -c "from ui.dialogs.spectral_deconv_inspector import SpectralDeconvInspectorDialog; print('import ok')"`
Expected: `import ok`

- [ ] **Step 5: Commit**

```bash
git add ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat(inspector): click handler maps scatter clicks to clusters"
```

---

### Task 3: Add cluster highlighting and clear-selection

**Files:**
- Modify: `ui/dialogs/spectral_deconv_inspector.py` — new `_highlight_cluster`, `_clear_cluster_selection` methods

When a cluster is selected, re-render the scatter plot with the selected cluster emphasized and others dimmed. Clearing selection restores the normal render.

- [ ] **Step 1: Implement `_highlight_cluster`**

Add this method after `_on_scatter_click`:

```python
def _highlight_cluster(self, cluster_idx: int):
    """Re-render scatter with the selected cluster highlighted."""
    if self._last_result is None:
        return

    intermediates = self._last_result['intermediates']
    rt_clusters = intermediates['rt_clusters']
    noise_peaks = intermediates['noise_peaks']
    model_peaks = intermediates['model_peaks']
    model_peak_ids = {id(mp) for mp in model_peaks}
    win_peaks = self._last_result['win_peaks']
    fid_rts = [p.retention_time for p in win_peaks]

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab10')

    self._ax_scatter.clear()

    # FID span shading
    if fid_rts:
        fid_min, fid_max = min(fid_rts), max(fid_rts)
        half_gap = (self._last_result['w_end'] - self._last_result['w_start']) * 0.05
        self._ax_scatter.axvspan(
            fid_min - half_gap, fid_max + half_gap,
            alpha=0.08, color='steelblue', zorder=1,
        )

    # Noise points — always dimmed
    if noise_peaks:
        self._ax_scatter.scatter(
            [p.rt_apex for p in noise_peaks],
            [p.mz for p in noise_peaks],
            color='gray', s=10, alpha=0.15, zorder=2,
        )

    # Clustered points — selected vs dimmed
    for ci, cluster in enumerate(rt_clusters):
        color = cmap(ci % 10)
        rts = [p.rt_apex for p in cluster]
        mzs = [p.mz for p in cluster]
        is_selected = (ci == cluster_idx)

        self._ax_scatter.scatter(
            rts, mzs,
            color=color,
            s=30 if is_selected else 18,
            alpha=1.0 if is_selected else 0.25,
            edgecolors='black' if is_selected else 'none',
            linewidths=0.8 if is_selected else 0,
            zorder=5 if is_selected else 3,
        )

        if is_selected:
            for peak in cluster:
                if id(peak) in model_peak_ids:
                    self._ax_scatter.scatter(
                        [peak.rt_apex], [peak.mz],
                        color=color, marker='*', s=120,
                        edgecolors='black', linewidths=0.8, zorder=6,
                    )
        else:
            for peak in cluster:
                if id(peak) in model_peak_ids:
                    self._ax_scatter.scatter(
                        [peak.rt_apex], [peak.mz],
                        color=color, marker='*', s=50, alpha=0.25, zorder=4,
                    )

    # FID peak RT lines
    for rt in fid_rts:
        self._ax_scatter.axvline(rt, color='steelblue', linestyle='--', alpha=0.7, linewidth=1)
        self._ax_scatter.text(
            rt, 1.01, f"{rt:.3f}",
            transform=self._ax_scatter.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=7, color='steelblue',
        )

    n_clusters = len(rt_clusters)
    n_noise = len(noise_peaks)
    self._ax_scatter.set_title(
        f"RT Clusters — {n_clusters} cluster(s), {n_noise} noise — "
        f"cluster {cluster_idx} selected"
    )
    self._ax_scatter.set_ylabel("m/z")
    self._canvas.draw()
```

- [ ] **Step 2: Implement `_clear_cluster_selection`**

Add this method after `_highlight_cluster`:

```python
def _clear_cluster_selection(self):
    """Remove cluster highlight and restore normal render."""
    if self._selected_cluster_idx is None:
        return
    self._selected_cluster_idx = None
    if self._last_result is not None:
        self._render_plots(self._last_result)
```

- [ ] **Step 3: Verify import succeeds**

Run: `conda run -n chromakit-env python -c "from ui.dialogs.spectral_deconv_inspector import SpectralDeconvInspectorDialog; print('import ok')"`
Expected: `import ok`

- [ ] **Step 4: Commit**

```bash
git add ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat(inspector): highlight selected cluster, dim others"
```

---

### Task 4: Add inline search results panel

**Files:**
- Modify: `ui/dialogs/spectral_deconv_inspector.py` — `_build_plot_panel`, new `show_search_results` method, `QTreeWidget` import

Add a `QTreeWidget` below the status label to display library search results. Provide a `show_search_results(results, rt)` method the app calls after searching.

- [ ] **Step 1: Add `QTreeWidget` and `QTreeWidgetItem` to imports**

Update the `QWidgets` import at line 11 to include `QTreeWidget` and `QTreeWidgetItem`:

```python
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QSplitter, QWidget,
    QGroupBox, QFormLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QComboBox, QCheckBox, QLineEdit, QPushButton, QProgressBar,
    QTreeWidget, QTreeWidgetItem, QHeaderView,
)
```

- [ ] **Step 2: Add the results tree to `_build_plot_panel`**

In `_build_plot_panel`, after the status label (`layout.addWidget(self._status_label)` at line 295), add:

```python
# Search results panel
self._results_label = QLabel("Click a cluster to search")
self._results_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
layout.addWidget(self._results_label)

self._results_tree = QTreeWidget()
self._results_tree.setHeaderLabels(["Rank", "Compound", "Score"])
self._results_tree.setRootIsDecorated(False)
self._results_tree.setMaximumHeight(140)
self._results_tree.header().setSectionResizeMode(1, QHeaderView.Stretch)
self._results_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
self._results_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
layout.addWidget(self._results_tree)
```

- [ ] **Step 3: Implement `show_search_results`**

Add this method after `_clear_cluster_selection`:

```python
def show_search_results(self, results: list, rt: float):
    """Display MS library search results in the inline panel.

    Args:
        results: list of (compound_name, match_score) tuples from ms_toolkit.
        rt: Component retention time in minutes.
    """
    self._results_tree.clear()

    if not results:
        self._results_label.setText(f"No matches for component at RT = {rt:.3f} min")
        return

    self._results_label.setText(f"Search results — component at RT = {rt:.3f} min")

    for rank, (name, score) in enumerate(results, start=1):
        item = QTreeWidgetItem([str(rank), str(name), f"{score:.4f}"])
        item.setTextAlignment(0, Qt.AlignCenter)
        item.setTextAlignment(2, Qt.AlignCenter)
        self._results_tree.addTopLevelItem(item)

    self._status_label.setText(
        f"Cluster search complete — {len(results)} result(s) at RT = {rt:.3f} min"
    )
```

- [ ] **Step 4: Clear results on new preview**

In `_on_preview_finished`, after `self._selected_cluster_idx = None`, add:

```python
self._results_tree.clear()
self._results_label.setText("Click a cluster to search")
```

- [ ] **Step 5: Verify import succeeds**

Run: `conda run -n chromakit-env python -c "from ui.dialogs.spectral_deconv_inspector import SpectralDeconvInspectorDialog; print('import ok')"`
Expected: `import ok`

- [ ] **Step 6: Commit**

```bash
git add ui/dialogs/spectral_deconv_inspector.py
git commit -m "feat(inspector): inline QTreeWidget for cluster search results"
```

---

### Task 5: Wire signal through `ChromaKitApp`

**Files:**
- Modify: `ui/app.py` — `_on_inspect_requested`, new `_on_cluster_search_requested` method

Connect the inspector's `cluster_search_requested` signal to a new app method that reads MSFrame's toolkit and search settings, runs the search, and sends results back.

- [ ] **Step 1: Connect signal in `_on_inspect_requested`**

In `_on_inspect_requested` (at line 2849), after the existing line:

```python
self._deconv_inspector.rerun_requested.connect(self._on_inspector_rerun_requested)
```

Add:

```python
self._deconv_inspector.cluster_search_requested.connect(self._on_cluster_search_requested)
```

- [ ] **Step 2: Implement `_on_cluster_search_requested`**

Add this method after `_on_inspector_rerun_requested` (after line 2889):

```python
def _on_cluster_search_requested(self, spectrum_dict: dict, rt: float):
    """Run MS library search for a deconvolved cluster spectrum.

    Called when user clicks a cluster in the inspector scatter plot.
    Uses MSFrame's ms_toolkit and search settings, then routes results
    back to the inspector's inline results panel.
    """
    if not hasattr(self.ms_frame, 'ms_toolkit') or self.ms_frame.ms_toolkit is None:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "MS Library Not Loaded",
            "The MS library must be loaded before searching.\n"
            "Go to the MS tab and load a library first.",
        )
        return

    if not self.ms_frame.library_loaded:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "MS Library Not Loaded",
            "The MS library must be loaded before searching.\n"
            "Go to the MS tab and load a library first.",
        )
        return

    # Build query spectrum as list of (mz, intensity) tuples
    query = [(float(mz), float(intensity))
             for mz, intensity in spectrum_dict.items()]

    if not query:
        if self._deconv_inspector:
            self._deconv_inspector.show_search_results([], rt)
        return

    options = self.ms_frame.search_options
    toolkit = self.ms_frame.ms_toolkit

    try:
        if options.get('search_method') == 'w2v':
            results = toolkit.search_w2v(
                query,
                top_n=options.get('top_n', 5),
                intensity_power=options.get('intensity_power', 0.6),
                top_k_clusters=options.get('top_k_clusters', 1),
            )
        elif options.get('search_method') == 'hybrid':
            results = toolkit.search_hybrid(
                query,
                method=options.get('hybrid_method', 'auto'),
                top_n=options.get('top_n', 5),
                intensity_power=options.get('intensity_power', 0.6),
                weighting_scheme=options.get('weighting', 'NIST_GC'),
                composite=(options.get('similarity') == 'composite'),
                unmatched_method=options.get('unmatched', 'keep_all'),
                top_k_clusters=options.get('top_k_clusters', 1),
            )
        else:
            # Default: vector search
            results = toolkit.search_vector(
                query,
                top_n=options.get('top_n', 5),
                composite=(options.get('similarity') == 'composite'),
                weighting_scheme=options.get('weighting', 'NIST_GC'),
                unmatched_method=options.get('unmatched', 'keep_all'),
                top_k_clusters=options.get('top_k_clusters', 1),
            )
    except Exception as e:
        if self._deconv_inspector:
            self._deconv_inspector.show_search_results([], rt)
            self._deconv_inspector._status_label.setText(
                f"Search error: {e}"
            )
        return

    if self._deconv_inspector:
        self._deconv_inspector.show_search_results(results, rt)
```

- [ ] **Step 3: Verify import succeeds**

Run: `conda run -n chromakit-env python -c "from ui.app import ChromaKitApp; print('import ok')"`
Expected: `import ok`

- [ ] **Step 4: Commit**

```bash
git add ui/app.py
git commit -m "feat(app): wire cluster_search_requested signal to MS library search"
```

---

### Task 6: End-to-end verification

**Files:**
- Read-only: all modified files

Verify everything imports cleanly and the existing test suite still passes.

- [ ] **Step 1: Run import check for all modified modules**

```bash
conda run -n chromakit-env python -c "
from ui.dialogs.spectral_deconv_inspector import SpectralDeconvInspectorDialog
from ui.app import ChromaKitApp
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 2: Run existing test suite**

```bash
conda run -n chromakit-env python -m pytest tests/ deconvolution/ -x -q 2>&1 | tail -20
```

Expected: All previously-passing tests still pass (52 + 41 = 93 pass, 3 pre-existing failures).

- [ ] **Step 3: Final commit (if any fixups needed)**

If any test failures or import errors were fixed, commit the fixes:

```bash
git add -u
git commit -m "fix: address test/import issues from cluster click-to-search"
```
