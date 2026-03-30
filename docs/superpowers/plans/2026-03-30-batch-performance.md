# Batch Processing Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add step-timing instrumentation and a skip-UI-updates mode to the batch processing loop, then decouple signal processing from the main Qt thread so that (in the common fast path) each file requires only one main-thread round-trip instead of three.

**Architecture:** Three bounded changes: (1) two new checkboxes in `BatchJobDialog` pass options through the existing `batch_options` dict; (2) `process_and_display` gains a `skip_render` flag and `integrate_peaks_no_ui` gains a `params` kwarg so both can be called safely from the worker thread; (3) `AutomationWorker` uses those flags to run signal math and integration directly on the worker thread when `skip_ui_updates=True`, and wraps each step in timestamps when `time_steps=True`.

**Tech Stack:** PySide6, numpy/scipy (already in use), Python `time.perf_counter`

---

## File Map

| File | Change |
|------|--------|
| `ui/dialogs/batch_job_dialog.py` | Add two checkboxes; include in emitted options dict |
| `ui/app.py` | `process_and_display`: add `skip_render=False` kwarg; `integrate_peaks_no_ui`: add `params=None` kwarg |
| `logic/automation_worker.py` | Timing infrastructure; params pre-fetch in `_load_file`; compute/render split in `_process_and_integrate` |

No new files. No changes to export logic, MS search, or signal processing math.

---

## Task 1: Add checkboxes to BatchJobDialog

**Files:**
- Modify: `ui/dialogs/batch_job_dialog.py:59-68` (options_layout block)

- [ ] **Step 1: Add the two checkboxes to `setup_ui`**

In `setup_ui`, after the existing `self.ms_search_check` block (around line 67), add:

```python
        # Timing option
        self.time_steps_check = QCheckBox("Time processing steps")
        self.time_steps_check.setChecked(False)
        options_layout.addWidget(self.time_steps_check)

        # Speed option
        self.skip_ui_check = QCheckBox("Skip UI updates during batch (faster)")
        self.skip_ui_check.setToolTip(
            "Skips chromatogram re-rendering between files. "
            "Uncheck to watch the display update as files are processed."
        )
        self.skip_ui_check.setChecked(True)
        options_layout.addWidget(self.skip_ui_check)
```

- [ ] **Step 2: Include new options in `on_start_processing`**

In `on_start_processing`, update the `options` dict:

```python
        options = {
            'integration': self.integration_check.isChecked(),
            'ms_search': self.ms_search_check.isChecked(),
            'save_results': self.save_results_check.isChecked(),
            'export_csv': self.export_csv_check.isChecked(),
            'overwrite_existing': self.overwrite_check.isChecked(),
            'time_steps': self.time_steps_check.isChecked(),
            'skip_ui_updates': self.skip_ui_check.isChecked(),
        }
```

- [ ] **Step 3: Launch the app and open the Batch Job dialog**

```bash
conda activate chromakit-env && python main.py
```

Verify: both new checkboxes appear under Processing Options. "Skip UI updates" is checked by default. "Time processing steps" is unchecked.

- [ ] **Step 4: Commit**

```bash
git add ui/dialogs/batch_job_dialog.py
git commit -m "feat: add time_steps and skip_ui_updates checkboxes to BatchJobDialog"
```

---

## Task 2: Thread-safe kwarg on `integrate_peaks_no_ui`

**Files:**
- Modify: `ui/app.py:1434` (`integrate_peaks_no_ui`)

`integrate_peaks_no_ui` currently calls `self.parameters_frame.get_parameters()` (a Qt widget access) unconditionally. We add a `params=None` kwarg so callers that already have params (e.g. the batch worker) can pass them in directly, making it safe to call off the main thread.

- [ ] **Step 1: Add `params` kwarg to `integrate_peaks_no_ui`**

Change the signature and the first widget-access line:

```python
    def integrate_peaks_no_ui(self, ms_data=None, quality_options=None, params=None):
        """Thread-safe version of integration without UI updates.

        Args:
            ms_data: Optional MS data dict.
            quality_options: Optional quality assessment options.
            params: Pre-fetched parameters dict. If None, fetches from parameters_frame
                    (requires main thread).
        """
        try:
            if not hasattr(self, 'current_processed') or self.current_processed is None:
                return None

            if params is None:
                params = self.parameters_frame.get_parameters()

            peaks_enabled = params['peaks']['enabled']
            neg_peaks_enabled = params.get('negative_peaks', {}).get('enabled', False)
            if not peaks_enabled and not neg_peaks_enabled:
                return None
            # ... rest of method unchanged ...
```

Leave the entire method body from `peak_mode = params['peaks'].get('mode', 'classical')` onward unchanged.

- [ ] **Step 2: Verify existing callers still work**

Search for all callers of `integrate_peaks_no_ui`:

```bash
grep -rn "integrate_peaks_no_ui" /Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS\ code\ development/chromakit-qt/
```

All existing callers pass zero positional args after `self` — the new `params=None` default keeps them working unchanged.

- [ ] **Step 3: Commit**

```bash
git add ui/app.py
git commit -m "refactor: add params kwarg to integrate_peaks_no_ui for off-main-thread use"
```

---

## Task 3: Add `skip_render` to `process_and_display`

**Files:**
- Modify: `ui/app.py:2110` (`process_and_display`)

- [ ] **Step 1: Add `skip_render=False` kwarg**

Change the signature:

```python
    def process_and_display(self, x, y, new_file=False, profile=None, skip_render=False):
```

After the `self.current_processed = processed` assignment (currently missing — add it just before the plot block), wrap the plot calls:

```python
        # --- Peak splitting mode ---
        if peak_mode == 'peak_splitting' and params['peaks']['enabled']:
            self._apply_peak_splitting(processed, params)

        # Always store for downstream use (integration, export)
        self.current_processed = processed

        if not skip_render:
            self.plot_frame.plot_chromatogram(
                processed,
                show_corrected=params['baseline']['show_corrected'],
                new_file=new_file
            )
            self.plot_frame.canvas.draw_idle()

        # Store the processed data for reference (keep existing line too)
        self.current_processed = processed
```

Note: the existing `self.current_processed = processed` line is at the end of the method — leave it there (duplicate assignment is harmless), and add the earlier one just before the `if not skip_render` block so it's set even when rendering is skipped.

- [ ] **Step 2: Verify no callers broke**

```bash
grep -rn "process_and_display" /Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS\ code\ development/chromakit-qt/
```

All existing callers pass positional args only — the new `skip_render=False` default keeps them working.

- [ ] **Step 3: Quick smoke test**

Launch the app, load a file manually, verify the chromatogram plots correctly (skip_render defaults to False so no behavior change).

- [ ] **Step 4: Commit**

```bash
git add ui/app.py
git commit -m "refactor: add skip_render kwarg to process_and_display"
```

---

## Task 4: Timing infrastructure in AutomationWorker

**Files:**
- Modify: `logic/automation_worker.py`

This task adds the timing data structure and the formatted table emitted at the end of a batch run. The per-step recording is wired in Task 5. This task is purely additive — no behavior changes.

- [ ] **Step 1: Add timing state to `__init__`**

In `AutomationWorker.__init__`, after `self.total_files = 0`:

```python
        # Timing: populated when self.app.batch_options['time_steps'] is True
        self._timings: list[dict] = []
```

- [ ] **Step 2: Add `_format_timing_table` static method**

Add this method anywhere in `AutomationWorker` (e.g. just before `_update_directory_progress`):

```python
    @staticmethod
    def _format_timing_table(timings: list[dict]) -> str:
        """Format per-file timing data as a log-friendly table.

        Each entry in timings: {
            'file': str,
            'load': float,        # seconds
            'process': float,     # seconds
            'ms_search': float | None,  # None if not applicable
            'save': float,
        }
        """
        if not timings:
            return "=== Batch Timing Report: no data ==="

        lines = ["", "=== Batch Timing Report ==="]
        header = f"{'File':<35} {'Load':>7} {'Process':>9} {'MS Search':>10} {'Save':>7} {'Total':>8}"
        lines.append(header)
        lines.append("-" * len(header))

        ms_times = []
        for t in timings:
            ms_val = t.get('ms_search')
            ms_str = f"{ms_val:>9.2f}s" if ms_val is not None else f"{'—':>9}"
            total = t['load'] + t['process'] + (ms_val or 0.0) + t['save']
            line = (
                f"{t['file']:<35}"
                f" {t['load']:>6.2f}s"
                f" {t['process']:>8.2f}s"
                f" {ms_str}"
                f" {t['save']:>6.2f}s"
                f" {total:>7.2f}s"
            )
            lines.append(line)
            if ms_val is not None:
                ms_times.append(ms_val)

        # Averages
        lines.append("-" * len(header))
        n = len(timings)
        avg_load = sum(t['load'] for t in timings) / n
        avg_proc = sum(t['process'] for t in timings) / n
        avg_save = sum(t['save'] for t in timings) / n
        avg_ms_str = f"{sum(ms_times)/len(ms_times):>9.2f}s" if ms_times else f"{'—':>9}"
        avg_total = avg_load + avg_proc + (sum(ms_times)/len(ms_times) if ms_times else 0.0) + avg_save
        lines.append(
            f"{'Average':<35}"
            f" {avg_load:>6.2f}s"
            f" {avg_proc:>8.2f}s"
            f" {avg_ms_str}"
            f" {avg_save:>6.2f}s"
            f" {avg_total:>7.2f}s"
        )
        lines.append("")
        return "\n".join(lines)
```

- [ ] **Step 3: Write a quick test for the formatter**

Create `tests/logic/test_batch_timing.py`:

```python
from logic.automation_worker import AutomationWorker


def test_format_timing_table_gcms():
    timings = [
        {'file': 'Sample_001.C', 'load': 0.31, 'process': 0.18, 'ms_search': 1.20, 'save': 0.02},
        {'file': 'Sample_002.C', 'load': 0.29, 'process': 0.17, 'ms_search': 1.18, 'save': 0.02},
    ]
    table = AutomationWorker._format_timing_table(timings)
    assert 'Batch Timing Report' in table
    assert 'Sample_001.C' in table
    assert 'Sample_002.C' in table
    assert 'Average' in table


def test_format_timing_table_mixed():
    """Non-GC-MS files show — in MS Search column."""
    timings = [
        {'file': 'Sample_001.C', 'load': 0.31, 'process': 0.18, 'ms_search': 1.20, 'save': 0.02},
        {'file': 'ReactIR_001.C', 'load': 0.20, 'process': 0.10, 'ms_search': None, 'save': 0.02},
    ]
    table = AutomationWorker._format_timing_table(timings)
    assert '—' in table
    assert 'Average' in table


def test_format_timing_table_empty():
    table = AutomationWorker._format_timing_table([])
    assert 'no data' in table
```

- [ ] **Step 4: Run the tests**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/logic/test_batch_timing.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add logic/automation_worker.py tests/logic/test_batch_timing.py
git commit -m "feat: add timing table formatter to AutomationWorker"
```

---

## Task 5: Wire timing into the batch loop + emit table

**Files:**
- Modify: `logic/automation_worker.py` (`run` method)

- [ ] **Step 1: Read `time_steps` and `skip_ui_updates` from batch_options at start of `run()`**

At the top of the `run()` method, just before the `c_files` scan, add:

```python
            batch_opts = getattr(self.app, 'batch_options', {}) or {}
            self._time_steps = batch_opts.get('time_steps', False)
            self._skip_ui_updates = batch_opts.get('skip_ui_updates', True)
```

- [ ] **Step 2: Wrap each step call in `run()` with timestamps**

Replace the three step calls inside the per-file `try` block (load → process → ms_search) with timed versions. The existing structure is:

```python
                success = self._load_file(file_path)
                ...
                success = self._process_and_integrate()
                ...
                success = self._run_ms_search()
                ...
```

Replace with:

```python
                # Step 1: Load
                t0 = time.perf_counter()
                self._update_directory_progress(filename, "Loading", 10)
                success = self._load_file(file_path)
                t_load = time.perf_counter() - t0

                if not success or self.cancelled:
                    self.signals.file_completed.emit(filename, False, "Failed to load file")
                    continue

                # Step 2: Process and integrate
                t1 = time.perf_counter()
                self._update_directory_progress(filename, step_label, 40)
                success = self._process_and_integrate()
                t_process = time.perf_counter() - t1

                if not success or self.cancelled:
                    self.signals.file_completed.emit(filename, False, "Failed to integrate")
                    continue

                # Step 2.5: Save integration results to JSON
                t2 = time.perf_counter()
                self._update_directory_progress(filename, "Saving integration results", 50)
                if hasattr(self.app, 'integrated_peaks') and self.app.integrated_peaks:
                    current_dir = self.app.data_handler.current_directory_path
                    integration_results = {'peaks': self.app.integrated_peaks}
                    json_saved = self._save_integration_json_no_ui(integration_results, current_dir)
                    if json_saved:
                        self.signals.log_message.emit(f"Integration results saved to JSON for {filename}")
                    else:
                        self.signals.log_message.emit(f"Warning: Failed to save JSON for {filename}")
                t_save_integration = time.perf_counter() - t2

                # Step 3: MS Library Search
                ms_search_enabled = True
                if hasattr(self.app, 'batch_options') and 'ms_search' in self.app.batch_options:
                    ms_search_enabled = self.app.batch_options['ms_search']

                ms_search_applicable = PipelineStage.MS_SEARCH in profile_stages
                t_ms = None

                if ms_search_enabled and ms_search_applicable:
                    self._update_directory_progress(filename, "MS Library Search", 70)
                    t3 = time.perf_counter()
                    success = self._run_ms_search()
                    t_ms = time.perf_counter() - t3

                    if not success or self.cancelled:
                        self.signals.file_completed.emit(filename, False, "Failed to run MS search")
                        continue
                elif ms_search_enabled:
                    self.signals.log_message.emit(
                        f"Skipping MS search: not applicable for signal type '{signal_type}'"
                    )
                else:
                    self.signals.log_message.emit("MS search disabled in options - skipping")

                # Record timing for this file
                if self._time_steps:
                    self._timings.append({
                        'file': filename,
                        'load': t_load,
                        'process': t_process,
                        'save': t_save_integration,
                        'ms_search': t_ms,
                    })

                # File completed successfully
                self._update_directory_progress(filename, "Completed", 100)
                self.signals.file_completed.emit(filename, True, "Processing completed")
                self.signals.log_message.emit(f"Completed processing {filename}")
```

- [ ] **Step 3: Emit timing table at end of `run()`**

Just before `self.signals.finished.emit()`, add:

```python
            # Emit timing report if enabled
            if self._time_steps and self._timings:
                table = self._format_timing_table(self._timings)
                self.signals.log_message.emit(table)
```

- [ ] **Step 4: Commit**

```bash
git add logic/automation_worker.py
git commit -m "feat: wire per-step timing into batch loop and emit table at completion"
```

---

## Task 6: Skip-UI-updates compute path in `_process_and_integrate`

**Files:**
- Modify: `logic/automation_worker.py` (`_load_file`, `_process_and_integrate`)

This is the main performance change: when `skip_ui_updates=True`, signal math and integration run directly on the worker thread. The main thread is only touched for the file load.

- [ ] **Step 1: Pre-fetch params in `_load_file`**

In `_load_file`, inside `main_thread_load` (which already runs on the main thread), add params pre-fetch at the end of the success path:

```python
            def main_thread_load():
                try:
                    status_text = self.app.status_bar.currentMessage()
                    self.app.on_file_selected(file_path, batch_mode=True)
                    self.load_success = True

                    # Pre-fetch params while we're on the main thread
                    # so _process_and_integrate can use them off-thread
                    try:
                        self._current_params = self.app.parameters_frame.get_parameters()
                    except Exception:
                        self._current_params = None

                    self.signals.log_message.emit(f"Successfully loaded {os.path.basename(file_path)}")
                except Exception as e:
                    # ... existing exception handling unchanged ...
```

Leave the rest of `_load_file` unchanged.

- [ ] **Step 2: Refactor `_process_and_integrate` to branch on `_skip_ui_updates`**

Replace the entire body of `_process_and_integrate` with:

```python
    def _process_and_integrate(self):
        """Process and integrate the current file."""
        try:
            filename = os.path.basename(
                self.app.data_handler.current_directory_path
            )

            self._update_directory_progress(filename, "Processing chromatogram", 20)

            if not (hasattr(self.app, 'current_x') and hasattr(self.app, 'current_y')
                    and self.app.current_x is not None and self.app.current_y is not None):
                self.signals.log_message.emit("No data available for processing")
                return False

            # Use pre-fetched params when available; fall back to main-thread fetch.
            params = getattr(self, '_current_params', None)
            if params is None:
                # Fetch on main thread (legacy path / params not pre-fetched)
                self._fetch_params_complete = False
                def fetch_params():
                    try:
                        self._current_params = self.app.parameters_frame.get_parameters()
                    finally:
                        self._fetch_params_complete = True
                main_thread_dispatcher.run_on_main_thread(fetch_params)
                timeout, start = 10, time.time()
                while not self._fetch_params_complete:
                    if self.cancelled or time.time() - start > timeout:
                        return False
                    time.sleep(0.05)
                params = self._current_params

            if not params['peaks']['enabled']:
                self.signals.log_message.emit("Peak detection is not enabled in parameters")
                return False

            if getattr(self, '_skip_ui_updates', True):
                # ── Fast path: run math directly on this worker thread ──────────
                self.signals.log_message.emit("Processing chromatogram...")
                try:
                    ms_range = None
                    if (hasattr(self.app, 'plot_frame')
                            and hasattr(self.app.plot_frame, 'tic_data')
                            and self.app.plot_frame.tic_data is not None):
                        td = self.app.plot_frame.tic_data
                        if 'x' in td and len(td['x']) > 0:
                            ms_range = (float(td['x'].min()), float(td['x'].max()))

                    profile = getattr(self.app, 'current_profile', None)
                    processed = self.app.processor.process(
                        self.app.current_x,
                        self.app.current_y,
                        params,
                        ms_range,
                        profile=profile,
                    )
                except Exception as e:
                    self.signals.log_message.emit(f"Error processing: {str(e)}")
                    return False

                # Store result so integrate_peaks_no_ui can read it
                self.app.current_processed = processed
                self.app.current_profile = profile

                self._update_directory_progress(filename, "Integrating peaks", 40)
                self.signals.log_message.emit("Integrating peaks...")

                try:
                    integration_results = self.app.integrate_peaks_no_ui(params=params)
                except Exception as e:
                    self.signals.log_message.emit(f"Error integrating: {str(e)}")
                    return False

            else:
                # ── Original path: dispatch to main thread ────────────────────
                self.process_complete = False
                self.process_success = False

                def main_thread_process():
                    try:
                        self.app.process_and_display(self.app.current_x, self.app.current_y)
                        self.process_success = True
                    except Exception as e:
                        self.signals.log_message.emit(f"Error processing: {str(e)}")
                    finally:
                        self.process_complete = True

                main_thread_dispatcher.run_on_main_thread(main_thread_process)
                timeout, start = 30, time.time()
                while not self.process_complete:
                    if self.cancelled or time.time() - start > timeout:
                        return False
                    time.sleep(0.1)
                if not self.process_success:
                    return False

                self._update_directory_progress(filename, "Integrating peaks", 40)
                self.signals.log_message.emit("Integrating peaks...")

                self.integrate_complete = False
                self.integration_results = None

                def main_thread_integrate():
                    try:
                        self.integration_results = self.app.integrate_peaks_no_ui()
                    finally:
                        self.integrate_complete = True

                main_thread_dispatcher.run_on_main_thread(main_thread_integrate)
                timeout, start = 30, time.time()
                while not self.integrate_complete:
                    if self.cancelled or time.time() - start > timeout:
                        return False
                    time.sleep(0.1)

                integration_results = self.integration_results

            # ── Common completion path ────────────────────────────────────────
            if not integration_results:
                self.signals.log_message.emit("Integration failed")
                return False

            if not self.app.integrated_peaks:
                self.signals.log_message.emit("No peaks found during integration")
                if getattr(self, "_is_chromatography", True):
                    return False
            else:
                if getattr(self, "_is_chromatography", True):
                    self._apply_manual_overrides()

            # Update plot shading (if UI updates are on)
            if not getattr(self, '_skip_ui_updates', True):
                def update_plot():
                    try:
                        self.app.plot_frame.shade_integration_areas(integration_results)
                    except Exception as e:
                        print(f"Error updating plot: {str(e)}")
                main_thread_dispatcher.run_on_main_thread(update_plot)

            self.signals.log_message.emit(
                f"Integration complete: {len(self.app.integrated_peaks)} peaks found"
            )
            self._update_directory_progress(
                filename, f"Integrated {len(self.app.integrated_peaks)} peaks", 50
            )
            return True

        except Exception as e:
            self.signals.log_message.emit(f"Error in processing/integration: {str(e)}")
            return False
```

- [ ] **Step 3: Verify `tic_data` attribute access is safe off-thread**

`self.app.plot_frame.tic_data` is a plain Python dict set during file load (on the main thread). By the time `_process_and_integrate` runs on the worker, load has completed — the read is safe. No change needed.

- [ ] **Step 4: Commit**

```bash
git add logic/automation_worker.py
git commit -m "perf: run signal processing and integration on worker thread when skip_ui_updates=True"
```

---

## Task 7: Manual end-to-end verification

- [ ] **Step 1: Run a small batch with default settings (skip UI on, timing off)**

Launch the app. Open Batch > Batch Job, add a directory with 3–5 `.C` files, leave "Skip UI updates" checked and "Time processing steps" unchecked. Start the run. Verify:
- All files complete successfully
- Log messages look normal
- No exceptions in console
- The chromatogram display does NOT cycle through files (it stays static or shows the last file)

- [ ] **Step 2: Run with timing enabled**

Same batch, check "Time processing steps". After the run completes, scroll the batch log to the end. Verify a timing table appears with one row per file, `—` in MS Search for any non-GC-MS files, and an Average row.

- [ ] **Step 3: Run with UI updates enabled (original behavior)**

Uncheck "Skip UI updates". Run the same batch. Verify the chromatogram updates for each file as before, and all results are identical to the skip-UI run.

- [ ] **Step 4: Run the full test suite**

```bash
conda activate chromakit-env && cd "/Users/ccoatney/Library/CloudStorage/OneDrive-NREL/GC-MS code development/chromakit-qt" && python -m pytest tests/ -v
```

Expected: all tests pass (including the new `test_batch_timing.py` tests).

- [ ] **Step 5: Final commit if any fixups were needed**

```bash
git add -p
git commit -m "fix: address issues found during batch performance manual testing"
```
