# Batch Processing Performance — Design Spec

**Date:** 2026-03-30
**Scope:** ChromaKit-MS batch loop (`AutomationWorker` + `BatchJobDialog`)

---

## Problem

For large batch jobs (~1000 files), total runtime is dominated by overhead rather than computation:

- Every processing step dispatches to the main Qt thread and spin-waits (`time.sleep(0.1)` loops)
- `process_and_display` does a full matplotlib re-render per file, even in batch mode
- All three of the above apply even when the user doesn't need the UI to update

The root cause is that signal math and plot rendering are coupled inside `process_and_display`, forcing all compute through the UI event loop.

---

## Goals

1. **Diagnose** where time actually goes via a per-step timing table
2. **Skip UI renders** in batch mode as a guaranteed, low-risk win
3. **Move compute off the main thread** to eliminate spin-wait round-trips

Non-goal: parallel file processing (deferred to a future phase pending timing data).

---

## Section 1: Timing Instrumentation

### UI

Add a **"Time processing steps"** checkbox to `BatchJobDialog` under Processing Options. Off by default.

### Behavior

When enabled, `AutomationWorker` records `time.perf_counter()` at each step boundary per file:

- **Load** — `_load_file` start → end
- **Process/Integrate** — `_process_and_integrate` start → end
- **MS Search** — `_run_ms_search` start → end (GC-MS files only)
- **Save** — save step start → end

After all files finish, emit a formatted table to the log:

```
=== Batch Timing Report ===
File                      Load    Process   MS Search  Save    Total
Sample_001.C              0.31s   0.22s     1.20s      0.02s   1.75s
Sample_002.C (no MS)      0.29s   0.20s     —          0.02s   0.51s
...
Average (all)             0.30s   0.21s     —          0.02s   —
Average (GC-MS only)      0.30s   0.21s     1.19s      0.02s   1.70s
```

### Implementation notes

- Store timings as a list of dicts on `AutomationWorker`: `[{'file': name, 'load': s, 'process': s, 'ms_search': s or None, 'save': s}]`
- Emit the table as a single `log_message` signal at the end of `run()`
- No file I/O — log window only

---

## Section 2: Skip UI Updates Option

### UI

Add a **"Skip UI updates during batch (faster)"** checkbox to `BatchJobDialog` under Processing Options. **Checked by default.**

Include a brief note: "Uncheck to watch the chromatogram update as files are processed."

Pass the setting through `options['skip_ui_updates']` in the `start_batch` signal.

### Behavior

When `skip_ui_updates=True`:

- `process_and_display` is **not called** on the main thread at all
- Signal processing (`processor.process()`) runs on the worker thread directly
- `canvas.draw_idle()` and `plot_chromatogram()` are skipped entirely
- The plot reflects the last-processed file once the batch completes

When `skip_ui_updates=False` (current behavior):
- No change — `process_and_display` is dispatched to the main thread as today

---

## Section 3: Decouple Compute from the Main Thread

### Current flow (per file)

```
worker thread:
  dispatch → main thread: process_and_display(x, y)   [math + render]
    spin-wait 100ms+
  dispatch → main thread: integrate_peaks_no_ui()
    spin-wait 100ms+
  dispatch → main thread: _create_batch_search_worker()
    spin-wait + QThreadPool + spin-wait
  dispatch → main thread: save_results()
    spin-wait
```

### New flow (skip_ui_updates=True)

```
worker thread:
  read params via main-thread dispatch (once per file, during load)
  processor.process(x, y, params)          ← direct call, no dispatch
  integrate_peaks(processed, params)       ← direct call, no dispatch
  [optional] apply_manual_overrides()      ← direct call, no dispatch
  BatchSearchWorker on QThreadPool         ← unchanged, already threaded
  save_results()                           ← dispatch to main thread (file I/O is fine here, but JSON exporter may touch app state)
  [optional] dispatch render to main thread (skip_ui_updates=False only)
```

### Key changes

**`_process_and_integrate` refactor:**

- Read `params` from `self.app.parameters_frame` during the file load step (main thread), store as `self._current_params`
- When `skip_ui_updates=True`: call `self.app.processor.process(x, y, params)` directly on the worker thread; store result in a local variable rather than going through `process_and_display`
- Call `integrate_peaks_no_ui` directly (verify it has no Qt widget access — expected to be clean)
- Store integrated peaks in `self.app.integrated_peaks` (plain Python attribute assignment, thread-safe for this single-writer pattern)
- When `skip_ui_updates=False`: dispatch `process_and_display` as today (no change)

**`_load_file` (unchanged in structure):**

- Still dispatches to main thread — file loading touches `on_file_selected` which updates UI state
- Reads `params` from `parameters_frame` at end of load and stores on `self`

**Save step:**

- `save_results` may touch `self.app` attributes — keep on main thread dispatch for safety

### Thread safety note

`self.app.integrated_peaks` is written by the worker and read by the save step (also dispatched to main thread, which runs after the worker signals completion). This is safe under the existing single-file-at-a-time model. No locking needed.

---

## What is NOT in scope

- Parallel file processing (multiple files concurrently) — deferred pending timing data from Section 1
- Changing `BatchSearchWorker` peak loop (already improved 90%)
- Any changes to export formats or post-processing logic

---

## Files Affected

| File | Change |
|------|--------|
| `ui/dialogs/batch_job_dialog.py` | Add two checkboxes; pass options |
| `logic/automation_worker.py` | Timing instrumentation; compute/render split; params pre-fetch |
| `ui/app.py` | `process_and_display` accepts `skip_render` flag |
