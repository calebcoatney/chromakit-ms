# Unreleased changes (since v0.2.5)

## Batch jobs: Front / Back detector positions
- New `logic/detector_position.py`. A GC batch row can name a **Front Detector** or **Back Detector** instead of a
  channel. A sample's channels are taken in the order its data file lists them (first = Front, second = Back), so one
  row matches each GC's channel whatever it is called (FID1A, TCD3C, ...).
- `logic/automation_worker.py`: `resolve_detector` resolves positions to the file's first/second channel (none if the
  sample has too few channels); the log notes which channel a position resolved to.
- `ui/dialogs/batch_job_dialog.py`: on GC1/GC2 rows the detector cell offers Front/Back (unique per GC); rows with no
  instrument keep a free-text channel name. Older setups with channel names map to a free position.
- `logic/gc12_workbook.py`: injections and results are keyed by the job's position but report/read each sample's own
  channel name (`_channels_for`, `_available_channels`).

## GC1&2 workbook: optional "NO INJECTION" blanks
- New checkbox in the batch dialog (off by default, enabled only for tagged GC folders). When on, a sample with no
  partner on the other GC gets a "NO INJECTION" blank block on the other sheet so the sheets line up block for block.
  Passed through as `gc12_add_placeholders` to `write_paired_workbook` (`ui/app.py`).

## Environment / packaging
- `run.bat` (Windows, needs [uv](https://docs.astral.sh/uv/)) creates a Python 3.13 `.venv`, installs dependencies on
  first run and launches the app; `.python-version` pins 3.13.
- `setup.py` / README: supported Python is now 3.10–3.13 (`rainbow-api` has no prebuilt wheels for newer versions).
- `main.py`: startup errors now also print a traceback.

## Tests
Updated `tests/logic/test_batch_detector_runs.py`, `tests/logic/test_gc12_workbook.py`,
`tests/ui/test_gc12_batch_export.py`, `tests/ui/dialogs/test_batch_job_detector_runs.py`.
