## ChromaKit for Windows (64-bit)

1. Download `ChromaKit-<version>-windows-x64.zip` below.
2. Right-click it → **Extract All…**, into a folder you can write to (for example
   `Documents\ChromaKit` or `D:\ChromaKit`), not Program Files.
3. Windows makes a folder named after the zip, and inside it a `ChromaKit` folder. Open that inner
   `ChromaKit` folder and double-click **ChromaKit.exe**.
4. Windows may say "Windows protected your PC". Click **More info**, then **Run anyway**. This app is
   not code-signed yet; you'll see this once per new version.

**Updating:** when a new version exists, ChromaKit shows an **Update available** button in its status
bar. Click it, then **Open release page**. Download the new zip, extract it to a new folder, then delete the old folder. Your settings are
kept (they are stored outside the app folder).

**Problems:** the log is at `%LOCALAPPDATA%\ChromaKit\logs\chromakit.log`. Running
`ChromaKit.exe --self-test "%TEMP%\ck"` writes a check to `%TEMP%\ck\self-test.txt`.

**Changes for existing users:**
- **Load Method now applies the method** to the open file straight away. If the method was saved with a
  different detector channel or signal factor, the file is reloaded on that channel. The status bar shows
  `Method: <name>`, with `*` when it has unsaved changes.
- **Save Method records the detector channel** that is open. The batch dialog fills its first per-detector
  row from the saved method, and the JSON→Excel converter picks a saved format preset whose name contains
  the files' detector (unless you already picked one or changed the format).
- **Manual assignment overrides are off in batch runs** unless you tick **Apply manual assignment
  overrides** in the batch dialog. An override with a stored spectrum that can't be compared (GC-only
  data) is now skipped and logged instead of being applied on retention time alone. Re-export batches
  from GC-only data if an unexpected compound name appeared among the Unknowns.
- **GC-only files hide the MS controls** (MS tab, MS Search All, Deconvolve MS, MS baseline). A GC-only
  file opened after a GC-MS file no longer reuses that file's MS time range, which could drop its peaks.
- **Show Corrected Signal** and the toolbar **Home** button now fit the trace on screen.
- **Enable Baseline Correction** checkbox in the baseline section.
- Peaks next to the solvent delay keep their correct start or end.
- The **Update available** button explains how to update for the way ChromaKit was installed.
