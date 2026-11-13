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

**Changes in 0.2.4:**
- The ⓘ help icons beside section titles no longer cover the end of the title on macOS.
- Changing a setting before opening a file no longer prints errors to the terminal.
- For copies run from a source folder, **Update available** now also shows the reinstall command to run
  after `git pull`. Without it, a copy installed before 0.2.0 can't run the update check.

**Changes in 0.2.3** (if you are updating from 0.2.2 or earlier):
- **Save Method always asks where to save.** The dialog opens on the loaded method file, so saving over
  it is still one click, but it is never silent. The method takes the name of the file you save to.
  Before, Save wrote over the loaded file without asking, which could change a shared method.
- **Tail skim** (new **Tail Skim** group in Parameters). A small peak riding the decaying tail of a
  bigger one is skimmed off that tail (dashed red line, shaded) instead of cut straight down at the
  valley; the bigger peak's area no longer includes it. Modes: *Automatic* (skim when the tail allows
  it), *Always skim*, *Always split*. New sessions start on *Automatic*. **Methods saved before 0.2.3
  open on *Always split*, so their results don't change** until you pick another mode and save.
- **Enable Shoulder Detection has a new detector.** It finds peaks merged into a bigger one (shoulders)
  or riding its tail, and keeps only those a peak-shape fit confirms. Off by default. If you had it on,
  expect different shoulder peaks.
- **GC1&2 batch.** In **Batch Process Directories…**, each folder is tagged GC1 or GC2 from its name or
  its runs' names (right-click → **Set instrument** when they don't say). Tick **Process each detector below with its own method** and give
  each detector row its instrument. The batch then pairs GC1 and GC2 injections by acquisition time and
  writes one paste workbook (GC1 and GC2 sheets plus a Pairing sheet). **Save setup…** stores the rows
  for next time. Batch reads `.C` folders: use **Tools → Batch Convert to .C…** first.
- **Peak windows (first part).** Each **Peak Range Filters** row is now a peak window and each **Signal
  Break Point** starts a baseline section; both can carry their own settings in the saved
  `.chromethod`. For now those settings are edited in the file itself; controls in the app come later.
  Methods without them work as before.
- Export to CSV keeps every column when rows have different fields.
