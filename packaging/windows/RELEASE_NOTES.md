## ChromaKit for Windows (64-bit)

1. Download `ChromaKit-<version>-windows-x64.zip` below.
2. Right-click it → **Extract All…**, into a folder you can write to (for example
   `Documents\ChromaKit` or `D:\ChromaKit`), not Program Files.
3. Windows makes a folder named after the zip, and inside it a `ChromaKit` folder. Open that inner
   `ChromaKit` folder and double-click **ChromaKit.exe**.
4. Windows may say "Windows protected your PC". Click **More info**, then **Run anyway**. This app is
   not code-signed yet; you'll see this once per new version.

**Updating:** when a new version exists, ChromaKit shows an **Update available** button in its status
bar. Download the new zip, extract it to a new folder, then delete the old folder. Your settings are
kept (they are stored outside the app folder).

**Problems:** the log is at `%LOCALAPPDATA%\ChromaKit\logs\chromakit.log`. Running
`ChromaKit.exe --self-test "%TEMP%\ck"` writes a check to `%TEMP%\ck\self-test.txt`.

**Changes for existing users:**
- GCxGC MGF entries now follow MZmine's GC (GNPS) export: `PEPMASS` is the base peak, plus `MSLEVEL=2`,
  `RTINSECONDS`, `CHARGE` and `Num peaks`.
- Exported MGF and MSP spectra are trimmed by **Export Min S/N** (default 5). Set it to 0 for the old
  behaviour (every ion kept).
- Saved MS time offsets now live in the `overrides` folder beside the program (source installs) or under
  `%LOCALAPPDATA%\ChromaKit` (the Windows app).
