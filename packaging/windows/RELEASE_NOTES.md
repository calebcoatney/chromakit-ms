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
- GCxGC peak lists change on TIC detection. **Ignore persistent bands** (new, on by default) stops air and
  column bleed from being reported as long peaks, so their false markers go and real streaks get their
  own. Peak volumes still integrate the full signal. Turn it off for dense samples, where real peaks can
  fill a 2D column for minutes. FID detection is unchanged.
- **Solvent Cut-off (min)** (new, blank by default) ignores peaks with an apex before that first-dimension
  time.
- A run's saved Phase Shift can cut through its peaks. If most of the signal sits against the top and
  bottom edges of the plot, enter a Phase Shift that moves it clear, then click Integrate.
- Re-export GCxGC results made with an earlier version: their peak lists can differ.
- Integrate now says why it found no peaks.
