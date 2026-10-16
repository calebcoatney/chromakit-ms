## ChromaKit for Windows (64-bit)

1. Download `ChromaKit-<version>-windows-x64.zip` below.
2. Right-click it → **Extract All…**, into a folder you can write to (for example
   `Documents\ChromaKit` or `D:\ChromaKit`), not Program Files.
3. Open the extracted `ChromaKit` folder and double-click **ChromaKit.exe**.
4. Windows may say "Windows protected your PC". Click **More info**, then **Run anyway**. This app is
   not code-signed yet; you'll see this once per new version.

**Updating:** when a new version exists, ChromaKit shows an **Update available** button in its status
bar. Download the new zip and extract it over the old folder (or beside it). Your settings are kept.

**Problems:** the log is at `%LOCALAPPDATA%\ChromaKit\logs\chromakit.log`. Running
`ChromaKit.exe --self-test "%TEMP%\ck"` writes a check to `%TEMP%\ck\self-test.txt`.
