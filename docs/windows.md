# ChromaKit on Windows

## The app (no Python needed)

1. Go to the [Releases page](https://github.com/calebcoatney/chromakit-ms/releases) and download
   `ChromaKit-<version>-windows-x64.zip` from the latest release.
2. Right-click the zip → **Extract All…**, into a folder you can write to (for example
   `Documents\ChromaKit` or `D:\ChromaKit`). Not Program Files.
3. Windows makes a folder named after the zip, and inside it a `ChromaKit` folder. Open that inner
   `ChromaKit` folder and double-click **ChromaKit.exe**.
4. If Windows says "Windows protected your PC": **More info** → **Run anyway**. The app is not
   code-signed yet, so this appears once per new version.

**Updating:** ChromaKit checks GitHub when it starts. If a newer version exists, an **Update
available** button appears in the status bar; it opens the release page. Download the new zip and
extract it to a **new** folder, then delete the old folder. Do not extract over the old folder. Settings
are kept because they live in the Windows registry, outside the app folder. Turn the check off under
**Settings → Check for Updates at Startup**.

**Where things are:** log file `%LOCALAPPDATA%\ChromaKit\logs\chromakit.log`; MS library cache and
manual overrides under `%LOCALAPPDATA%\ChromaKit\`.

**Check an install:** open a Command Prompt in the ChromaKit folder (the one that holds `ChromaKit.exe`)
and run `ChromaKit.exe --self-test "%TEMP%\ck"`. It can take a minute. When it finishes, `%TEMP%\ck\self-test.txt`
appears; its first line is `OK` or `FAIL`.

## A clone with its own Python environment

If ChromaKit was installed from a git clone into a conda/mamba environment, double-click
`tools\windows\update_chromakit.bat` in the clone. It pulls the latest code, installs it into that
environment and offers to start ChromaKit. The first time, it looks for the environment in the usual
Miniforge folders and, if it can't tell which one, asks you to drag its `python.exe` into the window; it
remembers the answer. `tools\windows\ChromaKit.bat` starts ChromaKit afterwards.

A clone made before this release does not have `tools\windows\update_chromakit.bat` yet. Run `git pull`
once in the clone (in Git Bash or a Command Prompt opened in that folder), or ask Caleb to, before the
first use.
