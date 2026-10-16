@echo off
setlocal EnableExtensions EnableDelayedExpansion
rem Update ChromaKit in this clone and its Python environment, then offer to start it.
rem   update_chromakit.bat [--no-pull] [--unattended] [--python "C:\path\to\python.exe"]
rem --unattended: no questions, no pause, no launch (for CI).

set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "SAVED=%~dp0chromakit-python.txt"
set "PULL=1"
set "ASK=1"
set "PY="

:args
if "%~1"=="" goto args_done
if /i "%~1"=="--no-pull" set "PULL=0" & shift & goto args
if /i "%~1"=="--unattended" set "ASK=0" & shift & goto args
if /i "%~1"=="--python" set "PY=%~2" & shift & shift & goto args
echo Unknown option %1
goto fail
:args_done

echo ChromaKit folder: "%REPO%"

rem ---- 1. Get the latest code -------------------------------------------------------------
if "%PULL%"=="1" (
  set "GIT=git"
  where git >nul 2>&1 || set "GIT=%ProgramFiles%\Git\cmd\git.exe"
  "!GIT!" -C "%REPO%" pull --ff-only
  if errorlevel 1 (
    echo.
    echo Could not update the code. Someone changed files in this folder, or the internet is down.
    echo Nothing was changed. Send Caleb a photo of this window.
    goto fail
  )
)

rem ---- 2. Find the Python that runs ChromaKit ---------------------------------------------
if not defined PY if exist "%SAVED%" set /p PY=<"%SAVED%"
if defined PY if not exist "%PY%" set "PY="
if not defined PY call :find_python
if not defined PY (
  if "%ASK%"=="0" ( echo No Python found. & goto fail )
  echo.
  echo Could not find the Python that runs ChromaKit.
  echo Drag python.exe from your ChromaKit environment into this window, then press Enter.
  set /p PY=python.exe: 
  set "PY=!PY:"=!"
)
if not exist "%PY%" ( echo "%PY%" does not exist. & goto fail )
> "%SAVED%" echo %PY%
echo Python: "%PY%"

rem ---- 3. Install what ChromaKit needs ----------------------------------------------------
pushd "%REPO%"
"%PY%" -m pip install --disable-pip-version-check -e ".[ms]"
if errorlevel 1 ( popd & echo. & echo Installing failed. Send Caleb a photo of this window. & goto fail )
"%PY%" -c "import ui.app, ms_toolkit.api; print('ChromaKit is ready.')"
if errorlevel 1 ( popd & echo. & echo ChromaKit did not start its check. Send Caleb a photo of this window. & goto fail )
popd

rem ---- 4. Start it ------------------------------------------------------------------------
if "%ASK%"=="0" exit /b 0
echo.
choice /m "Start ChromaKit now"
if errorlevel 2 exit /b 0
for %%I in ("%PY%") do set "PYW=%%~dpIpythonw.exe"
if not exist "%PYW%" set "PYW=%PY%"
start "" "%PYW%" "%REPO%\main.py"
exit /b 0

:find_python
set "HITS=0"
for %%R in ("D:\miniforge3" "D:\Miniforge3" "C:\miniforge3" "C:\ProgramData\miniforge3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniforge3") do (
  if exist "%%~R\envs" for /d %%E in ("%%~R\envs\*") do (
    if exist "%%~E\python.exe" (
      "%%~E\python.exe" -m pip show chromakit-ms >nul 2>&1 && (
        set /a HITS+=1
        set "FOUND=%%~E\python.exe"
      )
    )
  )
)
if "%HITS%"=="1" set "PY=%FOUND%"
exit /b 0

:fail
if "%ASK%"=="1" pause
exit /b 1
