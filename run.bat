@echo off
rem Launches ChromaKit-MS. On first run, creates .venv (Python 3.13) and installs dependencies.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where uv >nul 2>nul
    if errorlevel 1 (
        echo uv is required to create the environment: https://docs.astral.sh/uv/
        pause
        exit /b 1
    )
    uv venv .venv --python 3.13 --seed || (pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install -e . || (pause & exit /b 1)
)

".venv\Scripts\python.exe" main.py %*
if errorlevel 1 pause
