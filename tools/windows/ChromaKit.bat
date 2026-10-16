@echo off
rem Start ChromaKit from this clone with the Python that update_chromakit.bat found.
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "SAVED=%~dp0chromakit-python.txt"
if not exist "%SAVED%" ( echo Run update_chromakit.bat once first. & pause & exit /b 1 )
set /p PY=<"%SAVED%"
for %%I in ("%PY%") do set "PYW=%%~dpIpythonw.exe"
if not exist "%PYW%" set "PYW=%PY%"
start "" "%PYW%" "%REPO%\main.py"
