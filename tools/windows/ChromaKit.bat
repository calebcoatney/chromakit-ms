@echo off
rem Start ChromaKit from this clone with the Python that update_chromakit.bat found.
setlocal
set "REPO=%~dp0..\.."
for %%I in ("%REPO%") do set "REPO=%%~fI"
set "SAVED=%~dp0chromakit-python.txt"
if not exist "%SAVED%" ( echo Run update_chromakit.bat once first. & pause & exit /b 1 )
set /p PY=<"%SAVED%"
rem The env is not activated; some packages load DLLs from its Library\bin, so put its folders first on PATH.
for %%I in ("%PY%") do set "ENVDIR=%%~dpI"
set "PATH=%ENVDIR%;%ENVDIR%Library\bin;%ENVDIR%Scripts;%PATH%"
for %%I in ("%PY%") do set "PYW=%%~dpIpythonw.exe"
if not exist "%PYW%" set "PYW=%PY%"
start "" "%PYW%" "%REPO%\main.py"
