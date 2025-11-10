@echo off
echo Starting ChromaKit-MS API server...
echo.
echo API documentation will be available at:
echo   http://127.0.0.1:8000/docs
echo   http://127.0.0.1:8000/redoc
echo.

REM Change to project root directory
cd /d "%~dp0\.."

REM Run uvicorn with the api.main module
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

pause
