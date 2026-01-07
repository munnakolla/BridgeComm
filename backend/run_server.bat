@echo off
echo Starting BridgeComm Backend Server...
cd /d %~dp0
set PYTHONPATH=%~dp0

if exist "..\.venv\Scripts\python.exe" (
    echo Using virtual environment at ../.venv
    "../.venv/Scripts/python.exe" -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
) else (
    echo Virtual environment not found. Using system Python.
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
)
pause
