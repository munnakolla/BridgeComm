@echo off
REM ============================================================
REM BridgeComm Sign Language Models Setup Script
REM ============================================================
REM This script downloads and sets up pretrained models for
REM video-based sign language recognition.
REM
REM Models downloaded:
REM   - I3D-WLASL: Video-based sign recognition (100 ASL words)
REM   - Pose-LSTM: MediaPipe landmark-based recognition
REM   - WLASL Vocabulary: Word mappings
REM
REM Usage:
REM   setup_sign_models.bat
REM   setup_sign_models.bat --force  (re-download all)
REM   setup_sign_models.bat --verify (check status only)
REM ============================================================

echo.
echo ============================================================
echo  BridgeComm Sign Language Models Setup
echo ============================================================
echo.

REM Check if Python is available
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again.
    pause
    exit /b 1
)

REM Change to backend directory
cd /d "%~dp0"

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python
)

REM Run the download script
echo.
echo Running model download script...
echo.

if "%1"=="--verify" (
    python -m app.services.sign_language.download_models --verify
) else if "%1"=="--force" (
    python -m app.services.sign_language.download_models --force
) else (
    python -m app.services.sign_language.download_models
)

if errorlevel 1 (
    echo.
    echo WARNING: Some models may not have downloaded correctly.
    echo You can retry by running this script again.
    echo.
) else (
    echo.
    echo SUCCESS: All models are ready!
    echo.
)

REM Show model directory contents
echo.
echo Model files in backend\models\sign_language:
echo.
if exist "models\sign_language" (
    dir /b "models\sign_language"
) else (
    echo   [No models directory yet]
)

echo.
echo ============================================================
echo  Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Start the backend server: python -m uvicorn app.main:app --reload
echo   2. Test the API: GET http://localhost:8000/sign-video/status
echo   3. Send a video: POST http://localhost:8000/sign-video/recognize
echo.
pause
