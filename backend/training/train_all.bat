@echo off
echo ============================================================
echo BridgeCommApp Model Training Pipeline
echo ============================================================
echo.

cd /d "%~dp0"

echo Checking for Python virtual environment...
if exist "../.venv/Scripts/activate.bat" (
    call "../.venv/Scripts/activate.bat"
    echo Virtual environment activated.
) else (
    echo WARNING: Virtual environment not found. Using system Python.
)

echo.
echo Installing training dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ============================================================
echo STEP 1: Checking Datasets
echo ============================================================
echo.

if not exist "datasets\asl-alphabet\asl_alphabet_train\A" (
    echo ERROR: ASL Alphabet dataset not found!
    echo.
    echo Please download from:
    echo   https://www.kaggle.com/datasets/grassknoted/asl-alphabet
    echo.
    echo Extract to: training\datasets\asl-alphabet\
    echo.
    pause
    exit /b 1
)

if not exist "datasets\fer2013\train\angry" (
    echo ERROR: FER-2013 dataset not found!
    echo.
    echo Please download from:
    echo   https://www.kaggle.com/datasets/msambare/fer2013
    echo.
    echo Extract to: training\datasets\fer2013\
    echo.
    pause
    exit /b 1
)

echo Datasets found!
echo.

echo ============================================================
echo STEP 2: Training ASL Gesture Model
echo ============================================================
echo.
echo This will take 15-30 minutes...
python train_asl_gestures.py
if errorlevel 1 (
    echo WARNING: ASL training had errors
)

echo.
echo ============================================================
echo STEP 3: Training Emotion Detection Model  
echo ============================================================
echo.
echo This will take 30-60 minutes...
python train_emotion_model.py
if errorlevel 1 (
    echo WARNING: Emotion training had errors
)

echo.
echo ============================================================
echo STEP 4: Deploying Models
echo ============================================================
echo.
python deploy_models.py

echo.
echo ============================================================
echo TRAINING COMPLETE!
echo ============================================================
echo.
echo Models have been deployed to backend\models\
echo Please restart the backend server to use the new models.
echo.
pause
