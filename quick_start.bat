@echo off
REM Quick Start Script for ASD Video Classification
REM Windows PowerShell script to run the complete workflow

echo ========================================
echo ASD Video Classification - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please create it first:
    echo   python -m venv venv
    echo   .\venv\Scripts\Activate.ps1
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Step 1: Verify dataset
echo.
echo ========================================
echo Step 1: Verifying Dataset Structure
echo ========================================
echo.
python verify_dataset.py
if errorlevel 1 (
    echo.
    echo ERROR: Dataset verification failed!
    echo Please check your dataset structure and paths.
    pause
    exit /b 1
)

REM Pause to let user review verification results
echo.
echo Dataset verification complete. Press any key to continue with training...
pause >nul

REM Step 2: Train model
echo.
echo ========================================
echo Step 2: Training Model
echo ========================================
echo.
echo This may take 30 minutes to 2 hours depending on your hardware.
echo Feature extraction (first time only): ~30-60 minutes
echo Model training: ~10-60 minutes
echo.
python train_asd_model.py
if errorlevel 1 (
    echo.
    echo ERROR: Training failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

REM Step 3: Ask if user wants to run prediction
echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
set /p run_predict="Would you like to run predictions on the test set now? (Y/N): "
if /i "%run_predict%"=="Y" (
    echo.
    echo ========================================
    echo Step 3: Running Predictions
    echo ========================================
    echo.
    python predict_asd_model.py
    if errorlevel 1 (
        echo.
        echo ERROR: Prediction failed!
        echo Check the error messages above.
        pause
        exit /b 1
    )
    
    echo.
    echo ========================================
    echo All Steps Complete!
    echo ========================================
    echo.
    echo Check the following outputs:
    echo   - Model files: models\autism_data\
    echo   - Training plot: reports\autism_data\
    echo   - Predictions CSV: predictions_*.csv
    echo.
) else (
    echo.
    echo Training complete! You can run predictions later with:
    echo   python predict_asd_model.py
    echo.
)

echo Press any key to exit...
pause >nul
