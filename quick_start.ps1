# Quick Start Script for ASD Video Classification
# PowerShell script to run the complete workflow

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ASD Video Classification - Quick Start" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please create it first:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

# Step 1: Verify dataset
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 1: Verifying Dataset Structure" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

python verify_dataset.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Dataset verification failed!" -ForegroundColor Red
    Write-Host "Please check your dataset structure and paths." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Pause to let user review verification results
Write-Host ""
Read-Host "Dataset verification complete. Press Enter to continue with training"

# Step 2: Train model
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Step 2: Training Model" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This may take 30 minutes to 2 hours depending on your hardware." -ForegroundColor Yellow
Write-Host "Feature extraction (first time only): ~30-60 minutes" -ForegroundColor Yellow
Write-Host "Model training: ~10-60 minutes" -ForegroundColor Yellow
Write-Host ""

python train_asd_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Training failed!" -ForegroundColor Red
    Write-Host "Check the error messages above." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 3: Ask if user wants to run prediction
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$runPredict = Read-Host "Would you like to run predictions on the test set now? (Y/N)"
if ($runPredict -eq "Y" -or $runPredict -eq "y") {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Step 3: Running Predictions" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    python predict_asd_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: Prediction failed!" -ForegroundColor Red
        Write-Host "Check the error messages above." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "All Steps Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Check the following outputs:" -ForegroundColor Green
    Write-Host "  - Model files: models\autism_data\" -ForegroundColor White
    Write-Host "  - Training plot: reports\autism_data\" -ForegroundColor White
    Write-Host "  - Predictions CSV: predictions_*.csv" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Training complete! You can run predictions later with:" -ForegroundColor Yellow
    Write-Host "  python predict_asd_model.py" -ForegroundColor White
    Write-Host ""
}

Read-Host "Press Enter to exit"
