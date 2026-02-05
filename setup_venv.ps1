# ============================================================================
# ASD Screening Project - Virtual Environment Setup Script
# ============================================================================
# This script sets up a Python 3.11 environment with TensorFlow 2.15
# Compatible with CUDA 12.x for GPU acceleration
# ============================================================================

Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "ASD SCREENING PROJECT - ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "============================================================================`n" -ForegroundColor Cyan

# Step 1: Check Python version
Write-Host "[1/9] Checking Python version..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "Detected: $pythonVersion" -ForegroundColor Green
    
    # Extract version number
    if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        
        if ($major -eq 3 -and $minor -eq 11) {
            Write-Host "✓ Python 3.11 detected - PERFECT!" -ForegroundColor Green
        }
        elseif ($major -eq 3 -and $minor -eq 13) {
            Write-Host "`n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
            Write-Host "ERROR: Python 3.13 detected!" -ForegroundColor Red
            Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
            Write-Host "`nTensorFlow 2.15 requires Python 3.11" -ForegroundColor Yellow
            Write-Host "`nPlease download Python 3.11.8 from:" -ForegroundColor Yellow
            Write-Host "https://www.python.org/downloads/release/python-3118/" -ForegroundColor Cyan
            Write-Host "`nInstallation instructions:" -ForegroundColor Yellow
            Write-Host "1. Download 'Windows installer (64-bit)'" -ForegroundColor White
            Write-Host "2. Run installer" -ForegroundColor White
            Write-Host "3. Check 'Add Python 3.11 to PATH'" -ForegroundColor White
            Write-Host "4. Complete installation" -ForegroundColor White
            Write-Host "5. Restart PowerShell" -ForegroundColor White
            Write-Host "6. Run this script again" -ForegroundColor White
            Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
            $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
            exit 1
        }
        else {
            Write-Host "`nWARNING: Python $major.$minor detected" -ForegroundColor Yellow
            Write-Host "Recommended: Python 3.11" -ForegroundColor Yellow
            Write-Host "Download from: https://www.python.org/downloads/release/python-3118/" -ForegroundColor Cyan
            $continue = Read-Host "`nContinue anyway? (yes/no)"
            if ($continue -ne "yes") {
                Write-Host "Setup cancelled." -ForegroundColor Red
                exit 1
            }
        }
    }
}
catch {
    Write-Host "`nERROR: Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.11 from:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/release/python-3118/" -ForegroundColor Cyan
    Write-Host "`nPress any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Step 2: Create virtual environment
Write-Host "`n[2/9] Creating virtual environment 'venv_asd'..." -ForegroundColor Yellow

if (Test-Path "venv_asd") {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
    $recreate = Read-Host "Recreate it? (yes/no)"
    if ($recreate -eq "yes") {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force "venv_asd"
        python -m venv venv_asd
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    }
} else {
    python -m venv venv_asd
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Step 3: Activate virtual environment
Write-Host "`n[3/9] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv_asd\Scripts\Activate.ps1"
Write-Host "✓ Virtual environment activated" -ForegroundColor Green

# Step 4: Upgrade pip
Write-Host "`n[4/9] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
Write-Host "✓ Pip upgraded" -ForegroundColor Green

# Step 5: Install TensorFlow
Write-Host "`n[5/9] Installing TensorFlow 2.15.0..." -ForegroundColor Yellow
Write-Host "(This may take 5-10 minutes - downloading ~500 MB)" -ForegroundColor Cyan
pip install tensorflow==2.15.0
Write-Host "✓ TensorFlow installed" -ForegroundColor Green

# Step 6: Install other dependencies
Write-Host "`n[6/9] Installing other dependencies..." -ForegroundColor Yellow
pip install keras==2.15.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install scikit-learn==1.3.0
pip install scipy==1.11.0
pip install matplotlib==3.7.2
pip install opencv-contrib-python==4.8.0.74
pip install imutils==0.5.4
Write-Host "✓ All dependencies installed" -ForegroundColor Green

# Step 7: Test installations
Write-Host "`n[7/9] Testing installations..." -ForegroundColor Yellow

Write-Host "  Testing TensorFlow..." -ForegroundColor Cyan
python -c "import tensorflow as tf; print('  ✓ TensorFlow:', tf.__version__)"

Write-Host "  Testing Keras..." -ForegroundColor Cyan
python -c "import keras; print('  ✓ Keras:', keras.__version__)"

Write-Host "  Testing NumPy..." -ForegroundColor Cyan
python -c "import numpy as np; print('  ✓ NumPy:', np.__version__)"

Write-Host "  Testing OpenCV..." -ForegroundColor Cyan
python -c "import cv2; print('  ✓ OpenCV:', cv2.__version__)"

Write-Host "  Testing Pandas..." -ForegroundColor Cyan
python -c "import pandas as pd; print('  ✓ Pandas:', pd.__version__)"

Write-Host "  Testing Scikit-learn..." -ForegroundColor Cyan
python -c "import sklearn; print('  ✓ Scikit-learn:', sklearn.__version__)"

# Step 8: Test GPU detection
Write-Host "`n[8/9] Testing GPU detection..." -ForegroundColor Yellow
python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print('  GPUs detected:', len(gpus)); [print('  -', gpu) for gpu in gpus] if gpus else print('  ⚠ No GPU detected - will use CPU (slower but works)')"

# Step 9: Success message
Write-Host "`n============================================================================" -ForegroundColor Green
Write-Host "SETUP COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green

Write-Host "`nEnvironment: venv_asd" -ForegroundColor White
Write-Host "Location: $PWD\venv_asd" -ForegroundColor White

Write-Host "`n============================================================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "1. Create mini test dataset:" -ForegroundColor White
Write-Host "   python create_mini_dataset.py" -ForegroundColor Yellow
Write-Host "`n2. Run quick test (5-15 minutes):" -ForegroundColor White
Write-Host "   python quick_test.py" -ForegroundColor Yellow
Write-Host "`n3. For full training:" -ForegroundColor White
Write-Host "   - Edit config.py: USE_MINI_DATASET = False" -ForegroundColor Yellow
Write-Host "   - Run: python train_simple.py" -ForegroundColor Yellow
Write-Host "============================================================================`n" -ForegroundColor Cyan

Write-Host "Virtual environment is activated. Ready to use!" -ForegroundColor Green
Write-Host "`nTo activate later, run:" -ForegroundColor Yellow
Write-Host ".\venv_asd\Scripts\Activate.ps1`n" -ForegroundColor Cyan
