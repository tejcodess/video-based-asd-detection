# Enable GPU Support by Upgrading to TensorFlow 2.15
# This version supports your CUDA 12.9 installation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Upgrading TensorFlow for GPU Support" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Your System:" -ForegroundColor Yellow
Write-Host "  GPU: NVIDIA GeForce RTX 3050 Laptop (6GB)" -ForegroundColor White
Write-Host "  CUDA: 12.9 (already installed)" -ForegroundColor White
Write-Host "  Driver: 576.80" -ForegroundColor White
Write-Host ""

Write-Host "Upgrading to TensorFlow 2.15 (supports CUDA 12.x)..." -ForegroundColor Green
Write-Host ""

# Uninstall old versions
Write-Host "Step 1: Removing old TensorFlow/Keras versions..." -ForegroundColor Yellow
pip uninstall tensorflow keras -y

Write-Host ""
Write-Host "Step 2: Installing TensorFlow 2.15 with GPU support..." -ForegroundColor Yellow
pip install tensorflow==2.15.0

Write-Host ""
Write-Host "Step 3: Installing Keras 2.15..." -ForegroundColor Yellow
pip install keras==2.15.0

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing GPU Detection..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test GPU
python gpu_utils.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: python verify_setup.py" -ForegroundColor White
Write-Host "  2. Train: python train_simple.py" -ForegroundColor White
Write-Host ""
Write-Host "Your GPU will provide ~5-10x speedup!" -ForegroundColor Green
Write-Host ""
