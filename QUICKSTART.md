# 🚀 Getting Started - 5 Minute Quickstart

## Your Setup
- **Dataset**: `D:\projects\01\dataset\autism_data_anonymized`
- **Training videos**: 80 (40 ASD, 40 TD)
- **Testing videos**: 80 (40 ASD, 40 TD)
- **Python**: 3.9+
- **Environment**: Windows with virtual environment

---

## Step 1: Verify Everything is Ready (2 minutes)

```powershell
# Activate your virtual environment
.\venv\Scripts\Activate.ps1

# Run verification
python verify_dataset.py
```

**Expected output**:
- ✅ All prerequisites satisfied
- ✅ Training: 80 videos across 2 classes
- ✅ Testing: 80 videos across 2 classes
- ✅ Sample video is readable

If you see any ❌ errors, fix them before proceeding.

---

## Step 2: Train the Model (1-2 hours)

```powershell
python train_asd_model.py
```

**What to expect**:
1. **First 30-60 minutes**: Extracting VGG16 features from videos
   - Shows progress: "Processing class 'ASD': 40 videos"
   - Features are cached for future use

2. **Next 10-60 minutes**: Training LSTM model
   - Shows epochs: "Epoch 1/100"
   - Watch for increasing accuracy

3. **Output**:
   - Model saved: `models/autism_data/vgg16-lstm-hi-dim-weights.h5`
   - Plot saved: `reports/autism_data/vgg16-lstm-hi-dim-history.png`

**Good training signs**:
- Loss decreasing over time
- Validation accuracy > 60%
- No error messages

**Take a coffee break!** ☕ This will take a while.

---

## Step 3: Test the Model (20-40 minutes)

```powershell
python predict_asd_model.py
```

**What happens**:
- Loads trained model
- Processes 80 test videos
- Shows real-time progress and accuracy
- Saves results to CSV

**Expected output**:
```
[1/80] Processing: video_001.avi
  True: ASD | Predicted: ASD | Confidence: 0.8523 | ✓
  Running accuracy: 1.0000 (1/1)

...

Overall Accuracy: 0.6875 (55/80)

Per-Class Results:
  ASD: Accuracy: 0.7000 (28/40)
  TD: Accuracy: 0.6750 (27/40)

Predictions saved to: predictions_20260204_143025.csv
```

**Target accuracy**: 65-75% (comparable to original paper's 68.75%)

---

## Step 4: Review Results

### Check Training History
Open: `reports/autism_data/vgg16-lstm-hi-dim-history.png`
- Should show accuracy increasing and loss decreasing

### Check Predictions CSV
Open: `predictions_YYYYMMDD_HHMMSS.csv`

Example:
| Video Name | True Label | Predicted Label | Correct | Confidence |
|------------|------------|-----------------|---------|------------|
| video_001.avi | ASD | ASD | True | 0.8523 |
| video_002.avi | TD | TD | True | 0.7234 |

---

## 🎯 Success Criteria

Your model is working well if:
- ✅ Training completes without errors
- ✅ Validation accuracy reaches > 60%
- ✅ Test accuracy is 60-75%
- ✅ Both classes (ASD and TD) have similar accuracy
- ✅ Training plot shows learning (not flat lines)

---

## 🆘 Common Issues

### "Dataset not found"
- Check paths in `train_asd_model.py` and `predict_asd_model.py`
- Make sure dataset is at: `D:\projects\01\dataset\autism_data_anonymized`

### "Out of memory"
- Close other applications
- Edit `recurrent_networks.py`: Change `BATCH_SIZE = 625` to `BATCH_SIZE = 32`

### "Training is very slow"
- Check if GPU is being used: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- If no GPU, training will be slower but will still work

### "Low accuracy (<60%)"
- Try more epochs: Edit `train_asd_model.py`, change `NUM_EPOCHS = 100` to `NUM_EPOCHS = 150`
- Check training plot for overfitting
- Verify dataset videos are properly preprocessed

---

## 📚 More Information

- **Complete guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Implementation details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Original paper**: https://www.nature.com/articles/s41598-021-94378-z

---

## ⚡ One-Command Workflow

For automated execution of all steps:

```powershell
.\quick_start.ps1
```

This runs:
1. Dataset verification
2. Model training
3. Predictions (optional)

---

## 📊 Typical Timeline

| Task | First Run | Subsequent Runs |
|------|-----------|-----------------|
| Verification | 1-2 min | 1-2 min |
| Training | 1-2 hours | 10-60 min* |
| Testing | 20-40 min | 20-40 min |
| **Total** | **~2-3 hours** | **~30-90 min** |

*Subsequent training is faster because features are cached

---

## 🎓 What the Model Does

1. **VGG16**: Extracts visual features from each video frame
2. **LSTM**: Learns temporal patterns across frames
3. **Classification**: Predicts ASD vs TD for each video
4. **Output**: Confidence scores for each class

---

## ✨ Your Updated Files

The following scripts are ready to use:
- `train_asd_model.py` - Train the model
- `predict_asd_model.py` - Run predictions
- `verify_dataset.py` - Verify setup
- `quick_start.ps1` - Automated workflow

All paths are pre-configured for your dataset location!

---

**Ready? Start here:**

```powershell
# 1. Activate environment (if not already active)
.\venv\Scripts\Activate.ps1

# 2. Verify setup
python verify_dataset.py

# 3. Train model
python train_asd_model.py

# 4. Run predictions
python predict_asd_model.py
```

**Good luck! 🎯**
