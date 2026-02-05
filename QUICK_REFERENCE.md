# QUICK REFERENCE - ASD Video Classification

## Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Setup

```powershell
python verify_setup.py
```

## Check Configuration

```powershell
python config.py
```

## Training

```powershell
# Recommended (user-friendly with detailed output)
python train_simple.py

# Alternative: Original scripts
python vgg16_lstm_hi_dim_train.py
python vgg16_bidirectional_lstm_hi_dim_train.py
```

## Prediction/Testing

```powershell
# Recommended (user-friendly with detailed output)
python predict_simple.py

# Alternative: Original scripts
python vgg16_lstm_hi_dim_predict.py
python vgg16_bidirectional_lstm_hi_dim_predict.py
```

## Expected Dataset Structure

```
D:\projects\01\dataset\autism_data_anonymized\
├── training_set\
│   ├── ASD\        (Autism Spectrum Disorder videos)
│   └── TD\         (Typical Development videos)
└── testing_set\
    ├── ASD\
    └── TD\
```

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Configuration and paths |
| `verify_setup.py` | Check if everything is installed |
| `train_simple.py` | Train model (recommended) |
| `predict_simple.py` | Test model (recommended) |
| `recurrent_networks.py` | LSTM classifier core |
| `vgg16_feature_extractor.py` | Feature extraction |
| `UCF101_loader.py` | Dataset loading |

## Output Locations

| Type | Location |
|------|----------|
| Trained Models | `models/autism_data/` |
| Training Plots | `reports/autism_data/` |
| Extracted Features | `extracted_features/` |
| Prediction Results | `models/autism_data/prediction_results.txt` |

## Model Files

After training, you'll find these in `models/autism_data/`:
- `vgg16-lstm-hi-dim-config.npy` - Model configuration
- `vgg16-lstm-hi-dim-weights.h5` - Trained weights
- `vgg16-lstm-hi-dim-architecture.json` - Model architecture

## Common Issues

### "Directory does not exist"
→ Check paths in `config.py`
→ Ensure dataset is at: `D:\projects\01\dataset\autism_data_anonymized\`

### "No videos found"
→ Verify videos are in `training_set/ASD` and `training_set/TD`
→ Check file extensions: .mp4, .avi, .mov, .mkv

### "Model not found" (during prediction)
→ Train the model first: `python train_simple.py`

### Import errors
→ Reinstall: `pip install -r requirements.txt --force-reinstall`

## Training Details

- **Architecture**: VGG16 (feature extraction) + LSTM (classification)
- **Feature Dimension**: 25,088 (VGG16 without top layer)
- **Frame Rate**: 1 frame per second from videos
- **Epochs**: 100
- **Batch Size**: 625
- **Optimizer**: RMSprop
- **Loss**: Categorical cross-entropy

## Typical Workflow

1. **Setup**
   ```powershell
   python verify_setup.py
   ```

2. **Train**
   ```powershell
   python train_simple.py
   ```
   - Extracts features (cached for reuse)
   - Trains LSTM classifier
   - Saves model and plots

3. **Test**
   ```powershell
   python predict_simple.py
   ```
   - Loads trained model
   - Predicts on test videos
   - Shows accuracy metrics
   - Saves results

## Performance Expectations

| Dataset Size | Feature Extraction | Training Time |
|--------------|-------------------|---------------|
| 50-100 videos | 1-3 minutes | 5-15 minutes |
| 100-500 videos | 3-10 minutes | 15-60 minutes |
| 500+ videos | 10-30 minutes | 1-3 hours |

*Features are cached, so subsequent training runs skip extraction*

## Package Versions

| Package | Version |
|---------|---------|
| TensorFlow | 2.10.0 |
| Keras | 2.10.0 |
| NumPy | 1.23.5 |
| OpenCV | 4.5.5.64 |

## Support

- Full documentation: `PROJECT_UPDATE_GUIDE.md`
- Original README: `README.md`
- Project analysis: `PROJECT_ANALYSIS.txt`

## Tips

- Run `verify_setup.py` after any changes to configuration
- Check terminal output for detailed progress information
- Training history plots saved in `reports/autism_data/`
- Features are cached - delete cache to re-extract
- Use `train_simple.py` and `predict_simple.py` for best user experience

---

**Last Updated**: February 4, 2026
