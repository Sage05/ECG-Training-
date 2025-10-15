# Methodology

## Overview

This document details the methodology used to reproduce and extend the research paper "Interpretable Deep Learning for Automatic Diagnosis of 12-lead Electrocardiogram".

## 1. Data Preprocessing

### 1.1 Signal Normalization
- **Input Format:** 12-lead ECG signals stored in Parquet format
- **Signal Extraction:** Columns prefixed with `signal_*` are extracted
- **Length Standardization:** All signals are processed to 15,000 samples (30 seconds at 500 Hz)
  - Signals longer than 30s: Cropped (last 30s retained)
  - Signals shorter than 30s: Zero-padded

### 1.2 Data Augmentation (Training)
Applied during training to address data imbalance:
- **Scaling:** Random amplitude scaling
- **Shifting:** Temporal shifting of signals

### 1.3 Label Encoding
Multi-label binary encoding for 9 diagnostic classes:
```python
label_cols = [
    'Sinus Rhythm',                      # Normal
    'Atrial Fibrillation',              # AF
    '1 degree atrioventricular block',  # IAVB
    'left front bundle branch block',    # LBBB
    'right bundle branch block',         # RBBB
    'atrial premature beats',            # PAC
    'Ventricular Ectopics',             # PVC
    'ST drop down',                      # STD
    'ST tilt up'                         # STE
]
```

## 2. Model Architecture

### 2.1 Residual Block Design

```
Input → Conv1D(k=15) → BatchNorm → ReLU → Dropout(0.2) 
     → Conv1D(k=15) → BatchNorm → Add(Shortcut) → ReLU → Output
```

**Components:**
- Kernel size: 15
- Batch Normalization: Stabilizes training
- Dropout: 0.2 (prevents overfitting)
- Shortcut connection: Enables gradient flow

### 2.2 Complete Network Architecture

```
Input: [batch_size, 1, 15000]
    ↓
Conv1D (64 filters, k=15) → BatchNorm → ReLU
    ↓
ResBlock1 (64→128, stride=2)  [Output: 7500 samples]
    ↓
ResBlock2 (128→256, stride=2) [Output: 3750 samples]
    ↓
ResBlock3 (256→512, stride=2) [Output: 1875 samples]
    ↓
ResBlock4 (512→512, stride=2) [Output: 937 samples]
    ↓
AdaptiveAvgPool1D(1) + AdaptiveMaxPool1D(1)
    ↓
Concatenate → [batch_size, 1024]
    ↓
Fully Connected (1024 → 9)
    ↓
Sigmoid → [batch_size, 9]
```

**Design Rationale:**
- 1D convolutions capture temporal patterns in ECG signals
- Residual connections allow training of deep networks
- Dual pooling (avg + max) captures both average and salient features
- Sigmoid activation enables multi-label classification

## 3. Training Process

### 3.1 Data Split
- **Training:** 70% of data
- **Validation:** 20% of data (from training pool)
- **Test (Unseen):** 10% held out completely

### 3.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy (BCE) |
| Batch Size | 4 (limited by GPU memory) |
| Epochs | 10 |
| Dropout Rate | 0.3 |

### 3.3 Training Loop

```python
For each epoch:
    Training Phase:
        For each batch:
            1. Forward pass
            2. Calculate BCE loss
            3. Backpropagation
            4. Update weights with Adam optimizer
    
    Validation Phase:
        For each batch:
            1. Forward pass (no gradient)
            2. Calculate validation loss
        
        If validation_loss < best_validation_loss:
            Save model checkpoint
```

### 3.4 Early Stopping
Model with lowest validation loss is saved and used for final evaluation.

## 4. Evaluation Methodology

### 4.1 Metrics

**Sample-based Metrics (Overall):**
- Accuracy (Subset Accuracy)
- Precision
- Recall
- F1-Score

**Per-Class Metrics:**
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
- AUC: Area Under ROC Curve

### 4.2 Decision Threshold
- Default: 0.5
- Can be optimized per-class for better F1 score

### 4.3 Confusion Matrix Analysis
Normalized confusion matrices generated for each class to visualize:
- True Positive Rate (Sensitivity)
- True Negative Rate (Specificity)
- False Positive and False Negative patterns

## 5. Cross-Dataset Evaluation

### 5.1 Purpose
Evaluate model generalization by testing on MIT-BIH database (different from training data).

### 5.2 Domain Adaptation Strategy
```python
# Load pre-trained model
pretrained_dict = torch.load('best_ecg_model.pth')

# Remove final layer (class-specific)
pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                   if k not in ['fc.weight', 'fc.bias']}

# Load feature extraction layers only
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

This approach:
- Reuses learned ECG feature representations
- Adapts output layer for different number of classes
- Tests feature generalization across datasets

### 5.3 MIT-BIH Processing
- One-hot encoding of 5 heartbeat classes
- Same signal length (187 samples per heartbeat)
- Single-lead ECG (Lead II)

## 6. Reproducibility Considerations

### 6.1 Random Seeds
```python
random_state=42  # Used in train_test_split
torch.manual_seed(42)  # For weight initialization
```

### 6.2 Hardware Considerations
- CUDA-enabled GPU recommended
- Batch size adjusted based on GPU memory
- Training time: ~2-3 hours on modern GPU

### 6.3 Data Format
- Parquet format chosen for efficiency
- Large signal arrays stored column-wise
- Faster I/O compared to CSV

## 7. Key Differences from Original Paper

| Aspect | Original Paper | Our Implementation |
|--------|---------------|-------------------|
| Framework | PyTorch | PyTorch |
| Validation | 10-fold CV | Train-Val-Test split |
| Data Format | Raw WFDB | Parquet |
| Additional | SHAP interpretation | Metrics export system |
| Evaluation | Single dataset | Cross-dataset validation |

## 8. Challenges Faced

### 8.1 Data Imbalance
- **Problem:** Some classes (e.g., STE) have <300 samples
- **Solution:** Data augmentation, class-weighted sampling

### 8.2 Memory Constraints
- **Problem:** Large signal arrays (15,000 × 12 per sample)
- **Solution:** Reduced batch size, efficient data loading

### 8.3 Class Ambiguity
- **Problem:** Low performance on STE class
- **Reason:** High physician disagreement in labeling (as noted in paper)
- **Approach:** Acknowledged limitation, consistent with original findings

## 9. Validation Strategy

### 9.1 Internal Validation
- Held-out test set from same distribution
- Validates model performance on CPSC2018

### 9.2 External Validation
- MIT-BIH database (different hospital, different protocol)
- Tests generalization capability

### 9.3 Metrics Export
All metrics exported to JSON for:
- Reproducible reporting
- Dashboard visualization
- Performance tracking

## 10. Code Quality Measures

- Modular design (separate train/evaluate scripts)
- Comprehensive comments
- Progress bars for long operations
- Error handling for missing files
- Configurable paths and hyperparameters

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
3. Zhang et al. (2020). Original paper methodology section.
