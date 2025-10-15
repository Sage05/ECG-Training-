# Interpretable Deep Learning for 12-lead ECG Diagnosis

Reproduction and extension of the research paper "Interpretable Deep Learning for Automatic Diagnosis of 12-lead Electrocardiogram" by Zhang et al. (2020).

## 📄 Paper Information

**Title:** Interpretable Deep Learning for Automatic Diagnosis of 12-lead Electrocardiogram  
**Authors:** Dongdong Zhang, Xiaohui Yuan, Ping Zhang  
**Source:** arXiv:2010.10328 [cs.LG]  
**Original Repository:** https://github.com/onlyzdd/ecg-diagnosis

## 🎯 Project Overview

This project reproduces the deep learning methodology from the original paper and extends it by:
- Training on the CPSC2018 dataset (original paper's dataset)
- **Additional training on MIT-BIH Arrhythmia Database** (different dataset for extended validation)
- Cross-dataset evaluation between CPSC2018 and MIT-BIH
- Implementing a comprehensive evaluation pipeline
- Creating visualization dashboards for model performance

## 🏗️ Model Architecture

**ECGNet**: Deep Convolutional Neural Network with Residual Blocks
- **Input:** 12-lead ECG signals (30s duration, 500 Hz sampling rate = 15,000 samples)
- **Architecture:** 34-layer network with stacked residual blocks
- **Output:** Multi-label classification for 9 cardiac arrhythmias

### Architecture Details:
- Initial Conv1D layer (64 filters)
- 4 Residual Blocks (128 → 256 → 512 → 512 filters)
- Batch Normalization and Dropout (0.2)
- Adaptive Average + Max Pooling
- Fully Connected output layer with Sigmoid activation

## 📊 Datasets Used

### 1. CPSC2018 Dataset (Training)
- **Source:** China Physiological Signal Challenge 2018
- **Size:** 6,877 12-lead ECG recordings
- **Duration:** 6-60 seconds per recording
- **Sampling Rate:** 500 Hz
- **Classes:** 9 cardiac conditions

**Diagnostic Classes:**
1. Normal Sinus Rhythm (SNR)
2. Atrial Fibrillation (AF)
3. First-degree Atrioventricular Block (IAVB)
4. Left Bundle Branch Block (LBBB)
5. Right Bundle Branch Block (RBBB)
6. Premature Atrial Contraction (PAC)
7. Premature Ventricular Contraction (PVC)
8. ST-segment Depression (STD)
9. ST-segment Elevation (STE)

### 2. MIT-BIH Arrhythmia Database (Extended Training & Evaluation)
- **Source:** MIT-BIH Arrhythmia Database via Kaggle
- **Purpose:** Extended training on different dataset + cross-dataset evaluation
- **Size:** 87,554 training samples, 21,892 test samples
- **Classes:** 5 heartbeat types
  - **N:** Normal beats
  - **S:** Supraventricular ectopic beats
  - **V:** Ventricular ectopic beats
  - **F:** Fusion beats
  - **Q:** Unknown beats
- **Signal Length:** 187 samples per heartbeat
- **Lead:** Single-lead ECG (Lead II)

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (recommended)
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ecg-diagnosis-reproduction.git
cd ecg-diagnosis-reproduction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Dataset Setup

1. **CPSC2018 Dataset:**
   - Download from the official CPSC2018 challenge source
   - Convert to Parquet format (should contain `signal_*` columns and labels)
   - Place at: `M:\MODEL\cpsc_data.parquet` (or update path in `train_model.py` and `evaluate_cpsc.py`)

2. **MIT-BIH Dataset:**
   - Download from Kaggle: [ECG Heartbeat Categorization Dataset](https://www.kaggle.com/shayanfazeli/heartbeat)
   - Files needed: `mitbih_train.csv` and `mitbih_test.csv`
   - Update paths in `train_model.py` (if training) and `evaluate_mitbih.py`

## 📝 Usage

### 1. Train the Model on CPSC2018

```bash
python train_model.py
```

**Configuration Options:**
- Modify paths, batch size, epochs, learning rate in the script
- Default: 10 epochs, batch size 4, learning rate 0.001
- Model saved as `best_ecg_model.pth`
- Trains on 9-class CPSC2018 dataset

### 2. Train/Evaluate on MIT-BIH (Optional - Extended Validation)

The same model architecture can be trained on MIT-BIH for 5-class classification:

```bash
# Modify train_model.py to use MIT-BIH data
# Or use evaluate_mitbih.py with pre-trained weights
python evaluate_mitbih.py
```

**Note:** `evaluate_mitbih.py` loads CPSC-trained weights and adapts them for MIT-BIH evaluation.

### 3. Evaluate on CPSC2018

```bash
python evaluate_cpsc.py
```

Outputs:
- Performance metrics (Precision, Recall, F1, AUC)
- Confusion matrices visualization
- JSON metrics for dashboard

### 4. Evaluate on MIT-BIH

```bash
python evaluate_mitbih.py
```

Outputs:
- Cross-dataset evaluation results
- Confusion matrices for 5 heartbeat classes
- JSON metrics for dashboard

### 5. Generate Combined Dashboard

```bash
python merge_metrics.py
```

Then open `dashboard.html` in your browser to view interactive results.

## 📈 Results

### CPSC2018 Dataset Performance

| Class | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|----------|-----|
| SNR   | 0.814     | 0.800  | 0.805    | 0.974 |
| AF    | 0.920     | 0.918  | 0.919    | 0.988 |
| IAVB  | 0.868     | 0.865  | 0.864    | 0.987 |
| LBBB  | 0.844     | 0.894  | 0.866    | 0.980 |
| RBBB  | 0.911     | 0.942  | 0.926    | 0.987 |
| PAC   | 0.756     | 0.720  | 0.735    | 0.949 |
| PVC   | 0.869     | 0.839  | 0.851    | 0.976 |
| STD   | 0.808     | 0.826  | 0.814    | 0.971 |
| STE   | 0.603     | 0.504  | 0.535    | 0.923 |
| **AVG** | **0.821** | **0.812** | **0.813** | **0.970** |

### MIT-BIH Cross-Dataset Evaluation

Results demonstrate model generalization to different ECG datasets with different classification tasks:
- **Dataset Difference:** CPSC2018 uses 12-lead full ECG recordings, MIT-BIH uses single-lead segmented heartbeats
- **Task Difference:** CPSC2018 has 9 rhythm/morphology classes, MIT-BIH has 5 heartbeat classes
- **Performance:** Model trained on CPSC2018, adapted and evaluated on MIT-BIH

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal (N) | 0.887 | 0.945 | 0.915 |
| Supraventricular (S) | 0.756 | 0.692 | 0.723 |
| Ventricular (V) | 0.834 | 0.721 | 0.773 |
| Fusion (F) | 0.612 | 0.487 | 0.543 |
| Unknown (Q) | 0.698 | 0.756 | 0.726 |
| **AVG** | **0.757** | **0.720** | **0.736** |

## 📁 Project Structure

```
ecg-diagnosis-reproduction/
├── train_model.py              # Main training script
├── evaluate_cpsc.py            # CPSC dataset evaluation
├── evaluate_mitbih.py          # MIT-BIH dataset evaluation
├── metrics_exporter.py         # Utility for exporting metrics
├── merge_metrics.py            # Combine all metrics for dashboard
├── dashboard.html              # Interactive results visualization
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── METHODOLOGY.md              # Detailed methodology
├── best_ecg_model.pth          # Trained model weights
└── dashboard_data/             # Exported JSON metrics
    ├── cpsc_training_metrics.json
    ├── cpsc_evaluation.json
    ├── mitbih_evaluation.json
    └── combined_metrics.json
```

## 🔬 Key Differences from Original Paper

1. **Dataset:** 
   - Original: CPSC2018 only
   - Ours: CPSC2018 (primary) + MIT-BIH (extended validation)

2. **Cross-Dataset Validation:**
   - Original: Single dataset evaluation
   - Ours: Cross-dataset evaluation on MIT-BIH to test generalization

3. **Additional Features:**
   - Transfer learning approach (CPSC → MIT-BIH)
   - Automated metrics export system
   - Interactive dashboard for results visualization
   - Modular evaluation pipeline for multiple datasets

4. **Implementation Details:**
   - PyTorch implementation (same as original)
   - Added comprehensive data preprocessing for both datasets
   - Extended evaluation metrics
   - Confusion matrix visualization for all classes

## 🎓 Learning Outcomes

- Understanding of deep learning for ECG signal processing
- Experience with medical time-series data (both 12-lead and single-lead)
- Multi-label vs multi-class classification techniques
- Cross-dataset validation and transfer learning strategies
- Model evaluation and performance analysis across different domains
- Handling different ECG data formats (Parquet, CSV)
- Data imbalance and domain adaptation challenges

## 🛠️ Technologies Used

- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Data Format:** Parquet (for efficient storage)

## ⚠️ Known Issues

1. STE class shows lower F1 score (0.535) due to physician disagreement in diagnosis
2. Model performance varies with different data preprocessing approaches
3. Cross-dataset evaluation shows domain shift effects

## 🔮 Future Improvements

- [ ] Implement SHAP interpretability analysis
- [ ] Add single-lead model evaluation
- [ ] Incorporate data augmentation techniques
- [ ] Experiment with attention mechanisms
- [ ] Add real-time prediction interface

## 📚 References

1. Zhang, D., Yuan, X., & Zhang, P. (2020). Interpretable Deep Learning for Automatic Diagnosis of 12-lead Electrocardiogram. arXiv:2010.10328.
2. CPSC2018 Challenge: http://2018.icbeb.org/
3. MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/

## 👥 Contributors

- [Your Name] - Group Member 1
- [Partner Name] - Group Member 2

## 📄 License

This project is for educational purposes as part of the Machine Learning course mini-project.

## 🙏 Acknowledgments

- Original authors: Dongdong Zhang, Xiaohui Yuan, Ping Zhang
- CPSC2018 Challenge organizers
- PhysioNet for MIT-BIH database

---

**Course:** Machine Learning Laboratory  
**Institution:** Somaiya Vidyavihar University  
**Academic Year:** 2024-25
