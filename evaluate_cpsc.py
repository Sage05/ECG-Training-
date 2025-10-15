import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from metrics_exporter import MetricsExporter

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# =================================================================================
# SECTION 1: MODEL DEFINITION (Required to load the saved model)
# =================================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return self.relu(out)

class ECGNet(nn.Module):
    def __init__(self, num_classes=9, input_channels=12, kernel_size=15, dropout=0.2):
        super(ECGNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size, stride=1, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(64, 128, kernel_size, stride=2, dropout=dropout)
        self.layer2 = ResidualBlock(128, 256, kernel_size, stride=2, dropout=dropout)
        self.layer3 = ResidualBlock(256, 512, kernel_size, stride=2, dropout=dropout)
        self.layer4 = ResidualBlock(512, 512, kernel_size, stride=2, dropout=dropout)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * 2, num_classes)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return torch.sigmoid(out)

# =================================================================================
# SECTION 2: DATA LOADING & PREDICTION
# =================================================================================

class ECGParquetDataset(Dataset):
    def __init__(self, dataframe, label_cols, max_signal_length=None):
        self.df = dataframe
        self.label_cols = label_cols
        self.max_signal_length = max_signal_length
        self.signal_cols = [col for col in self.df.columns if col.startswith('signal_')]
        signal_data = self.df[self.signal_cols].values.astype(np.float32)
        if self.max_signal_length and signal_data.shape[1] > self.max_signal_length:
            signal_data = signal_data[:, :self.max_signal_length]
        self.signals = np.expand_dims(signal_data, axis=1)
        self.labels = self.df[self.label_cols].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label

def predict_on_new_data(model, test_loader, device):
    print("\n--- Applying model to new data ---")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for signals, labels in tqdm(test_loader, desc="Predicting"):
            signals = signals.to(device)
            outputs = model(signals)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.vstack(all_preds), np.vstack(all_labels)

# =================================================================================
# SECTION 3: EVALUATION & PLOTTING
# =================================================================================

def evaluate_performance(y_true, y_pred_proba, class_names, threshold=0.5):
    print(f"\n--- Model Performance Evaluation (Threshold = {threshold}) ---")
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='samples', zero_division=0)
    print(f"Overall Sample-based Metrics:")
    print(f"  - Accuracy (Subset): {accuracy_score(y_true, y_pred_binary):.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-Score: {f1:.4f}\n")
    print("Per-Class Metrics:")
    for i, class_name in enumerate(class_names):
        p, r, f, _ = precision_recall_fscore_support(y_true[:, i], y_pred_binary[:, i], average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            print(f"  - {class_name}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, AUC={auc:.4f}")
        except ValueError:
            print(f"  - {class_name}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, AUC=N/A")

def plot_and_save_confusion_matrices(y_true, y_pred_proba, class_names, threshold=0.5, save_path="confusion_matrix_results.png"):
    print(f"\nGenerating and saving confusion matrix plot to {save_path}...")
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    # Set up the plot grid
    num_classes = len(class_names)
    grid_size = int(np.ceil(np.sqrt(num_classes)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 3.5), sharey=True)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_classes:
            class_name = class_names[i]
            cm = confusion_matrix(y_true[:, i], y_pred_binary[:, i], normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot(include_values=True, cmap='Blues', ax=ax, colorbar=False, values_format=".3f")
            ax.set_title(class_name)
            if i < (grid_size * (grid_size - 1)): ax.set_xlabel('')
            if i % grid_size != 0: ax.set_ylabel('')
        else:
            ax.axis('off') # Hide unused subplots
    
    fig.subplots_adjust(right=0.85) 
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(disp.im_, cax=cbar_ax)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Predicted label", labelpad=20, fontsize=14)
    plt.ylabel("True label", labelpad=30, fontsize=14)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print("Plot saved successfully.")

# =================================================================================
# MAIN EXECUTION BLOCK
# =================================================================================
def main():
    # --- 1. CONFIGURATION ---
    # ** Paths to your trained model and the NEW dataset you want to evaluate **
    MODEL_PATH = "best_ecg_model.pth"
    NEW_DATA_PATH = r"M:\MODEL\cpsc_data.parquet" # <-- IMPORTANT: Point this to your new dataset
    
    # ** Model & Data Parameters (Must match the trained model) **
    NUM_CLASSES = 9  # Adjust if your model was trained on a different number of classes
    INPUT_CHANNELS = 1
    MAX_SIGNAL_LENGTH = 30000
    BATCH_SIZE = 32 # Can be larger for evaluation as it uses less memory than training
    
    # --- 2. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. LOAD DATASET ---
    print(f"Loading new dataset from {NEW_DATA_PATH}...")
    df = pd.read_parquet(NEW_DATA_PATH)
    
    label_cols = [
        'Sinus Rhythm', 'Atrial Fibrillation', '1 degree atrioventricular block',
        'left front bundle branch block', 'right bundle branch block', 'atrial premature beats',
        'Ventricular Ectopics', 'ST drop down', 'ST tilt up'
    ]
    existing_label_cols = [col for col in label_cols if col in df.columns]
    
    # Update NUM_CLASSES based on the labels found in the new dataset
    if len(existing_label_cols) != NUM_CLASSES:
        print(f"Warning: Found {len(existing_label_cols)} labels, but model expects {NUM_CLASSES}. Adjusting.")
        NUM_CLASSES = len(existing_label_cols)

    dataset = ECGParquetDataset(df, label_cols=existing_label_cols, max_signal_length=MAX_SIGNAL_LENGTH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 4. LOAD PRE-TRAINED MODEL ---
    print(f"Loading pre-trained model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please make sure the path is correct.")
        return
        
    model = ECGNet(num_classes=NUM_CLASSES, input_channels=INPUT_CHANNELS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # --- 5. GET PREDICTIONS AND EVALUATE ---
    predictions_proba, true_labels = predict_on_new_data(model, data_loader, device)
    
    evaluate_performance(true_labels, predictions_proba, class_names=existing_label_cols)
    
    plot_and_save_confusion_matrices(true_labels, predictions_proba, class_names=existing_label_cols)

    # --- 6. EXPORT METRICS FOR DASHBOARD ---
    print("\n--- Exporting metrics to JSON for dashboard ---")
    exporter = MetricsExporter(output_dir="dashboard_data")
    exporter.add_evaluation_metrics("cpsc", true_labels, predictions_proba, existing_label_cols)
    exporter.save_to_json("cpsc_evaluation.json")
    print("✅ CPSC metrics exported successfully!")

if __name__ == '__main__':
    main()