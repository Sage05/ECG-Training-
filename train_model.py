import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import os
import warnings
from metrics_exporter import MetricsExporter

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# =================================================================================
# SECTION 1: MODEL DEFINITION
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
    def __init__(self, num_classes=9, input_channels=1, kernel_size=15, dropout=0.2):
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
# SECTION 2: DATA LOADING
# =================================================================================

class ECGParquetDataset(Dataset):
    """Dataset for 'wide' format Parquet files where signal is in many columns."""
    def __init__(self, dataframe, label_cols, max_signal_length=None):
        self.df = dataframe
        self.label_cols = label_cols
        self.max_signal_length = max_signal_length

        self.signal_cols = [col for col in self.df.columns if col.startswith('signal_')]

        print(f"Found {len(self.signal_cols)} signal columns. Stacking into memory...")
        signal_data = self.df[self.signal_cols].values.astype(np.float32)
        
        if self.max_signal_length and signal_data.shape[1] > self.max_signal_length:
            print(f"Truncating signal length from {signal_data.shape[1]} to {self.max_signal_length}")
            signal_data = signal_data[:, :self.max_signal_length]

        self.signals = np.expand_dims(signal_data, axis=1)
        self.labels = self.df[self.label_cols].values.astype(np.float32)
        print("Done stacking.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label

def load_data_from_parquet(path, test_size=0.2, new_data_fraction=0.1):
    """Loads wide-format data and identifies signal vs. label columns."""
    print(f"Loading and splitting data from {path}...")
    df = pd.read_parquet(path)

    label_cols = [
        'Sinus Rhythm', 'Atrial Fibrillation', '1 degree atrioventricular block',
        'left front bundle branch block', 'right bundle branch block', 'atrial premature beats',
        'Ventricular Ectopics', 'ST drop down', 'ST tilt up'
    ]

    existing_label_cols = [col for col in label_cols if col in df.columns]
    if len(existing_label_cols) != len(label_cols):
        print("Warning: Not all defined label columns were found in the Parquet file.")
    label_cols = existing_label_cols

    main_df, new_unseen_df = train_test_split(df, test_size=new_data_fraction, random_state=42)
    train_df, val_df = train_test_split(main_df, test_size=test_size, random_state=42)

    print(f"Using {len(label_cols)} label columns: {label_cols}")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"New (unseen) test set size: {len(new_unseen_df)}")

    return train_df, val_df, new_unseen_df, len(label_cols), label_cols

# =================================================================================
# SECTIONS 3, 5, 6: TRAIN, APPLY, EVALUATE
# =================================================================================

def train_model(train_loader, val_loader, model, criterion, optimizer, device, epochs, model_save_path, exporter=None):
    """Main training and validation loop."""
    best_val_loss = float('inf')
    print("\n--- Starting Model Training ---")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * signals.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for signals, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * signals.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if exporter:
            exporter.add_epoch_metrics(epoch + 1, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved. Model saved to {model_save_path}")

    print("--- Finished Training ---")

def predict_on_new_data(model, test_loader, device):
    """Applies the trained model to a new dataset to get predictions."""
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

def evaluate_performance(y_true, y_pred_proba, class_names, threshold=0.5):
    """Calculates and prints classification metrics."""
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

# =================================================================================
# MAIN EXECUTION BLOCK
# =================================================================================

def main():
    # --- 1. CONFIGURATION ---
    PARQUET_PATH = r"M:\MODEL\cpsc_data.parquet"  # ⚠️ UPDATE THIS PATH
    MODEL_SAVE_PATH = "best_ecg_model.pth"
    
    BATCH_SIZE = 4
    MAX_SIGNAL_LENGTH = 30000
    
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DROPOUT = 0.3
    INPUT_CHANNELS = 1
    DO_TRAINING = True

    # --- 2. SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize metrics exporter
    exporter = MetricsExporter(output_dir="dashboard_data")

    # --- 3. LOAD AND PREPARE DATA ---
    train_df, val_df, test_df, num_classes, class_names = load_data_from_parquet(PARQUET_PATH)

    train_dataset = ECGParquetDataset(train_df, label_cols=class_names, max_signal_length=MAX_SIGNAL_LENGTH)
    val_dataset = ECGParquetDataset(val_df, label_cols=class_names, max_signal_length=MAX_SIGNAL_LENGTH)
    test_dataset = ECGParquetDataset(test_df, label_cols=class_names, max_signal_length=MAX_SIGNAL_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 4. INITIALIZE MODEL ---
    model = ECGNet(num_classes=num_classes, input_channels=INPUT_CHANNELS, dropout=DROPOUT).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. TRAIN THE MODEL ---
    if DO_TRAINING:
        train_model(train_loader, val_loader, model, criterion, optimizer, device, EPOCHS, MODEL_SAVE_PATH, exporter=exporter)
    else:
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"Error: Training is skipped but no model found at {MODEL_SAVE_PATH}.")
            return
        print(f"Skipping training as requested.")

    # --- 6. LOAD BEST MODEL AND EVALUATE ---
    print(f"\nLoading best model from {MODEL_SAVE_PATH} for final evaluation.")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    predictions_proba, true_labels = predict_on_new_data(model, test_loader, device)

    evaluate_performance(true_labels, predictions_proba, class_names)
    
    # Export evaluation metrics
    exporter.add_evaluation_metrics("cpsc_test", true_labels, predictions_proba, class_names)
    exporter.save_to_json("cpsc_training_metrics.json")
    
    print("\n" + "="*70)
    print("✅ Training complete! Metrics saved to dashboard_data/")
    print("="*70)

if __name__ == '__main__':
    main() 