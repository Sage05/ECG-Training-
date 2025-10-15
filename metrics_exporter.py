import json
import os
from datetime import datetime

class MetricsExporter:
    """Export model metrics to JSON format for dashboard integration"""
    
    def __init__(self, output_dir="dashboard_data"):
        self.output_dir = output_dir
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "training": [],
            "evaluation": {}
        }
        
        os.makedirs(output_dir, exist_ok=True)
    
    def add_epoch_metrics(self, epoch, train_loss, val_loss):
        """Add training/validation metrics for an epoch"""
        self.metrics["training"].append({
            "epoch": epoch,
            "trainLoss": float(train_loss),
            "valLoss": float(val_loss)
        })
    
    def add_evaluation_metrics(self, dataset_name, y_true, y_pred_proba, 
                              class_names, threshold=0.5):
        """Add evaluation metrics for a dataset"""
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                                     roc_auc_score)
        import numpy as np
        
        y_pred_binary = (y_pred_proba > threshold).astype(int)
        
        # Overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='samples', zero_division=0
        )
        accuracy = accuracy_score(y_true, y_pred_binary)
        
        # Per-class metrics
        per_class_metrics = []
        for i, class_name in enumerate(class_names):
            p, r, f, _ = precision_recall_fscore_support(
                y_true[:, i], y_pred_binary[:, i], 
                average='binary', zero_division=0
            )
            
            try:
                auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
            except ValueError:
                auc = None
            
            per_class_metrics.append({
                "name": class_name,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "auc": float(auc) if auc is not None else None
            })
        
        self.metrics["evaluation"][dataset_name] = {
            "overall": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1Score": float(f1)
            },
            "perClass": per_class_metrics,
            "threshold": threshold,
            "numClasses": len(class_names)
        }
    
    def save_to_json(self, filename="model_metrics.json"):
        """Save metrics to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to {filepath}")
        return filepath