import json
import os
from datetime import datetime

def merge_all_metrics(training_file="dashboard_data/cpsc_training_metrics.json", 
                     cpsc_eval_file="dashboard_data/cpsc_evaluation.json",
                     mitbih_eval_file="dashboard_data/mitbih_evaluation.json",
                     output_file="dashboard_data/combined_metrics.json"):
    """
    Merge all exported metrics into a single JSON file for the dashboard
    """
    combined = {
        "timestamp": datetime.now().isoformat(),
        "training": [],
        "datasets": {}
    }
    
    # Load training metrics
    if os.path.exists(training_file):
        with open(training_file, 'r') as f:
            training_data = json.load(f)
            combined["training"] = training_data.get("training", [])
            if "evaluation" in training_data:
                combined["datasets"].update(training_data["evaluation"])
        print(f"✅ Loaded training metrics from {training_file}")
    else:
        print(f"⚠️  Training file not found: {training_file}")
    
    # Load CPSC evaluation
    if os.path.exists(cpsc_eval_file):
        with open(cpsc_eval_file, 'r') as f:
            cpsc_data = json.load(f)
            if "evaluation" in cpsc_data:
                combined["datasets"].update(cpsc_data["evaluation"])
        print(f"✅ Loaded CPSC evaluation from {cpsc_eval_file}")
    else:
        print(f"⚠️  CPSC evaluation file not found: {cpsc_eval_file}")
    
    # Load MIT-BIH evaluation
    if os.path.exists(mitbih_eval_file):
        with open(mitbih_eval_file, 'r') as f:
            mitbih_data = json.load(f)
            if "evaluation" in mitbih_data:
                combined["datasets"].update(mitbih_data["evaluation"])
        print(f"✅ Loaded MIT-BIH evaluation from {mitbih_eval_file}")
    else:
        print(f"⚠️  MIT-BIH evaluation file not found: {mitbih_eval_file}")
    
    # Save combined metrics
    with open(output_file, 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ All metrics merged into {output_file}")
    print(f"   Datasets included: {list(combined['datasets'].keys())}")
    print(f"   Training epochs: {len(combined['training'])}")
    print(f"{'='*70}\n")
    
    return output_file


if __name__ == "__main__":
    print("="*70)
    print("Merging All Metrics for Dashboard")
    print("="*70 + "\n")
    
    merge_all_metrics()
    
    print("Next step: Open dashboard.html in your browser!")