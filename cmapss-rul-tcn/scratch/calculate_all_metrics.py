import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.inference import predict_pipeline

def evaluate_dataset(dataset_name):
    test_file = PROJECT_ROOT / "data" / dataset_name / f"test_{dataset_name}.txt"
    rul_file = PROJECT_ROOT / "data" / dataset_name / f"RUL_{dataset_name}.txt"
    
    if not test_file.exists() or not rul_file.exists():
        print(f"[{dataset_name}] Error: Data files not found.")
        return None
        
    try:
        # Run inference using the webapp pipeline (handles rolling features dynamically)
        res = predict_pipeline(str(test_file), dataset=dataset_name, bandwidth_limit=100)
        engines = sorted(res["engines"], key=lambda x: x["engine_id"])
        pred_ruls = np.array([e["predicted_rul"] for e in engines], dtype=np.float32)
        
        # Load ground truth
        with open(rul_file, "r") as f:
            true_ruls = np.array([float(line.strip()) for line in f.readlines() if line.strip()], dtype=np.float32)
            
        # We use a threshold of 50 for both true RUL and predicted RUL 
        # to create a meaningful binary urgency split (Urgent vs Normal)
        y_true = np.where(true_ruls <= 50, "Urgent", "Normal")
        y_pred = np.where(pred_ruls <= 50, "Urgent", "Normal")
        
        # Calculate classification metrics
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, pos_label="Urgent", zero_division=0) * 100
        rec = recall_score(y_true, y_pred, pos_label="Urgent", zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, pos_label="Urgent", zero_division=0) * 100
        cm = confusion_matrix(y_true, y_pred, labels=["Urgent", "Normal"])
        
        # Calculate regression RMSE
        rmse = np.sqrt(np.mean((true_ruls - pred_ruls) ** 2))
        
        # Check if the checkpoint used rolling features
        ckpt_path = PROJECT_ROOT / "outputs" / dataset_name / f"tcn_{dataset_name}.pt"
        has_rolling = False
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            has_rolling = any("_roll" in c for c in ckpt.get("feature_cols", []))
            
        print(f"==================================================")
        print(f"DATASET: {dataset_name} (rolling_features={has_rolling})")
        print(f"==================================================")
        print(f"  * RMSE (Regression Error): {rmse:.2f} cycles")
        print(f"  * Binary Urgency Classification (Threshold <= 50):")
        print(f"    - Accuracy:  {acc:.2f}%")
        print(f"    - Precision: {prec:.2f}%")
        print(f"    - Recall:    {rec:.2f}%")
        print(f"    - F1 Score:  {f1:.2f}%")
        print(f"    - Confusion Matrix (Urgent vs Normal):\n{cm}")
        print()
        
        return {"dataset": dataset_name, "rmse": rmse, "accuracy": acc, "precision": prec, "recall": rec, "rolling": has_rolling}
        
    except Exception as e:
        print(f"[{dataset_name}] Error during evaluation: {e}")
        return None

def main():
    print("Evaluating active models on all CMAPSS subsets...")
    print("Note: If a dataset shows rolling_features=False, it is still training in the background.\n")
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    for ds in datasets:
        evaluate_dataset(ds)

if __name__ == "__main__":
    main()

