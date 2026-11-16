import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.inference import predict_pipeline

def classify_tier(rul):
    if rul < 20:
        return "Critical"
    elif rul < 50:
        return "High"
    elif rul < 90:
        return "Medium"
    else:
        return "Low"

def main():
    print("Running telemetry predictions...")
    # 1. Run inference using pipeline
    test_file = PROJECT_ROOT / "data" / "FD001" / "test_FD001.txt"
    if not test_file.exists():
        print(f"Error: {test_file} not found.")
        return

    res = predict_pipeline(str(test_file), dataset="FD001", bandwidth_limit=100)
    engines = res["engines"]
    
    # Sort by engine_id to match RUL_FD001.txt order (Engine 1, 2, ..., 100)
    engines_sorted = sorted(engines, key=lambda x: x["engine_id"])
    pred_ruls = [e["predicted_rul"] for e in engines_sorted]
    
    # 2. Load ground truth RUL
    rul_file = PROJECT_ROOT / "data" / "FD001" / "RUL_FD001.txt"
    if not rul_file.exists():
        print(f"Error: {rul_file} not found.")
        return
        
    with open(rul_file, "r") as f:
        true_ruls = [float(line.strip()) for line in f.readlines() if line.strip()]

    # 3. Classify into tiers
    labels = ["Critical", "High", "Medium", "Low"]
    y_true = [classify_tier(r) for r in true_ruls]
    y_pred = [classify_tier(r) for r in pred_ruls]

    # 4. Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize by row (true class size) to show percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent = np.nan_to_num(cm_percent) # handle divide by zero if any tier is empty

    # 5. Plot
    plt.figure(figsize=(8, 6.5))
    
    # Setup dark theme style matching the dashboard
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor='#090d14')
    ax.set_facecolor('#0f1624')
    
    # Construct labels with counts and percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    sns.heatmap(
        cm_percent,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=True,
        linewidths=1.5,
        linecolor='#1e293b',
        annot_kws={"size": 11, "weight": "bold", "color": "#ffffff"}
    )

    # Styling annotations
    plt.title("Priority Urgency Tier Confusion Matrix (FD001)", fontsize=13, pad=20, weight="bold", color="#ffffff")
    plt.xlabel("Predicted Priority Tier", fontsize=11, labelpad=12, color="#94a3b8")
    plt.ylabel("True Priority Tier", fontsize=11, labelpad=12, color="#94a3b8")
    
    # Style colorbar ticks
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9, colors="#94a3b8")
    
    plt.xticks(fontsize=10, color="#94a3b8")
    plt.yticks(fontsize=10, color="#94a3b8")
    
    plt.tight_layout()
    
    # 6. Save image
    out_dir = PROJECT_ROOT / "outputs" / "FD001"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = out_dir / "confusion_matrix_FD001.png"
    plt.savefig(img_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Confusion matrix generated and saved successfully to: {img_path}")

if __name__ == "__main__":
    main()

