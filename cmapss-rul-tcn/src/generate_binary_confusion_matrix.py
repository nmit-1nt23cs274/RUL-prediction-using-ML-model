import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.inference import predict_pipeline

def main():
    print("Running predictions...")
    test_file = PROJECT_ROOT / "data" / "FD001" / "test_FD001.txt"
    if not test_file.exists():
        print(f"Error: {test_file} not found.")
        return

    res = predict_pipeline(str(test_file), dataset="FD001", bandwidth_limit=100)
    engines = res["engines"]
    
    # Sort by engine_id to match RUL_FD001.txt order
    engines_sorted = sorted(engines, key=lambda x: x["engine_id"])
    pred_ruls = [e["predicted_rul"] for e in engines_sorted]
    
    # Load ground truth RUL
    rul_file = PROJECT_ROOT / "data" / "FD001" / "RUL_FD001.txt"
    if not rul_file.exists():
        print(f"Error: {rul_file} not found.")
        return
        
    with open(rul_file, "r") as f:
        true_ruls = [float(line.strip()) for line in f.readlines() if line.strip()]

    # We use a threshold of 50 for both true RUL and predicted RUL 
    # to create a meaningful binary urgency split (Urgent vs Normal)
    y_true = ["Urgent" if r <= 50 else "Normal" for r in true_ruls]
    y_pred = ["Urgent" if r <= 50 else "Normal" for r in pred_ruls]

    labels = ["Urgent", "Normal"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("Calculated Confusion Matrix:")
    print(cm)

    # Recreate the exact style of the reference image
    # Style: Green on diagonal, Purple/Blue on off-diagonal
    
    # Set style
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#090d14')
    ax.set_facecolor('#0f1624')
    
    # Draw custom colored rectangles to match the green/purple layout exactly
    colors = [
        ['#10b981', '#6366f1'], # Row 0: Green (TP), Purple (FN)
        ['#6366f1', '#10b981']  # Row 1: Purple (FP), Green (TN)
    ]
    
    # Plot using a custom patch drawing approach for exact styling matching
    for i in range(2):
        for j in range(2):
            # Draw cell background
            rect = plt.Rectangle((j, 1-i), 1, 1, facecolor=colors[i][j], edgecolor='#1e293b', linewidth=2)
            ax.add_patch(rect)
            # Add text inside cell
            ax.text(j + 0.5, 1-i + 0.5, str(cm[i, j]),
                    ha='center', va='center', color='#ffffff', 
                    fontsize=24, weight='bold')
            
    # Set limits and ticks
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Urgent", "Normal"], fontsize=12, color="#94a3b8")
    
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Normal", "Urgent"], fontsize=12, color="#94a3b8") # inverted for plot coordinates
    
    # Titles & Labels
    plt.title("Confusion Matrix - Output", fontsize=18, pad=22, weight="bold", color="#ffffff")
    
    # Predicted label at the top
    ax.text(1.0, 2.15, "Predicted", ha='center', va='center', fontsize=12, color="#94a3b8")
    
    # Actual label on the left
    fig.text(-0.05, 0.5, "Actual", ha='center', va='center', rotation='vertical', fontsize=12, color="#94a3b8")
    
    # Add a custom colorbar representation on the right matching the image layout
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    # Draw green-to-light-green colorbar gradient
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cbar_ax.imshow(gradient, aspect='auto', cmap=plt.cm.colors.ListedColormap(
        sns.light_palette("#10b981", n_colors=256)
    ))
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([0, 128, 255])
    cbar_ax.set_yticklabels(["5.0", "3.0", "1.0"], color="#94a3b8", fontsize=9)
    cbar_ax.tick_params(size=0)
    
    # Clean main axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(size=0)
    
    # Save confusion matrix
    out_dir = PROJECT_ROOT / "outputs" / "FD001"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    img_path = out_dir / "confusion_matrix_styled.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Styled confusion matrix saved successfully to: {img_path}")

if __name__ == "__main__":
    main()

