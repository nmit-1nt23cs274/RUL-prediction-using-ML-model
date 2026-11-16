import sys
from pathlib import Path
import torch
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    for ds in datasets:
        path = str(PROJECT_ROOT / f'outputs/{ds}/tcn_{ds}.pt')
        if not os.path.exists(path):
            print(f"{ds}: Checkpoint does not exist.")
            continue
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            cols = ckpt.get('feature_cols', [])
            has_roll = any('_roll' in c for c in cols)
            print(f"{ds}: has_rolling={has_roll}, num_features={len(cols)}")
        except Exception as e:
            print(f"{ds}: Error loading checkpoint: {e}")

if __name__ == "__main__":
    main()

