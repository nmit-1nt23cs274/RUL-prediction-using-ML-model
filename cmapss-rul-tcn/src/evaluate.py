from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data_utils import load_raw, load_rul, make_test_sequences, phm08_score, rmse


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved TCN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="FD001")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_cols = checkpoint["feature_cols"]
    scaler = checkpoint["normalizer"]
    model_args = checkpoint["args"]

    from src.tcn_model import TCN

    model = TCN(
        input_size=len(feature_cols),
        num_channels=model_args.get("channels", [32, 32, 64, 64]),
        kernel_size=model_args.get("kernel_size", 3),
        dropout=model_args.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    data_path = Path(args.data_dir) / args.dataset
    test_df = load_raw(str(data_path / f"test_{args.dataset}.txt"))
    true_rul = load_rul(str(data_path / f"RUL_{args.dataset}.txt"))

    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))
    test_df = test_df.copy()
    for i, col in enumerate(feature_cols):
        test_df[col] = X_test[:, i]

    seq_test_X, engine_ids = make_test_sequences(test_df, feature_cols, args.window)
    with torch.no_grad():
        preds = model(torch.tensor(seq_test_X, dtype=torch.float32)).numpy()

    pred_vals = np.clip(preds.astype(np.float32), 0.0, None)
    true_lookup = {int(unit): float(rul) for unit, rul in zip(test_df["unit"].unique(), true_rul)}
    true_vals = np.array([true_lookup[int(e)] for e in engine_ids], dtype=np.float32)

    rmse_value = rmse(true_vals, pred_vals)
    phm_score = phm08_score(true_vals, pred_vals)
    print(f"RMSE={rmse_value:.6f}")
    print(f"PHM08={phm_score:.6f}")

    results = pd.DataFrame({
        "engine_id": engine_ids,
        "true_RUL": true_vals,
        "pred_RUL": pred_vals,
        "error": pred_vals - true_vals,
    }).sort_values("true_RUL").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(results["true_RUL"], label="True RUL", color="tab:blue")
    axes[0].plot(results["pred_RUL"], label="Predicted RUL", color="tab:orange")
    axes[0].legend()
    axes[0].set_xlabel("Engine")
    axes[0].set_ylabel("RUL")
    axes[0].grid(alpha=0.3)

    min_v = min(results["true_RUL"].min(), results["pred_RUL"].min())
    max_v = max(results["true_RUL"].max(), results["pred_RUL"].max())
    axes[1].scatter(results["true_RUL"], results["pred_RUL"], alpha=0.7)
    axes[1].plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black")
    axes[1].set_xlabel("True RUL")
    axes[1].set_ylabel("Predicted RUL")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(args.output) / args.dataset / f"eval_{args.dataset}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
