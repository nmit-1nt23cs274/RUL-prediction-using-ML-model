from __future__ import annotations

import argparse
import os
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
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.data_utils import Normalizer, add_train_rul, drop_low_info_sensors, add_rolling_features, load_raw, load_rul, make_sequences, make_test_sequences, phm08_score, rmse
from src.tcn_model import TCN


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_by_unit(df: pd.DataFrame, val_split: float, seed: int):
    units = df["unit"].unique()
    rng = np.random.default_rng(seed)
    units = rng.permutation(units)
    val_count = max(1, int(len(units) * val_split))
    val_units = set(units[:val_count])
    train_df = df[~df["unit"].isin(val_units)].copy()
    val_df = df[df["unit"].isin(val_units)].copy()
    return train_df, val_df


def save_training_curve(train_losses, val_losses, path: Path):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(train_losses, label="Train MSE", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train MSE", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(val_losses, label="Val RMSE", color="tab:orange")
    ax2.set_ylabel("Val RMSE", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def train_model(args):
    set_seed(args.seed)
    dataset_path = Path(args.data_dir) / args.dataset
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_raw(str(dataset_path / f"train_{args.dataset}.txt"))
    test_df = load_raw(str(dataset_path / f"test_{args.dataset}.txt"))
    train_df = add_train_rul(train_df)
    test_df = add_train_rul(test_df)

    sensor_cols = [f"s{i}" for i in range(1, 22)]
    train_df = drop_low_info_sensors(train_df, sensor_cols)
    test_df = test_df[[c for c in test_df.columns if c in train_df.columns or c in ["unit", "cycle"]]].copy()

    active_sensors = [c for c in train_df.columns if c in sensor_cols]
    train_df = add_rolling_features(train_df, active_sensors, window=10)
    test_df = add_rolling_features(test_df, active_sensors, window=10)

    feature_cols = [c for c in train_df.columns if c not in ["unit", "cycle", "RUL"]]
    if not feature_cols:
        raise ValueError("No feature columns selected for training.")

    train_df, val_df = split_by_unit(train_df, args.val_split, args.seed)
    scaler = Normalizer()
    train_values = train_df[feature_cols].to_numpy(dtype=np.float32)
    scaler.fit(train_values)

    X_train = scaler.transform(train_df[feature_cols].to_numpy(dtype=np.float32))
    X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float32))
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float32))

    seq_train_X, seq_train_y = make_sequences(train_df.assign(**{c: X_train[:, i] for i, c in enumerate(feature_cols)}), feature_cols, args.window)
    seq_val_X, seq_val_y = make_sequences(val_df.assign(**{c: X_val[:, i] for i, c in enumerate(feature_cols)}), feature_cols, args.window)
    seq_test_X, test_units = make_test_sequences(test_df.assign(**{c: X_test[:, i] for i, c in enumerate(feature_cols)}), feature_cols, args.window)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCN(input_size=len(feature_cols), num_channels=args.channels, kernel_size=args.kernel_size, dropout=args.dropout)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience // 2)

    train_losses = []
    val_losses = []
    best_val = np.inf
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0
        for start in range(0, len(seq_train_X), args.batch_size):
            xb = torch.tensor(seq_train_X[start:start + args.batch_size], dtype=torch.float32, device=device)
            yb = torch.tensor(seq_train_y[start:start + args.batch_size], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_train_loss += float(loss.item()) * len(xb)

        train_loss = epoch_train_loss / max(1, len(seq_train_X))
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_preds = model(torch.tensor(seq_val_X, dtype=torch.float32, device=device))
            val_rmse = rmse(seq_val_y, val_preds.cpu().numpy())
            val_losses.append(val_rmse)

        scheduler.step(val_rmse)

        if val_rmse < best_val:
            best_val = val_rmse
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "normalizer": scaler,
                "args": vars(args),
            }, out_dir / f"tcn_{args.dataset}.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    # Refresh best model state for final evaluation.
    checkpoint = torch.load(out_dir / f"tcn_{args.dataset}.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with torch.no_grad():
        test_preds = model(torch.tensor(seq_test_X, dtype=torch.float32, device=device)).cpu().numpy()

    true_rul = load_rul(str(dataset_path / f"RUL_{args.dataset}.txt"))
    true_rul_lookup = {int(unit): float(rul) for unit, rul in zip(test_df["unit"].unique(), true_rul)}
    true_vals = np.array([true_rul_lookup[int(u)] for u in test_units], dtype=np.float32)
    pred_vals = np.clip(test_preds.astype(np.float32), 0.0, None)

    pred_df = pd.DataFrame({
        "engine_id": test_units.astype(int),
        "true_RUL": true_vals,
        "pred_RUL": pred_vals,
        "error": pred_vals - true_vals,
    })
    pred_df.to_csv(out_dir / f"predictions_{args.dataset}.csv", index=False)

    rmse_value = rmse(true_vals, pred_vals)
    score_value = phm08_score(true_vals, pred_vals)

    with open(out_dir / f"metrics_{args.dataset}.txt", "w", encoding="utf-8") as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"RMSE={rmse_value:.6f}\n")
        f.write(f"PHM08={score_value:.6f}\n")
        f.write(f"window={args.window}\n")
        f.write(f"epochs={len(train_losses)}\n")

    save_training_curve(train_losses, val_losses, out_dir / f"training_curve_{args.dataset}.png")
    print(f"RMSE={rmse_value:.6f}")
    print(f"PHM08={score_value:.6f}")
    return rmse_value, score_value


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TCN for C-MAPSS RUL prediction")
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 32, 64, 64])
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
