"""
Inference module for the C-MAPSS FD001 RUL web application.

Loads the trained TCN model checkpoint (if available) or falls back to
demo mode.  Applies the IIoT priority scheduler with four urgency tiers
(Critical / High / Medium / Low) and bandwidth-limited transmission
scheduling.

Public API
----------
predict_pipeline(file_path, bandwidth_limit=10) -> dict
    End-to-end: load data ➜ preprocess ➜ predict ➜ schedule ➜ return result dict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from src.data_utils import (
    Normalizer,
    drop_low_info_sensors,
    load_raw,
    make_test_sequences,
)
from src.priority import priority_score as _base_priority_score
from src.priority import rank_engines
from src.tcn_model import TCN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Try multiple possible checkpoint locations (train.py may save to either)
_CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "tcn_rul_model.pt",       # spec location
    PROJECT_ROOT / "outputs" / "FD001" / "tcn_FD001.pt",  # legacy location
]

# Four-tier urgency thresholds (matching the project spec)
TIER_CRITICAL = 20   # RUL < 20
TIER_HIGH = 50       # 20 <= RUL < 50
TIER_MEDIUM = 90     # 50 <= RUL < 90
                     # RUL >= 90  → Low


# ---------------------------------------------------------------------------
# Tier classification (4-tier, independent of src/priority.py 3-tier)
# ---------------------------------------------------------------------------

def classify_tier(rul: float) -> str:
    """Classify predicted RUL into one of four urgency tiers."""
    if rul < TIER_CRITICAL:
        return "critical"
    if rul < TIER_HIGH:
        return "high"
    if rul < TIER_MEDIUM:
        return "medium"
    return "low"


TIER_DISPLAY = {
    "critical": "CRITICAL",
    "high":     "HIGH",
    "medium":   "MEDIUM",
    "low":      "LOW",
}


# ---------------------------------------------------------------------------
# Model loading (lazy / cached)
# ---------------------------------------------------------------------------

_cached_models = {}  # key: dataset_name, value: (model, feature_cols, normalizer, window)


def _find_checkpoint(dataset: str = "FD001") -> Path | None:
    """Return the first existing checkpoint path for the given dataset, or None."""
    candidates = [
        PROJECT_ROOT / "outputs" / dataset / f"tcn_{dataset}.pt",
        PROJECT_ROOT / "outputs" / f"tcn_rul_model_{dataset}.pt",
        PROJECT_ROOT / "outputs" / f"tcn_{dataset}.pt",
    ]
    if dataset == "FD001":
        candidates.append(PROJECT_ROOT / "outputs" / "tcn_rul_model.pt")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_model(dataset: str = "FD001"):
    """Load TCN checkpoint into module-level cache.  Safe to call repeatedly."""
    global _cached_models

    if dataset in _cached_models:
        return _cached_models[dataset]

    ckpt_path = _find_checkpoint(dataset)
    if ckpt_path is None:
        _cached_models[dataset] = (None, None, None, None)
        return None, None, None, None

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        _cached_models[dataset] = (None, None, None, None)
        return None, None, None, None

    if not isinstance(checkpoint, dict):
        _cached_models[dataset] = (None, None, None, None)
        return None, None, None, None

    # --- Extract config ---------------------------------------------------
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Feature columns & normalizer
    feature_cols = checkpoint.get("feature_cols")
    normalizer = checkpoint.get("normalizer")

    # Model hyper-parameters (support both flat keys and nested "args"/"model_config")
    cfg = checkpoint.get("model_config", checkpoint.get("args", {}))
    channels = cfg.get("channels", cfg.get("num_channels", [32, 32, 64, 64]))
    kernel_size = cfg.get("kernel_size", 3)
    dropout = cfg.get("dropout", 0.2)
    window = cfg.get("window", cfg.get("window_size", 30))

    if feature_cols is None:
        feature_cols = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]

    # Build and load model
    model = TCN(
        input_size=len(feature_cols),
        num_channels=channels if isinstance(channels, list) else list(channels),
        kernel_size=kernel_size,
        dropout=dropout,
    )
    model.load_state_dict(state_dict)
    model.eval()

    res = (model, feature_cols, normalizer, window)
    _cached_models[dataset] = res
    return res


# ---------------------------------------------------------------------------
# Public prediction pipeline
# ---------------------------------------------------------------------------

def predict_pipeline(
    file_path: str,
    dataset: str = "FD001",
    bandwidth_limit: int = 10,
) -> Dict[str, Any]:
    """
    End-to-end inference pipeline.

    Parameters
    ----------
    file_path : str
        Path to an uploaded C-MAPSS format text file (26 whitespace-separated columns).
    dataset : str
        The specific subdataset (FD001, FD002, FD003, FD004) to load the model checkpoint for.
    bandwidth_limit : int
        Number of transmission slots available this cycle.  The top-N engines
        by priority score are marked ``transmit_this_cycle = True``; the rest
        are deferred.

    Returns
    -------
    dict
        Rich result dictionary consumed by ``results.html``.
    """
    df = load_raw(file_path)
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)

    model, feature_cols, normalizer, window = load_model(dataset)

    # ------------------------------------------------------------------
    # Run model or demo fallback
    # ------------------------------------------------------------------
    if model is not None and feature_cols is not None:
        window_size = window if window is not None else 30

        # Calculate rolling features if expected by the model
        active_sensors = [c for c in feature_cols if not c.endswith("_roll_mean") and not c.endswith("_roll_std")]
        test_data = df.copy()
        for col in active_sensors:
            mean_col = f"{col}_roll_mean"
            std_col = f"{col}_roll_std"
            if mean_col in feature_cols:
                test_data[mean_col] = test_data.groupby("unit")[col].transform(
                    lambda x: x.rolling(10, min_periods=1).mean()
                )
            if std_col in feature_cols:
                test_data[std_col] = test_data.groupby("unit")[col].transform(
                    lambda x: x.rolling(10, min_periods=1).std().fillna(0)
                )

        # Handle potential column-name mismatch
        missing = [c for c in feature_cols if c not in test_data.columns]
        if missing:
            sensor_cols = [f"s{i}" for i in range(1, 22)]
            df_filtered = drop_low_info_sensors(test_data, sensor_cols)
            feature_cols = [
                c for c in df_filtered.columns if c not in ("unit", "cycle", "RUL")
            ]

        if normalizer is not None:
            try:
                norm_vals = normalizer.transform(
                    test_data[feature_cols].to_numpy(dtype=np.float32)
                )
                for i, col in enumerate(feature_cols):
                    test_data[col] = norm_vals[:, i]
            except Exception:
                pass  # proceed with unnormalized data rather than crashing

        X, engine_ids = make_test_sequences(test_data, feature_cols, window=window_size)

        with torch.no_grad():
            preds = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
            pred_vals = np.clip(preds.astype(np.float32).ravel(), 0.0, None)

        is_real_model = True
        ckpt_name = (_find_checkpoint(dataset) or Path("unknown")).name
        status_msg = f"Inference completed using trained TCN checkpoint ({ckpt_name}) for {dataset}"
    else:
        # Demo / preview mode
        sensor_cols = [f"s{i}" for i in range(1, 22)]
        X, engine_ids = make_test_sequences(df, sensor_cols, window=30)
        rng = np.random.default_rng(42)
        pred_vals = np.clip(
            rng.normal(loc=75, scale=35, size=len(engine_ids)), 5.0, 140.0
        ).astype(np.float32)
        is_real_model = False
        status_msg = (
            f"No trained model checkpoint found for {dataset}. Displaying simulated preview. "
            f"Place your checkpoint at outputs/{dataset}/tcn_{dataset}.pt and restart the app."
        )

    # ------------------------------------------------------------------
    # Build priority-ranked results with 4-tier classification
    # ------------------------------------------------------------------
    # Use src/priority.rank_engines for base sorting (by 1/RUL score)
    ranked = rank_engines(dict(zip(engine_ids.tolist(), pred_vals.tolist())))

    # Counters per tier
    tier_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    engines_list: List[Dict[str, Any]] = []

    for rank_idx, (engine_id, rul, score, _old_tier) in enumerate(ranked, start=1):
        tier = classify_tier(rul)
        tier_counts[tier] += 1
        transmit = rank_idx <= bandwidth_limit
        engines_list.append(
            {
                "rank": rank_idx,
                "engine_id": int(engine_id),
                "predicted_rul": round(float(rul), 2),
                "priority_score": f"{score:.5f}",
                "tier": tier,
                "tier_badge": TIER_DISPLAY[tier],
                "transmit": transmit,
                "transmit_label": "TRANSMIT" if transmit else "DEFERRED",
            }
        )

    total = len(engines_list)
    transmit_count = min(bandwidth_limit, total)
    deferred_count = max(0, total - transmit_count)
    avg_rul = round(float(np.mean(pred_vals)), 2) if len(pred_vals) > 0 else 0.0

    if len(pred_vals) > 0:
        best_idx = int(np.argmax(pred_vals))
        worst_idx = int(np.argmin(pred_vals))
        best_engine_id = int(engine_ids[best_idx])
        best_engine_rul = round(float(pred_vals[best_idx]), 2)
        worst_engine_id = int(engine_ids[worst_idx])
        worst_engine_rul = round(float(pred_vals[worst_idx]), 2)
        median_rul = round(float(np.median(pred_vals)), 2)
    else:
        best_engine_id = 0
        best_engine_rul = 0.0
        worst_engine_id = 0
        worst_engine_rul = 0.0
        median_rul = 0.0

    # JSON-safe lists for Chart.js
    chart_labels = [str(e["engine_id"]) for e in engines_list]
    chart_ruls = [e["predicted_rul"] for e in engines_list]
    chart_colors = [
        "#ef4444" if e["tier"] == "critical"
        else "#f97316" if e["tier"] == "high"
        else "#eab308" if e["tier"] == "medium"
        else "#22c55e"
        for e in engines_list
    ]

    return {
        "is_real_model": is_real_model,
        "status": status_msg,
        "file": Path(file_path).name,
        "dataset": dataset,
        "bandwidth_limit": bandwidth_limit,
        "total_engines": total,
        "transmit_count": transmit_count,
        "deferred_count": deferred_count,
        "critical_count": tier_counts["critical"],
        "high_count": tier_counts["high"],
        "medium_count": tier_counts["medium"],
        "low_count": tier_counts["low"],
        "avg_rul": avg_rul,
        "best_engine_id": best_engine_id,
        "best_engine_rul": best_engine_rul,
        "worst_engine_id": worst_engine_id,
        "worst_engine_rul": worst_engine_rul,
        "median_rul": median_rul,
        "engines": engines_list,
        # Chart.js data (pre-serialized for the template)
        "chart_labels_json": json.dumps(chart_labels),
        "chart_ruls_json": json.dumps(chart_ruls),
        "chart_colors_json": json.dumps(chart_colors),
        "chart_tier_counts_json": json.dumps([
            tier_counts["critical"],
            tier_counts["high"],
            tier_counts["medium"],
            tier_counts["low"],
        ]),
    }

