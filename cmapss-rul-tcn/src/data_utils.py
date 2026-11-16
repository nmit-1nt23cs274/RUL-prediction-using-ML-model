from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


CMAPSS_COLUMNS = [
    "unit",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
    *[f"s{i}" for i in range(1, 22)],
]


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw C-MAPSS text file as a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    return df.reset_index(drop=True)


def load_rul(path: str) -> np.ndarray:
    """Load the RUL file into a 1D NumPy array."""
    data = np.loadtxt(path, dtype=np.float32)
    return np.asarray(data, dtype=np.float32).reshape(-1)


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Create training RUL labels using the standard piecewise-linear target."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = np.minimum(np.maximum(max_cycle - out["cycle"], 0), 125).astype(np.float32)
    return out


def drop_low_info_sensors(df: pd.DataFrame, sensor_cols: List[str]) -> pd.DataFrame:
    """Drop sensors with near-zero standard deviation."""
    keep = [col for col in sensor_cols if df[col].std(ddof=0) > 1e-6]
    return df[[c for c in df.columns if c not in sensor_cols or c in keep]]


def add_rolling_features(df: pd.DataFrame, sensor_cols: List[str], window: int = 10) -> pd.DataFrame:
    """Add rolling mean and rolling standard deviation features for active sensors."""
    out = df.copy()
    for col in sensor_cols:
        # Rolling mean
        out[f"{col}_roll_mean"] = out.groupby("unit")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        # Rolling standard deviation
        out[f"{col}_roll_std"] = out.groupby("unit")[col].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
    return out


class Normalizer:
    """Simple min-max scaler fitted on training data only."""

    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> "Normalizer":
        X = np.asarray(X, dtype=np.float32)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        diff = self.max_ - self.min_
        diff[diff == 0] = 1.0
        self.max_ = self.max_.astype(np.float32)
        self.min_ = self.min_.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.min_ is None or self.max_ is None:
            raise ValueError("Normalizer has not been fitted yet.")
        denom = self.max_ - self.min_
        denom[denom == 0] = 1.0
        return (X - self.min_) / denom

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _pad_front(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) >= window:
        return values
    pad_count = window - len(values)
    first_row = np.repeat(values[:1], pad_count, axis=0)
    return np.concatenate([first_row, values], axis=0)


def make_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences per engine unit."""
    X_parts = []
    y_parts = []

    for _, group in df.groupby("unit", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=np.float32)
        targets = group["RUL"].to_numpy(dtype=np.float32)

        if len(values) < window:
            seq = _pad_front(values, window)
            X_parts.append(seq[np.newaxis, :, :])
            y_parts.append(np.array([targets[-1]], dtype=np.float32))
            continue

        for start in range(0, len(values) - window + 1, stride):
            seq = values[start : start + window]
            X_parts.append(seq[np.newaxis, :, :])
            y_parts.append(np.array([targets[start + window - 1]], dtype=np.float32))

    if not X_parts:
        return np.empty((0, window, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.float32)
    return X, y


def make_test_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return exactly one sequence per engine using the last window."""
    X_parts = []
    engine_ids = []

    for unit in sorted(df["unit"].unique()):
        group = df[df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
        values = group[feature_cols].to_numpy(dtype=np.float32)
        seq = _pad_front(values, window)
        if len(seq) > window:
            seq = seq[-window:]
        X_parts.append(seq[np.newaxis, :, :])
        engine_ids.append(int(unit))

    if not X_parts:
        return np.empty((0, window, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    return X, np.asarray(engine_ids, dtype=np.int32)


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric PHM08 score using the maintenance-oriented sign convention.

    Here, d = y_true - y_pred. A late prediction means the model predicts too much
    remaining life relative to the true RUL, which is penalized more heavily.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    d = y_true - y_pred
    score = np.where(d < 0, np.exp((-d) / 10.0) - 1.0, np.exp(d / 13.0) - 1.0)
    return float(np.sum(score))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
