"""
Feature Engineering Module (single source of truth for features, offline and online).

- load_clean_weather_data: Quality Gate (2023-2025) + strictly causal imputation.
- build_features: batch lag / rolling / cyclical features for window L, target y_{t+1}.
  Note: rows are dropped (lag warm-up, missing targets) and the result is RE-INDEXED, so its
  index does not align with the clean DataFrame; look up timestamps via its 'time' / 'target_time'.
- split_time_series: chronological Train / Validation / Test split by target_time.
- extract_single_sample_features / extract_features_from_array: online single-sample extractors,
  bit-for-bit parity with build_features (see tests/test_pipeline.py).

Feature count: 6*L lags + 5 rolling stats (only if L >= 3) + 4 cyclical encodings
(L=1 -> 10 features, L=6 -> 45 features).
"""

import math
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/raw/weather_meteostat_2021_2026.csv")
PROCESSED_DATA_DIR = Path("data/processed")

# Base physical meteorological parameters
BASE_NUMERIC_COLS = ["temp", "dwpt", "rhum", "pres", "wspd"]
# Raw per-hour inputs used to build lag features (order defines feature column order)
RAW_WINDOW_COLS = BASE_NUMERIC_COLS + ["prcp_clean"]


def load_clean_weather_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw Meteostat CSV and apply Quality Gate filtering (2023-2025).
    Uses strictly past-only / causal imputation to prevent future data leakage:
    - Continuous physical variables (temp, dwpt, rhum, pres, wspd): forward-fill up to 2 hours.
    - Precipitation: forward-fill short gaps up to 1 hour, remaining missing treated as 0.0 for feature lagging.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}. Run 'python src/data/download.py' first.")

    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["time"])

    # Quality Gate: Filter clean period (2023-01-01 to 2025-12-31)
    df = df[(df["time"] >= "2023-01-01") & (df["time"] <= "2025-12-31 23:59:59")].copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Strictly causal forward-fill for continuous weather variables (no future values accessed)
    df[BASE_NUMERIC_COLS] = df[BASE_NUMERIC_COLS].ffill(limit=2)

    # Clean precipitation for feature lagging: past-only forward-fill (limit=1), remainder as 0.0
    # Note: ground-truth target is derived from raw df['prcp'] before fillna, and missing targets are dropped
    df["prcp_clean"] = df["prcp"].ffill(limit=1).fillna(0.0)

    return df


def build_features(df: pd.DataFrame, L: int = 6) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct lag, rolling, and cyclical features for a given historical window length L.
    
    Parameters:
        df: Clean hourly DataFrame.
        L: Historical window length in hours (L in [1, 72]).
        
    Returns:
        df_features: DataFrame containing features, 'target' (y_t+1), 'time', and 'target_time'.
        feature_cols: List of feature column names.
    """
    if L < 1:
        raise ValueError("Window length L must be at least 1.")

    data_dict = {}

    # Target: 1 hour ahead precipitation (y_t+1)
    data_dict["target"] = df["prcp"].shift(-1)
    data_dict["time"] = df["time"]
    # Explicit target timestamp to eliminate boundary leakage
    data_dict["target_time"] = df["time"] + pd.to_timedelta(1, unit="h")

    feature_cols = []

    # 1. Lag features: values from t (lag 0) back to t-(L-1) (lag L-1)
    for lag in range(L):
        for col in RAW_WINDOW_COLS:
            col_name = f"{col}_lag_{lag}" if col != "prcp_clean" else f"prcp_lag_{lag}"
            data_dict[col_name] = df[col].shift(lag)
            feature_cols.append(col_name)

    # 2. Rolling window statistics over window L (for L >= 3)
    if L >= 3:
        rolling_prcp = df["prcp_clean"].rolling(window=L)
        data_dict[f"prcp_rolling_mean_{L}"] = rolling_prcp.mean()
        data_dict[f"prcp_rolling_max_{L}"] = rolling_prcp.max()
        data_dict[f"prcp_rolling_std_{L}"] = rolling_prcp.std().fillna(0.0)

        data_dict[f"temp_rolling_mean_{L}"] = df["temp"].rolling(window=L).mean()
        data_dict[f"rhum_rolling_mean_{L}"] = df["rhum"].rolling(window=L).mean()

        feature_cols.extend([
            f"prcp_rolling_mean_{L}",
            f"prcp_rolling_max_{L}",
            f"prcp_rolling_std_{L}",
            f"temp_rolling_mean_{L}",
            f"rhum_rolling_mean_{L}",
        ])

    # 3. Cyclical temporal encodings (Hour of day and Month of year)
    hour = df["time"].dt.hour
    month = df["time"].dt.month
    data_dict["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    data_dict["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    data_dict["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    data_dict["cos_month"] = np.cos(2 * np.pi * month / 12.0)

    feature_cols.extend(["sin_hour", "cos_hour", "sin_month", "cos_month"])

    # Combine into DataFrame
    df_feat = pd.DataFrame(data_dict)

    # Drop any row with NaN in features (lag warm-up) or NaN in target (missing ground truth)
    df_feat = df_feat.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    return df_feat, feature_cols


def split_time_series(df_feat: pd.DataFrame, feature_cols: List[str]) -> Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    """
    Split time series chronologically by target_time to eliminate boundary leakage:
    - Train: target_time <= 2024-12-31 23:59:59 (features up to 22:00 predicting 23:00)
    - Validation: 2025-01-01 00:00:00 <= target_time <= 2025-06-30 23:59:59
    - Test: 2025-07-01 00:00:00 <= target_time <= 2025-12-31 23:59:59
    """
    train_mask = (df_feat["target_time"] >= "2023-01-01 01:00:00") & (df_feat["target_time"] <= "2024-12-31 23:59:59")
    val_mask = (df_feat["target_time"] >= "2025-01-01 00:00:00") & (df_feat["target_time"] <= "2025-06-30 23:59:59")
    test_mask = (df_feat["target_time"] >= "2025-07-01 00:00:00") & (df_feat["target_time"] <= "2025-12-31 23:59:59")

    train_df = df_feat[train_mask]
    val_df = df_feat[val_mask]
    test_df = df_feat[test_mask]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def extract_features_from_array(arr: np.ndarray, hour: int, month: int, L: int) -> np.ndarray:
    """
    Fast-path single-sample feature extractor operating on a raw NumPy buffer.
    Input: float array of shape (L, 6) with columns RAW_WINDOW_COLS, oldest row first,
           plus the hour/month of the latest observation (time t).
    Output: 2D float32 array of shape (1, num_features_D), column order identical to build_features.
    """
    if arr.shape != (L, len(RAW_WINDOW_COLS)):
        raise ValueError(f"Expected window of shape ({L}, {len(RAW_WINDOW_COLS)}), got {arr.shape}")

    n_rolling = 5 if L >= 3 else 0
    out = np.empty((1, L * len(RAW_WINDOW_COLS) + n_rolling + 4), dtype=np.float32)

    # Lags from lag 0 (latest, row L-1) down to lag L-1 (oldest, row 0)
    # Row reversal arr[::-1] followed by flatten matches build_features column order exactly
    n_lag = L * len(RAW_WINDOW_COLS)
    out[0, :n_lag] = arr[::-1].ravel()

    # Rolling statistics (for L >= 3)
    if n_rolling:
        prcp_series = arr[:, 5]
        out[0, n_lag] = prcp_series.mean()
        out[0, n_lag + 1] = prcp_series.max()
        # Use ddof=1 to match pandas rolling().std() in build_features exactly
        out[0, n_lag + 2] = prcp_series.std(ddof=1)
        out[0, n_lag + 3] = arr[:, 0].mean()
        out[0, n_lag + 4] = arr[:, 2].mean()

    # Cyclic time features from current timestamp (time t); math.* is cheaper than np.* on scalars
    out[0, -4] = math.sin(2.0 * math.pi * hour / 24.0)
    out[0, -3] = math.cos(2.0 * math.pi * hour / 24.0)
    out[0, -2] = math.sin(2.0 * math.pi * month / 12.0)
    out[0, -1] = math.cos(2.0 * math.pi * month / 12.0)

    return out


def extract_single_sample_features(raw_window_slice: pd.DataFrame, L: int) -> np.ndarray:
    """
    Vectorized feature extractor for single-sample end-to-end latency measurement.
    Input: DataFrame of shape (L, num_features) representing the last L hours up to time t.
    Output: 2D numpy array of shape (1, num_features_D).
    """
    # Pull each column's backing array directly; selecting a column subset (df[[...]]) builds a new
    # DataFrame per call and dominated single-sample latency.
    arr = np.column_stack([raw_window_slice[c].to_numpy(dtype=np.float32) for c in RAW_WINDOW_COLS])
    curr_time = raw_window_slice["time"].iloc[-1]
    return extract_features_from_array(arr, curr_time.hour, curr_time.month, L)


if __name__ == "__main__":
    print("=" * 65)
    print("Executing build_features.py standalone check...")
    print("=" * 65)

    clean_df = load_clean_weather_data()
    print(f"[+] Loaded clean weather data: {len(clean_df):,} rows.")

    # Demonstration with L = 6 and L = 72
    for test_L in [3, 6, 24, 72]:
        df_feat, fcols = build_features(clean_df, L=test_L)
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_time_series(df_feat, fcols)
        print(f"[*] Window L={test_L:>2}: Total Features={len(fcols):>3} | Train={X_tr.shape} | Val={X_v.shape} | Test={X_te.shape}")

    # Save feature names for reference
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DATA_DIR / "feature_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(fcols))
    print(f"\n[+] Saved feature list ({len(fcols)} features for L=72) to {PROCESSED_DATA_DIR / 'feature_list.txt'}")
