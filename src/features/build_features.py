

from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("data/raw/weather_meteostat_2021_2026.csv")
PROCESSED_DATA_DIR = Path("data/processed")

# Base physical meteorological parameters
BASE_NUMERIC_COLS = ["temp", "dwpt", "rhum", "pres", "wspd"]


def load_clean_weather_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw Meteostat CSV and apply Quality Gate filtering (2023-2025).
    Applies short-gap linear interpolation for continuous sensors (<= 2h).
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}. Run 'python src/data/download.py' first.")

    df = pd.read_csv(filepath)
    df["time"] = pd.to_datetime(df["time"])

    # Quality Gate: Filter clean period (2023-01-01 to 2025-12-31)
    df = df[(df["time"] >= "2023-01-01") & (df["time"] <= "2025-12-31 23:59:59")].copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Linear interpolation for continuous weather variables (gaps <= 2h)
    df[BASE_NUMERIC_COLS] = df[BASE_NUMERIC_COLS].interpolate(method="linear", limit=2)

    # Clean precipitation: fill remaining missing values with 0.0 for lagging
    df["prcp_clean"] = df["prcp"].fillna(0.0)

    return df


def build_features(df: pd.DataFrame, L: int = 6) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct lag, rolling, and cyclical features for a given historical window length L.
    
    Parameters:
        df: Clean hourly DataFrame.
        L: Historical window length in hours (L in [1, 72]).
        
    Returns:
        df_features: DataFrame containing features, 'target' (y_t+1), and 'time'.
        feature_cols: List of feature column names.
    """
    if L < 1:
        raise ValueError("Window length L must be at least 1.")

    data_dict = {}

    # Target: 1 hour ahead precipitation (y_t+1)
    data_dict["target"] = df["prcp"].shift(-1)
    data_dict["time"] = df["time"]

    base_cols = ["temp", "dwpt", "rhum", "pres", "wspd", "prcp_clean"]
    feature_cols = []

    # 1. Lag features: values from t (lag 0) back to t-(L-1) (lag L-1)
    for lag in range(L):
        for col in base_cols:
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

    # Drop invalid rows: first L-1 rows (due to lag) and last row (target is NaN)
    df_feat = df_feat.dropna(subset=["target"]).iloc[L - 1:].reset_index(drop=True)

    return df_feat, feature_cols


def split_time_series(df_feat: pd.DataFrame, feature_cols: List[str]) -> Tuple[
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series],
]:
    """
    Split time series chronologically without data leakage:
    - Train: 2023-01-01 to 2024-12-31 (17,544 hours due to 2024 leap year, ~66.7%)
    - Validation: 2025-01-01 to 2025-06-30 (4,344 hours, ~16.5%)
    - Test: 2025-07-01 to 2025-12-31 (4,416 hours, ~16.8%)
    """
    train_mask = (df_feat["time"] >= "2023-01-01") & (df_feat["time"] <= "2024-12-31 23:59:59")
    val_mask = (df_feat["time"] >= "2025-01-01") & (df_feat["time"] <= "2025-06-30 23:59:59")
    test_mask = (df_feat["time"] >= "2025-07-01") & (df_feat["time"] <= "2025-12-31 23:59:59")

    train_df = df_feat[train_mask]
    val_df = df_feat[val_mask]
    test_df = df_feat[test_mask]

    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]
    X_test, y_test = test_df[feature_cols], test_df["target"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def extract_single_sample_features(raw_window_slice: pd.DataFrame, L: int) -> np.ndarray:
    """
    Vectorized feature extractor for single-sample end-to-end latency measurement.
    Input: DataFrame of shape (L, num_features) representing the last L hours up to time t.
    Output: 2D numpy array of shape (1, num_features_D).
    """
    # Extract matrix of raw climate columns: temp, dwpt, rhum, pres, wspd, prcp_clean
    arr = raw_window_slice[["temp", "dwpt", "rhum", "pres", "wspd", "prcp_clean"]].to_numpy(dtype=np.float32)

    # Lags from lag 0 (latest, row L-1) down to lag L-1 (oldest, row 0)
    # Row reversal arr[::-1] followed by flatten matches build_features column order exactly
    lag_feats = arr[::-1].ravel()
    parts = [lag_feats]

    # Rolling statistics (for L >= 3)
    if L >= 3:
        prcp_series = arr[:, 5]
        mean_p = float(np.mean(prcp_series))
        max_p = float(np.max(prcp_series))
        # Use ddof=1 to match pandas rolling().std() in build_features exactly
        std_p = float(np.std(prcp_series, ddof=1)) if len(prcp_series) > 1 else 0.0
        mean_t = float(np.mean(arr[:, 0]))
        mean_r = float(np.mean(arr[:, 2]))
        rolling_feats = np.array([mean_p, max_p, std_p, mean_t, mean_r], dtype=np.float32)
        parts.append(rolling_feats)

    # Cyclic time features from current timestamp (time t)
    curr_time = raw_window_slice["time"].iloc[-1]
    h, m = curr_time.hour, curr_time.month
    cyclic_feats = np.array([
        np.sin(2.0 * np.pi * h / 24.0),
        np.cos(2.0 * np.pi * h / 24.0),
        np.sin(2.0 * np.pi * m / 12.0),
        np.cos(2.0 * np.pi * m / 12.0),
    ], dtype=np.float32)
    parts.append(cyclic_feats)

    return np.concatenate(parts).reshape(1, -1)


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
