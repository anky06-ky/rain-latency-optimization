"""
Standalone Model Training Script.

Trains a LightGBM regressor on the training split (2023-01 to 2024-12)
using specified or optimal hyperparameters, and reports metrics across splits.
"""

import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    load_clean_weather_data,
    build_features,
    split_time_series,
)
from src.evaluation.latency import benchmark_end_to_end_latency, LATENCY_CONSTRAINT_MS


def parse_args():
    parser = argparse.ArgumentParser(description="Train LightGBM rainfall forecasting model.")
    parser.add_argument("--L", type=int, default=1, help="Historical window length in hours (default: 1)")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees (default: 200)")
    parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth (default: 6)")
    parser.add_argument("--num-leaves", type=int, default=15, help="Number of leaves (default: 15)")
    parser.add_argument("--learning-rate", type=float, default=0.0235, help="Learning rate (default: 0.0235)")
    parser.add_argument("--min-child-samples", type=int, default=10, help="Min child samples (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("TRAINING LIGHTGBM RAINFALL FORECASTING MODEL")
    print(f"Window: L={args.L}h | Trees={args.n_estimators} | Depth={args.max_depth} | Leaves={args.num_leaves} | LR={args.learning_rate}")
    print("=" * 65)

    clean_df = load_clean_weather_data()
    df_feat, fcols = build_features(clean_df, L=args.L)
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_time_series(df_feat, fcols)

    model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        min_child_samples=args.min_child_samples,
        random_state=args.seed,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X_tr, y_tr)

    # Compute metrics across splits
    splits = [("Train (2023-2024)", X_tr, y_tr), ("Validation (2025-H1)", X_v, y_v), ("Test (2025-H2)", X_te, y_te)]
    print("\n--- Model Evaluation Across Chronological Splits ---")
    for name, X, y in splits:
        preds = model.predict(X)
        rmse = root_mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        print(f"{name:>22}: RMSE = {rmse:.4f} mm/h | MAE = {mae:.4f} mm/h | R^2 = {r2:.4f}")

    # Latency benchmark
    raw_window = clean_df.iloc[-args.L:]
    lat_res = benchmark_end_to_end_latency(model, raw_window, L=args.L, n_runs=200, warmup=20)
    print(f"\nEnd-to-End Latency P95: {lat_res['p95_latency_ms']:.3f} ms (Budget <= {LATENCY_CONSTRAINT_MS} ms) -> Feasible: {lat_res['is_feasible']}")


if __name__ == "__main__":
    main()
