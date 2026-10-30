"""
Baseline Optimization Module: Constrained Random Search.

Searches decision variables x = (L, theta) over a bounded space:
    - L: Historical window length in hours
    - theta: LightGBM hyperparameters
Objective: Minimize Validation RMSE subject to P95 End-to-End Latency <= 5.0 ms.
Budget: 40 trials per seed across 5 random seeds (total 200 evaluations).
"""

import argparse
from pathlib import Path
import math
import random
import sys
import time
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.features.build_features import (
    load_clean_weather_data,
    build_features,
    split_time_series,
)

from src.evaluation.latency import benchmark_end_to_end_latency, LATENCY_CONSTRAINT_MS


RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "optimization_history_random.csv"

# Pre-defined search space
WINDOW_CANDIDATES = [1, 2, 3, 6, 12, 24, 36, 48, 72]
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]


def sample_random_configuration(seed: int) -> Dict[str, Any]:
    """Sample a random candidate solution x = (L, theta) using identical prior distribution as Optuna."""
    rng = random.Random(seed)
    # Log-uniform sampling for learning_rate in [0.01, 0.25] to match Optuna log=True exactly
    log_lr = rng.uniform(math.log(0.01), math.log(0.25))
    lr = round(math.exp(log_lr), 4)

    return {
        "L": rng.choice(WINDOW_CANDIDATES),
        "n_estimators": rng.choice([20, 50, 100, 150, 200, 300, 400, 500]),
        "max_depth": rng.choice([2, 3, 4, 6, 8, 10, 12]),
        "num_leaves": rng.choice([7, 15, 31, 63, 127]),
        "learning_rate": lr,
        "min_child_samples": rng.choice([10, 20, 50, 100]),
    }


def evaluate_candidate(
    clean_df: pd.DataFrame,
    config: Dict[str, Any],
    eval_seed: int = 42,
) -> Dict[str, Any]:
    """
    Train model on candidate configuration and evaluate Validation error and Latency.
    """
    L = config["L"]

    # 1. Feature extraction
    df_feat, fcols = build_features(clean_df, L=L)
    (X_tr, y_tr), (X_v, y_v), _ = split_time_series(df_feat, fcols)

    # 2. Train LightGBM model
    model = lgb.LGBMRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        num_leaves=config["num_leaves"],
        learning_rate=config["learning_rate"],
        min_child_samples=config["min_child_samples"],
        random_state=eval_seed,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X_tr, y_tr)

    # 3. Validation Performance
    y_val_pred = model.predict(X_v)
    val_rmse = float(root_mean_squared_error(y_v, y_val_pred))
    val_mae = float(mean_absolute_error(y_v, y_val_pred))

    # 4. End-to-End Latency Measurement strictly on Validation slice (0% contact with Test set)
    val_slice = clean_df[(clean_df["time"] >= "2025-01-01") & (clean_df["time"] <= "2025-06-30 23:59:59")]
    raw_window = val_slice.iloc[-L:]
    lat_res = benchmark_end_to_end_latency(model, raw_window, L=L, n_runs=200, warmup=20)

    return {
        "L": L,
        "n_features": len(fcols),
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "num_leaves": config["num_leaves"],
        "learning_rate": config["learning_rate"],
        "min_child_samples": config["min_child_samples"],
        "val_rmse": round(val_rmse, 4),
        "val_mae": round(val_mae, 4),
        "mean_latency_ms": lat_res["mean_latency_ms"],
        "p95_latency_ms": lat_res["p95_latency_ms"],
        "is_feasible": lat_res["is_feasible"],
    }


def run_random_search(n_trials: int = 40, seeds: List[int] = DEFAULT_SEEDS) -> pd.DataFrame:
    """Execute baseline Random Search across specified trials and seeds."""
    print("=" * 65)
    print("STARTING BASELINE OPTIMIZATION: CONSTRAINED RANDOM SEARCH")
    print(f"Budget: {n_trials} trials/seed across {len(seeds)} seeds (Total: {n_trials * len(seeds)} evaluations)")
    print(f"Constraint: End-to-End Latency P95 <= {LATENCY_CONSTRAINT_MS} ms")
    print("=" * 65)

    clean_df = load_clean_weather_data()
    all_trials = []

    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"\n---> [Seed {seed_idx}/{len(seeds)}: {seed}] Running {n_trials} trials...")
        best_val_rmse = float("inf")
        best_config = None

        for trial in range(1, n_trials + 1):
            t0 = time.time()
            trial_seed = seed * 1000 + trial
            config = sample_random_configuration(trial_seed)
            result = evaluate_candidate(clean_df, config, eval_seed=seed)

            elapsed = time.time() - t0
            result["trial"] = trial
            result["seed"] = seed
            result["method"] = "RandomSearch"

            # Check feasibility and track best-so-far
            feasible_str = "FEASIBLE" if result["is_feasible"] else "VIOLATED"
            if result["is_feasible"] and result["val_rmse"] < best_val_rmse:
                best_val_rmse = result["val_rmse"]
                best_config = result

            all_trials.append(result)

            if trial % 10 == 0 or trial == n_trials:
                best_str = f"Best Feasible RMSE={best_val_rmse:.4f}" if best_config else "No Feasible Yet"
                print(f"    Trial {trial:>2}/{n_trials} | L={result['L']:>2} | RMSE={result['val_rmse']:.4f} | Latency={result['p95_latency_ms']:.2f}ms [{feasible_str}] | {best_str} ({elapsed:.1f}s)")

    results_df = pd.DataFrame(all_trials)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Random Search finished. Logged {len(results_df)} evaluations to {OUTPUT_CSV}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Constrained Random Search Baseline (RQ1 & RQ2).")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per seed (default: 100 for 500 scaling trials; set 40 for RQ1 fair comparison)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds (default: 5 seeds)")
    args = parser.parse_args()

    run_random_search(n_trials=args.trials, seeds=args.seeds)
