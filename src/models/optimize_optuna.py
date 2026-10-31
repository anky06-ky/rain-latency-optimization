"""
Main Optimization Method: Constrained Bayesian Optimization via Optuna (TPE).

Searches decision variables x = (L, theta) over a bounded space:
    - L: Historical window length in hours
    - theta: LightGBM hyperparameters
Objective: Minimize Validation RMSE subject to P95 End-to-End Latency <= 5.0 ms.
Budget: Default 100 trials per seed across 5 random seeds (total 500 evaluations for RQ2 scaling);
        can be set to 40 trials per seed (200 evaluations) for RQ1 fair budget comparison.
"""

import argparse
from pathlib import Path
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
import optuna
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.features.build_features import (
    load_clean_weather_data,
    build_features,
    split_time_series,
)

from src.evaluation.latency import benchmark_end_to_end_latency, LATENCY_CONSTRAINT_MS


# Suppress verbose Optuna logging during search
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "optimization_history_optuna.csv"

WINDOW_CANDIDATES = [1, 2, 3, 6, 12, 24, 36, 48, 72]
DEFAULT_SEEDS = [42, 123, 456, 789, 2026]


def run_optuna_study(
    clean_df: pd.DataFrame,
    seed: int,
    n_trials: int = 40,
) -> List[Dict[str, Any]]:
    """Execute one Optuna study for a given random seed."""
    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        # 1. Suggest decision variables
        L = trial.suggest_categorical("L", WINDOW_CANDIDATES)
        n_estimators = trial.suggest_categorical("n_estimators", [20, 50, 100, 150, 200, 300, 400, 500])
        max_depth = trial.suggest_categorical("max_depth", [2, 3, 4, 6, 8, 10, 12])
        num_leaves = trial.suggest_categorical("num_leaves", [7, 15, 31, 63, 127])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
        min_child_samples = trial.suggest_categorical("min_child_samples", [10, 20, 50, 100])

        # 2. Build features for window L
        df_feat, fcols = build_features(clean_df, L=L)
        (X_tr, y_tr), (X_v, y_v), _ = split_time_series(df_feat, fcols)

        # 3. Fit LightGBM model
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            min_child_samples=min_child_samples,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(X_tr, y_tr)

        # 4. Compute validation metrics
        y_val_pred = model.predict(X_v)
        val_rmse = float(root_mean_squared_error(y_v, y_val_pred))
        val_mae = float(mean_absolute_error(y_v, y_val_pred))

        # 5. Measure single-sample End-to-End Latency strictly on Validation slice (0% contact with Test set)
        val_slice = clean_df[(clean_df["time"] >= "2025-01-01") & (clean_df["time"] <= "2025-06-30 23:59:59")]
        raw_window = val_slice.iloc[-L:]
        lat_res = benchmark_end_to_end_latency(model, raw_window, L=L, n_runs=200, warmup=20)

        # 6. Set constraint in Optuna: violation <= 0 means feasible (P95 <= 5.0 ms)
        p95_lat = lat_res["p95_latency_ms"]
        latency_violation = float(p95_lat - LATENCY_CONSTRAINT_MS)
        trial.set_user_attr("constraint", [latency_violation])

        record = {
            "trial": trial.number + 1,
            "seed": seed,
            "method": "OptunaTPE",
            "L": L,
            "n_features": len(fcols),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": round(learning_rate, 4),
            "min_child_samples": min_child_samples,
            "val_rmse": round(val_rmse, 4),
            "val_mae": round(val_mae, 4),
            "mean_latency_ms": lat_res["mean_latency_ms"],
            "p95_latency_ms": p95_lat,
            "is_feasible": lat_res["is_feasible"],
        }
        trial_records.append(record)

        return val_rmse

    def constraints_func(trial: optuna.trial.FrozenTrial):
        """Optuna TPESampler constraint function: returns list of floats, <= 0 is feasible."""
        return trial.user_attrs.get("constraint", [0.0])

    # Create study with Constrained TPE sampler
    sampler = optuna.samplers.TPESampler(seed=seed, constraints_func=constraints_func)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return trial_records


def run_optuna_search(n_trials: int = 40, seeds: List[int] = DEFAULT_SEEDS) -> pd.DataFrame:
    """Execute Optuna TPE search across all seeds."""
    print("=" * 65)
    print("STARTING MAIN OPTIMIZATION: CONSTRAINED OPTUNA (TPE)")
    print(f"Budget: {n_trials} trials/seed across {len(seeds)} seeds (Total: {n_trials * len(seeds)} evaluations)")
    print(f"Constraint: End-to-End Latency P95 <= {LATENCY_CONSTRAINT_MS} ms")
    print("=" * 65)

    clean_df = load_clean_weather_data()
    all_trials = []

    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"\n---> [Seed {seed_idx}/{len(seeds)}: {seed}] Running {n_trials} trials with TPE...")
        t0 = time.time()
        records = run_optuna_study(clean_df, seed=seed, n_trials=n_trials)
        elapsed = time.time() - t0

        all_trials.extend(records)

        # Print seed summary
        feasible_records = [r for r in records if r["is_feasible"]]
        if feasible_records:
            best_feasible = min(feasible_records, key=lambda r: r["val_rmse"])
            print(f"    Completed in {elapsed:.1f}s | Best Feasible: L={best_feasible['L']} | RMSE={best_feasible['val_rmse']:.4f} | Latency={best_feasible['p95_latency_ms']}ms")
        else:
            print(f"    Completed in {elapsed:.1f}s | No feasible trials found.")

    results_df = pd.DataFrame(all_trials)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[+] Optuna TPE finished. Logged {len(results_df)} evaluations to {OUTPUT_CSV}")
    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Constrained Optuna TPE Optimization (RQ1 & RQ2).")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per seed (default: 100 for 500 scaling trials; set 40 for RQ1 fair comparison)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds (default: 5 seeds)")
    args = parser.parse_args()

    run_optuna_search(n_trials=args.trials, seeds=args.seeds)
