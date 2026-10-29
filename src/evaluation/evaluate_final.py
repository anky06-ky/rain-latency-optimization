"""
Final Evaluation & Visualization Module.

1. Identifies the best feasible configuration (L*, theta*) on Validation.
2. Evaluates the selected model on the held-out Test set (2025-07 to 2025-12) EXACTLY ONCE.
3. Generates Pareto Frontier and Convergence comparison figures.
"""

import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from src.features.build_features import (
    load_clean_weather_data,
    build_features,
    split_time_series,
)

from src.evaluation.latency import benchmark_end_to_end_latency, LATENCY_CONSTRAINT_MS


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_CSV = RESULTS_DIR / "optimization_history_random.csv"
OPTUNA_CSV = RESULTS_DIR / "optimization_history_optuna.csv"
FINAL_METRICS_JSON = RESULTS_DIR / "final_test_metrics.json"


def plot_pareto_frontier(df_all: pd.DataFrame):
    """Plot Validation RMSE vs Latency, showing the 5.0 ms constraint boundary and Pareto front."""
    plt.figure(figsize=(10, 6))

    feasible = df_all[df_all["is_feasible"]]
    infeasible = df_all[~df_all["is_feasible"]]

    # Scatter plot
    plt.scatter(
        feasible["p95_latency_ms"],
        feasible["val_rmse"],
        c="#2b8cbe",
        alpha=0.7,
        edgecolors="k",
        s=50,
        label=f"Nghiệm khả thi (<= {LATENCY_CONSTRAINT_MS} ms)",
    )
    plt.scatter(
        infeasible["p95_latency_ms"],
        infeasible["val_rmse"],
        c="#e34a33",
        alpha=0.4,
        edgecolors="k",
        s=40,
        marker="x",
        label=f"Vi phạm ràng buộc (> {LATENCY_CONSTRAINT_MS} ms)",
    )

    # Constraint threshold line
    plt.axvline(
        LATENCY_CONSTRAINT_MS,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Ràng buộc Latency ({LATENCY_CONSTRAINT_MS} ms)",
    )

    # Highlight best feasible
    if not feasible.empty:
        best_row = feasible.loc[feasible["val_rmse"].idxmin()]
        plt.scatter(
            [best_row["p95_latency_ms"]],
            [best_row["val_rmse"]],
            color="#fd8d3c",
            edgecolors="black",
            s=180,
            marker="*",
            zorder=5,
            label=f"Nghiệm tối ưu: L={int(best_row['L'])} (RMSE={best_row['val_rmse']:.4f})",
        )

    plt.title("Biên tối ưu Pareto: Sai số dự báo (RMSE) vs Độ trễ suy luận (Latency)", fontsize=12, fontweight="bold")
    plt.xlabel("End-to-End Latency P95 (ms)", fontsize=11)
    plt.ylabel("Validation RMSE (mm/h)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "pareto_frontier.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved Pareto frontier plot to {out_path}")


def plot_convergence_comparison(df_random: pd.DataFrame, df_optuna: pd.DataFrame):
    """Plot best-so-far curve over 40 trials comparing Random Search vs Optuna TPE."""
    plt.figure(figsize=(9, 5))

    def get_convergence_matrix(df: pd.DataFrame) -> np.ndarray:
        seeds = df["seed"].unique()
        n_trials = df["trial"].max()
        matrix = []

        for s in seeds:
            sub = df[df["seed"] == s].sort_values("trial")
            best_so_far = []
            curr_best = np.nan
            for _, row in sub.iterrows():
                # Feasible candidates only
                if row["is_feasible"]:
                    if np.isnan(curr_best) or row["val_rmse"] < curr_best:
                        curr_best = row["val_rmse"]
                best_so_far.append(curr_best)
            # Backfill any initial infeasible trials with the first feasible value
            series_best = pd.Series(best_so_far).bfill().ffill()
            matrix.append(series_best.values)

        return np.array(matrix, dtype=np.float64)

    mat_random = get_convergence_matrix(df_random)
    mat_optuna = get_convergence_matrix(df_optuna)


    trials_axis = np.arange(1, mat_random.shape[1] + 1)

    # Random Search Mean ± Std
    mean_r, std_r = np.nanmean(mat_random, axis=0), np.nanstd(mat_random, axis=0)
    plt.plot(trials_axis, mean_r, label="Baseline: Random Search", color="#756bb1", linewidth=2.2)
    plt.fill_between(trials_axis, mean_r - std_r, mean_r + std_r, color="#756bb1", alpha=0.2)

    # Optuna TPE Mean ± Std
    mean_o, std_o = np.nanmean(mat_optuna, axis=0), np.nanstd(mat_optuna, axis=0)
    plt.plot(trials_axis, mean_o, label="Main Method: Optuna (TPE)", color="#31a354", linewidth=2.2)
    plt.fill_between(trials_axis, mean_o - std_o, mean_o + std_o, color="#31a354", alpha=0.2)

    plt.title("Hội tụ tối ưu: So sánh Best-so-far Validation RMSE qua 40 Trials (5 Seeds)", fontsize=12, fontweight="bold")
    plt.xlabel("Số lượng Trials", fontsize=11)
    plt.ylabel("Validation RMSE tốt nhất", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = FIGURES_DIR / "convergence_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved Convergence comparison plot to {out_path}")


def evaluate_final_test_set():
    """Select best feasible model and evaluate on the held-out Test set once."""
    if not RANDOM_CSV.exists() or not OPTUNA_CSV.exists():
        raise FileNotFoundError("Missing optimization logs. Please run optimize_random.py and optimize_optuna.py first.")

    df_random = pd.read_csv(RANDOM_CSV)
    df_optuna = pd.read_csv(OPTUNA_CSV)
    df_all = pd.concat([df_random, df_optuna], ignore_index=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate Figures
    plot_pareto_frontier(df_all)
    plot_convergence_comparison(df_random, df_optuna)

    # 2. Select overall best feasible configuration on Validation
    feasible_all = df_all[df_all["is_feasible"]]
    if feasible_all.empty:
        raise RuntimeError("No feasible configurations found across all trials.")

    best_config = feasible_all.loc[feasible_all["val_rmse"].idxmin()]
    print("\n" + "=" * 65)
    print("BEST FEASIBLE CONFIGURATION SELECTED (ON VALIDATION SET):")
    print(f"  - Method          : {best_config['method']}")
    print(f"  - Historical L    : {int(best_config['L'])} hours")
    print(f"  - Features        : {int(best_config['n_features'])}")
    print(f"  - n_estimators    : {int(best_config['n_estimators'])}")
    print(f"  - max_depth       : {int(best_config['max_depth'])}")
    print(f"  - num_leaves      : {int(best_config['num_leaves'])}")
    print(f"  - learning_rate   : {best_config['learning_rate']}")
    print(f"  - Validation RMSE : {best_config['val_rmse']:.4f} mm/h")
    print(f"  - Val Latency P95 : {best_config['p95_latency_ms']:.3f} ms (<= 5.0 ms constraint)")
    print("=" * 65)

    # 3. Final Test Evaluation (Done EXACTLY ONCE on Test set)
    clean_df = load_clean_weather_data()
    best_L = int(best_config["L"])
    df_feat, fcols = build_features(clean_df, L=best_L)
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_time_series(df_feat, fcols)

    # Train final model on Train set
    final_model = lgb.LGBMRegressor(
        n_estimators=int(best_config["n_estimators"]),
        max_depth=int(best_config["max_depth"]),
        num_leaves=int(best_config["num_leaves"]),
        learning_rate=float(best_config["learning_rate"]),
        min_child_samples=int(best_config["min_child_samples"]),
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    final_model.fit(X_tr, y_tr)

    # Predict on held-out Test set (2025-07 to 2025-12)
    y_test_pred = final_model.predict(X_te)
    test_rmse = float(root_mean_squared_error(y_te, y_test_pred))
    test_mae = float(mean_absolute_error(y_te, y_test_pred))
    test_r2 = float(r2_score(y_te, y_test_pred))

    # Test Latency benchmark
    raw_window = clean_df.iloc[-best_L:]
    test_lat = benchmark_end_to_end_latency(final_model, raw_window, L=best_L, n_runs=300, warmup=20)

    final_report = {
        "best_configuration": {
            "method": str(best_config["method"]),
            "L_hours": best_L,
            "n_features": int(best_config["n_features"]),
            "n_estimators": int(best_config["n_estimators"]),
            "max_depth": int(best_config["max_depth"]),
            "num_leaves": int(best_config["num_leaves"]),
            "learning_rate": float(best_config["learning_rate"]),
            "min_child_samples": int(best_config["min_child_samples"]),
        },
        "validation_metrics": {
            "val_rmse": float(best_config["val_rmse"]),
            "val_mae": float(best_config["val_mae"]),
            "val_p95_latency_ms": float(best_config["p95_latency_ms"]),
        },
        "final_test_evaluation": {
            "test_period": "2025-07-01 to 2025-12-31",
            "test_samples": len(y_te),
            "test_rmse": round(test_rmse, 4),
            "test_mae": round(test_mae, 4),
            "test_r2": round(test_r2, 4),
            "test_mean_latency_ms": test_lat["mean_latency_ms"],
            "test_p95_latency_ms": test_lat["p95_latency_ms"],
            "test_feasible": test_lat["is_feasible"],
            "constraint_satisfied": test_lat["is_feasible"],
        },
    }

    with open(FINAL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("FINAL TEST EVALUATION METRICS (EVALUATED EXACTLY ONCE):")
    print(f"  - Test RMSE         : {test_rmse:.4f} mm/h")
    print(f"  - Test MAE          : {test_mae:.4f} mm/h")
    print(f"  - Test R^2          : {test_r2:.4f}")
    print(f"  - Test Latency P95  : {test_lat['p95_latency_ms']:.3f} ms (<= 5.0 ms constraint)")
    print(f"  - Feasible on Test  : {test_lat['is_feasible']}")
    print(f"[+] Saved final evaluation report to {FINAL_METRICS_JSON}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_final_test_set()
