"""
Final Evaluation & Multi-Model Visualization Module.

1. Visualizes chronological Train / Validation / Test data split.
2. Identifies the best feasible configuration (L*, theta*) on Validation.
3. Evaluates the selected model on the held-out Test set (2025-07 to 2025-12) EXACTLY ONCE.
4. Performs comprehensive multi-model benchmark (Persistence, Linear, Ridge, Default LightGBM, Random, Optuna).
5. Generates Pareto Frontier, Search Convergence, Split Timeline, and Actual vs Predicted figures.
"""

import json
from pathlib import Path
import sys
from typing import Any, Tuple, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score

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
BENCHMARK_CSV = RESULTS_DIR / "multi_model_benchmark.csv"
FINAL_METRICS_JSON = RESULTS_DIR / "final_test_metrics.json"


def plot_train_val_test_split(clean_df: pd.DataFrame):
    """Plot the chronological time-series splitting timeline across 2023-2025."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    daily_df = clean_df.copy()
    daily_df["time"] = daily_df["time"].dt.tz_localize(None)
    daily_df = daily_df.set_index("time").resample("D")["prcp"].sum().reset_index()

    train_end = pd.to_datetime("2024-12-31 23:59:59")
    val_end = pd.to_datetime("2025-06-30 23:59:59")

    ax.plot(daily_df["time"], daily_df["prcp"], color="#4a7bb0", alpha=0.6, label="Lượng mưa ngày (mm)")

    # Highlight split regions with exact hour counts and leap year detail
    ax.axvspan(daily_df["time"].min(), train_end, color="#74c476", alpha=0.3, label="Train (17.544 giờ, 66.7% - 2023-2024, năm 2024 nhuận)")
    ax.axvspan(train_end, val_end, color="#fed976", alpha=0.4, label="Validation (4.344 giờ, 16.5% - T1-T6/2025)")
    ax.axvspan(val_end, daily_df["time"].max(), color="#fc9272", alpha=0.4, label="Test (4.416 giờ, 16.8% - T7-T12/2025, đỉnh mùa mưa)")

    ax.axvline(train_end, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(val_end, color="black", linestyle="--", linewidth=1.2)

    ax.set_title("Phân chia tập dữ liệu theo dòng thời gian (Chronological Split - Không rò rỉ dữ liệu)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Thời gian (Năm - Tháng)", fontsize=11)
    ax.set_ylabel("Tổng lượng mưa ngày (mm)", fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = FIGURES_DIR / "train_val_test_split.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved Train/Val/Test split plot to {out_path}")


def plot_pareto_frontier(df_all: pd.DataFrame):
    """Plot Validation RMSE vs Latency trade-off, highlighting the actual non-dominated Pareto frontier."""
    plt.figure(figsize=(10, 6))

    feasible = df_all[df_all["is_feasible"]].copy()
    infeasible = df_all[~df_all["is_feasible"]].copy()

    # Scatter plot of all trials
    plt.scatter(
        feasible["p95_latency_ms"],
        feasible["val_rmse"],
        c="#2b8cbe",
        alpha=0.55,
        edgecolors="k",
        s=45,
        label=f"Nghiệm khả thi (P95 <= {LATENCY_CONSTRAINT_MS} ms)",
    )
    plt.scatter(
        infeasible["p95_latency_ms"],
        infeasible["val_rmse"],
        c="#e34a33",
        alpha=0.35,
        edgecolors="k",
        s=35,
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

    # Compute actual Non-Dominated Pareto Frontier among feasible points
    if not feasible.empty:
        feasible_sorted = feasible.sort_values("p95_latency_ms").reset_index(drop=True)
        pareto_records = []
        min_so_far = float("inf")
        for _, row in feasible_sorted.iterrows():
            if row["val_rmse"] < min_so_far:
                min_so_far = row["val_rmse"]
                pareto_records.append(row)

        pareto_df = pd.DataFrame(pareto_records)
        if not pareto_df.empty:
            plt.step(
                pareto_df["p95_latency_ms"],
                pareto_df["val_rmse"],
                where="post",
                color="#e6550d",
                linewidth=2.2,
                linestyle="-",
                label="Biên Pareto Non-Dominated (Khả thi)",
                zorder=3,
            )
            plt.scatter(
                pareto_df["p95_latency_ms"],
                pareto_df["val_rmse"],
                color="#e6550d",
                edgecolors="black",
                s=70,
                zorder=4,
            )

        best_row = feasible.loc[feasible["val_rmse"].idxmin()]
        plt.scatter(
            [best_row["p95_latency_ms"]],
            [best_row["val_rmse"]],
            color="#fd8d3c",
            edgecolors="black",
            s=220,
            marker="*",
            zorder=5,
            label=f"Nghiệm tối ưu: L={int(best_row['L'])}h (RMSE={best_row['val_rmse']:.4f})",
        )

    plt.title("Đánh đổi Khả thi: Sai số Validation (RMSE) vs Độ trễ suy luận ($T_{e2e}$)", fontsize=12, fontweight="bold")
    plt.xlabel("End-to-End Latency P95 (ms)", fontsize=11)
    plt.ylabel("Validation RMSE (mm/h)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "pareto_frontier.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved Feasible RMSE-Latency Trade-Off plot to {out_path}")


def plot_convergence_comparison(df_random: pd.DataFrame, df_optuna: pd.DataFrame):
    """Plot best-so-far curve over 40 trials comparing Random Search vs Optuna TPE."""
    plt.figure(figsize=(9, 5))

    def get_convergence_matrix(df: pd.DataFrame) -> np.ndarray:
        seeds = df["seed"].unique()
        matrix = []

        for s in seeds:
            sub = df[df["seed"] == s].sort_values("trial")
            best_so_far = []
            curr_best = np.nan
            for _, row in sub.iterrows():
                if row["is_feasible"]:
                    if np.isnan(curr_best) or row["val_rmse"] < curr_best:
                        curr_best = row["val_rmse"]
                best_so_far.append(curr_best)
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
    plt.ylabel("Validation RMSE tốt nhất (mm/h)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = FIGURES_DIR / "convergence_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved Convergence comparison plot to {out_path}")


def plot_actual_vs_predicted(clean_df: pd.DataFrame, model: Any, L: int):
    """Plot Actual vs Predicted rainfall on a representative storm event in the Test set."""
    df_feat, fcols = build_features(clean_df, L=L)
    _, _, (X_te, y_te) = split_time_series(df_feat, fcols)

    test_plot_df = df_feat[(df_feat["target_time"] >= "2025-07-01 00:00:00") & (df_feat["target_time"] <= "2025-12-31 23:59:59")].copy()
    test_plot_df["actual"] = y_te.values
    test_plot_df["pred"] = model.predict(X_te)
    test_plot_df["target_time"] = test_plot_df["target_time"].dt.tz_localize(None)

    # Select October 2025 (peak storm period)
    oct_sample = test_plot_df[(test_plot_df["target_time"] >= "2025-10-01") & (test_plot_df["target_time"] <= "2025-10-15 23:59:59")]

    plt.figure(figsize=(13, 5))
    plt.plot(oct_sample["target_time"], oct_sample["actual"], label="Lượng mưa thực tế (Actual prcp)", color="#2b8cbe", linewidth=2.2, marker="o", markersize=4)
    plt.plot(oct_sample["target_time"], oct_sample["pred"], label="Mô hình tối ưu dự báo (Predicted by Optuna TPE)", color="#e6550d", linestyle="--", linewidth=2, marker="s", markersize=3)
    plt.title("Đối chiếu Thực tế vs Dự báo trên Tập Test (Nửa đầu Tháng 10/2025 - Đỉnh mùa mưa)", fontsize=12, fontweight="bold")
    plt.xlabel("Thời gian", fontsize=11)
    plt.ylabel("Lượng mưa 1h tới (mm)", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = FIGURES_DIR / "actual_vs_predicted_test.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved Actual vs Predicted plot to {out_path}")


def compute_split_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression and domain-specific flood warning metrics."""
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Wet-hour metrics (hours with actual precipitation > 0 mm)
    wet_mask = y_true.values > 0.0
    wet_rmse = float(root_mean_squared_error(y_true[wet_mask], y_pred[wet_mask])) if wet_mask.sum() > 0 else np.nan
    wet_mae = float(mean_absolute_error(y_true[wet_mask], y_pred[wet_mask])) if wet_mask.sum() > 0 else np.nan

    # Rain event detection (threshold >= 0.5 mm/h)
    y_true_rain = (y_true.values >= 0.5).astype(int)
    y_pred_rain = (y_pred >= 0.5).astype(int)
    prec = float(precision_score(y_true_rain, y_pred_rain, zero_division=0))
    rec = float(recall_score(y_true_rain, y_pred_rain, zero_division=0))
    f1 = float(f1_score(y_true_rain, y_pred_rain, zero_division=0))

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "wet_rmse": round(wet_rmse, 4),
        "wet_mae": round(wet_mae, 4),
        "rain_precision": round(prec, 4),
        "rain_recall": round(rec, 4),
        "rain_f1": round(f1, 4),
    }


def run_multi_model_benchmark(
    clean_df: pd.DataFrame,
    best_opt_config: pd.Series,
    best_rnd_config: pd.Series,
) -> Tuple[pd.DataFrame, Any]:
    """Benchmark all baseline and optimized models across Train, Validation, and Test sets."""
    print("\n" + "=" * 70)
    print("RUNNING MULTI-MODEL BENCHMARK (TRAIN vs VAL vs TEST + WET HOURS & F1)")
    print("=" * 70)

    benchmark_rows = []

    # 1. Persistence Baseline (y_t+1 = prcp_t)
    df_feat_1, fcols_1 = build_features(clean_df, L=1)
    (X_tr_1, y_tr_1), (X_v_1, y_v_1), (X_te_1, y_te_1) = split_time_series(df_feat_1, fcols_1)
    tr_m1 = compute_split_metrics(y_tr_1, X_tr_1["prcp_lag_0"].values)
    val_m1 = compute_split_metrics(y_v_1, X_v_1["prcp_lag_0"].values)
    te_m1 = compute_split_metrics(y_te_1, X_te_1["prcp_lag_0"].values)
    benchmark_rows.append({
        "Model": "1. Persistence (Ngây thơ)",
        "L": 1,
        "Cấu hình": "y(t+1) = prcp(t)",
        "Train_RMSE": tr_m1["rmse"],
        "Val_RMSE": val_m1["rmse"],
        "Test_RMSE": te_m1["rmse"],
        "Test_MAE": te_m1["mae"],
        "Wet_Test_RMSE": te_m1["wet_rmse"],
        "Rain_Event_F1": te_m1["rain_f1"],
        "Latency_P95": 0.05,
        "Feasible": True,
    })

    # Sliced strictly from Validation set to prevent any data snooping into Test period
    val_slice = clean_df[(clean_df["time"] >= "2025-01-01") & (clean_df["time"] <= "2025-06-30 23:59:59")]

    # 2. Linear Regression (L=6)
    df_feat_6, fcols_6 = build_features(clean_df, L=6)
    (X_tr_6, y_tr_6), (X_v_6, y_v_6), (X_te_6, y_te_6) = split_time_series(df_feat_6, fcols_6)
    lr = LinearRegression()
    lr.fit(X_tr_6, y_tr_6)
    tr_m2 = compute_split_metrics(y_tr_6, lr.predict(X_tr_6))
    val_m2 = compute_split_metrics(y_v_6, lr.predict(X_v_6))
    te_m2 = compute_split_metrics(y_te_6, lr.predict(X_te_6))
    lat_lr = benchmark_end_to_end_latency(lr, val_slice.iloc[-6:], L=6, n_runs=200, warmup=20)
    benchmark_rows.append({
        "Model": "2. Linear Regression",
        "L": 6,
        "Cấu hình": "Default OLS",
        "Train_RMSE": tr_m2["rmse"],
        "Val_RMSE": val_m2["rmse"],
        "Test_RMSE": te_m2["rmse"],
        "Test_MAE": te_m2["mae"],
        "Wet_Test_RMSE": te_m2["wet_rmse"],
        "Rain_Event_F1": te_m2["rain_f1"],
        "Latency_P95": lat_lr["p95_latency_ms"],
        "Feasible": lat_lr["is_feasible"],
    })

    # 3. Ridge Regression (L=6, alpha=1.0)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_6, y_tr_6)
    tr_m3 = compute_split_metrics(y_tr_6, ridge.predict(X_tr_6))
    val_m3 = compute_split_metrics(y_v_6, ridge.predict(X_v_6))
    te_m3 = compute_split_metrics(y_te_6, ridge.predict(X_te_6))
    lat_ridge = benchmark_end_to_end_latency(ridge, val_slice.iloc[-6:], L=6, n_runs=200, warmup=20)
    benchmark_rows.append({
        "Model": "3. Ridge Regression",
        "L": 6,
        "Cấu hình": "alpha=1.0",
        "Train_RMSE": tr_m3["rmse"],
        "Val_RMSE": val_m3["rmse"],
        "Test_RMSE": te_m3["rmse"],
        "Test_MAE": te_m3["mae"],
        "Wet_Test_RMSE": te_m3["wet_rmse"],
        "Rain_Event_F1": te_m3["rain_f1"],
        "Latency_P95": lat_ridge["p95_latency_ms"],
        "Feasible": lat_ridge["is_feasible"],
    })

    # 4. LightGBM Default (L=6, 100 trees, depth=-1)
    lgb_def = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=1)
    lgb_def.fit(X_tr_6, y_tr_6)
    tr_m4 = compute_split_metrics(y_tr_6, lgb_def.predict(X_tr_6))
    val_m4 = compute_split_metrics(y_v_6, lgb_def.predict(X_v_6))
    te_m4 = compute_split_metrics(y_te_6, lgb_def.predict(X_te_6))
    lat_def = benchmark_end_to_end_latency(lgb_def, val_slice.iloc[-6:], L=6, n_runs=200, warmup=20)
    benchmark_rows.append({
        "Model": "4. LightGBM (Mặc định)",
        "L": 6,
        "Cấu hình": "100 trees, depth=-1",
        "Train_RMSE": tr_m4["rmse"],
        "Val_RMSE": val_m4["rmse"],
        "Test_RMSE": te_m4["rmse"],
        "Test_MAE": te_m4["mae"],
        "Wet_Test_RMSE": te_m4["wet_rmse"],
        "Rain_Event_F1": te_m4["rain_f1"],
        "Latency_P95": lat_def["p95_latency_ms"],
        "Feasible": lat_def["is_feasible"],
    })

    # 5. Random Search Best (Loaded dynamically from CSV)
    rnd_L = int(best_rnd_config["L"])
    df_feat_rnd, fcols_rnd = build_features(clean_df, L=rnd_L)
    (X_tr_r, y_tr_r), (X_v_r, y_v_r), (X_te_r, y_te_r) = split_time_series(df_feat_rnd, fcols_rnd)
    lgb_rnd = lgb.LGBMRegressor(
        n_estimators=int(best_rnd_config["n_estimators"]),
        max_depth=int(best_rnd_config["max_depth"]),
        num_leaves=int(best_rnd_config["num_leaves"]),
        learning_rate=float(best_rnd_config["learning_rate"]),
        min_child_samples=int(best_rnd_config["min_child_samples"]),
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    lgb_rnd.fit(X_tr_r, y_tr_r)
    tr_m5 = compute_split_metrics(y_tr_r, lgb_rnd.predict(X_tr_r))
    val_m5 = compute_split_metrics(y_v_r, lgb_rnd.predict(X_v_r))
    te_m5 = compute_split_metrics(y_te_r, lgb_rnd.predict(X_te_r))
    lat_rnd = benchmark_end_to_end_latency(lgb_rnd, val_slice.iloc[-rnd_L:], L=rnd_L, n_runs=200, warmup=20)
    benchmark_rows.append({
        "Model": "5. Random Search Best",
        "L": rnd_L,
        "Cấu hình": f"{int(best_rnd_config['n_estimators'])} trees, depth={int(best_rnd_config['max_depth'])}, lr={float(best_rnd_config['learning_rate'])}",
        "Train_RMSE": tr_m5["rmse"],
        "Val_RMSE": val_m5["rmse"],
        "Test_RMSE": te_m5["rmse"],
        "Test_MAE": te_m5["mae"],
        "Wet_Test_RMSE": te_m5["wet_rmse"],
        "Rain_Event_F1": te_m5["rain_f1"],
        "Latency_P95": lat_rnd["p95_latency_ms"],
        "Feasible": lat_rnd["is_feasible"],
    })

    # 6. Optuna TPE Best (Overall Best Feasible Model)
    best_L = int(best_opt_config["L"])
    df_feat_opt, fcols_opt = build_features(clean_df, L=best_L)
    (X_tr_o, y_tr_o), (X_v_o, y_v_o), (X_te_o, y_te_o) = split_time_series(df_feat_opt, fcols_opt)
    lgb_opt = lgb.LGBMRegressor(
        n_estimators=int(best_opt_config["n_estimators"]),
        max_depth=int(best_opt_config["max_depth"]),
        num_leaves=int(best_opt_config["num_leaves"]),
        learning_rate=float(best_opt_config["learning_rate"]),
        min_child_samples=int(best_opt_config["min_child_samples"]),
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    lgb_opt.fit(X_tr_o, y_tr_o)
    tr_m6 = compute_split_metrics(y_tr_o, lgb_opt.predict(X_tr_o))
    val_m6 = compute_split_metrics(y_v_o, lgb_opt.predict(X_v_o))
    te_m6 = compute_split_metrics(y_te_o, lgb_opt.predict(X_te_o))
    lat_opt = benchmark_end_to_end_latency(lgb_opt, val_slice.iloc[-best_L:], L=best_L, n_runs=200, warmup=20)
    benchmark_rows.append({
        "Model": "6. Optuna TPE (Đề tài tối ưu)",
        "L": best_L,
        "Cấu hình": f"{int(best_opt_config['n_estimators'])} trees, depth={int(best_opt_config['max_depth'])}, lr={float(best_opt_config['learning_rate'])}",
        "Train_RMSE": tr_m6["rmse"],
        "Val_RMSE": val_m6["rmse"],
        "Test_RMSE": te_m6["rmse"],
        "Test_MAE": te_m6["mae"],
        "Wet_Test_RMSE": te_m6["wet_rmse"],
        "Rain_Event_F1": te_m6["rain_f1"],
        "Latency_P95": lat_opt["p95_latency_ms"],
        "Feasible": lat_opt["is_feasible"],
    })

    bench_df = pd.DataFrame(benchmark_rows)
    bench_df.to_csv(BENCHMARK_CSV, index=False)
    print(bench_df.to_string(index=False))
    print(f"\n[+] Saved multi-model benchmark table to {BENCHMARK_CSV}")

    return bench_df, lgb_opt


def evaluate_final_test_set():
    """Main evaluation pipeline orchestrating all figures and final metrics."""
    if not RANDOM_CSV.exists() or not OPTUNA_CSV.exists():
        raise FileNotFoundError("Missing optimization logs. Please run optimize_random.py and optimize_optuna.py first.")

    df_random = pd.read_csv(RANDOM_CSV)
    df_optuna = pd.read_csv(OPTUNA_CSV)
    df_all = pd.concat([df_random, df_optuna], ignore_index=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clean_df = load_clean_weather_data()

    # 1. Figure: Chronological Data Split Timeline
    plot_train_val_test_split(clean_df)

    # 2. Figure: Pareto Frontier Plot
    plot_pareto_frontier(df_all)

    # 3. Figure: Convergence Comparison Plot
    plot_convergence_comparison(df_random, df_optuna)

    # 4. Extract best feasible configurations from both methods
    feasible_rnd = df_random[df_random["is_feasible"]]
    feasible_opt = df_optuna[df_optuna["is_feasible"]]
    if feasible_rnd.empty or feasible_opt.empty:
        raise RuntimeError("Feasible configurations not found in optimization logs.")

    best_rnd_config = feasible_rnd.loc[feasible_rnd["val_rmse"].idxmin()]
    best_opt_config = feasible_opt.loc[feasible_opt["val_rmse"].idxmin()]

    # 5. Compute rigorous seed-level statistics (mean +/- std) and paired t-test
    rnd_seed_best = feasible_rnd.groupby("seed")["val_rmse"].min()
    opt_seed_best = feasible_opt.groupby("seed")["val_rmse"].min()

    common_seeds = sorted(list(set(rnd_seed_best.index).intersection(set(opt_seed_best.index))))
    rnd_vals = [float(rnd_seed_best[s]) for s in common_seeds]
    opt_vals = [float(opt_seed_best[s]) for s in common_seeds]
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(rnd_vals, opt_vals)

    # 6. Multi-Model Benchmark across Train, Validation, and Test
    bench_df, final_model = run_multi_model_benchmark(clean_df, best_opt_config, best_rnd_config)

    # 7. Figure: Actual vs Predicted Rainfall on Test Set
    plot_actual_vs_predicted(clean_df, final_model, L=int(best_opt_config["L"]))

    # 8. Final JSON Report
    best_row = bench_df.iloc[-1]
    final_report = {
        "best_configuration": {
            "method": "OptunaTPE",
            "L_hours": int(best_opt_config["L"]),
            "n_features": int(best_opt_config["n_features"]),
            "n_estimators": int(best_opt_config["n_estimators"]),
            "max_depth": int(best_opt_config["max_depth"]),
            "num_leaves": int(best_opt_config["num_leaves"]),
            "learning_rate": float(best_opt_config["learning_rate"]),
            "min_child_samples": int(best_opt_config["min_child_samples"]),
        },
        "optimization_comparison_across_seeds": {
            "seeds_evaluated": common_seeds,
            "random_search_val_rmse_mean": round(float(rnd_seed_best.mean()), 4),
            "random_search_val_rmse_std": round(float(rnd_seed_best.std()), 4),
            "optuna_tpe_val_rmse_mean": round(float(opt_seed_best.mean()), 4),
            "optuna_tpe_val_rmse_std": round(float(opt_seed_best.std()), 4),
            "val_rmse_relative_difference_pct": round(float((rnd_seed_best.mean() - opt_seed_best.mean()) / rnd_seed_best.mean() * 100), 2),
            "paired_t_statistic": round(float(t_stat), 4),
            "paired_t_p_value": round(float(p_val), 4),
        },
        "performance_across_splits": {
            "train_rmse": float(best_row["Train_RMSE"]),
            "val_rmse": float(best_row["Val_RMSE"]),
            "test_rmse": float(best_row["Test_RMSE"]),
            "test_mae": float(best_row["Test_MAE"]),
            "wet_test_rmse": float(best_row["Wet_Test_RMSE"]),
            "rain_event_f1": float(best_row["Rain_Event_F1"]),
            "train_val_gap": round(float(best_row["Val_RMSE"] - best_row["Train_RMSE"]), 4),
            "train_test_gap": round(float(best_row["Test_RMSE"] - best_row["Train_RMSE"]), 4),
        },
        "latency_compliance": {
            "test_p95_latency_ms": float(best_row["Latency_P95"]),
            "constraint_threshold_ms": LATENCY_CONSTRAINT_MS,
            "constraint_satisfied": bool(best_row["Latency_P95"] <= LATENCY_CONSTRAINT_MS),
        },
        "data_split_specification": {
            "train_hours": 17544,
            "train_pct": 66.7,
            "train_period": "2023-01-01 to 2024-12-31 (leap year 2024)",
            "val_hours": 4344,
            "val_pct": 16.5,
            "val_period": "2025-01-01 to 2025-06-30",
            "test_hours": 4416,
            "test_pct": 16.8,
            "test_period": "2025-07-01 to 2025-12-31 (monsoon peak)",
        },
    }

    with open(FINAL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Saved final evaluation report to {FINAL_METRICS_JSON}")


if __name__ == "__main__":
    evaluate_final_test_set()
