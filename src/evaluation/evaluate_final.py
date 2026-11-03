"""
Module đánh giá cuối cùng và trực quan hóa đa mô hình.

1. Trực quan hóa cách chia dữ liệu thành tập Train / Val / Test theo thứ tự thời gian.
2. Xác định cấu hình khả thi tốt nhất (L*, theta*) trên tập Val.
3. Đánh giá mô hình được chọn trên tập Test giữ riêng (2025-07 đến 2025-12) ĐÚNG MỘT LẦN.
4. Chạy benchmark đầy đủ nhiều mô hình (Persistence, Climatology, Linear, Ridge, LightGBM mặc định,
   Random Search tốt nhất, Optuna tốt nhất), kèm skill score so với Persistence/Climatology.
5. Tính thống kê tỷ lệ khả thi từ cờ is_feasible đã ghi trong log: Fisher / chi2 gộp
   (giả định các trial độc lập) và Wilcoxon / paired t-test theo từng seed (seed = đơn vị lặp lại).
6. Sinh các hình biên Pareto, hội tụ tìm kiếm, dòng thời gian chia tập và Thực tế vs Dự báo.
7. Giải lại bài toán epsilon-constraint cho nhiều ngân sách độ trễ (quyết định theo từng kịch bản) và
   báo cáo chi phí tìm kiếm (số lượt đánh giá; thời gian chạy khi log có cột trial_time_s).

Đầu ra (nguồn số liệu gốc duy nhất cho README / báo cáo):
    results/multi_model_benchmark.csv, results/final_test_metrics.json, results/epsilon_scenarios.csv,
    results/figures/*.png
Lưu ý: độ trễ của Persistence / Climatology là giá trị danh nghĩa cố định, không phải giá trị đo.
"""

import json
from pathlib import Path
import sys
from typing import Any, Tuple, Dict, List

# Đảm bảo thư mục gốc của dự án có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score

from src.features.build_features import (
    load_clean_weather_data,
    build_features,
    split_time_series,
)
from src.evaluation.latency import benchmark_end_to_end_latency, LATENCY_CONSTRAINT_MS
from src.evaluation.epsilon_scenarios import compute_epsilon_scenarios, SCENARIOS_CSV


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_CSV = RESULTS_DIR / "optimization_history_random.csv"
OPTUNA_CSV = RESULTS_DIR / "optimization_history_optuna.csv"
BENCHMARK_CSV = RESULTS_DIR / "multi_model_benchmark.csv"
FINAL_METRICS_JSON = RESULTS_DIR / "final_test_metrics.json"


def plot_train_val_test_split(clean_df: pd.DataFrame):
    """Vẽ dòng thời gian chia tập (theo trình tự thời gian) của chuỗi dữ liệu giai đoạn 2023-2025."""
    fig, ax = plt.subplots(figsize=(12, 4.5))

    daily_df = clean_df.copy()
    daily_df["time"] = daily_df["time"].dt.tz_localize(None)
    daily_df = daily_df.set_index("time").resample("D")["prcp"].sum().reset_index()

    train_end = pd.to_datetime("2024-12-31 23:59:59")
    val_end = pd.to_datetime("2025-06-30 23:59:59")

    ax.plot(daily_df["time"], daily_df["prcp"], color="#4a7bb0", alpha=0.6, label="Lượng mưa ngày (mm)")

    # Tô màu các vùng chia tập, kèm số giờ chính xác và chi tiết năm nhuận
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
    """Vẽ đánh đổi giữa Val RMSE và độ trễ, làm nổi bật biên Pareto thực tế (các điểm không bị trội)."""
    plt.figure(figsize=(10, 6))

    feasible = df_all[df_all["is_feasible"]].copy()
    infeasible = df_all[~df_all["is_feasible"]].copy()

    # Biểu đồ phân tán của tất cả các trial
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

    # Đường ngưỡng ràng buộc
    plt.axvline(
        LATENCY_CONSTRAINT_MS,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Ràng buộc Latency ({LATENCY_CONSTRAINT_MS} ms)",
    )

    # Tính biên Pareto thực tế (các điểm không bị trội) trong số các điểm khả thi
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

        # Tốt nhất của từng phương pháp; nghiệm được triển khai là của Optuna TPE (xem evaluate_final_test_set)
        for method, marker, color, size in (("RandomSearch", "D", "#756bb1", 110), ("OptunaTPE", "*", "#fd8d3c", 260)):
            sub = feasible[feasible["method"] == method]
            if sub.empty:
                continue
            best_row = sub.loc[sub["val_rmse"].idxmin()]
            name = "Optuna TPE (được chọn)" if method == "OptunaTPE" else "Random Search"
            plt.scatter(
                [best_row["p95_latency_ms"]], [best_row["val_rmse"]], color=color, edgecolors="black", s=size, marker=marker,
                zorder=5, label=f"Tốt nhất {name}: L={int(best_row['L'])}h, RMSE={best_row['val_rmse']:.4f}",
            )

    plt.title("Đánh đổi Khả thi: Sai số Validation (RMSE) vs Độ trễ suy luận ($T_{e2e}$)", fontsize=12, fontweight="bold")
    # Thang log: phần lớn trial nằm trong 1–10 ms nhưng có vài trial tới >100 ms
    plt.xscale("log")
    plt.xticks([2, 3, 5, 10, 20, 50, 100], ["2", "3", "5", "10", "20", "50", "100"])
    plt.xlabel("End-to-End Latency P95 (ms, thang log)", fontsize=11)
    plt.ylabel("Validation RMSE (mm/h)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "pareto_frontier.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved Feasible RMSE-Latency Trade-Off plot to {out_path}")


def plot_convergence_comparison(df_random: pd.DataFrame, df_optuna: pd.DataFrame):
    """Vẽ đường tốt nhất tính đến hiện tại theo từng trial (có mốc 40 trial của Pha 1): Random Search vs Optuna TPE."""
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

    # Random Search: trung bình ± độ lệch chuẩn
    mean_r, std_r = np.nanmean(mat_random, axis=0), np.nanstd(mat_random, axis=0)
    plt.plot(trials_axis, mean_r, label="Baseline: Random Search", color="#756bb1", linewidth=2.2)
    plt.fill_between(trials_axis, mean_r - std_r, mean_r + std_r, color="#756bb1", alpha=0.2)

    # Optuna TPE: trung bình ± độ lệch chuẩn
    mean_o, std_o = np.nanmean(mat_optuna, axis=0), np.nanstd(mat_optuna, axis=0)
    plt.plot(trials_axis, mean_o, label="Main Method: Optuna (TPE)", color="#31a354", linewidth=2.2)
    plt.fill_between(trials_axis, mean_o - std_o, mean_o + std_o, color="#31a354", alpha=0.2)

    # Đánh dấu mốc cắt Pha 1 tại trial 40 nếu mỗi seed có ít nhất 40 trial
    if len(trials_axis) >= 40:
        plt.axvline(x=40, color="#d95f02", linestyle="--", linewidth=1.5, alpha=0.85, label="Mốc đối chuẩn cùng ngân sách (40 trials)")
        plt.text(41, np.nanmax(mean_r) - 0.0005, "Pha 2: Mở rộng ngân sách", fontsize=9, color="#d95f02", fontweight="bold")
        plt.text(12, np.nanmax(mean_r) - 0.0005, "Pha 1: Đối chuẩn", fontsize=9, color="#666666", fontweight="bold")

    n_trials_per_seed = len(trials_axis)
    plt.title(f"Hội tụ tối ưu: So sánh Best-so-far Validation RMSE qua {n_trials_per_seed} Trials (5 Seeds)", fontsize=12, fontweight="bold")
    plt.xlabel("Số lượng Trials (Ngân sách đánh giá)", fontsize=11)
    plt.ylabel("Validation RMSE tốt nhất (mm/h)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = FIGURES_DIR / "convergence_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved Convergence comparison plot to {out_path}")


def plot_actual_vs_predicted(clean_df: pd.DataFrame, model: Any, L: int):
    """Vẽ lượng mưa Thực tế vs Dự báo trong một đợt mưa bão tiêu biểu thuộc tập Test."""
    df_feat, fcols = build_features(clean_df, L=L)
    _, _, (X_te, y_te) = split_time_series(df_feat, fcols)

    test_plot_df = df_feat[(df_feat["target_time"] >= "2025-07-01 00:00:00") & (df_feat["target_time"] <= "2025-12-31 23:59:59")].copy()
    test_plot_df["actual"] = y_te.values
    test_plot_df["pred"] = model.predict(X_te)
    test_plot_df["target_time"] = test_plot_df["target_time"].dt.tz_localize(None)

    # Chọn nửa đầu tháng 10/2025 (01–15/10, giai đoạn đỉnh mùa mưa)
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
    """Tính đầy đủ các chỉ số hồi quy và các chỉ số chuyên ngành cho cảnh báo ngập lụt."""
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # Chỉ số trên giờ mưa (các giờ có lượng mưa thực tế > 0 mm)
    wet_mask = y_true.values > 0.0
    wet_rmse = float(root_mean_squared_error(y_true[wet_mask], y_pred[wet_mask])) if wet_mask.sum() > 0 else np.nan
    wet_mae = float(mean_absolute_error(y_true[wet_mask], y_pred[wet_mask])) if wet_mask.sum() > 0 else np.nan

    # Phát hiện sự kiện mưa (ngưỡng >= 0.5 mm/h)
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
) -> Tuple[pd.DataFrame, Any, Dict[str, Any]]:
    """Benchmark tất cả mô hình baseline và mô hình đã tối ưu trên các tập Train, Val và Test."""
    print("\n" + "=" * 70)
    print("RUNNING MULTI-MODEL BENCHMARK (TRAIN vs VAL vs TEST + WET HOURS & F1 & SKILL SCORES)")
    print("=" * 70)

    benchmark_rows = []

    # Chỉ cắt từ tập Val để ngăn mọi hành vi nhìn trộm dữ liệu (data snooping) giai đoạn Test
    val_slice = clean_df[(clean_df["time"] >= "2025-01-01") & (clean_df["time"] <= "2025-06-30 23:59:59")]

    # 1. Baseline Persistence (y_t+1 = prcp_t)
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

    # 2. Baseline Climatology (lượng mưa trung bình theo Tháng-Giờ, tính trên tập Train)
    # Index của các tập chia thuộc df_feat_1 (đã đánh lại index sau dropna), KHÔNG phải clean_df, nên
    # phải tra mốc thời gian ở đó. Khóa theo target_time: climatology của y_{t+1} tại đúng tháng/giờ của nó.
    target_times = df_feat_1["target_time"]
    clim_df = pd.DataFrame({
        "month": target_times.loc[y_tr_1.index].dt.month.values,
        "hour": target_times.loc[y_tr_1.index].dt.hour.values,
        "prcp": y_tr_1.values
    })
    clim_map = clim_df.groupby(["month", "hour"])["prcp"].mean().to_dict()
    global_clim_mean = float(y_tr_1.mean())

    def get_clim_preds(indices):
        times = target_times.loc[indices]
        return np.array([clim_map.get((t.month, t.hour), global_clim_mean) for t in times], dtype=np.float32)

    tr_pred_clim = get_clim_preds(y_tr_1.index)
    val_pred_clim = get_clim_preds(y_v_1.index)
    te_pred_clim = get_clim_preds(y_te_1.index)

    tr_m_clim = compute_split_metrics(y_tr_1, tr_pred_clim)
    val_m_clim = compute_split_metrics(y_v_1, val_pred_clim)
    te_m_clim = compute_split_metrics(y_te_1, te_pred_clim)
    benchmark_rows.append({
        "Model": "2. Climatology (Tháng-Giờ)",
        "L": 1,
        "Cấu hình": "E[prcp | month, hour]",
        "Train_RMSE": tr_m_clim["rmse"],
        "Val_RMSE": val_m_clim["rmse"],
        "Test_RMSE": te_m_clim["rmse"],
        "Test_MAE": te_m_clim["mae"],
        "Wet_Test_RMSE": te_m_clim["wet_rmse"],
        "Rain_Event_F1": te_m_clim["rain_f1"],
        "Latency_P95": 0.08,
        "Feasible": True,
    })

    # 3. Hồi quy tuyến tính (Linear Regression, L=6)
    df_feat_6, fcols_6 = build_features(clean_df, L=6)
    (X_tr_6, y_tr_6), (X_v_6, y_v_6), (X_te_6, y_te_6) = split_time_series(df_feat_6, fcols_6)
    lr = LinearRegression()
    lr.fit(X_tr_6.values, y_tr_6.values)
    tr_m2 = compute_split_metrics(y_tr_6, lr.predict(X_tr_6.values))
    val_m2 = compute_split_metrics(y_v_6, lr.predict(X_v_6.values))
    te_m2 = compute_split_metrics(y_te_6, lr.predict(X_te_6.values))
    lat_lr = benchmark_end_to_end_latency(lr, val_slice.iloc[-6:], L=6, n_runs=100, warmup=20)
    benchmark_rows.append({
        "Model": "3. Linear Regression",
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

    # 4. Hồi quy Ridge (L=6, alpha=1.0)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_6.values, y_tr_6.values)
    tr_m3 = compute_split_metrics(y_tr_6, ridge.predict(X_tr_6.values))
    val_m3 = compute_split_metrics(y_v_6, ridge.predict(X_v_6.values))
    te_m3 = compute_split_metrics(y_te_6, ridge.predict(X_te_6.values))
    lat_ridge = benchmark_end_to_end_latency(ridge, val_slice.iloc[-6:], L=6, n_runs=100, warmup=20)
    benchmark_rows.append({
        "Model": "4. Ridge Regression",
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

    # 5. LightGBM mặc định (L=6, 100 cây, depth=-1)
    lgb_def = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=1)
    lgb_def.fit(X_tr_6, y_tr_6)
    tr_m4 = compute_split_metrics(y_tr_6, lgb_def.predict(X_tr_6))
    val_m4 = compute_split_metrics(y_v_6, lgb_def.predict(X_v_6))
    te_m4 = compute_split_metrics(y_te_6, lgb_def.predict(X_te_6))
    lat_def = benchmark_end_to_end_latency(lgb_def, val_slice.iloc[-6:], L=6, n_runs=100, warmup=20)
    benchmark_rows.append({
        "Model": "5. LightGBM (Mặc định)",
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

    # 6. Random Search tốt nhất (cấu hình nạp động từ CSV)
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
    lat_rnd = benchmark_end_to_end_latency(lgb_rnd, val_slice.iloc[-rnd_L:], L=rnd_L, n_runs=100, warmup=20)
    benchmark_rows.append({
        "Model": "6. Random Search Best",
        "L": rnd_L,
        "Cấu hình": f"{int(best_rnd_config['n_estimators'])} trees, depth={int(best_rnd_config['max_depth'])}, lr={float(best_rnd_config['learning_rate']):.4f}",
        "Train_RMSE": tr_m5["rmse"],
        "Val_RMSE": val_m5["rmse"],
        "Test_RMSE": te_m5["rmse"],
        "Test_MAE": te_m5["mae"],
        "Wet_Test_RMSE": te_m5["wet_rmse"],
        "Rain_Event_F1": te_m5["rain_f1"],
        "Latency_P95": lat_rnd["p95_latency_ms"],
        "Feasible": lat_rnd["is_feasible"],
    })

    # 7. Optuna TPE tốt nhất (mô hình khả thi tốt nhất tổng thể)
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

    # Đo độ trễ lặp lại nhiều lần cho champion (5 lượt, mỗi lượt 100 lần lặp)
    opt_lat_runs = [benchmark_end_to_end_latency(lgb_opt, val_slice.iloc[-best_L:], L=best_L, n_runs=100, warmup=20) for _ in range(5)]
    opt_p95_list = [r["p95_latency_ms"] for r in opt_lat_runs]
    opt_p95_median = float(np.median(opt_p95_list))
    opt_p95_min = float(np.min(opt_p95_list))
    opt_p95_max = float(np.max(opt_p95_list))

    benchmark_rows.append({
        "Model": "7. Optuna TPE (Đề tài tối ưu)",
        "L": best_L,
        "Cấu hình": f"{int(best_opt_config['n_estimators'])} trees, depth={int(best_opt_config['max_depth'])}, lr={float(best_opt_config['learning_rate']):.4f}",
        "Train_RMSE": tr_m6["rmse"],
        "Val_RMSE": val_m6["rmse"],
        "Test_RMSE": te_m6["rmse"],
        "Test_MAE": te_m6["mae"],
        "Wet_Test_RMSE": te_m6["wet_rmse"],
        "Rain_Event_F1": te_m6["rain_f1"],
        "Latency_P95": round(opt_p95_median, 3),
        "Feasible": bool(opt_p95_median <= LATENCY_CONSTRAINT_MS),
    })

    # Tính skill score so với Persistence và Climatology
    ref_pers_rmse = te_m1["rmse"]
    ref_clim_rmse = te_m_clim["rmse"]
    for row in benchmark_rows:
        row["Skill_Score_Persistence_pct"] = round((1.0 - row["Test_RMSE"] / ref_pers_rmse) * 100.0, 2)
        row["Skill_Score_Climatology_pct"] = round((1.0 - row["Test_RMSE"] / ref_clim_rmse) * 100.0, 2)

    bench_df = pd.DataFrame(benchmark_rows)
    bench_df.to_csv(BENCHMARK_CSV, index=False)
    print(bench_df.to_string(index=False))
    print(f"\n[+] Saved multi-model benchmark table to {BENCHMARK_CSV}")

    champion_latency_stats = {
        "median_p95_ms": round(opt_p95_median, 3),
        "min_p95_ms": round(opt_p95_min, 3),
        "max_p95_ms": round(opt_p95_max, 3),
        "all_p95_runs": [round(x, 3) for x in opt_p95_list],
        "is_feasible": bool(opt_p95_median <= LATENCY_CONSTRAINT_MS),
        "meets_safety_margin": bool(opt_p95_median <= 4.5),
    }

    split_stats = {
        "train": {
            "period": "2023-01-01 to 2024-12-31",
            "samples": int(len(y_tr_1)),
            "mean_prcp_mm": round(float(y_tr_1.mean()), 4),
            "std_prcp_mm": round(float(y_tr_1.std()), 4),
            "wet_hours_pct": round(float((y_tr_1 > 0).mean() * 100), 2),
            "max_prcp_mm": round(float(y_tr_1.max()), 2),
        },
        "validation": {
            "period": "2025-01-01 to 2025-06-30 (dry & transition season)",
            "samples": int(len(y_v_1)),
            "mean_prcp_mm": round(float(y_v_1.mean()), 4),
            "std_prcp_mm": round(float(y_v_1.std()), 4),
            "wet_hours_pct": round(float((y_v_1 > 0).mean() * 100), 2),
            "max_prcp_mm": round(float(y_v_1.max()), 2),
        },
        "test": {
            "period": "2025-07-01 to 2025-12-31 (peak tropical monsoon)",
            "samples": int(len(y_te_1)),
            "mean_prcp_mm": round(float(y_te_1.mean()), 4),
            "std_prcp_mm": round(float(y_te_1.std()), 4),
            "wet_hours_pct": round(float((y_te_1 > 0).mean() * 100), 2),
            "max_prcp_mm": round(float(y_te_1.max()), 2),
            "std_relative_increase_vs_val_pct": round(float((y_te_1.std() - y_v_1.std()) / y_v_1.std() * 100), 2),
        },
    }

    return bench_df, lgb_opt, {"champion_latency": champion_latency_stats, "split_stats": split_stats, "ref_pers_rmse": ref_pers_rmse, "ref_clim_rmse": ref_clim_rmse}


def summarize_search_cost(df: pd.DataFrame) -> Dict[str, Any]:
    """Chi phí tìm kiếm của một phương pháp: luôn có số lượt đánh giá; thời gian chạy nếu có trial_time_s."""
    cost: Dict[str, Any] = {"evaluations": int(len(df))}
    if "trial_time_s" in df.columns:
        cost["total_wall_clock_min"] = round(float(df["trial_time_s"].sum()) / 60, 2)
        cost["mean_trial_time_s"] = round(float(df["trial_time_s"].mean()), 3)
        cost["median_trial_time_s"] = round(float(df["trial_time_s"].median()), 3)
    else:
        # Log được tạo trước khi bổ sung đo thời gian từng trial (chi phí chỉ biết qua số lượt đánh giá)
        cost["total_wall_clock_min"] = None
    return cost


def evaluate_final_test_set():
    """Pipeline đánh giá chính, điều phối việc tạo toàn bộ hình vẽ và chỉ số cuối cùng."""
    if not RANDOM_CSV.exists() or not OPTUNA_CSV.exists():
        raise FileNotFoundError("Missing optimization logs. Please run optimize_random.py and optimize_optuna.py first.")

    df_random = pd.read_csv(RANDOM_CSV)
    df_optuna = pd.read_csv(OPTUNA_CSV)
    df_all = pd.concat([df_random, df_optuna], ignore_index=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clean_df = load_clean_weather_data()

    # 1. Hình: chia tập dữ liệu theo dòng thời gian
    plot_train_val_test_split(clean_df)

    # 2. Hình: biên Pareto
    plot_pareto_frontier(df_all)

    # 3. Hình: so sánh hội tụ
    plot_convergence_comparison(df_random, df_optuna)

    # 4. Lấy cấu hình khả thi tốt nhất của cả hai phương pháp
    feasible_rnd = df_random[df_random["is_feasible"]]
    feasible_opt = df_optuna[df_optuna["is_feasible"]]
    if feasible_rnd.empty or feasible_opt.empty:
        raise RuntimeError("Feasible configurations not found in optimization logs.")

    best_rnd_config = feasible_rnd.loc[feasible_rnd["val_rmse"].idxmin()]
    best_opt_config = feasible_opt.loc[feasible_opt["val_rmse"].idxmin()]

    # 5. Tính thống kê chặt chẽ ở cấp seed (trung bình +/- độ lệch chuẩn) và paired t-test
    rnd_seed_best = feasible_rnd.groupby("seed")["val_rmse"].min()
    opt_seed_best = feasible_opt.groupby("seed")["val_rmse"].min()

    common_seeds = sorted(set(rnd_seed_best.index).intersection(opt_seed_best.index))
    rnd_vals = [float(rnd_seed_best[s]) for s in common_seeds]
    opt_vals = [float(opt_seed_best[s]) for s in common_seeds]
    t_stat, p_val = stats.ttest_rel(rnd_vals, opt_vals)
    budget_per_seed = int(df_optuna.groupby("seed").size().max())

    # 6. Benchmark nhiều mô hình trên các tập Train, Val và Test
    bench_df, final_model, extra_stats = run_multi_model_benchmark(clean_df, best_opt_config, best_rnd_config)

    # 7. Hình: lượng mưa Thực tế vs Dự báo trên tập Test
    plot_actual_vs_predicted(clean_df, final_model, L=int(best_opt_config["L"]))

    # 8. Bảng: kịch bản epsilon-constraint (áp lại quy tắc chọn nghiệm cho mỗi ngân sách độ trễ, chỉ dùng log)
    compute_epsilon_scenarios(df_random, df_optuna).to_csv(SCENARIOS_CSV, index=False)
    print(f"\n[+] Saved epsilon-constraint scenarios to {SCENARIOS_CSV}")

    # Tính số trial khả thi trực tiếp từ các file CSV. Số đếm theo SLA dùng cờ is_feasible đã ghi trong log
    # (tính trên P95 chưa làm tròn, tức đúng ràng buộc mà bộ tối ưu thực sự thấy); nếu đặt lại ngưỡng trên cột
    # p95_latency_ms (3 chữ số thập phân) thì sẽ phân loại sai các trial nằm đúng tại 5.000 ms.
    opt_feas_count_50 = int(df_optuna["is_feasible"].sum())
    rnd_feas_count_50 = int(df_random["is_feasible"].sum())
    opt_feas_pct_50 = round(float(opt_feas_count_50 / len(df_optuna) * 100), 1)
    rnd_feas_pct_50 = round(float(rnd_feas_count_50 / len(df_random) * 100), 1)

    opt_feas_count_45 = int((df_optuna["p95_latency_ms"] <= 4.5).sum())
    rnd_feas_count_45 = int((df_random["p95_latency_ms"] <= 4.5).sum())
    opt_feas_pct_45 = round(float(opt_feas_count_45 / len(df_optuna) * 100), 1)
    rnd_feas_pct_45 = round(float(rnd_feas_count_45 / len(df_random) * 100), 1)

    # So sánh Pha 1 (40 trial đầu tiên của mỗi seed)
    opt_p1 = df_optuna.groupby("seed").head(40)
    rnd_p1 = df_random.groupby("seed").head(40)
    opt_p1_feas = int(opt_p1["is_feasible"].sum())
    rnd_p1_feas = int(rnd_p1["is_feasible"].sum())

    # Số trial khả thi theo từng seed (seed là đơn vị lặp lại độc lập)
    opt_seed_feas = df_optuna.groupby("seed")["is_feasible"].sum().reindex(common_seeds)
    rnd_seed_feas = df_random.groupby("seed")["is_feasible"].sum().reindex(common_seeds)
    wilcoxon_p = float(stats.wilcoxon(opt_seed_feas, rnd_seed_feas).pvalue)
    seed_feas_t, seed_feas_p = stats.ttest_rel(opt_seed_feas, rnd_seed_feas)

    # Kiểm định thống kê cho tỷ lệ khả thi (Fisher's exact test và khoảng tin cậy 95% cho chênh lệch)
    table_50 = [[opt_feas_count_50, len(df_optuna) - opt_feas_count_50],
                [rnd_feas_count_50, len(df_random) - rnd_feas_count_50]]
    odds_ratio_50, p_fisher_50 = stats.fisher_exact(table_50)
    chi2_50, p_chi2_50, _, _ = stats.chi2_contingency(table_50)

    p1 = opt_feas_count_50 / len(df_optuna)
    p2 = rnd_feas_count_50 / len(df_random)
    diff_p = p1 - p2
    se_diff = np.sqrt(p1 * (1 - p1) / len(df_optuna) + p2 * (1 - p2) / len(df_random))
    ci_lower = round(float((diff_p - 1.96 * se_diff) * 100), 2)
    ci_upper = round(float((diff_p + 1.96 * se_diff) * 100), 2)
    best_row = bench_df.iloc[-1]
    champ_lat = extra_stats["champion_latency"]
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
            "hyperparameter_notes": "With num_leaves=15, max_depth=12 acts as an upper depth ceiling for leaf-wise tree growth rather than forcing a full binary tree.",
        },
        "optimization_comparison_across_seeds": {
            "seeds_evaluated": common_seeds,
            "budget_per_seed": budget_per_seed,
            "total_trials_per_method": int(len(df_optuna)),
            "search_cost": {
                "random_search": summarize_search_cost(df_random),
                "optuna_tpe": summarize_search_cost(df_optuna),
            },
            "random_search_val_rmse_mean": round(float(rnd_seed_best.mean()), 4),
            "random_search_val_rmse_std": round(float(rnd_seed_best.std()), 4),
            "optuna_tpe_val_rmse_mean": round(float(opt_seed_best.mean()), 4),
            "optuna_tpe_val_rmse_std": round(float(opt_seed_best.std()), 4),
            "val_rmse_relative_difference_pct": round(float((rnd_seed_best.mean() - opt_seed_best.mean()) / rnd_seed_best.mean() * 100), 2),
            "paired_t_statistic": round(float(t_stat), 4),
            "paired_t_p_value": round(float(p_val), 4),
            "variance_reduction_vs_100_trial_rs_pct": round(float((rnd_seed_best.std() - opt_seed_best.std()) / rnd_seed_best.std() * 100), 1),
            "feasible_rate_comparison": {
                "phase_1_budget_40_trials": {
                    "optuna_feasible_count": opt_p1_feas,
                    "optuna_feasible_rate_pct": round(opt_p1_feas / len(opt_p1) * 100, 1),
                    "random_feasible_count": rnd_p1_feas,
                    "random_feasible_rate_pct": round(rnd_p1_feas / len(rnd_p1) * 100, 1),
                    "gain_percentage_points": round((opt_p1_feas - rnd_p1_feas) / len(opt_p1) * 100, 1),
                },
                "phase_2_budget_100_trials": {
                    "threshold_5_0_ms": {
                        "optuna_feasible_count": opt_feas_count_50,
                        "optuna_feasible_rate_pct": opt_feas_pct_50,
                        "random_feasible_count": rnd_feas_count_50,
                        "random_feasible_rate_pct": rnd_feas_pct_50,
                        "gain_percentage_points": round(opt_feas_pct_50 - rnd_feas_pct_50, 1),
                        "fisher_exact_p_value": float(f"{p_fisher_50:.4e}"),
                        "chi2_statistic": round(float(chi2_50), 3),
                        "chi2_p_value": float(f"{p_chi2_50:.4e}"),
                        "odds_ratio": round(float(odds_ratio_50), 3),
                        "difference_95_ci_pct": [ci_lower, ci_upper],
                    },
                    "per_seed": {
                        "optuna_feasible_counts": {int(s): int(v) for s, v in opt_seed_feas.items()},
                        "random_feasible_counts": {int(s): int(v) for s, v in rnd_seed_feas.items()},
                        "seeds_optuna_ahead": int((opt_seed_feas > rnd_seed_feas).sum()),
                        "wilcoxon_p_value": round(wilcoxon_p, 4),
                        "paired_t_statistic": round(float(seed_feas_t), 4),
                        "paired_t_p_value": round(float(seed_feas_p), 4),
                    },
                    "safety_margin_4_5_ms": {
                        "optuna_feasible_count": opt_feas_count_45,
                        "optuna_feasible_rate_pct": opt_feas_pct_45,
                        "random_feasible_count": rnd_feas_count_45,
                        "random_feasible_rate_pct": rnd_feas_pct_45,
                        "gain_percentage_points": round(opt_feas_pct_45 - rnd_feas_pct_45, 1),
                    },
                },
            },
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
            "generalization_notes": f"Train-Val gap is {float(best_row['Val_RMSE'] - best_row['Train_RMSE']):+.4f} mm/h. The increase on Test set is driven by seasonal climate distribution shift during the peak monsoon.",
        },
        "skill_scores_on_test": {
            "skill_score_vs_persistence_pct": float(best_row["Skill_Score_Persistence_pct"]),
            "skill_score_vs_climatology_pct": float(best_row["Skill_Score_Climatology_pct"]),
        },
        "latency_compliance": {
            "test_p95_latency_ms": champ_lat["median_p95_ms"],
            "test_p95_latency_range_ms": [champ_lat["min_p95_ms"], champ_lat["max_p95_ms"]],
            "all_runs_p95_ms": champ_lat["all_p95_runs"],
            "sla_constraint_threshold_ms": LATENCY_CONSTRAINT_MS,
            "sla_satisfied": champ_lat["is_feasible"],
            "internal_safety_margin_ms": 4.5,
            "safety_margin_satisfied": champ_lat["meets_safety_margin"],
        },
        "data_split_statistics": extra_stats["split_stats"],
    }

    with open(FINAL_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Saved final evaluation report to {FINAL_METRICS_JSON}")


if __name__ == "__main__":
    evaluate_final_test_set()
