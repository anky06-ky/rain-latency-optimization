"""
Phương pháp tối ưu chính: Tối ưu Bayes có ràng buộc bằng Optuna (TPE).

Tìm biến quyết định x = (L, theta) trên một không gian tìm kiếm có giới hạn:
    - L: Độ dài cửa sổ lịch sử tính theo giờ
    - theta: Các siêu tham số của LightGBM
Hàm mục tiêu: cực tiểu hóa RMSE trên tập Val với ràng buộc độ trễ end-to-end P95 <= 5.0 ms,
        đo theo giao thức pha tìm kiếm (SEARCH_PROTOCOL trong src/evaluation/latency.py), giống lúc sinh log.
Ngân sách: mặc định 100 trial/seed với 5 seed ngẫu nhiên (tổng 500 lượt đánh giá, giai đoạn 2 - mở rộng ngân sách);
        có thể đặt 40 trial/seed (200 lượt đánh giá) để so sánh công bằng cùng ngân sách ở giai đoạn 1.
Chi phí tìm kiếm: mỗi trial trong log đều ghi trial_time_s (số giây thực tế cho lấy mẫu TPE, huấn luyện,
        đánh giá trên tập Val và benchmark độ trễ).
"""

import argparse
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Đảm bảo thư mục gốc của dự án có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.features.build_features import load_clean_weather_data
from src.models.optimize_random import get_train_val_split, N_ESTIMATORS_CANDIDATES
from src.evaluation.latency import benchmark_end_to_end_latency, LATENCY_CONSTRAINT_MS, SEARCH_PROTOCOL


# Tắt log chi tiết của Optuna trong quá trình tìm kiếm
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
    """Chạy một study Optuna với một seed ngẫu nhiên cho trước."""
    trial_records = []
    # Độ trễ chỉ được đo trên phần dữ liệu của tập Val (hoàn toàn không chạm vào tập Test)
    val_slice = clean_df[(clean_df["time"] >= "2025-01-01") & (clean_df["time"] <= "2025-06-30 23:59:59")]

    def objective(trial: optuna.Trial) -> float:
        # Đồng hồ đo chi phí tìm kiếm. TPE lấy mẫu bên trong suggest_* (lấy mẫu độc lập) nên được tính vào;
        # trial đầu tiên của mỗi L còn tính cả thời gian tạo đặc trưng (được lưu đệm).
        t0 = time.perf_counter()

        # 1. Đề xuất các biến quyết định
        L = trial.suggest_categorical("L", WINDOW_CANDIDATES)
        n_estimators = trial.suggest_categorical("n_estimators", N_ESTIMATORS_CANDIDATES)
        max_depth = trial.suggest_categorical("max_depth", [2, 3, 4, 6, 8, 10, 12])
        num_leaves = trial.suggest_categorical("num_leaves", [7, 15, 31, 63, 127])
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
        min_child_samples = trial.suggest_categorical("min_child_samples", [10, 20, 50, 100])

        # 2. Tạo đặc trưng cho cửa sổ L (lưu đệm: chỉ có len(WINDOW_CANDIDATES) ma trận khác nhau)
        fcols, (X_tr, y_tr), (X_v, y_v) = get_train_val_split(clean_df, L)

        # 3. Huấn luyện mô hình LightGBM
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

        # 4. Tính các chỉ số trên tập Val
        y_val_pred = model.predict(X_v)
        val_rmse = float(root_mean_squared_error(y_v, y_val_pred))
        val_mae = float(mean_absolute_error(y_v, y_val_pred))

        # 5. Đo độ trễ end-to-end trên một mẫu đơn, theo giao thức pha tìm kiếm đã sinh ra log
        raw_window = val_slice.iloc[-L:]
        lat_res = benchmark_end_to_end_latency(model, raw_window, L=L, n_runs=200, warmup=20, protocol=SEARCH_PROTOCOL)

        # 6. Đặt ràng buộc cho Optuna: mức vi phạm <= 0 nghĩa là khả thi (P95 <= 5.0 ms)
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
            # Giữ nguyên độ chính xác: làm tròn ở đây từng khiến huấn luyện lại từ CSV cho ra mô hình khác
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "val_rmse": round(val_rmse, 4),
            "val_mae": round(val_mae, 4),
            "mean_latency_ms": lat_res["mean_latency_ms"],
            "p95_latency_ms": p95_lat,
            "is_feasible": lat_res["is_feasible"],
            "trial_time_s": round(time.perf_counter() - t0, 3),
        }
        trial_records.append(record)

        return val_rmse

    def constraints_func(trial: optuna.trial.FrozenTrial):
        """Hàm ràng buộc cho TPESampler của Optuna: trả về danh sách số thực, <= 0 là khả thi."""
        return trial.user_attrs.get("constraint", [0.0])

    # Tạo study với TPE sampler có ràng buộc
    sampler = optuna.samplers.TPESampler(seed=seed, constraints_func=constraints_func)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return trial_records


def run_optuna_search(n_trials: int = 40, seeds: List[int] = DEFAULT_SEEDS) -> pd.DataFrame:
    """Chạy tìm kiếm Optuna TPE trên tất cả các seed."""
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

        # In tóm tắt kết quả của seed
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
    parser = argparse.ArgumentParser(description="Run Constrained Optuna TPE Optimization.")
    parser.add_argument("--trials", type=int, default=100, help="Number of trials per seed (default: 100 for 500 scaling trials; set 40 for the phase-1 equal-budget comparison)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds (default: 5 seeds)")
    args = parser.parse_args()

    run_optuna_search(n_trials=args.trials, seeds=args.seeds)
