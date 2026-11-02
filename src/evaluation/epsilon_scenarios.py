"""
Phân tích kịch bản epsilon-constraint (quyết định theo từng ngân sách độ trễ).

Bài toán của dự án là cách phát biểu epsilon-constraint cho sự đánh đổi Độ chính xác - Độ trễ:
    min Val RMSE(x)  với ràng buộc  P95 latency(x) <= eps,   trong đó eps = 5.0 ms.
Module này giải lại bài toán cho nhiều ngân sách triển khai eps, chỉ dùng các trial đã ghi log trong
results/optimization_history_*.csv. Không huấn luyện lại mô hình nào và không hề chạm tới tập Test, nên
bảng này thuần túy là việc đọc lại lịch sử tìm kiếm và có thể sinh lại trong vài giây.

Lưu ý: các giá trị P95 trong log đến từ pha tìm kiếm (wrapper scikit-learn, commit 1da4ab6) và cao hơn
đường triển khai booster_.predict (P95 của champion ~0.68 ms). Vì vậy kịch bản rỗng nghĩa là "không có
trial nào trong log đạt eps theo cách đo của pha tìm kiếm", chứ không phải eps không thể đạt khi triển khai.

Đầu ra: results/epsilon_scenarios.csv (mỗi dòng ứng với một cặp eps x phương pháp).
"""

from pathlib import Path
import sys
from typing import List

# Đảm bảo thư mục gốc của dự án có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.latency import LATENCY_CONSTRAINT_MS, SAFETY_MARGIN_MS


RESULTS_DIR = Path("results")
RANDOM_CSV = RESULTS_DIR / "optimization_history_random.csv"
OPTUNA_CSV = RESULTS_DIR / "optimization_history_optuna.csv"
SCENARIOS_CSV = RESULTS_DIR / "epsilon_scenarios.csv"

EPSILON_SCENARIOS_MS = [1.0, 2.0, 3.0, SAFETY_MARGIN_MS, LATENCY_CONSTRAINT_MS]
CONFIG_COLS = ["L", "n_estimators", "max_depth", "num_leaves", "learning_rate", "min_child_samples"]
# Kiểu int nullable: nếu không, các kịch bản rỗng sẽ biến các cột này thành float (789.0) trong CSV
INT_COLS = ["seed", "trial", "L", "n_estimators", "max_depth", "num_leaves", "min_child_samples"]


def feasible_mask(df: pd.DataFrame, eps_ms: float) -> pd.Series:
    """
    Các trial thỏa P95 <= eps. Tại ngưỡng chính thức, dùng cờ is_feasible đã ghi trong log: cờ này được
    tính trên P95 chưa làm tròn mà bộ tối ưu thực sự thấy, còn cột p95_latency_ms (3 chữ số thập phân)
    phân loại sai các trial nằm đúng tại 5.000 ms.
    """
    if eps_ms == LATENCY_CONSTRAINT_MS:
        return df["is_feasible"].astype(bool)
    return df["p95_latency_ms"] <= eps_ms


def compute_epsilon_scenarios(
    df_random: pd.DataFrame,
    df_optuna: pd.DataFrame,
    eps_list: List[float] = EPSILON_SCENARIOS_MS,
) -> pd.DataFrame:
    """
    Với mỗi ngân sách eps và mỗi phương pháp, báo cáo số lượng/tỷ lệ trial khả thi và cấu hình mà quy tắc
    chọn nghiệm của dự án (Val RMSE thấp nhất trong các trial khả thi) sẽ chọn. Các dòng không có trial
    khả thi vẫn giữ số đếm và để trống các cột cấu hình.
    """
    rows = []
    for eps in eps_list:
        for method, df in (("RandomSearch", df_random), ("OptunaTPE", df_optuna)):
            feas = df[feasible_mask(df, eps)]
            row = {
                "epsilon_ms": eps,
                "method": method,
                "feasible_count": int(len(feas)),
                "total_trials": int(len(df)),
                "feasible_rate_pct": round(len(feas) / len(df) * 100, 1),
            }
            if not feas.empty:
                best = feas.loc[feas["val_rmse"].idxmin()]
                row.update({
                    "seed": int(best["seed"]),
                    "trial": int(best["trial"]),
                    **{c: best[c] for c in CONFIG_COLS},
                    "val_rmse": float(best["val_rmse"]),
                    "p95_latency_ms": float(best["p95_latency_ms"]),
                })
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.astype({c: "Int64" for c in INT_COLS if c in df.columns})


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not RANDOM_CSV.exists() or not OPTUNA_CSV.exists():
        raise FileNotFoundError("Missing optimization logs. Please run optimize_random.py and optimize_optuna.py first.")

    scenarios = compute_epsilon_scenarios(pd.read_csv(RANDOM_CSV), pd.read_csv(OPTUNA_CSV))
    scenarios.to_csv(SCENARIOS_CSV, index=False)
    print(scenarios.to_string(index=False))
    print(f"\n[+] Saved epsilon-constraint scenarios to {SCENARIOS_CSV}")
