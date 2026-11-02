"""
Module đánh giá độ trễ suy luận.

Đo độ trễ suy luận end-to-end trên một mẫu đơn:
    T_e2e = T_feature_transform(L) + T_predict(model)
Đánh giá việc tuân thủ ràng buộc của dự án: T_e2e <= 5.0 ms
(và biên an toàn nội bộ: T_e2e <= 4.5 ms).

Quy trình đo: `warmup` lần gọi không bấm giờ, rồi `n_runs` lần gọi có bấm giờ khi GC của Python tạm dừng;
tính khả thi được xét trên P95 của các lần đo đó. Dùng model.booster_.predict trên NumPy nếu có.
Lịch sử: các lượt tìm kiếm đã ghi log (results/optimization_history_*.csv, commit 1da4ab6) được đo
bằng wrapper scikit-learn model.predict, vốn chậm hơn đường suy luận hiện tại.
"""

import gc
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.features.build_features import extract_single_sample_features

LATENCY_CONSTRAINT_MS = 5.0
SAFETY_MARGIN_MS = 4.5


def benchmark_end_to_end_latency(
    model: Any,
    raw_window_slice: pd.DataFrame,
    L: int,
    n_runs: int = 200,
    warmup: int = 20,
) -> Dict[str, float]:
    """
    Đo độ trễ suy luận end-to-end cho một yêu cầu gồm một mẫu đơn:
        T_e2e = T_feature_transform(L) + T_predict(model)
    
    Tham số:
        model: Mô hình đã huấn luyện, hỗ trợ .predict(X) hoặc .booster_.predict(X).
        raw_window_slice: DataFrame chứa L quan trắc thời tiết theo giờ gần nhất.
        L: Độ dài cửa sổ, tính bằng giờ.
        n_runs: Số lần lặp có bấm giờ.
        warmup: Số lần chạy warmup không bấm giờ, để làm nóng cache CPU.
        
    Trả về:
        Dict chứa các chỉ số độ trễ:
            - 'mean_latency_ms': Độ trễ trung bình, tính bằng mili giây.
            - 'p95_latency_ms': Độ trễ phân vị thứ 95, tính bằng mili giây.
            - 'p99_latency_ms': Độ trễ phân vị thứ 99, tính bằng mili giây.
            - 'is_feasible': Giá trị boolean cho biết p95_latency_ms <= 5.0 ms hay không.
    """
    predict_fn = model.booster_.predict if hasattr(model, "booster_") else model.predict

    # 1. Các lần chạy warmup (làm nóng cache, JIT, threadpool của OpenMP)
    for _ in range(warmup):
        feat = extract_single_sample_features(raw_window_slice, L)
        _ = predict_fn(feat)

    # 2. Đo chính xác. Tạm dừng bộ thu gom rác (như timeit vẫn làm) để các lần dừng do GC, phát sinh
    #    từ việc cấp phát bộ nhớ ở nơi khác trong tiến trình, không rơi vào đuôi P95.
    latencies_arr = np.empty(n_runs, dtype=np.float64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(n_runs):
            t0 = time.perf_counter_ns()
            feat = extract_single_sample_features(raw_window_slice, L)
            _ = predict_fn(feat)
            latencies_arr[i] = (time.perf_counter_ns() - t0) / 1_000_000.0
    finally:
        if gc_was_enabled:
            gc.enable()

    mean_lat = float(np.mean(latencies_arr))
    median_lat = float(np.median(latencies_arr))
    p95_lat = float(np.percentile(latencies_arr, 95))
    p99_lat = float(np.percentile(latencies_arr, 99))
    min_lat = float(np.min(latencies_arr))
    max_lat = float(np.max(latencies_arr))
    is_feasible = bool(p95_lat <= LATENCY_CONSTRAINT_MS)
    meets_safety = bool(p95_lat <= SAFETY_MARGIN_MS)

    return {
        "mean_latency_ms": round(mean_lat, 3),
        "median_latency_ms": round(median_lat, 3),
        "p95_latency_ms": round(p95_lat, 3),
        "p99_latency_ms": round(p99_lat, 3),
        "min_latency_ms": round(min_lat, 3),
        "max_latency_ms": round(max_lat, 3),
        "is_feasible": is_feasible,
        "meets_safety_margin": meets_safety,
        "safety_margin_ms": SAFETY_MARGIN_MS,
        "constraint_threshold_ms": LATENCY_CONSTRAINT_MS,
    }


if __name__ == "__main__":
    print("Latency module ready.")
