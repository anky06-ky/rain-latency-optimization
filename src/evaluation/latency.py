"""
Module đánh giá độ trễ suy luận.

Đo độ trễ suy luận end-to-end trên một mẫu đơn:
    T_e2e = T_feature_transform(L) + T_predict(model)
Đánh giá việc tuân thủ ràng buộc của dự án: T_e2e <= 5.0 ms
(và biên an toàn nội bộ: T_e2e <= 4.5 ms).

Quy trình đo: `warmup` lần gọi không bấm giờ, rồi `n_runs` lần gọi có bấm giờ; tính khả thi được xét
trên P95 của các lần đo đó. Có hai giao thức đo, khác nhau ở đường suy luận được bấm giờ:
- SEARCH_PROTOCOL ("search"): giao thức đã sinh ra results/optimization_history_*.csv (commit 1da4ab6),
  dùng trong optimize_*.py. Chọn tập con cột DataFrame ở mỗi lần gọi, wrapper scikit-learn model.predict,
  GC của Python vẫn bật. Chạy lại tìm kiếm bằng giao thức này mới so sánh được với log hiện có.
- DEPLOY_PROTOCOL ("deploy", mặc định): đường triển khai, dùng cho mô hình champion và bảng đối chuẩn.
  Trích đặc trưng bằng NumPy, model.booster_.predict nếu có, GC tạm dừng trong vòng đo. Nhanh hơn "search".
"""

import gc
import time
import warnings
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.features.build_features import (
    extract_single_sample_features,
    extract_features_from_array,
    RAW_WINDOW_COLS,
)

LATENCY_CONSTRAINT_MS = 5.0
SAFETY_MARGIN_MS = 4.5
SEARCH_PROTOCOL = "search"
DEPLOY_PROTOCOL = "deploy"


def extract_features_search_protocol(raw_window_slice: pd.DataFrame, L: int) -> np.ndarray:
    """
    Bộ trích đặc trưng của giao thức "search" (commit 1da4ab6). Cho cùng vector đặc trưng như
    extract_single_sample_features, nhưng chọn tập con cột (df[[...]]) nên tạo DataFrame mới ở mỗi lần gọi;
    đây là phần chi phí mà log pha tìm kiếm đã đo.
    """
    arr = raw_window_slice[RAW_WINDOW_COLS].to_numpy(dtype=np.float32)
    curr_time = raw_window_slice["time"].iloc[-1]
    return extract_features_from_array(arr, curr_time.hour, curr_time.month, L)


def benchmark_end_to_end_latency(
    model: Any,
    raw_window_slice: pd.DataFrame,
    L: int,
    n_runs: int = 200,
    warmup: int = 20,
    protocol: str = DEPLOY_PROTOCOL,
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
        protocol: SEARCH_PROTOCOL (pha tìm kiếm, khớp log) hoặc DEPLOY_PROTOCOL (đường triển khai).
        
    Trả về:
        Dict chứa các chỉ số độ trễ:
            - 'mean_latency_ms': Độ trễ trung bình, tính bằng mili giây.
            - 'p95_latency_ms': Độ trễ phân vị thứ 95, tính bằng mili giây.
            - 'p99_latency_ms': Độ trễ phân vị thứ 99, tính bằng mili giây.
            - 'is_feasible': Giá trị boolean cho biết p95_latency_ms <= 5.0 ms hay không.
    """
    if protocol == SEARCH_PROTOCOL:
        extract_fn, predict_fn, pause_gc = extract_features_search_protocol, model.predict, False
    elif protocol == DEPLOY_PROTOCOL:
        predict_fn = model.booster_.predict if hasattr(model, "booster_") else model.predict
        extract_fn, pause_gc = extract_single_sample_features, True
    else:
        raise ValueError(f"Unknown latency protocol: {protocol!r}")

    latencies_arr = np.empty(n_runs, dtype=np.float64)
    gc_was_enabled = gc.isenabled()
    with warnings.catch_warnings():
        # Mô hình học trên DataFrame nhưng dự báo trên mảng NumPy: bỏ cảnh báo tên cột của scikit-learn
        # (commit 1da4ab6 cũng bỏ cảnh báo này)
        warnings.simplefilter("ignore", UserWarning)

        # 1. Các lần chạy warmup (làm nóng cache, JIT, threadpool của OpenMP)
        for _ in range(warmup):
            feat = extract_fn(raw_window_slice, L)
            _ = predict_fn(feat)

        # 2. Đo chính xác. Với "deploy", tạm dừng bộ thu gom rác (như timeit vẫn làm) để các lần dừng do GC,
        #    phát sinh từ việc cấp phát bộ nhớ ở nơi khác trong tiến trình, không rơi vào đuôi P95.
        #    "search" giữ GC bật như lúc sinh log.
        if pause_gc:
            gc.disable()
        try:
            for i in range(n_runs):
                t0 = time.perf_counter_ns()
                feat = extract_fn(raw_window_slice, L)
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
        "protocol": protocol,
    }


if __name__ == "__main__":
    print("Latency module ready.")
