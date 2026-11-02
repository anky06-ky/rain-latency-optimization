"""
Inference Latency Evaluation Module.

Measures single-sample End-to-End Inference Latency:
    T_e2e = T_feature_transform(L) + T_predict(model)
Evaluates compliance with the project constraint: T_e2e <= 5.0 ms
(and the internal safety margin: T_e2e <= 4.5 ms).

Measurement protocol: `warmup` untimed calls, then `n_runs` timed calls with Python GC paused;
feasibility is judged on the P95 of those runs. Uses model.booster_.predict on NumPy when available.
History: the logged search runs (results/optimization_history_*.csv, commit 1da4ab6) were measured
with the scikit-learn wrapper model.predict, which is slower than the current path.
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
    Measure end-to-end inference latency on a single-sample request:
        T_e2e = T_feature_transform(L) + T_predict(model)
    
    Parameters:
        model: Fitted estimator supporting .predict(X) or .booster_.predict(X).
        raw_window_slice: DataFrame containing the last L hourly weather observations.
        L: Window length in hours.
        n_runs: Number of timed repetitions.
        warmup: Number of untimed warmup runs to prime CPU caches.
        
    Returns:
        Dictionary with latency metrics:
            - 'mean_latency_ms': Average latency in milliseconds.
            - 'p95_latency_ms': 95th percentile latency in milliseconds.
            - 'p99_latency_ms': 99th percentile latency in milliseconds.
            - 'is_feasible': Boolean indicating whether p95_latency_ms <= 5.0 ms.
    """
    predict_fn = model.booster_.predict if hasattr(model, "booster_") else model.predict

    # 1. Warmup runs (prime cache, JIT, OpenMP threadpool)
    for _ in range(warmup):
        feat = extract_single_sample_features(raw_window_slice, L)
        _ = predict_fn(feat)

    # 2. Precision benchmarking. Garbage collection is paused (as timeit does) so that GC pauses
    #    triggered by allocations elsewhere in the process do not land in the P95 tail.
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
