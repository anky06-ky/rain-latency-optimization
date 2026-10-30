"""
Interactive Demo Script for Next-Hour Rainfall Forecasting.

Loads the trained champion model (Optuna TPE), performs end-to-end single-sample
inference on realistic weather observations, and reports prediction latency.
"""

import sys
import time
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

from src.features.build_features import extract_single_sample_features, load_clean_weather_data

MODEL_PATH = PROJECT_ROOT / "models" / "best_optuna_lgb.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "best_optuna_lgb.json"


def run_demo():
    print("=" * 68)
    print("DEMO: DỰ BÁO LƯỢNG MƯA THỜI GIAN THỰC & ĐO ĐỘ TRỄ SUY LUẬN")
    print("Trạm quan trắc Tân Sơn Nhất (ID: 48900) - Ràng buộc: T_e2e <= 5.0 ms")
    print("=" * 68)

    # 1. Load model artifact & metadata
    if not MODEL_PATH.exists():
        print(f"[-] Model artifact not found at {MODEL_PATH}. Run 'python src/models/train.py' first.")
        return

    model = joblib.load(MODEL_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    L = meta.get("L_hours", 1)

    print(f"[+] Loaded Model: {meta['model_type']} (L={L}h, Trees={meta['hyperparameters']['n_estimators']}, Depth={meta['hyperparameters']['max_depth']}, lr={meta['hyperparameters']['learning_rate']})")
    print(f"    Features in model ({meta['n_features']}): {meta['feature_names']}")

    # 2. Get realistic sample from clean weather observations
    clean_df = load_clean_weather_data()
    # Pick a sample from monsoon season with rain (e.g. October 2025)
    sample_window = clean_df[(clean_df["time"] >= "2025-10-15 14:00:00")].head(L).copy()
    if sample_window.empty or len(sample_window) < L:
        sample_window = clean_df.iloc[-L:].copy()

    latest_time = sample_window["time"].iloc[-1]
    target_time = latest_time + pd.Timedelta(hours=1)

    print(f"\n--- [1] Dữ liệu quan trắc đầu vào tại thời điểm t ({latest_time.strftime('%Y-%m-%d %H:%M')}) ---")
    latest_row = sample_window.iloc[-1]
    print(f"    Nhiệt độ (temp)     : {latest_row['temp']:.1f} °C")
    print(f"    Độ ẩm (rhum)        : {latest_row['rhum']:.1f} %")
    print(f"    Điểm sương (dwpt)   : {latest_row['dwpt']:.1f} °C")
    print(f"    Áp suất khí quyển   : {latest_row['pres']:.1f} hPa")
    print(f"    Tốc độ gió (wspd)   : {latest_row['wspd']:.1f} km/h")
    print(f"    Mưa giờ trước (prcp): {latest_row['prcp_clean']:.1f} mm/h")

    # 3. Timed End-to-End Inference
    # Warmup
    for _ in range(10):
        feat = extract_single_sample_features(sample_window, L=L)
        _ = model.predict(feat)

    # Measure high precision
    t0 = time.perf_counter_ns()
    feat_vector = extract_single_sample_features(sample_window, L=L)
    t_feat = time.perf_counter_ns()

    pred_rain = float(model.predict(feat_vector)[0])
    t_pred = time.perf_counter_ns()

    feat_time_ms = (t_feat - t0) / 1e6
    pred_time_ms = (t_pred - t_feat) / 1e6
    total_time_ms = (t_pred - t0) / 1e6

    pred_rain = max(0.0, pred_rain)  # Non-negative precipitation

    # 4. Display Results & Flood Alert Assessment
    print(f"\n--- [2] Kết quả dự báo cho giờ kế tiếp ({target_time.strftime('%Y-%m-%d %H:%M')}) ---")
    print(f"    Lượng mưa dự báo (y_hat) : {pred_rain:.2f} mm/h")
    if pred_rain >= 10.0:
        alert = "🔴 BÁO ĐỘNG ĐỎ: Mưa rất to, nguy cơ ngập lụt đô thị nghiêm trọng!"
    elif pred_rain >= 0.5:
        alert = "🟡 CẢNH BÁO VÀNG: Có mưa dông, cần kích hoạt hệ thống thoát nước trạm bơm!"
    else:
        alert = "🟢 BÌNH THƯỜNG: Không mưa hoặc mưa không đáng kể."
    print(f"    Đánh giá cảnh báo ngập   : {alert}")

    print(f"\n--- [3] Kiểm toán độ trễ suy luận End-to-End ---")
    print(f"    Thời gian trích xuất đặc trưng : {feat_time_ms:.3f} ms")
    print(f"    Thời gian dự báo mô hình       : {pred_time_ms:.3f} ms")
    print(f"    Tổng độ trễ End-to-End (T_e2e) : {total_time_ms:.3f} ms")
    print(f"    Ràng buộc phần cứng (<= 5.0 ms): {'THỎA MÃN (FEASIBLE)' if total_time_ms <= 5.0 else 'VI PHẠM (VIOLATED)'}")
    print("=" * 68)


if __name__ == "__main__":
    run_demo()
