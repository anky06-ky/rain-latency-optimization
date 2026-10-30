"""
Interactive Web Application Server for Rainfall Nowcasting under Latency Constraints.
Uses Python built-in http.server to provide REST APIs and serve the frontend dashboard.
"""

import sys
import json
import time
import math
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import extract_single_sample_features

WEB_DIR = PROJECT_ROOT / "web"
MODEL_PATH = PROJECT_ROOT / "models" / "best_optuna_lgb.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "best_optuna_lgb.json"
BENCHMARK_PATH = PROJECT_ROOT / "results" / "multi_model_benchmark.csv"

# Global model state
MODEL = None
METADATA = None
FEATURE_COLS = None
BENCHMARK_DATA = []

# Pre-allocate template DataFrame for fast-path inference
TEMPLATE_DF = None

def load_app_resources():
    global MODEL, METADATA, FEATURE_COLS, BENCHMARK_DATA, TEMPLATE_DF
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        print(f"[+] Loaded Champion Model from {MODEL_PATH}")
    else:
        print(f"[-] Warning: Model not found at {MODEL_PATH}")

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
            FEATURE_COLS = METADATA.get("feature_names", [
                "temp_lag_0", "dwpt_lag_0", "rhum_lag_0", "pres_lag_0", "wspd_lag_0",
                "prcp_lag_0", "sin_hour", "cos_hour", "sin_month", "cos_month"
            ])
            TEMPLATE_DF = pd.DataFrame(np.zeros((1, len(FEATURE_COLS)), dtype=np.float32), columns=FEATURE_COLS)
            print(f"[+] Loaded Model Metadata (L={METADATA.get('L_hours', 1)}h, {len(FEATURE_COLS)} features)")

    if BENCHMARK_PATH.exists():
        try:
            df = pd.read_csv(BENCHMARK_PATH)
            BENCHMARK_DATA = df.to_dict(orient="records")
        except Exception as e:
            print(f"[-] Failed to load benchmark CSV: {e}")

    # Warmup model once at boot
    if MODEL is not None and TEMPLATE_DF is not None:
        for _ in range(20):
            _ = MODEL.predict(TEMPLATE_DF)
        print("[+] Model warmup completed (20 iterations)")


def predict_single(data: dict):
    """
    Run timed end-to-end inference on input weather observation.
    Input dictionary: temp, dwpt, rhum, pres, wspd, prcp, hour, month.
    """
    global MODEL, METADATA, FEATURE_COLS, TEMPLATE_DF

    t0 = time.perf_counter_ns()

    hour = int(data.get("hour", 14))
    month = int(data.get("month", 10))
    temp = float(data.get("temp", 32.0))
    dwpt = float(data.get("dwpt", 24.0))
    rhum = float(data.get("rhum", 75.0))
    pres = float(data.get("pres", 1008.0))
    wspd = float(data.get("wspd", 10.0))
    prcp = float(data.get("prcp", 0.0))

    # Fast NumPy Vectorized Feature Extraction (Single Sample)
    sin_h = math.sin(2.0 * math.pi * hour / 24.0)
    cos_h = math.cos(2.0 * math.pi * hour / 24.0)
    sin_m = math.sin(2.0 * math.pi * month / 12.0)
    cos_m = math.cos(2.0 * math.pi * month / 12.0)

    feat_arr = np.array([[temp, dwpt, rhum, pres, wspd, prcp, sin_h, cos_h, sin_m, cos_m]], dtype=np.float32)
    t_feat = time.perf_counter_ns()

    # Reuse pre-allocated DataFrame for zero allocations
    if TEMPLATE_DF is not None:
        TEMPLATE_DF.iloc[0] = feat_arr[0]
        input_data = TEMPLATE_DF
    else:
        input_data = feat_arr

    pred_raw = float(MODEL.predict(input_data)[0]) if MODEL is not None else 0.0
    t_pred = time.perf_counter_ns()

    pred_rain = max(0.0, round(pred_raw, 2))
    t_feature_ms = (t_feat - t0) / 1e6
    t_predict_ms = (t_pred - t_feat) / 1e6
    t_total_ms = (t_pred - t0) / 1e6

    # Determine Flood Alert Category
    if pred_rain >= 10.0:
        alert_level = "RED"
        alert_title = "BÁO ĐỘNG ĐỎ: Mưa Cực Đoan"
        alert_desc = "Lượng mưa giờ tới >= 10.0 mm/h. Nguy cơ ngập sâu các tuyến đường trũng tại TP.HCM. Kích hoạt toàn bộ công suất trạm bơm."
    elif pred_rain >= 3.0:
        alert_level = "ORANGE"
        alert_title = "CẢNH BÁO CAM: Mưa Rào Nặng Hạt"
        alert_desc = "Lượng mưa giờ tới từ 3.0 - 9.9 mm/h. Nguy cơ ngập cục bộ trong giờ cao điểm. Chuẩn bị đóng cống ngăn triều."
    elif pred_rain >= 0.5:
        alert_level = "YELLOW"
        alert_title = "CẢNH BÁO VÀNG: Mưa Dông Nhẹ"
        alert_desc = "Lượng mưa giờ tới từ 0.5 - 2.9 mm/h. Khả năng phát triển mây đối lưu nhiệt đới. Giám sát mực nước kênh rạch."
    else:
        alert_level = "GREEN"
        alert_title = "BÌNH THƯỜNG: Không Mưa / Mưa Không Đáng Kể"
        alert_desc = "Lượng mưa < 0.5 mm/h. Hệ thống tiêu thoát nước vận hành ở trạng thái bình thường."

    is_feasible = t_total_ms <= 5.0

    return {
        "pred_rain_mm": pred_rain,
        "alert": {
            "level": alert_level,
            "title": alert_title,
            "desc": alert_desc
        },
        "latency": {
            "feature_ms": round(t_feature_ms, 3),
            "predict_ms": round(t_predict_ms, 3),
            "total_ms": round(t_total_ms, 3),
            "constraint_ms": 5.0,
            "is_feasible": is_feasible,
            "utilization_pct": round((t_total_ms / 5.0) * 100, 1)
        },
        "model_info": {
            "model_type": METADATA.get("model_type", "LightGBMRegressor") if METADATA else "LightGBM",
            "L_hours": 1,
            "n_estimators": METADATA.get("hyperparameters", {}).get("n_estimators", 150) if METADATA else 150,
            "max_depth": METADATA.get("hyperparameters", {}).get("max_depth", 12) if METADATA else 12,
            "learning_rate": METADATA.get("hyperparameters", {}).get("learning_rate", 0.0215) if METADATA else 0.0215,
        }
    }


def run_latency_benchmark(data: dict, n_runs: int = 200, warmup: int = 20):
    """Run full statistical latency benchmark on current hardware."""
    global MODEL, METADATA, FEATURE_COLS, TEMPLATE_DF

    hour = int(data.get("hour", 14))
    month = int(data.get("month", 10))
    temp = float(data.get("temp", 32.0))
    dwpt = float(data.get("dwpt", 24.0))
    rhum = float(data.get("rhum", 75.0))
    pres = float(data.get("pres", 1008.0))
    wspd = float(data.get("wspd", 10.0))
    prcp = float(data.get("prcp", 0.0))

    sin_h = math.sin(2.0 * math.pi * hour / 24.0)
    cos_h = math.cos(2.0 * math.pi * hour / 24.0)
    sin_m = math.sin(2.0 * math.pi * month / 12.0)
    cos_m = math.cos(2.0 * math.pi * month / 12.0)

    feat_arr = np.array([[temp, dwpt, rhum, pres, wspd, prcp, sin_h, cos_h, sin_m, cos_m]], dtype=np.float32)
    bench_df = pd.DataFrame(feat_arr, columns=FEATURE_COLS) if FEATURE_COLS else feat_arr

    # Warmup
    for _ in range(warmup):
        _ = MODEL.predict(bench_df)

    # Measurement repetitions
    durations = []
    for _ in range(n_runs):
        t0 = time.perf_counter_ns()
        _ = MODEL.predict(bench_df)
        t_end = time.perf_counter_ns()
        durations.append((t_end - t0) / 1e6)

    durations = np.array(durations)
    p50 = float(np.percentile(durations, 50))
    p95 = float(np.percentile(durations, 95))
    p99 = float(np.percentile(durations, 99))
    mean = float(np.mean(durations))
    std = float(np.std(durations))

    return {
        "n_runs": n_runs,
        "warmup": warmup,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "is_feasible": p95 <= 5.0,
        "distribution_sample": [round(float(x), 3) for x in durations[:30]]
    }


class AppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(BENCHMARK_DATA).encode("utf-8"))
            return
        elif parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            status = {
                "status": "online",
                "station": "Tan Son Nhat (WMO 48900)",
                "target": "Next-hour rainfall (prcp t+1)",
                "constraint": "P95 <= 5.0 ms",
                "model_ready": MODEL is not None
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return
        
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)

        try:
            payload = json.loads(post_body.decode("utf-8")) if post_body else {}
        except Exception:
            payload = {}

        if parsed.path == "/api/predict":
            res = predict_single(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        elif parsed.path == "/api/benchmark":
            n_runs = int(payload.get("n_runs", 200))
            res = run_latency_benchmark(payload, n_runs=n_runs)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def start_server(port: int = 8000):
    load_app_resources()
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, AppRequestHandler)
    print("=" * 65)
    print(f"[*] RAINFALL LATENCY OPTIMIZATION WEB DASHBOARD ONLINE")
    print(f"[*] Access URL: http://127.0.0.1:{port}")
    print(f"[*] Serving frontend directory: {WEB_DIR}")
    print(f"[*] Press Ctrl+C to terminate.")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    start_server(port)
