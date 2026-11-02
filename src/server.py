"""
Interactive Web Application Server for Rainfall Nowcasting under Latency Constraints.
Uses Python built-in http.server to provide REST APIs and serve the frontend dashboard.

The dashboard follows the course demo order: problem -> search results (GET /api/optimization,
read from results/) -> epsilon scenarios -> deployment check of the champion (POST /api/predict,
/api/benchmark) -> multi-model benchmark (GET /api/models).
"""

import gc
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

from src.features.build_features import extract_features_from_array, RAW_WINDOW_COLS
from src.evaluation.latency import LATENCY_CONSTRAINT_MS, SAFETY_MARGIN_MS
from src.evaluation.epsilon_scenarios import compute_epsilon_scenarios, CONFIG_COLS

WEB_DIR = PROJECT_ROOT / "web"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = PROJECT_ROOT / "models" / "best_optuna_lgb.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "best_optuna_lgb.json"
BENCHMARK_PATH = RESULTS_DIR / "multi_model_benchmark.csv"
FINAL_METRICS_PATH = RESULTS_DIR / "final_test_metrics.json"
HISTORY_PATHS = {
    "RandomSearch": RESULTS_DIR / "optimization_history_random.csv",
    "OptunaTPE": RESULTS_DIR / "optimization_history_optuna.csv",
}
HISTORY_COLS = ["seed", "trial", *CONFIG_COLS, "val_rmse", "p95_latency_ms", "is_feasible"]

MAX_BENCHMARK_RUNS = 5000

# Global model state
MODEL = None
PREDICT_FN = None
METADATA = None
BENCHMARK_DATA = []
OPTIMIZATION_DATA = None


class BadRequest(ValueError):
    """Invalid client payload (mapped to HTTP 400)."""


def _records(df: pd.DataFrame):
    """JSON-safe rows: to_json maps NaN / pd.NA to null and NumPy scalars to plain numbers."""
    return json.loads(df.to_json(orient="records"))


def build_optimization_payload():
    """
    Search history, epsilon scenarios and the evaluation summary, all read from results/.
    Scenarios are recomputed with the same tested function that writes results/epsilon_scenarios.csv,
    so the dashboard and README cannot disagree. Returns None when the search logs are missing.
    """
    if not all(p.exists() for p in HISTORY_PATHS.values()):
        return None
    histories = {method: pd.read_csv(path) for method, path in HISTORY_PATHS.items()}
    summary = None
    if FINAL_METRICS_PATH.exists():
        with open(FINAL_METRICS_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
    return {
        "constraint_ms": LATENCY_CONSTRAINT_MS,
        "safety_margin_ms": SAFETY_MARGIN_MS,
        "summary": summary,
        "scenarios": _records(compute_epsilon_scenarios(histories["RandomSearch"], histories["OptunaTPE"])),
        # Column-oriented rows keep ~1,000 trials compact on the wire
        "history": {
            "columns": HISTORY_COLS,
            **{method: json.loads(df[HISTORY_COLS].to_json(orient="values")) for method, df in histories.items()},
        },
    }


def load_app_resources():
    global MODEL, PREDICT_FN, METADATA, BENCHMARK_DATA, OPTIMIZATION_DATA
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            METADATA = json.load(f)
        # The API accepts a single hourly observation, so it can only serve an L=1 model.
        L = METADATA.get("L_hours", 1)
        if L != 1:
            raise RuntimeError(f"Dashboard API serves single-observation (L=1) models only; artifact has L={L}.")
        print(f"[+] Loaded Model Metadata (L={L}h, {METADATA.get('n_features')} features)")

    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
        # Call the native booster on NumPy directly (no sklearn validation / DataFrame overhead),
        # the same path used by the latency benchmark.
        PREDICT_FN = MODEL.booster_.predict if hasattr(MODEL, "booster_") else MODEL.predict
        print(f"[+] Loaded Champion Model from {MODEL_PATH}")
    else:
        print(f"[-] Warning: Model not found at {MODEL_PATH}")

    if BENCHMARK_PATH.exists():
        try:
            df = pd.read_csv(BENCHMARK_PATH)
            BENCHMARK_DATA = df.to_dict(orient="records")
        except Exception as e:
            print(f"[-] Failed to load benchmark CSV: {e}")

    OPTIMIZATION_DATA = build_optimization_payload()
    if OPTIMIZATION_DATA is None:
        print("[-] Warning: optimization logs not found in results/ (search sections will be empty)")
    else:
        n = sum(len(OPTIMIZATION_DATA["history"][m]) for m in HISTORY_PATHS)
        print(f"[+] Loaded optimization history ({n} trials) and epsilon scenarios")

    # Warmup model once at boot
    if PREDICT_FN is not None:
        warm = build_feature_vector({})
        for _ in range(20):
            _ = PREDICT_FN(warm)
        print("[+] Model warmup completed (20 iterations)")


def _num(data: dict, key: str, default: float) -> float:
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number")
    if not math.isfinite(value):
        raise BadRequest(f"'{key}' must be finite")
    return value


def build_feature_vector(data: dict) -> np.ndarray:
    """
    Validate input (temp, dwpt, rhum, pres, wspd, prcp, hour, month) and build the model's
    feature vector through the same extractor covered by the batch/single-sample parity test.
    """
    hour = int(_num(data, "hour", 14))
    month = int(_num(data, "month", 10))
    if not 0 <= hour <= 23:
        raise BadRequest("'hour' must be in [0, 23]")
    if not 1 <= month <= 12:
        raise BadRequest("'month' must be in [1, 12]")

    defaults = {"temp": 32.0, "dwpt": 24.0, "rhum": 75.0, "pres": 1008.0, "wspd": 10.0, "prcp_clean": 0.0}
    payload_keys = {"prcp_clean": "prcp"}
    row = [_num(data, payload_keys.get(c, c), defaults[c]) for c in RAW_WINDOW_COLS]
    if row[RAW_WINDOW_COLS.index("prcp_clean")] < 0:
        raise BadRequest("'prcp' must be non-negative")

    return extract_features_from_array(np.array([row], dtype=np.float32), hour, month, L=1)


def predict_single(data: dict):
    """
    Run timed end-to-end inference on input weather observation.
    Input dictionary: temp, dwpt, rhum, pres, wspd, prcp, hour, month.
    """
    t0 = time.perf_counter_ns()
    feat_arr = build_feature_vector(data)
    t_feat = time.perf_counter_ns()

    pred_raw = float(PREDICT_FN(feat_arr)[0])
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

    meta = METADATA or {}
    hyper = meta.get("hyperparameters", {})
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
            "constraint_ms": LATENCY_CONSTRAINT_MS,
            "is_feasible": t_total_ms <= LATENCY_CONSTRAINT_MS,
            "utilization_pct": round((t_total_ms / LATENCY_CONSTRAINT_MS) * 100, 1)
        },
        "model_info": {
            "model_type": meta.get("model_type", "LightGBMRegressor"),
            "L_hours": meta.get("L_hours", 1),
            "n_estimators": hyper.get("n_estimators"),
            "max_depth": hyper.get("max_depth"),
            "learning_rate": hyper.get("learning_rate"),
        }
    }


def run_latency_benchmark(data: dict, n_runs: int = 200, warmup: int = 20):
    """Run statistical end-to-end latency benchmark (feature build + predict) on current hardware."""
    if not 1 <= n_runs <= MAX_BENCHMARK_RUNS:
        raise BadRequest(f"'n_runs' must be in [1, {MAX_BENCHMARK_RUNS}]")
    build_feature_vector(data)  # validate once before timing

    # Warmup
    for _ in range(warmup):
        _ = PREDICT_FN(build_feature_vector(data))

    # Measurement repetitions (GC paused, as in src/evaluation/latency.py)
    durations = np.empty(n_runs, dtype=np.float64)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for i in range(n_runs):
            t0 = time.perf_counter_ns()
            _ = PREDICT_FN(build_feature_vector(data))
            durations[i] = (time.perf_counter_ns() - t0) / 1e6
    finally:
        if gc_was_enabled:
            gc.enable()

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
        "is_feasible": p95 <= LATENCY_CONSTRAINT_MS,
        "distribution_sample": [round(float(x), 3) for x in durations[:30]]
    }


class AppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def _send_json(self, status: int, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/models":
            self._send_json(200, BENCHMARK_DATA)
            return
        elif parsed.path == "/api/optimization":
            if OPTIMIZATION_DATA is None:
                self._send_json(503, {"error": "Missing optimization logs. Run optimize_random.py and optimize_optuna.py first."})
            else:
                self._send_json(200, OPTIMIZATION_DATA)
            return
        elif parsed.path == "/api/status":
            self._send_json(200, {
                "status": "online",
                "station": "Tan Son Nhat (WMO 48900)",
                "target": "Next-hour rainfall (prcp t+1)",
                "constraint": f"P95 <= {LATENCY_CONSTRAINT_MS} ms",
                "model_ready": PREDICT_FN is not None
            })
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/api/predict", "/api/benchmark"):
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length)
        try:
            payload = json.loads(post_body.decode("utf-8")) if post_body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "Malformed JSON body"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "JSON body must be an object"})
            return

        if PREDICT_FN is None:
            self._send_json(503, {"error": "Model not loaded. Run 'python src/models/train.py' first."})
            return

        try:
            if parsed.path == "/api/predict":
                res = predict_single(payload)
            else:
                res = run_latency_benchmark(payload, n_runs=int(_num(payload, "n_runs", 200)))
        except BadRequest as e:
            self._send_json(400, {"error": str(e)})
            return
        self._send_json(200, res)

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
