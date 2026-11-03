"""
Unit test cho pipeline dữ liệu thời tiết, điền khuyết nhân quả và độ trễ suy luận.
Gồm kiểm tra rò rỉ dữ liệu nghiêm ngặt, kiểm tra khớp đặc trưng giữa batch và mẫu đơn, và xác minh độ trễ.
"""

import sys
import time
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

# Đảm bảo thư mục gốc của dự án có trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    load_clean_weather_data,
    build_features,
    split_time_series,
    extract_single_sample_features,
    extract_features_from_array,
    check_quality_gate,
    RAW_WINDOW_COLS,
    MAX_PRCP_MISSING_RATE,
)
from src.evaluation.latency import (
    benchmark_end_to_end_latency,
    extract_features_search_protocol,
    LATENCY_CONSTRAINT_MS,
    SEARCH_PROTOCOL,
)
from src.evaluation.epsilon_scenarios import compute_epsilon_scenarios
from src.models.optimize_random import sample_random_configuration, N_ESTIMATORS_CANDIDATES


@pytest.fixture(scope="session")
def clean_data():
    """Fixture cung cấp dữ liệu quan trắc thời tiết đã làm sạch (2023-2025)."""
    return load_clean_weather_data()


class TestPipeline:

    def test_data_quality_and_causal_imputation(self, clean_data):
        """Kiểm tra phạm vi dữ liệu (2023-2025) và các biến liên tục bắt buộc không còn giá trị thiếu."""
        assert len(clean_data) == 26304, f"Expected 26,304 hours (2023-2025), got {len(clean_data)}"
        assert clean_data["time"].dt.year.min() == 2023
        assert clean_data["time"].dt.year.max() == 2025

        # Kiểm tra điền khuyết nhân quả
        for col in ["temp", "dwpt", "rhum", "pres", "wspd", "prcp_clean"]:
            missing_count = clean_data[col].isna().sum()
            assert missing_count == 0, f"Column {col} still has {missing_count} NaN values"

    def test_feature_engineering_dimensions(self, clean_data):
        """Kiểm tra số chiều đặc trưng với L=1 và L=6."""
        # L=1: 6 đặc trưng trễ + 4 đặc trưng chu kỳ = 10 đặc trưng
        df_feat_1, fcols_1 = build_features(clean_data, L=1)
        assert len(fcols_1) == 10, f"Expected 10 features for L=1, got {len(fcols_1)}"
        assert not df_feat_1[fcols_1].isna().any().any(), "Feature matrix contains NaN values"

        # L=6: 6*6 đặc trưng trễ + 5 đặc trưng thống kê trượt + 4 đặc trưng chu kỳ = 45 đặc trưng
        df_feat_6, fcols_6 = build_features(clean_data, L=6)
        assert len(fcols_6) == 45, f"Expected 45 features for L=6, got {len(fcols_6)}"
        assert not df_feat_6[fcols_6].isna().any().any(), "Feature matrix contains NaN values"

    def test_chronological_splits_leakage_free(self, clean_data):
        """Kiểm tra các tập chia theo thời gian không chồng lấn mốc thời gian nào (0%) và có số mẫu hợp lệ."""
        df_feat, fcols = build_features(clean_data, L=1)
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_time_series(df_feat, fcols)

        # Đảm bảo không có tập rỗng và độ dài hợp lệ
        assert len(X_tr) > 0 and len(X_v) > 0 and len(X_te) > 0
        assert len(X_tr) == len(y_tr)
        assert len(X_v) == len(y_v)
        assert len(X_te) == len(y_te)

        # Kiểm tra tách biệt mốc thời gian nghiêm ngặt: max(Train) < min(Val) < max(Val) < min(Test)
        train_target_max = df_feat.loc[X_tr.index, "target_time"].max()
        val_target_min = df_feat.loc[X_v.index, "target_time"].min()
        val_target_max = df_feat.loc[X_v.index, "target_time"].max()
        test_target_min = df_feat.loc[X_te.index, "target_time"].min()

        assert train_target_max < val_target_min, f"Train overlap: {train_target_max} >= {val_target_min}"
        assert val_target_max < test_target_min, f"Val overlap: {val_target_max} >= {test_target_min}"

    def test_feature_parity_batch_vs_single_sample(self, clean_data):
        """
        Kiểm tra sự khớp nhau về mặt toán học giữa trích xuất đặc trưng theo batch (build_features)
        và suy luận mẫu đơn đã vector hóa (extract_single_sample_features).
        Kiểm tra then chốt, đảm bảo triển khai online khớp chính xác với huấn luyện offline.
        """
        for test_L in [1, 6, 24, 72]:
            df_feat, fcols = build_features(clean_data, L=test_L)
            sample_subset = df_feat.sample(n=10, random_state=42)

            for _, row in sample_subset.iterrows():
                target_time = row["time"]
                batch_vector = row[fcols].to_numpy(dtype=np.float32).reshape(1, -1)

                # Tìm cửa sổ kết thúc tại target_time trong clean_data
                idx = clean_data[clean_data["time"] == target_time].index[0]
                window = clean_data.iloc[idx - test_L + 1 : idx + 1]

                single_vector = extract_single_sample_features(window, L=test_L)

                # Khẳng định các đặc trưng khớp nhau trong dung sai số học của float32
                assert np.allclose(batch_vector, single_vector, atol=1e-5), (
                    f"Feature mismatch for L={test_L} at timestamp {target_time}: "
                    f"max diff = {np.max(np.abs(batch_vector - single_vector))}"
                )

        # Kiểm tra dự báo khớp nhau trên mô hình champion đã nạp (L=1)
        model_path = PROJECT_ROOT / "models" / "best_optuna_lgb.joblib"
        assert model_path.exists(), f"Champion model artifact missing at {model_path}"
        champ_model = joblib.load(model_path)
        predict_fn = champ_model.booster_.predict if hasattr(champ_model, "booster_") else champ_model.predict
        df_feat_1, fcols_1 = build_features(clean_data, L=1)
        for _, row in df_feat_1.sample(n=25, random_state=123).iterrows():
            b_vec = row[fcols_1].to_numpy(dtype=np.float32).reshape(1, -1)
            idx = clean_data[clean_data["time"] == row["time"]].index[0]
            w = clean_data.iloc[idx : idx + 1]
            s_vec = extract_single_sample_features(w, L=1)
            p_batch = float(predict_fn(b_vec)[0])
            p_single = float(predict_fn(s_vec)[0])
            assert abs(p_batch - p_single) < 1e-5, f"Prediction mismatch: batch={p_batch}, single={p_single}"

    def test_array_fast_path_parity_and_validation(self, clean_data):
        """Nhánh nhanh NumPy phải khớp đúng với bộ trích xuất DataFrame và từ chối cửa sổ sai định dạng."""
        for test_L in [1, 2, 3, 24]:
            window = clean_data.iloc[1000 - test_L : 1000]
            t = window["time"].iloc[-1]
            arr = window[RAW_WINDOW_COLS].to_numpy(dtype=np.float32)
            np.testing.assert_array_equal(
                extract_features_from_array(arr, t.hour, t.month, test_L),
                extract_single_sample_features(window, L=test_L),
            )
        with pytest.raises(ValueError):
            extract_features_from_array(np.zeros((2, len(RAW_WINDOW_COLS)), dtype=np.float32), 0, 1, L=3)

    def test_server_feature_vector_matches_pipeline(self, clean_data):
        """API của dashboard phải tạo vector đặc trưng giống pipeline huấn luyện và phải kiểm tra đầu vào."""
        from src.server import build_feature_vector, BadRequest

        row = clean_data.iloc[5000]
        payload = {c: float(row[c]) for c in ["temp", "dwpt", "rhum", "pres", "wspd"]}
        payload.update(prcp=float(row["prcp_clean"]), hour=row["time"].hour, month=row["time"].month)
        np.testing.assert_allclose(
            build_feature_vector(payload),
            extract_single_sample_features(clean_data.iloc[5000:5001], L=1),
            atol=1e-5,
        )
        for bad in ({"hour": 24}, {"month": 0}, {"temp": "abc"}, {"prcp": -1}, {"rhum": float("nan")}):
            with pytest.raises(BadRequest):
                build_feature_vector(bad)

    def test_single_sample_vectorization_latency(self, clean_data):
        """Kiểm tra (theo thống kê) việc trích xuất mẫu đơn chạy ổn định trong ngân sách ràng buộc."""
        sample_window = clean_data.iloc[-1:].copy()

        # Chạy warmup
        for _ in range(20):
            _ = extract_single_sample_features(sample_window, L=1)

        durations_ms = []
        for _ in range(50):
            t0 = time.perf_counter_ns()
            feat_vector = extract_single_sample_features(sample_window, L=1)
            durations_ms.append((time.perf_counter_ns() - t0) / 1e6)

        median_ms = np.median(durations_ms)
        p95_ms = np.percentile(durations_ms, 95)

        assert feat_vector.shape == (1, 10), f"Expected shape (1, 10), got {feat_vector.shape}"
        assert not np.isnan(feat_vector).any(), "Extracted features contain NaN"
        assert p95_ms < 5.0, f"P95 Feature extraction too slow: {p95_ms:.2f} ms"
        assert median_ms < 2.0, f"Median Feature extraction too slow: {median_ms:.2f} ms"

    def test_champion_model_artifact(self, clean_data):
        """Kiểm tra tệp mô hình champion đã lưu có thể nạp được và hoạt động nhất quán."""
        model_path = PROJECT_ROOT / "models" / "best_optuna_lgb.joblib"
        meta_path = PROJECT_ROOT / "models" / "best_optuna_lgb.json"

        assert model_path.exists(), f"Model artifact not found at {model_path}"
        assert meta_path.exists(), f"Metadata JSON not found at {meta_path}"

        model = joblib.load(model_path)
        sample_window = clean_data.iloc[-1:].copy()
        feat_vector = extract_single_sample_features(sample_window, L=1)

        predict_fn = model.booster_.predict if hasattr(model, "booster_") else model.predict
        prediction = predict_fn(feat_vector)
        assert len(prediction) == 1, "Expected single prediction"
        assert prediction[0] >= -0.05, f"Prediction is unreasonably negative: {prediction[0]}"

        # Kiểm tra dự báo trên một quan trắc thời tiết khô thực tế lấy từ bộ dữ liệu
        dry_candidates = clean_data[(clean_data["prcp_clean"] == 0.0) & (clean_data["temp"] >= 30.0) & (clean_data["rhum"] <= 55.0)]
        assert not dry_candidates.empty, "No realistic dry sample found in data"
        dry_window = dry_candidates.iloc[-1:]
        dry_vector = extract_single_sample_features(dry_window, L=1)
        dry_pred = float(predict_fn(dry_vector)[0])
        assert dry_pred < 0.5, f"Expected dry forecast (< 0.5 mm/h) for clear dry weather, got {dry_pred:.2f} mm/h"

    def test_epsilon_scenarios_select_best_feasible_per_budget(self):
        """Mỗi ngân sách độ trễ chọn trial có RMSE thấp nhất với P95 <= eps; eps = 5 ms theo cờ is_feasible."""
        history = pd.DataFrame({
            "seed": [1, 1, 1, 1], "trial": [1, 2, 3, 4],
            "L": [1, 2, 6, 24], "n_estimators": [50, 100, 200, 300], "max_depth": [2, 4, 8, 12],
            "num_leaves": [7, 15, 31, 127], "learning_rate": [0.1, 0.05, 0.03, 0.02], "min_child_samples": [10, 20, 50, 100],
            "val_rmse": [0.90, 0.85, 0.80, 0.70],
            "p95_latency_ms": [1.2, 2.5, 4.9, 5.2],
            "is_feasible": [True, True, True, False],
        })
        scen = compute_epsilon_scenarios(history, history, eps_list=[1.0, 2.0, 3.0, LATENCY_CONSTRAINT_MS])
        opt = scen[scen["method"] == "OptunaTPE"].set_index("epsilon_ms")

        assert opt["feasible_count"].tolist() == [0, 1, 2, 3], "Feasible count must grow with the budget"
        assert pd.isna(opt.loc[1.0, "val_rmse"]), "No trial meets 1 ms, so no configuration is selected"
        assert opt.loc[2.0, "trial"] == 1 and opt.loc[3.0, "trial"] == 2
        # Trial có RMSE 0.70 vi phạm ràng buộc 5 ms và tuyệt đối không được chọn
        assert opt.loc[LATENCY_CONSTRAINT_MS, "trial"] == 3
        assert opt.loc[LATENCY_CONSTRAINT_MS, "val_rmse"] == pytest.approx(0.80)

    def test_quality_gate_missing_rate(self, clean_data):
        """Cổng chất lượng: mỗi năm 2023-2025 thiếu prcp <= 5%; một năm vượt ngưỡng phải bị chặn."""
        rates = check_quality_gate(clean_data)
        assert sorted(rates.index) == [2023, 2024, 2025]
        assert (rates <= MAX_PRCP_MISSING_RATE).all(), f"Missing rate above threshold: {rates.to_dict()}"

        # Xóa toàn bộ lượng mưa năm 2024 (thiếu 100%) thì cổng phải báo lỗi
        broken = clean_data[["time", "prcp"]].copy()
        broken.loc[broken["time"].dt.year == 2024, "prcp"] = np.nan
        with pytest.raises(ValueError, match="Quality gate failed"):
            check_quality_gate(broken)

    def test_search_space_matches_logged_trials(self):
        """Không gian tìm kiếm trong code phải là không gian đã sinh ra log results/optimization_history_*.csv."""
        cols = ["L", "n_estimators", "max_depth", "num_leaves", "learning_rate", "min_child_samples"]
        rs = pd.read_csv(PROJECT_ROOT / "results" / "optimization_history_random.csv")
        # Sampler Random Search là tất định theo (seed, trial): phải tái tạo đúng từng cấu hình đã ghi
        for row in rs.itertuples():
            logged = {c: getattr(row, c) for c in cols}
            assert sample_random_configuration(row.seed * 1000 + row.trial) == logged, f"seed={row.seed} trial={row.trial}"

        opt = pd.read_csv(PROJECT_ROOT / "results" / "optimization_history_optuna.csv")
        for df in (rs, opt):
            assert sorted(df["n_estimators"].unique()) == N_ESTIMATORS_CANDIDATES

    def test_search_protocol_latency(self, clean_data):
        """Giao thức pha tìm kiếm cho cùng vector đặc trưng như đường triển khai và báo đúng giao thức đã dùng."""
        for test_L in [1, 3, 24]:
            window = clean_data.iloc[1000 - test_L : 1000]
            np.testing.assert_array_equal(
                extract_features_search_protocol(window, test_L), extract_single_sample_features(window, L=test_L)
            )
        model = joblib.load(PROJECT_ROOT / "models" / "best_optuna_lgb.joblib")
        res = benchmark_end_to_end_latency(model, clean_data.iloc[-1:], L=1, n_runs=20, warmup=5, protocol=SEARCH_PROTOCOL)
        assert res["protocol"] == SEARCH_PROTOCOL and res["p95_latency_ms"] > 0
        with pytest.raises(ValueError):
            benchmark_end_to_end_latency(model, clean_data.iloc[-1:], L=1, protocol="sklearn")
