"""
Unit Tests for Weather Data Pipeline, Causal Imputation, and Inference Latency.
Includes strict leakage checks, batch vs single-sample feature parity, and latency verification.
"""

import sys
import time
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import (
    load_clean_weather_data,
    build_features,
    split_time_series,
    extract_single_sample_features,
)
from src.evaluation.latency import benchmark_end_to_end_latency, LATENCY_CONSTRAINT_MS


@pytest.fixture(scope="session")
def clean_data():
    """Fixture providing clean weather observations (2023-2025)."""
    return load_clean_weather_data()


class TestPipeline:

    def test_data_quality_and_causal_imputation(self, clean_data):
        """Verify data span (2023-2025) and ensure zero missing values in required continuous variables."""
        assert len(clean_data) == 26304, f"Expected 26,304 hours (2023-2025), got {len(clean_data)}"
        assert clean_data["time"].dt.year.min() == 2023
        assert clean_data["time"].dt.year.max() == 2025

        # Causal imputation check
        for col in ["temp", "dwpt", "rhum", "pres", "wspd", "prcp_clean"]:
            missing_count = clean_data[col].isna().sum()
            assert missing_count == 0, f"Column {col} still has {missing_count} NaN values"

    def test_feature_engineering_dimensions(self, clean_data):
        """Verify feature dimensions for L=1 and L=6."""
        # L=1: 6 lag features + 4 cyclical features = 10 features
        df_feat_1, fcols_1 = build_features(clean_data, L=1)
        assert len(fcols_1) == 10, f"Expected 10 features for L=1, got {len(fcols_1)}"
        assert not df_feat_1[fcols_1].isna().any().any(), "Feature matrix contains NaN values"

        # L=6: 6*6 lag features + 5 rolling features + 4 cyclical features = 45 features
        df_feat_6, fcols_6 = build_features(clean_data, L=6)
        assert len(fcols_6) == 45, f"Expected 45 features for L=6, got {len(fcols_6)}"
        assert not df_feat_6[fcols_6].isna().any().any(), "Feature matrix contains NaN values"

    def test_chronological_splits_leakage_free(self, clean_data):
        """Verify chronological splits have strict 0% timestamp overlap and valid sample counts."""
        df_feat, fcols = build_features(clean_data, L=1)
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_time_series(df_feat, fcols)

        # Ensure no empty splits and valid lengths
        assert len(X_tr) > 0 and len(X_v) > 0 and len(X_te) > 0
        assert len(X_tr) == len(y_tr)
        assert len(X_v) == len(y_v)
        assert len(X_te) == len(y_te)

        # Strict timestamp segregation check: max(Train) < min(Val) < max(Val) < min(Test)
        train_target_max = df_feat.loc[X_tr.index, "target_time"].max()
        val_target_min = df_feat.loc[X_v.index, "target_time"].min()
        val_target_max = df_feat.loc[X_v.index, "target_time"].max()
        test_target_min = df_feat.loc[X_te.index, "target_time"].min()

        assert train_target_max < val_target_min, f"Train overlap: {train_target_max} >= {val_target_min}"
        assert val_target_max < test_target_min, f"Val overlap: {val_target_max} >= {test_target_min}"

    def test_feature_parity_batch_vs_single_sample(self, clean_data):
        """
        Verify mathematical parity between batch feature extraction (build_features)
        and vectorized single-sample inference (extract_single_sample_features).
        Critical check ensuring online deployment matches offline training exactly.
        """
        for test_L in [1, 6, 24, 72]:
            df_feat, fcols = build_features(clean_data, L=test_L)
            sample_subset = df_feat.sample(n=10, random_state=42)

            for _, row in sample_subset.iterrows():
                target_time = row["time"]
                batch_vector = row[fcols].to_numpy(dtype=np.float32).reshape(1, -1)

                # Locate window ending at target_time in clean_data
                idx = clean_data[clean_data["time"] == target_time].index[0]
                window = clean_data.iloc[idx - test_L + 1 : idx + 1]

                single_vector = extract_single_sample_features(window, L=test_L)

                # Assert feature parity within float32 numerical tolerance
                assert np.allclose(batch_vector, single_vector, atol=1e-5), (
                    f"Feature mismatch for L={test_L} at timestamp {target_time}: "
                    f"max diff = {np.max(np.abs(batch_vector - single_vector))}"
                )

        # Test model prediction parity on the loaded Champion model (L=1)
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

    def test_single_sample_vectorization_latency(self, clean_data):
        """Verify statistical single-sample extraction runs reliably within constraint budget."""
        sample_window = clean_data.iloc[-1:].copy()

        # Warmup
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
        """Verify the saved champion model artifact can be loaded and behaves consistently."""
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

        # Check prediction on realistic dry weather observation from dataset
        dry_candidates = clean_data[(clean_data["prcp_clean"] == 0.0) & (clean_data["temp"] >= 30.0) & (clean_data["rhum"] <= 55.0)]
        assert not dry_candidates.empty, "No realistic dry sample found in data"
        dry_window = dry_candidates.iloc[-1:]
        dry_vector = extract_single_sample_features(dry_window, L=1)
        dry_pred = float(predict_fn(dry_vector)[0])
        assert dry_pred < 0.5, f"Expected dry forecast (< 0.5 mm/h) for clear dry weather, got {dry_pred:.2f} mm/h"
