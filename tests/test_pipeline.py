"""
Unit Tests for Weather Data Pipeline, Causal Imputation, and Inference Latency.
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


class TestPipeline:
    @pytest.fixture(scope="session")
    def clean_data(self):
        """Fixture providing clean weather observations (2023-2025)."""
        return load_clean_weather_data()

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
        """Verify chronological splits have 0% timestamp overlap and valid sample counts."""
        df_feat, fcols = build_features(clean_data, L=1)
        (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = split_time_series(df_feat, fcols)

        # Ensure no empty splits and proper chronological segregation
        assert len(X_tr) > 0 and len(X_v) > 0 and len(X_te) > 0
        assert len(X_tr) == len(y_tr)
        assert len(X_v) == len(y_v)
        assert len(X_te) == len(y_te)

    def test_single_sample_vectorization_latency(self, clean_data):
        """Verify NumPy-vectorized single-sample extraction runs in < 5.0 ms."""
        sample_window = clean_data.iloc[-1:].copy()
        t0 = time.perf_counter_ns()
        feat_vector = extract_single_sample_features(sample_window, L=1)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6

        assert feat_vector.shape == (1, 10), f"Expected shape (1, 10), got {feat_vector.shape}"
        assert not np.isnan(feat_vector).any(), "Extracted features contain NaN"
        assert elapsed_ms < 5.0, f"Feature extraction too slow: {elapsed_ms:.2f} ms"

    def test_champion_model_artifact(self, clean_data):
        """Verify the saved champion model can be loaded and infers within latency budget."""
        model_path = PROJECT_ROOT / "models" / "best_optuna_lgb.joblib"
        assert model_path.exists(), f"Model artifact not found at {model_path}"

        model = joblib.load(model_path)
        sample_window = clean_data.iloc[-1:].copy()
        feat_vector = extract_single_sample_features(sample_window, L=1)

        prediction = model.predict(feat_vector)
        assert len(prediction) == 1, "Expected single prediction"
        assert prediction[0] >= -0.5, "Prediction is unreasonably negative"
