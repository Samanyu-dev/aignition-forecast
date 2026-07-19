import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from forecasting import (
    forecast,
    _draw_block_residuals,
    compute_elasticity_scale,
    update_model_from_features,
    HORIZONS,
    SEED,
)


def test_hill_saturation_curve():
    # mult <= 0 returns 0
    assert compute_elasticity_scale(0.0, 1.0) == 0.0
    # mult == 1.0 returns 1.0
    assert compute_elasticity_scale(1.0, 0.8) == 1.0
    # mult < 1.0 returns mult ** beta
    assert pytest.approx(compute_elasticity_scale(0.5, 0.8)) == 0.5 ** 0.8
    # mult > 1.0 saturates gracefully (less than pure exponential mult ** beta)
    raw_power = 2.0 ** 1.5  # ~2.828
    saturated = compute_elasticity_scale(2.0, 1.5, gamma=0.5)
    assert 1.0 < saturated < raw_power


def test_draw_block_residuals():
    rng = np.random.default_rng(SEED)
    pool = np.arange(100, dtype=float)
    draws = _draw_block_residuals(pool, horizon_days=30, n_sims=10, rng=rng, block_size=7)
    
    assert draws.shape == (10, 30)
    # Check block structure: adjacent elements within a block of 7 should be contiguous in the pool
    for i in range(10):
        diffs = np.diff(draws[i, :7])
        assert np.all(diffs == 1.0) or len(pool) < 7


def test_update_model_from_features():
    dummy_features = pd.DataFrame({
        "date": pd.date_range("2026-07-01", periods=10, freq="D"),
        "channel": ["google"] * 10,
        "campaign_id": ["101"] * 10,
        "campaign_name": ["Search_TM_101"] * 10,
        "campaign_type": ["SEARCH"] * 10,
        "spend": [10.0] * 10,
        "revenue": [50.0] * 10,
    })
    
    model = {"segments": {}, "campaign_segments": {}}
    updated_model, max_date = update_model_from_features(model, dummy_features)
    
    assert max_date == pd.Timestamp("2026-07-10")
    assert ("google", "SEARCH") in updated_model["segments"]
    assert ("google", "SEARCH", "101") in updated_model["campaign_segments"]
    assert updated_model["segments"][("google", "SEARCH")]["avg_daily_revenue"] == 50.0


def test_forecast_output_schema():
    dummy_features = pd.DataFrame({
        "date": pd.date_range("2026-06-01", periods=15, freq="D"),
        "channel": ["bing"] * 15,
        "campaign_id": ["bing_1"] * 15,
        "campaign_name": ["Search_TM_bing"] * 15,
        "campaign_type": ["Search"] * 15,
        "spend": [20.0] * 15,
        "revenue": [80.0] * 15,
    })
    
    df = forecast(model={}, eval_features=dummy_features, horizons=[30, 60], n_sims=100, seed=SEED)
    
    assert not df.empty
    expected_cols = ["channel", "campaign_type", "campaign_id", "horizon_days", "metric", "p10", "p50", "p90"]
    assert list(df.columns) == expected_cols
    
    # Verify P10 <= P50 <= P90
    for _, row in df.iterrows():
        assert row["p10"] <= row["p50"] <= row["p90"]
