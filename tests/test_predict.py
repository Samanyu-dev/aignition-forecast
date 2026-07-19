import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from forecasting import forecast


def test_predict_dynamic_forecast():
    # Verify that passing features to forecast produces predictions aligned with features' max date
    features = pd.DataFrame({
        "date": pd.date_range("2026-08-01", periods=10, freq="D"),
        "channel": ["google"] * 10,
        "campaign_id": ["999"] * 10,
        "campaign_name": ["Search_TM_999"] * 10,
        "campaign_type": ["SEARCH"] * 10,
        "spend": [15.0] * 10,
        "revenue": [60.0] * 10,
    })
    
    predictions = forecast(model={}, eval_features=features, horizons=[30], n_sims=50)
    assert not predictions.empty
    assert "google" in predictions["channel"].values
    assert "SEARCH" in predictions["campaign_type"].values
    assert "999" in predictions["campaign_id"].values
