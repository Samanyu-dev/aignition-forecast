import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from generate_features import infer_funnel_stage, _find_file, load_bing, load_google, load_meta


def test_infer_funnel_stage():
    assert infer_funnel_stage("Search_TM_Branded") == "high_intent"
    assert infer_funnel_stage("Remarketing_Retarget_Audience") == "high_intent"
    assert infer_funnel_stage("Prospecting_Generic_Keyword") == "prospecting"


def test_find_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        google_file = os.path.join(tmpdir, "google_ads_campaign_stats.csv")
        with open(google_file, "w") as f:
            f.write("header\n")
        
        found = _find_file(tmpdir, "google")
        assert found == google_file


def test_find_file_missing_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            _find_file(tmpdir, "nonexistent")


def test_load_meta_treats_conversion_as_revenue():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_file = os.path.join(tmpdir, "meta_ads_campaign_stats.csv")
        df = pd.DataFrame({
            "date_start": ["2026-01-01"],
            "campaign_id": ["123"],
            "campaign_name": ["Prospecting_Campaign_123"],
            "spend": [10.0],
            "conversion": [40.99],
            "clicks": [5],
            "impressions": [100],
            "daily_budget": [20.0],
        })
        df.to_csv(meta_file, index=False)
        
        loaded = load_meta(meta_file)
        assert loaded.iloc[0]["revenue"] == 40.99
        assert loaded.iloc[0]["channel"] == "meta"
        assert loaded.iloc[0]["campaign_type"] == "Prospecting"
