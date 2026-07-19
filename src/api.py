"""
FastAPI demo layer -- NOT part of run.sh's critical path. Wraps forecasting.py
and (optionally) llm_summary.py behind a single POST /forecast endpoint used
by app_streamlit.py.

Run locally: uvicorn src.api:app --reload --port 8000
"""
import os
import pickle
import sys
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from forecasting import forecast, HORIZONS
from generate_features import _find_file, load_bing, load_google, load_meta, infer_funnel_stage
from llm_summary import generate_causal_summary
from validate_consistency import run_validation
from recommendations import generate_reallocation_candidates, compute_segment_multipliers, SHIFT_FRACTION

DATA_DIR = os.environ.get("DATA_DIR", "./data")
MODEL_PATH = os.environ.get("MODEL_PATH", "./pickle/model.pkl")

app = FastAPI(title="AIgnition Forecast API")

_model = None
_features = None
_validation_report = None


def _load_state():
    global _model, _features, _validation_report
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    bing = load_bing(_find_file(DATA_DIR, "bing"))
    google = load_google(_find_file(DATA_DIR, "google"))
    meta = load_meta(_find_file(DATA_DIR, "meta"))
    features = pd.concat([bing, google, meta], ignore_index=True)
    features["funnel_stage"] = features["campaign_name"].apply(infer_funnel_stage)
    features["date"] = pd.to_datetime(features["date"])
    _features = features

    _validation_report = run_validation(DATA_DIR)


@app.on_event("startup")
def startup():
    _load_state()


class ForecastRequest(BaseModel):
    channel: Optional[str] = None
    campaign_type: Optional[str] = None
    horizons: List[int] = list(HORIZONS)
    budget_multipliers: Dict[str, float] = {}
    include_narrative: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "segments": len(_model["segments"]) if _model else 0}


@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    baseline = forecast(_model, horizons=req.horizons)

    scenario = None
    result_df = baseline
    if req.budget_multipliers and any(v != 1.0 for v in req.budget_multipliers.values()):
        scenario = forecast(_model, horizons=req.horizons, budget_multipliers=req.budget_multipliers)
        result_df = scenario

    filtered = result_df
    if req.channel:
        filtered = filtered[filtered.channel == req.channel]
    if req.campaign_type:
        filtered = filtered[filtered.campaign_type == req.campaign_type]

    response = {"forecast": filtered.to_dict(orient="records")}

    if req.include_narrative:
        summary = generate_causal_summary(_features, _model, baseline, scenario,
                                           validation_report=_validation_report)
        response["narrative"] = summary["narrative"]
        response["risk_flags"] = summary["risk_flags"]
        response["narrative_source"] = summary["source"]
        response["stats"] = summary["stats"]

    return response


@app.get("/validation-report")
def get_validation_report():
    return _validation_report


@app.get("/top-recommendation-forecast")
def get_top_recommendation_forecast():
    """Baseline (current run-rate) vs. the top confidence-gated budget
    recommendation applied, at all 3 horizons -- for the Streamlit side-by-
    side scenario comparison view. Reuses recommendations.py's exact pricing
    formula rather than re-deriving it, so this always matches the number
    the 'Recommended budget shifts' table shows."""
    top = generate_reallocation_candidates(_model, top_n=1)
    if not top:
        return {"available": False}
    rec = top[0]

    d_channel, d_type = rec["from"].split("/", 1)
    r_channel, r_type = rec["to"].split("/", 1)
    donor = (d_channel, d_type, _model["segments"][(d_channel, d_type)])
    receiver = (r_channel, r_type, _model["segments"][(r_channel, r_type)])
    seg_mults, _ = compute_segment_multipliers(donor, receiver, SHIFT_FRACTION)

    baseline = forecast(_model, horizons=list(HORIZONS))
    recommended = forecast(_model, horizons=list(HORIZONS), segment_budget_multipliers=seg_mults)

    baseline_df = baseline[(baseline.channel == "blended")]
    recommended_df = recommended[(recommended.channel == "blended")]

    return {
        "available": True,
        "recommendation": rec,
        "baseline_forecast": baseline_df.to_dict(orient="records"),
        "recommended_forecast": recommended_df.to_dict(orient="records"),
    }
