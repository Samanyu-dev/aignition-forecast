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

DATA_DIR = os.environ.get("DATA_DIR", "./data")
MODEL_PATH = os.environ.get("MODEL_PATH", "./pickle/model.pkl")

app = FastAPI(title="AIgnition Forecast API")

_model = None
_features = None


def _load_state():
    global _model, _features
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    bing = load_bing(_find_file(DATA_DIR, "bing"))
    google = load_google(_find_file(DATA_DIR, "google"))
    meta = load_meta(_find_file(DATA_DIR, "meta"))
    features = pd.concat([bing, google, meta], ignore_index=True)
    features["funnel_stage"] = features["campaign_name"].apply(infer_funnel_stage)
    features["date"] = pd.to_datetime(features["date"])
    _features = features


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
        summary = generate_causal_summary(_features, _model, baseline, scenario)
        response["narrative"] = summary["narrative"]
        response["risk_flags"] = summary["risk_flags"]
        response["narrative_source"] = summary["source"]
        response["stats"] = summary["stats"]

    return response
