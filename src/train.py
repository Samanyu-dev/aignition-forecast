"""
Fit and pickle the forecasting model. NOT called by run.sh — this is how
pickle/model.pkl was produced; the artifact is committed to the repo and
run.sh only loads + predicts from it.

Per (channel, campaign_type) segment, fits a lightweight trend + day-of-week
seasonality decomposition with an empirical residual pool (used for Monte
Carlo bootstrap uncertainty at forecast time — see src/forecasting.py), plus
a log-log spend->revenue elasticity used for budget-scenario simulation.

Holt-Winters/ARIMA-style models were considered but many segments here are
sparse and zero-inflated at the daily campaign_type grain, which makes MLE-based
seasonal models numerically unstable. A trend+seasonal decomposition with
empirical (bootstrapped, not Gaussian) residuals is more robust to that and
still yields genuine data-driven prediction intervals. See docs/TECHNICAL_DOC.md.
"""
import argparse
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from generate_features import _find_file, load_bing, load_google, load_meta, infer_funnel_stage

SEED = 42
TREND_WINDOW_DAYS = 120
# Reflects schema-interpretation risk on Meta's revenue field (see generate_features.py
# data validation note) — a modest inflation of Meta's residual bootstrap noise so its
# forecast intervals are appropriately wider than Bing/Google's directly-reported revenue.
CHANNEL_UNCERTAINTY_INFLATION = {"bing": 1.0, "google": 1.0, "meta": 1.3}


def build_features(data_dir: str) -> pd.DataFrame:
    bing = load_bing(_find_file(data_dir, "bing"))
    google = load_google(_find_file(data_dir, "google"))
    meta = load_meta(_find_file(data_dir, "meta"))
    features = pd.concat([bing, google, meta], ignore_index=True)
    features["funnel_stage"] = features["campaign_name"].apply(infer_funnel_stage)
    features["date"] = pd.to_datetime(features["date"])
    return features


def daily_segment_series(features: pd.DataFrame, channel: str, campaign_type: str) -> pd.DataFrame:
    seg = features[(features.channel == channel) & (features.campaign_type == campaign_type)]
    daily = seg.groupby("date").agg(revenue=("revenue", "sum"), spend=("spend", "sum")).reset_index()
    full_range = pd.date_range(daily.date.min(), daily.date.max(), freq="D")
    daily = daily.set_index("date").reindex(full_range, fill_value=0.0).rename_axis("date").reset_index()
    return daily


def fit_trend_seasonal(daily: pd.DataFrame) -> dict:
    n = len(daily)
    window = daily.tail(min(n, TREND_WINDOW_DAYS)).reset_index(drop=True)
    x = np.arange(len(window))
    y = window["revenue"].values
    if len(window) >= 2 and y.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(y.mean() if len(y) else 0.0)
    # anchor intercept to the end of the window so forecasting starts from "today"
    intercept_at_end = slope * (len(window) - 1) + intercept

    overall_mean = daily["revenue"].mean() if daily["revenue"].mean() > 0 else 1.0
    dow = daily.copy()
    dow["dow"] = dow["date"].dt.dayofweek
    dow_factor = (dow.groupby("dow")["revenue"].mean() / overall_mean).to_dict()
    dow_factor = {int(k): (float(v) if np.isfinite(v) and v > 0 else 1.0) for k, v in dow_factor.items()}
    for d in range(7):
        dow_factor.setdefault(d, 1.0)

    fitted = slope * x + intercept
    fitted_seasonal = fitted * window["date"].dt.dayofweek.map(dow_factor).values
    residuals = y - fitted_seasonal
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        residuals = np.array([0.0])

    return {
        "trend_slope": float(slope),
        "trend_intercept_at_end": float(intercept_at_end),
        "dow_factor": dow_factor,
        "residual_pool": residuals.astype(float),
        "last_train_date": daily["date"].max(),
        "avg_daily_spend": float(daily["spend"].tail(min(n, 28)).mean()),
        "avg_daily_revenue": float(daily["revenue"].tail(min(n, 28)).mean()),
        "n_days_observed": int(n),
    }


def fit_elasticity(daily: pd.DataFrame) -> dict:
    pos = daily[(daily.spend > 0) & (daily.revenue > 0)]
    if len(pos) >= 10:
        log_spend = np.log(pos["spend"].values)
        log_revenue = np.log(pos["revenue"].values)
        beta, alpha = np.polyfit(log_spend, log_revenue, 1)
        beta = float(np.clip(beta, 0.0, 2.0))  # clip to plausible diminishing/linear returns range
        n_obs = len(pos)
    else:
        beta, alpha, n_obs = 1.0, 0.0, len(pos)  # fallback: assume linear pass-through, low confidence
    return {"elasticity_beta": beta, "elasticity_alpha": float(alpha), "elasticity_n_obs": int(n_obs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model-out", default="./pickle/model.pkl")
    args = ap.parse_args()

    np.random.seed(SEED)

    print(f"Building features from {args.data_dir}", file=sys.stderr)
    features = build_features(args.data_dir)

    segments = {}
    combos = features[["channel", "campaign_type"]].drop_duplicates().itertuples(index=False)
    for channel, campaign_type in combos:
        daily = daily_segment_series(features, channel, campaign_type)
        seg_model = fit_trend_seasonal(daily)
        seg_model.update(fit_elasticity(daily))
        seg_model["channel"] = channel
        seg_model["campaign_type"] = campaign_type
        seg_model["uncertainty_inflation"] = CHANNEL_UNCERTAINTY_INFLATION.get(channel, 1.0)
        segments[(channel, campaign_type)] = seg_model
        print(
            f"  fit {channel}/{campaign_type}: {seg_model['n_days_observed']}d, "
            f"elasticity={seg_model['elasticity_beta']:.2f} (n={seg_model['elasticity_n_obs']})",
            file=sys.stderr,
        )

    model = {
        "segments": segments,
        "channel_uncertainty_inflation": CHANNEL_UNCERTAINTY_INFLATION,
        "feature_schema_version": 1,
        "trained_at": datetime.utcnow().isoformat(),
        "seed": SEED,
    }

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(model, f)
    print(f"Wrote model with {len(segments)} segments to {args.model_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
