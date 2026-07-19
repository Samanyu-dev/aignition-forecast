"""
Shared forecasting core: turns a trained model dict (see src/train.py) into
probabilistic (P10/P50/P90) revenue & ROAS forecasts over 30/60/90-day
horizons, at campaign_type / channel / blended rollup levels, with optional
per-channel OR per-(channel, campaign_type)-segment budget scenario
overrides (the latter is what src/recommendations.py uses to price a
"shift $X from segment A to segment B" reallocation precisely, rather than
only being able to scale an entire channel at once).

Used by both src/predict.py (the run.sh critical path — always multiplier=1.0,
i.e. "continue current run-rate") and src/api.py (the interactive demo layer,
where a user can pass budget_multipliers to simulate a different future spend).

Monte Carlo, not closed-form: for each segment we simulate many draws of daily
revenue (trend * day-of-week seasonality + a bootstrap draw from that segment's
empirical residual pool, scaled by a channel uncertainty inflation factor), sum
across the horizon and across segments, then take percentiles of the resulting
distribution. Segments are treated as independent draws (a simplifying
assumption — see docs/TECHNICAL_DOC.md limitations).
"""
from typing import Optional

import numpy as np
import pandas as pd

HORIZONS = (30, 60, 90)
DEFAULT_N_SIMS = 2000
SEED = 42

# Nov 20 - Dec 31 each year: covers Black Friday/Cyber Monday and the December
# holiday shopping runup. Derived from an observed +5.1 sigma Black Friday 2024
# spike and a repeated Dec 2025 Q4 pattern in this dataset (see TECHNICAL_DOC.md).
HOLIDAY_START = (11, 20)
HOLIDAY_END = (12, 31)


def is_holiday(ts: pd.Timestamp) -> bool:
    md = (ts.month, ts.day)
    return HOLIDAY_START <= md <= HOLIDAY_END


def _simulate_segment_daily(seg: dict, horizon_days: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Returns an (n_sims, horizon_days) array of simulated daily revenue."""
    k = np.arange(1, horizon_days + 1)
    future_dates = pd.date_range(seg["last_train_date"] + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    if seg.get("method") == "holt_winters":
        hw_fc = seg["hw_point_forecast"]
        idx = np.clip(k - 1, 0, len(hw_fc) - 1)
        point = np.clip(hw_fc[idx], 0.0, None)
    else:
        trend = seg["trend_intercept_at_end"] + seg["trend_slope"] * k
        trend = np.clip(trend, 0.0, None)
        dow_mult = np.array([seg["dow_factor"][d] for d in future_dates.dayofweek])
        holiday_factor = seg.get("holiday_factor", 1.0)
        holiday_mult = np.array([holiday_factor if is_holiday(d) else 1.0 for d in future_dates])
        point = trend * dow_mult * holiday_mult  # shape (horizon_days,)

    residual_pool = seg["residual_pool"]
    inflation = seg.get("uncertainty_inflation", 1.0)
    draws = rng.choice(residual_pool, size=(n_sims, horizon_days), replace=True) * inflation

    daily = point[None, :] + draws
    return np.clip(daily, 0.0, None)


def _percentiles(arr: np.ndarray) -> dict:
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def forecast(
    model: dict,
    horizons=HORIZONS,
    budget_multipliers: Optional[dict] = None,
    segment_budget_multipliers: Optional[dict] = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = SEED,
) -> pd.DataFrame:
    """budget_multipliers: {channel: multiplier}, applied to every segment in
    that channel. segment_budget_multipliers: {(channel, campaign_type):
    multiplier}, overrides budget_multipliers for that specific segment only
    -- lets a scenario move spend between segments within the same channel,
    not just scale a whole channel uniformly."""
    budget_multipliers = budget_multipliers or {}
    segment_budget_multipliers = segment_budget_multipliers or {}
    rng = np.random.default_rng(seed)
    segments = model["segments"]

    rows = []
    channel_sims = {}  # (channel, horizon) -> list of (revenue_sim_array, spend_total)
    blended_sims = {}  # horizon -> list of (revenue_sim_array, spend_total)

    for (channel, campaign_type), seg in segments.items():
        mult = float(segment_budget_multipliers.get(
            (channel, campaign_type), budget_multipliers.get(channel, 1.0)
        ))
        elasticity = seg.get("elasticity_beta_for_scenario", seg["elasticity_beta"])
        scenario_scale = mult ** elasticity if mult > 0 else 0.0

        for horizon in horizons:
            daily = _simulate_segment_daily(seg, horizon, n_sims, rng)
            revenue_sim = daily.sum(axis=1) * scenario_scale
            spend_total = seg["avg_daily_spend"] * mult * horizon

            rev_pct = _percentiles(revenue_sim)
            for metric_name, pct in (("revenue", rev_pct),):
                rows.append({
                    "channel": channel, "campaign_type": campaign_type, "campaign_id": "",
                    "horizon_days": horizon, "metric": metric_name, **pct,
                })
            if spend_total > 0:
                roas_sim = revenue_sim / spend_total
                rows.append({
                    "channel": channel, "campaign_type": campaign_type, "campaign_id": "",
                    "horizon_days": horizon, "metric": "roas", **_percentiles(roas_sim),
                })

            channel_sims.setdefault((channel, horizon), []).append((revenue_sim, spend_total))
            blended_sims.setdefault(horizon, []).append((revenue_sim, spend_total))

    # Campaign-level rows are an additional, finer-grained breakdown of the same
    # underlying data already counted at the campaign_type level above -- they
    # are NOT folded into channel_sims/blended_sims (that would double-count).
    for (channel, campaign_type, campaign_id), seg in model.get("campaign_segments", {}).items():
        mult = float(segment_budget_multipliers.get(
            (channel, campaign_type), budget_multipliers.get(channel, 1.0)
        ))
        elasticity = seg.get("elasticity_beta_for_scenario", seg["elasticity_beta"])
        scenario_scale = mult ** elasticity if mult > 0 else 0.0

        for horizon in horizons:
            daily = _simulate_segment_daily(seg, horizon, n_sims, rng)
            revenue_sim = daily.sum(axis=1) * scenario_scale
            spend_total = seg["avg_daily_spend"] * mult * horizon

            rows.append({
                "channel": channel, "campaign_type": campaign_type, "campaign_id": campaign_id,
                "horizon_days": horizon, "metric": "revenue", **_percentiles(revenue_sim),
            })
            if spend_total > 0:
                roas_sim = revenue_sim / spend_total
                rows.append({
                    "channel": channel, "campaign_type": campaign_type, "campaign_id": campaign_id,
                    "horizon_days": horizon, "metric": "roas", **_percentiles(roas_sim),
                })

    for (channel, horizon), pairs in channel_sims.items():
        revenue_sim = np.sum([p[0] for p in pairs], axis=0)
        spend_total = sum(p[1] for p in pairs)
        rows.append({
            "channel": channel, "campaign_type": "", "campaign_id": "",
            "horizon_days": horizon, "metric": "revenue", **_percentiles(revenue_sim),
        })
        if spend_total > 0:
            rows.append({
                "channel": channel, "campaign_type": "", "campaign_id": "",
                "horizon_days": horizon, "metric": "roas", **_percentiles(revenue_sim / spend_total),
            })

    for horizon, pairs in blended_sims.items():
        revenue_sim = np.sum([p[0] for p in pairs], axis=0)
        spend_total = sum(p[1] for p in pairs)
        rows.append({
            "channel": "blended", "campaign_type": "", "campaign_id": "",
            "horizon_days": horizon, "metric": "revenue", **_percentiles(revenue_sim),
        })
        if spend_total > 0:
            rows.append({
                "channel": "blended", "campaign_type": "", "campaign_id": "",
                "horizon_days": horizon, "metric": "roas", **_percentiles(revenue_sim / spend_total),
            })

    df = pd.DataFrame(rows)
    return df[["channel", "campaign_type", "campaign_id", "horizon_days", "metric", "p10", "p50", "p90"]]
