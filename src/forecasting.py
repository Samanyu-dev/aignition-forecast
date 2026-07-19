"""
Shared forecasting core: turns a trained model dict (see src/train.py) or live
feature evaluation dataframe into probabilistic (P10/P50/P90) revenue & ROAS
forecasts over 30/60/90-day horizons, at campaign_type / channel / blended
rollup levels, with optional per-channel OR per-(channel, campaign_type)-segment
budget scenario overrides.

Used by both src/predict.py (the run.sh critical path) and src/api.py.

Upgraded with:
  - Block Bootstrap residual sampling (7-day contiguous blocks) to preserve
    temporal autocorrelation and improve interval calibration.
  - Dynamic evaluation dataset alignment: ingests features to set the forecast
    origin dynamically from features["date"].max() and dynamically fits
    unseen/sparse campaigns via hierarchical shrinkage.
  - Hill saturation curve for spend elasticity scaling to prevent exponential
    explosion under large budget multipliers.
"""
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

HORIZONS = (30, 60, 90)
DEFAULT_N_SIMS = 2000
SEED = 42
BLOCK_SIZE = 7

# Nov 20 - Dec 31 each year: covers Black Friday/Cyber Monday and the December
# holiday shopping runup.
HOLIDAY_START = (11, 20)
HOLIDAY_END = (12, 31)


def is_holiday(ts: pd.Timestamp) -> bool:
    md = (ts.month, ts.day)
    return HOLIDAY_START <= md <= HOLIDAY_END


def compute_elasticity_scale(mult: float, beta: float, gamma: float = 0.5) -> float:
    """Computes spend scaling using a Hill saturation curve.
    Prevents exponential explosion when mult > 1.0 while maintaining linear/power-law
    response for modest spend shifts."""
    if mult <= 0.0:
        return 0.0
    if mult == 1.0:
        return 1.0
    if mult < 1.0:
        return float(mult ** beta)
    
    # Saturation curve for mult > 1.0
    excess = (mult ** beta) - 1.0
    saturated_excess = excess / (1.0 + gamma * excess)
    return float(1.0 + saturated_excess)


def _draw_block_residuals(
    residual_pool: np.ndarray,
    horizon_days: int,
    n_sims: int,
    rng: np.random.Generator,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Block Bootstrap: draws contiguous blocks of residuals to preserve time-series
    autocorrelation across multi-day horizons.

    Vectorized via fancy indexing instead of a per-(sim, block) Python double loop --
    profiling showed the old loop was >90% of forecast() runtime (see docs/TECHNICAL_DOC.md).
    Draws the same `starts` values in the same order from `rng`, so output is bit-identical
    to the original loop implementation, just computed without per-element Python overhead."""
    pool = np.asarray(residual_pool, dtype=float)
    pool = pool[np.isfinite(pool)]
    if len(pool) == 0:
        return np.zeros((n_sims, horizon_days))
    if len(pool) <= block_size:
        return rng.choice(pool, size=(n_sims, horizon_days), replace=True)

    n_blocks = (horizon_days + block_size - 1) // block_size
    max_start = len(pool) - block_size
    starts = rng.integers(0, max_start + 1, size=(n_sims, n_blocks))

    offsets = np.arange(block_size)
    idx = starts[:, :, None] + offsets[None, None, :]  # (n_sims, n_blocks, block_size)
    draws = pool[idx].reshape(n_sims, n_blocks * block_size)

    return draws[:, :horizon_days]


def _simulate_segment_daily(
    seg: dict,
    horizon_days: int,
    n_sims: int,
    rng: np.random.Generator,
    last_eval_date: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """Returns an (n_sims, horizon_days) array of simulated daily revenue."""
    k = np.arange(1, horizon_days + 1)
    base_date = last_eval_date if last_eval_date is not None else seg.get("last_train_date", pd.Timestamp("2026-06-05"))
    future_dates = pd.date_range(base_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    if seg.get("method") == "holt_winters" and "hw_point_forecast" in seg:
        hw_fc = seg["hw_point_forecast"]
        idx = np.clip(k - 1, 0, len(hw_fc) - 1)
        point = np.clip(hw_fc[idx], 0.0, None)
    else:
        intercept = seg.get("trend_intercept_at_end", seg.get("avg_daily_revenue", 0.0))
        slope = seg.get("trend_slope", 0.0)
        trend = intercept + slope * k
        trend = np.clip(trend, 0.0, None)

        dow_map = seg.get("dow_factor", {d: 1.0 for d in range(7)})
        dow_mult = np.array([dow_map.get(d, 1.0) for d in future_dates.dayofweek])
        holiday_factor = seg.get("holiday_factor", 1.0)
        holiday_mult = np.array([holiday_factor if is_holiday(d) else 1.0 for d in future_dates])
        point = trend * dow_mult * holiday_mult

    residual_pool = seg.get("residual_pool", np.array([0.0]))
    inflation = seg.get("uncertainty_inflation", 1.0)
    
    # Block Bootstrap sampling instead of i.i.d.
    draws = _draw_block_residuals(residual_pool, horizon_days, n_sims, rng) * inflation

    daily = point[None, :] + draws
    return np.clip(daily, 0.0, None)


def _percentiles(arr: np.ndarray) -> dict:
    p10, p50, p90 = np.percentile(arr, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def update_model_from_features(model: dict, features: pd.DataFrame) -> Tuple[dict, pd.Timestamp]:
    """Dynamically updates model segment baselines, campaign rosters, and last_train_date
    from the freshly generated evaluation features table.
    Ensures predictions respond dynamically to whatever data is passed into run.sh."""
    updated_model = dict(model)
    segments = dict(model.get("segments", {}))
    campaign_segments = dict(model.get("campaign_segments", {}))

    if features.empty or "date" not in features.columns:
        return updated_model, pd.Timestamp("2026-06-05")

    max_date = pd.to_datetime(features["date"]).max()

    # Update segment run-rates from the latest 28 days of features
    recent_start = max_date - pd.Timedelta(days=28)
    recent = features[features["date"] > recent_start]

    for (channel, campaign_type), group in features.groupby(["channel", "campaign_type"]):
        key = (channel, campaign_type)
        rec_group = recent[(recent.channel == channel) & (recent.campaign_type == campaign_type)]
        avg_rev = float(rec_group["revenue"].mean()) if len(rec_group) > 0 else float(group["revenue"].mean() or 0.0)
        avg_spend = float(rec_group["spend"].mean()) if len(rec_group) > 0 else float(group["spend"].mean() or 0.0)

        if key in segments:
            seg = dict(segments[key])
            seg["last_train_date"] = max_date
            seg["avg_daily_revenue"] = avg_rev
            seg["avg_daily_spend"] = avg_spend
            # Adjust intercept if current run-rate differs from model intercept
            if seg.get("method") != "holt_winters":
                seg["trend_intercept_at_end"] = avg_rev
            segments[key] = seg
        else:
            # Unseen segment in features: initialize via default heuristics
            segments[key] = {
                "method": "empirical",
                "trend_slope": 0.0,
                "trend_intercept_at_end": avg_rev,
                "dow_factor": {d: 1.0 for d in range(7)},
                "holiday_factor": 1.0,
                "residual_pool": np.array([0.05 * avg_rev if avg_rev > 0 else 1.0]),
                "last_train_date": max_date,
                "avg_daily_spend": avg_spend,
                "avg_daily_revenue": avg_rev,
                "n_days_observed": len(group),
                "elasticity_beta": 1.0,
                "elasticity_beta_for_scenario": 1.0,
                "uncertainty_inflation": 1.0,
            }

    # Dynamically handle all campaigns present in features
    for (channel, campaign_type, campaign_id), group in features.groupby(["channel", "campaign_type", "campaign_id"]):
        c_key = (channel, str(campaign_type), str(campaign_id))
        rec_c = recent[(recent.channel == channel) & (recent.campaign_id.astype(str) == str(campaign_id))]
        c_rev = float(rec_c["revenue"].mean()) if len(rec_c) > 0 else float(group["revenue"].mean() or 0.0)
        c_spend = float(rec_c["spend"].mean()) if len(rec_c) > 0 else float(group["spend"].mean() or 0.0)

        parent_seg = segments.get((channel, campaign_type), {})
        parent_avg_rev = parent_seg.get("avg_daily_revenue", 1.0) or 1.0
        scale = (c_rev / parent_avg_rev) if parent_avg_rev > 0 else 1.0

        if c_key in campaign_segments:
            c_seg = dict(campaign_segments[c_key])
            c_seg["last_train_date"] = max_date
            c_seg["avg_daily_revenue"] = c_rev
            c_seg["avg_daily_spend"] = c_spend
            if c_seg.get("method") != "holt_winters":
                c_seg["trend_intercept_at_end"] = c_rev
            campaign_segments[c_key] = c_seg
        else:
            # Unseen campaign: dynamic hierarchical shrinkage from parent segment
            res_pool = parent_seg.get("residual_pool", np.array([0.0])) * scale
            campaign_segments[c_key] = {
                "method": parent_seg.get("method", "empirical"),
                "is_shrinkage": True,
                "trend_slope": parent_seg.get("trend_slope", 0.0) * scale,
                "trend_intercept_at_end": c_rev,
                "dow_factor": parent_seg.get("dow_factor", {d: 1.0 for d in range(7)}),
                "holiday_factor": parent_seg.get("holiday_factor", 1.0),
                "residual_pool": res_pool,
                "last_train_date": max_date,
                "avg_daily_spend": c_spend,
                "avg_daily_revenue": c_rev,
                "n_days_observed": len(group),
                "elasticity_beta": parent_seg.get("elasticity_beta", 1.0),
                "elasticity_beta_for_scenario": parent_seg.get("elasticity_beta_for_scenario", 1.0),
                "uncertainty_inflation": parent_seg.get("uncertainty_inflation", 1.0),
            }

    updated_model["segments"] = segments
    updated_model["campaign_segments"] = campaign_segments
    return updated_model, max_date


def forecast(
    model: dict,
    horizons=HORIZONS,
    budget_multipliers: Optional[dict] = None,
    segment_budget_multipliers: Optional[dict] = None,
    eval_features: Optional[pd.DataFrame] = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Main forecasting function. If eval_features is provided, dynamically aligns
    the forecast origin and campaign parameters to the evaluation data."""
    budget_multipliers = budget_multipliers or {}
    segment_budget_multipliers = segment_budget_multipliers or {}
    rng = np.random.default_rng(seed)

    if eval_features is not None and not eval_features.empty:
        model_to_use, last_eval_date = update_model_from_features(model, eval_features)
    else:
        model_to_use = model
        last_eval_date = None

    segments = model_to_use["segments"]
    rows = []
    channel_sims = {}  # (channel, horizon) -> list of (revenue_sim_array, spend_total)
    blended_sims = {}  # horizon -> list of (revenue_sim_array, spend_total)

    for (channel, campaign_type), seg in segments.items():
        mult = float(segment_budget_multipliers.get(
            (channel, campaign_type), budget_multipliers.get(channel, 1.0)
        ))
        elasticity = seg.get("elasticity_beta_for_scenario", seg.get("elasticity_beta", 1.0))
        scenario_scale = compute_elasticity_scale(mult, elasticity)

        for horizon in horizons:
            daily = _simulate_segment_daily(seg, horizon, n_sims, rng, last_eval_date=last_eval_date)
            revenue_sim = daily.sum(axis=1) * scenario_scale
            spend_total = seg.get("avg_daily_spend", 0.0) * mult * horizon

            rev_pct = _percentiles(revenue_sim)
            rows.append({
                "channel": channel, "campaign_type": campaign_type, "campaign_id": "",
                "horizon_days": horizon, "metric": "revenue", **rev_pct,
            })
            if spend_total > 0:
                roas_sim = revenue_sim / spend_total
                rows.append({
                    "channel": channel, "campaign_type": campaign_type, "campaign_id": "",
                    "horizon_days": horizon, "metric": "roas", **_percentiles(roas_sim),
                })

            channel_sims.setdefault((channel, horizon), []).append((revenue_sim, spend_total))
            blended_sims.setdefault(horizon, []).append((revenue_sim, spend_total))

    # Campaign-level rows
    for (channel, campaign_type, campaign_id), seg in model_to_use.get("campaign_segments", {}).items():
        mult = float(segment_budget_multipliers.get(
            (channel, campaign_type), budget_multipliers.get(channel, 1.0)
        ))
        elasticity = seg.get("elasticity_beta_for_scenario", seg.get("elasticity_beta", 1.0))
        scenario_scale = compute_elasticity_scale(mult, elasticity)

        for horizon in horizons:
            daily = _simulate_segment_daily(seg, horizon, n_sims, rng, last_eval_date=last_eval_date)
            revenue_sim = daily.sum(axis=1) * scenario_scale
            spend_total = seg.get("avg_daily_spend", 0.0) * mult * horizon

            rows.append({
                "channel": channel, "campaign_type": campaign_type, "campaign_id": str(campaign_id),
                "horizon_days": horizon, "metric": "revenue", **_percentiles(revenue_sim),
            })
            if spend_total > 0:
                roas_sim = revenue_sim / spend_total
                rows.append({
                    "channel": channel, "campaign_type": campaign_type, "campaign_id": str(campaign_id),
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

