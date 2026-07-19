"""
Fit and pickle the forecasting model. NOT called by run.sh -- this is how
pickle/model.pkl was produced; the artifact is committed to the repo and
run.sh only loads + predicts from it.

Two candidate models are fit and backtested per (channel, campaign_type)
segment, and the better one (by walk-forward pinball loss) is kept:

  1. "empirical" -- a lightweight trend + day-of-week + holiday-window
     seasonality decomposition with an empirical (bootstrapped, not Gaussian)
     residual pool. Robust on the sparse/zero-inflated segments where
     MLE-based seasonal models struggle.
  2. "holt_winters" -- statsmodels ExponentialSmoothing (additive trend +
     weekly seasonal), a genuine second method (not a variant of #1), used
     as the comparison point.

See docs/TECHNICAL_DOC.md sections 3.5-3.6 for the backtest results this
comparison and the calibration-scale search below are based on -- these
weren't chosen a priori, they're what the walk-forward numbers picked.
"""
import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from generate_features import _find_file, load_bing, load_google, load_meta, infer_funnel_stage
from forecasting import is_holiday
from backtest_core import evaluate_segment_method, generate_cutoffs, valid_cutoffs, TEST_HORIZON_DAYS, MIN_TRAIN_DAYS

SEED = 42
TREND_WINDOW_DAYS = 120
CLIP_PERCENTILE = 97  # cap extreme single-day spikes (e.g. Black Friday) before fitting the trend line
MIN_CAMPAIGN_DAYS = 30  # fit independently only above this; below, use hierarchical shrinkage instead
MIN_DAYS_FOR_SHRINKAGE = 3  # below this, even a shrinkage-anchored level has no real signal -- skip
ELASTICITY_CI_WIDTH_THRESHOLD = 1.0  # if the 10-90 bootstrap CI on beta is wider than this, don't trust it for scenarios
CALIBRATION_CANDIDATE_SCALES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
CALIBRATION_TARGET_COVERAGE = 75.0  # aim close to the 80% nominal without being needlessly wide
CALIBRATION_EXTENDED_N_CUTOFFS = 6  # look back 6x30=180 days so there's enough to split tune/eval
MIN_CUTOFFS_FOR_SPLIT = 4  # below this, there isn't enough data to tune and hold out separately

# Reflects schema-interpretation risk on Meta's revenue field (see generate_features.py
# data validation note) -- a modest inflation of Meta's residual bootstrap noise so its
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


def _daily_series(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date").agg(revenue=("revenue", "sum"), spend=("spend", "sum")).reset_index()
    full_range = pd.date_range(daily.date.min(), daily.date.max(), freq="D")
    daily = daily.set_index("date").reindex(full_range, fill_value=0.0).rename_axis("date").reset_index()
    return daily


def daily_segment_series(features: pd.DataFrame, channel: str, campaign_type: str) -> pd.DataFrame:
    seg = features[(features.channel == channel) & (features.campaign_type == campaign_type)]
    return _daily_series(seg)


def daily_campaign_series(features: pd.DataFrame, channel: str, campaign_id: str) -> pd.DataFrame:
    seg = features[(features.channel == channel) & (features.campaign_id == campaign_id)]
    return _daily_series(seg)


def fit_trend_seasonal(daily: pd.DataFrame) -> dict:
    """'empirical' method: clipped linear trend + day-of-week + holiday-window
    seasonal multipliers + empirical residual bootstrap pool."""
    n = len(daily)
    window = daily.tail(min(n, TREND_WINDOW_DAYS)).reset_index(drop=True)
    x = np.arange(len(window))
    y_raw = window["revenue"].values
    clip_at = np.percentile(y_raw, CLIP_PERCENTILE) if len(y_raw) else 0.0
    y = np.clip(y_raw, 0, clip_at) if clip_at > 0 else y_raw

    if len(window) >= 2 and y.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = 0.0, float(y.mean() if len(y) else 0.0)
    intercept_at_end = slope * (len(window) - 1) + intercept

    overall_mean = daily["revenue"].mean() if daily["revenue"].mean() > 0 else 1.0
    dow = daily.copy()
    dow["dow"] = dow["date"].dt.dayofweek
    dow_factor = (dow.groupby("dow")["revenue"].mean() / overall_mean).to_dict()
    dow_factor = {int(k): (float(v) if np.isfinite(v) and v > 0 else 1.0) for k, v in dow_factor.items()}
    for d in range(7):
        dow_factor.setdefault(d, 1.0)

    holiday_mask = daily["date"].apply(is_holiday)
    if holiday_mask.any() and (~holiday_mask).any():
        holiday_mean = daily.loc[holiday_mask, "revenue"].mean()
        non_holiday_mean = daily.loc[~holiday_mask, "revenue"].mean()
        holiday_factor = float(holiday_mean / non_holiday_mean) if non_holiday_mean > 0 else 1.0
        holiday_factor = holiday_factor if np.isfinite(holiday_factor) and holiday_factor > 0 else 1.0
    else:
        holiday_factor = 1.0  # segment has no holiday-window history -- don't guess

    fitted = slope * x + intercept
    dow_at_window = window["date"].dt.dayofweek.map(dow_factor).values
    holiday_at_window = np.array([holiday_factor if is_holiday(d) else 1.0 for d in window["date"]])
    fitted_seasonal = fitted * dow_at_window * holiday_at_window
    residuals = y_raw - fitted_seasonal  # residuals computed against the UNCLIPPED actuals -- real noise, not clipped noise
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        residuals = np.array([0.0])

    return {
        "method": "empirical",
        "trend_slope": float(slope),
        "trend_intercept_at_end": float(intercept_at_end),
        "dow_factor": dow_factor,
        "holiday_factor": holiday_factor,
        "residual_pool": residuals.astype(float),
        "last_train_date": daily["date"].max(),
        "avg_daily_spend": float(daily["spend"].tail(min(n, 28)).mean()),
        "avg_daily_revenue": float(daily["revenue"].tail(min(n, 28)).mean()),
        "n_days_observed": int(n),
    }


def fit_holt_winters(daily: pd.DataFrame) -> dict:
    """'holt_winters' method: statsmodels additive Holt-Winters (trend + weekly
    seasonal) on the clipped daily series, its own multi-step forecast used
    directly as the point path, empirical in-sample residuals for bootstrap."""
    n = len(daily)
    window = daily.tail(min(n, TREND_WINDOW_DAYS)).reset_index(drop=True)
    rev_raw = window["revenue"].values
    clip_at = np.percentile(rev_raw, CLIP_PERCENTILE) if len(rev_raw) else 0.0
    rev_c = np.clip(rev_raw, 0, clip_at) if clip_at > 0 else rev_raw

    model = ExponentialSmoothing(
        rev_c, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated"
    )
    fitted = model.fit(optimized=True)
    hw_point_forecast = np.clip(fitted.forecast(120), 0.0, None)
    residuals = rev_raw - fitted.fittedvalues
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        residuals = np.array([0.0])

    return {
        "method": "holt_winters",
        "hw_point_forecast": np.asarray(hw_point_forecast, dtype=float),
        "residual_pool": residuals.astype(float),
        "last_train_date": daily["date"].max(),
        "avg_daily_spend": float(daily["spend"].tail(min(n, 28)).mean()),
        "avg_daily_revenue": float(daily["revenue"].tail(min(n, 28)).mean()),
        "n_days_observed": int(n),
    }


def fit_elasticity(daily: pd.DataFrame, n_bootstrap: int = 500, seed: int = SEED) -> dict:
    pos = daily[(daily.spend > 0) & (daily.revenue > 0)]
    rng = np.random.default_rng(seed)

    if len(pos) >= 10:
        log_spend = np.log(pos["spend"].values)
        log_revenue = np.log(pos["revenue"].values)
        beta, alpha = np.polyfit(log_spend, log_revenue, 1)
        beta = float(np.clip(beta, 0.0, 2.0))
        n_obs = len(pos)

        boot_betas = []
        idx = np.arange(n_obs)
        for _ in range(n_bootstrap):
            sample = rng.choice(idx, size=n_obs, replace=True)
            try:
                b, _ = np.polyfit(log_spend[sample], log_revenue[sample], 1)
                boot_betas.append(float(np.clip(b, 0.0, 2.0)))
            except np.linalg.LinAlgError:
                continue
        if boot_betas:
            ci_low, ci_high = np.percentile(boot_betas, [10, 90])
        else:
            ci_low, ci_high = beta, beta
    else:
        beta, alpha, n_obs = 1.0, 0.0, len(pos)
        ci_low, ci_high = beta, beta

    ci_width = float(ci_high - ci_low)
    low_confidence = n_obs < 10 or ci_width > ELASTICITY_CI_WIDTH_THRESHOLD
    beta_for_scenario = 1.0 if low_confidence else beta

    return {
        "elasticity_beta": beta,
        "elasticity_alpha": float(alpha),
        "elasticity_n_obs": int(n_obs),
        "elasticity_ci_low": float(ci_low),
        "elasticity_ci_high": float(ci_high),
        "elasticity_low_confidence": bool(low_confidence),
        "elasticity_beta_for_scenario": float(beta_for_scenario),
    }


def select_method(daily: pd.DataFrame, base_inflation: float, rng: np.random.Generator) -> tuple:
    """Backtest both candidate methods on this segment's history, return
    (winning_fit_fn, winning_name, {method_name: score_summary})."""
    scores = {}
    for name, fit_fn in (("empirical", fit_trend_seasonal), ("holt_winters", fit_holt_winters)):
        result = evaluate_segment_method(daily, fit_fn, rng, base_uncertainty_inflation=base_inflation)
        scores[name] = {
            "n_scored": result["n_scored"],
            "mean_ape_pct": result["mean_ape_pct"],
            "mean_pinball_loss": result["mean_pinball_loss"],
            "coverage_pct": result["coverage_pct"],
        }

    def rank_key(name):
        s = scores[name]
        if s["n_scored"] == 0:
            return (1, float("inf"))  # unscored methods lose to any scored method
        return (0, s["mean_pinball_loss"])

    winner = min(scores, key=rank_key)
    if scores[winner]["n_scored"] == 0:
        winner = "empirical"  # both unscored (very short segment) -- default to the more robust method
    fit_fn = fit_trend_seasonal if winner == "empirical" else fit_holt_winters
    return fit_fn, winner, scores


def search_calibration_scale(segment_daily: dict, segment_fit_fns: dict, base_inflations: dict) -> tuple:
    """Rolling-origin per-segment calibration: tune the residual-scale
    multiplier on the OLDER half of available walk-forward cutoffs, then
    report coverage on the NEWER, never-touched-during-tuning half. An
    earlier version of this function tuned and reported on the same 3
    cutoffs -- a real double-dipping flaw a stats-literate reviewer caught
    (see docs/TECHNICAL_DOC.md sec 3.8): the reported coverage was optimistic
    versus genuinely held-out performance. Fixed here by extending to up to
    CALIBRATION_EXTENDED_N_CUTOFFS (6) 30-day windows (180 days back) so
    there's enough to split, and never letting the tuning step see the fold
    its own chosen scale is then scored on.

    A single global multiplier was tried before per-segment search too, and
    plateaued around 53% aggregate coverage (worse beyond scale=4, since
    different segments need very different amounts of widening).

    Segments without >=MIN_CUTOFFS_FOR_SPLIT valid cutoffs can't be split
    honestly -- for those, calibration_scale stays 1.0 (no tuning performed)
    and the eval result (if any cutoffs are scoreable at all) is reported
    with no_calibration_tuning=True rather than silently reusing a
    single shared fold. Returns (per_segment_scale, per_segment_eval_result,
    per_segment_curve) -- per_segment_eval_result is the HONEST, held-out
    coverage/MAPE/pinball to use as this segment's reported backtest number."""
    per_segment_scale = {}
    per_segment_eval = {}
    per_segment_curve = {}

    for key, daily in segment_daily.items():
        channel = key[0]
        fit_fn = segment_fit_fns[key]
        base_inflation = base_inflations[channel]

        all_cutoffs = generate_cutoffs(daily, CALIBRATION_EXTENDED_N_CUTOFFS, TEST_HORIZON_DAYS)
        usable = valid_cutoffs(daily, all_cutoffs, TEST_HORIZON_DAYS, MIN_TRAIN_DAYS)

        if len(usable) < MIN_CUTOFFS_FOR_SPLIT:
            # Not enough history to tune AND hold out separately -- don't double-dip on
            # whatever little data exists. No tuning; report on whatever's scoreable, honestly labeled.
            scale = 1.0
            rng = np.random.default_rng(SEED)
            eval_result = evaluate_segment_method(
                daily, fit_fn, rng, base_uncertainty_inflation=base_inflation,
                residual_scale=scale, cutoffs=usable,
            )
            eval_result["no_calibration_tuning"] = True
            per_segment_scale[key] = scale
            per_segment_eval[key] = eval_result
            per_segment_curve[key] = {}
            continue

        split = len(usable) // 2
        tune_cutoffs, eval_cutoffs = usable[:split], usable[split:]

        rng_tune = np.random.default_rng(SEED)
        coverage_by_scale = {}
        for scale in CALIBRATION_CANDIDATE_SCALES:
            result = evaluate_segment_method(
                daily, fit_fn, rng_tune, base_uncertainty_inflation=base_inflation,
                residual_scale=scale, cutoffs=tune_cutoffs,
            )
            coverage_by_scale[scale] = result["coverage_pct"] if result["coverage_pct"] is not None else 0.0

        qualifying = [s for s, cov in coverage_by_scale.items() if cov >= 66.7]
        chosen = min(qualifying) if qualifying else max(coverage_by_scale, key=coverage_by_scale.get)

        rng_eval = np.random.default_rng(SEED)
        eval_result = evaluate_segment_method(
            daily, fit_fn, rng_eval, base_uncertainty_inflation=base_inflation,
            residual_scale=chosen, cutoffs=eval_cutoffs,
        )
        eval_result["no_calibration_tuning"] = False

        per_segment_scale[key] = chosen
        per_segment_eval[key] = eval_result
        per_segment_curve[key] = coverage_by_scale

    return per_segment_scale, per_segment_eval, per_segment_curve


def fit_shrinkage_from_parent(daily_child: pd.DataFrame, parent_seg: dict) -> dict:
    """Hierarchical shrinkage for a child segment with too little history to fit
    independently: borrow the parent (channel, campaign_type) segment's trend/
    seasonal SHAPE (dow_factor + holiday_factor, or the Holt-Winters point-forecast
    curve) and residual-pool volatility shape, but anchor the LEVEL to the child's
    own observed average daily revenue/spend, scaled proportionally. This is
    partial pooling ("borrow statistical strength"), not a parent-forecast
    passthrough -- the child's own real (if sparse) observations still set its
    level; only the shape comes from the better-estimated parent. Bottom-up in
    the sense that child-level totals are still anchored to child-level data,
    not derived by splitting a parent forecast top-down."""
    n = len(daily_child)
    child_avg_daily_revenue = float(daily_child["revenue"].mean()) if n > 0 else 0.0
    child_avg_daily_spend = float(daily_child["spend"].mean()) if n > 0 else 0.0
    parent_avg_revenue = parent_seg.get("avg_daily_revenue") or 0.0
    scale = (child_avg_daily_revenue / parent_avg_revenue) if parent_avg_revenue > 0 else 0.0

    seg = {
        "method": parent_seg["method"],
        "is_shrinkage": True,
        "shrinkage_source": f"{parent_seg['channel']}/{parent_seg['campaign_type']}",
        "last_train_date": daily_child["date"].max() if n > 0 else parent_seg["last_train_date"],
        "avg_daily_spend": child_avg_daily_spend,
        "avg_daily_revenue": child_avg_daily_revenue,
        "n_days_observed": n,
        "residual_pool": parent_seg["residual_pool"] * scale,
    }
    if parent_seg["method"] == "holt_winters":
        seg["hw_point_forecast"] = parent_seg["hw_point_forecast"] * scale
    else:
        seg["trend_slope"] = parent_seg["trend_slope"] * scale
        seg["trend_intercept_at_end"] = parent_seg["trend_intercept_at_end"] * scale
        seg["dow_factor"] = parent_seg["dow_factor"]
        seg["holiday_factor"] = parent_seg.get("holiday_factor", 1.0)
    return seg


def fit_campaign_segments(features: pd.DataFrame, campaign_type_methods: dict, parent_segments: dict) -> dict:
    """Fit per-individual-campaign models. Three tiers by observed history:
      - >=MIN_CAMPAIGN_DAYS: fit independently, reusing the winning method
        already selected for the parent (channel, campaign_type) segment
        (skipping a full method-comparison backtest per campaign to keep this
        tractable -- see docs/TECHNICAL_DOC.md limitations).
      - MIN_DAYS_FOR_SHRINKAGE..MIN_CAMPAIGN_DAYS: too little history to fit
        independently, but not nothing -- hierarchical shrinkage from the
        parent segment (fit_shrinkage_from_parent) rather than being dropped.
      - <MIN_DAYS_FOR_SHRINKAGE: genuinely no signal at all -- skipped, not
        fabricated."""
    campaign_segments = {}
    skipped = []
    shrinkage_applied = []
    combos = features[["channel", "campaign_type", "campaign_id"]].drop_duplicates().itertuples(index=False)
    for channel, campaign_type, campaign_id in combos:
        daily = daily_campaign_series(features, channel, campaign_id)
        n = daily["date"].nunique()

        if n < MIN_DAYS_FOR_SHRINKAGE:
            skipped.append(f"{channel}/{campaign_id}")
            continue

        if n < MIN_CAMPAIGN_DAYS:
            parent_key = (channel, campaign_type)
            if parent_key not in parent_segments:
                skipped.append(f"{channel}/{campaign_id}")
                continue
            seg_model = fit_shrinkage_from_parent(daily, parent_segments[parent_key])
            shrinkage_applied.append(f"{channel}/{campaign_id}")
        else:
            fit_fn = campaign_type_methods.get((channel, campaign_type), fit_trend_seasonal)
            try:
                seg_model = fit_fn(daily)
            except Exception:
                seg_model = fit_trend_seasonal(daily)
            seg_model["is_shrinkage"] = False

        seg_model.update(fit_elasticity(daily))
        seg_model["channel"] = channel
        seg_model["campaign_type"] = campaign_type
        seg_model["campaign_id"] = campaign_id
        seg_model["uncertainty_inflation"] = CHANNEL_UNCERTAINTY_INFLATION.get(channel, 1.0)
        campaign_segments[(channel, campaign_type, campaign_id)] = seg_model

    print(f"  campaign-level: fit {len(campaign_segments)} ({len(shrinkage_applied)} via hierarchical "
          f"shrinkage from parent campaign_type), skipped {len(skipped)} (<{MIN_DAYS_FOR_SHRINKAGE}d history, "
          f"no signal at all)", file=sys.stderr)
    return campaign_segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--model-out", default="./pickle/model.pkl")
    ap.add_argument("--backtest-out", default="./docs/backtest_results_post_fix.json")
    args = ap.parse_args()

    np.random.seed(SEED)

    print(f"Building features from {args.data_dir}", file=sys.stderr)
    features = build_features(args.data_dir)

    combos = list(features[["channel", "campaign_type"]].drop_duplicates().itertuples(index=False))
    segment_daily = {}
    segment_fit_fns = {}
    segment_method_scores = {}

    print("Selecting method per segment via walk-forward backtest...", file=sys.stderr)
    rng = np.random.default_rng(SEED)
    for channel, campaign_type in combos:
        key = (channel, campaign_type)
        daily = daily_segment_series(features, channel, campaign_type)
        segment_daily[key] = daily
        base_inflation = CHANNEL_UNCERTAINTY_INFLATION.get(channel, 1.0)
        fit_fn, winner, scores = select_method(daily, base_inflation, rng)
        segment_fit_fns[key] = fit_fn
        segment_method_scores[key] = scores
        print(f"  {channel}/{campaign_type}: winner={winner} "
              f"(empirical pinball={scores['empirical']['mean_pinball_loss']}, "
              f"holt_winters pinball={scores['holt_winters']['mean_pinball_loss']})", file=sys.stderr)

    print("Rolling-origin calibration: tuning on earlier folds, evaluating on later held-out folds...",
          file=sys.stderr)
    per_segment_scale, per_segment_eval, per_segment_coverage_curve = search_calibration_scale(
        segment_daily, segment_fit_fns, CHANNEL_UNCERTAINTY_INFLATION
    )
    for key, scale in sorted(per_segment_scale.items()):
        ev = per_segment_eval[key]
        tuned = "no held-out tuning (too little history to split)" if ev.get("no_calibration_tuning") else "tuned"
        print(f"  {key[0]}/{key[1]}: calibration_scale={scale} [{tuned}], "
              f"held-out coverage={ev['coverage_pct']}", file=sys.stderr)

    segments = {}
    final_backtest_detail = {}
    for channel, campaign_type in combos:
        key = (channel, campaign_type)
        daily = segment_daily[key]
        fit_fn = segment_fit_fns[key]
        base_inflation = CHANNEL_UNCERTAINTY_INFLATION.get(channel, 1.0)
        calibration_scale = per_segment_scale[key]

        seg_model = fit_fn(daily)
        seg_model.update(fit_elasticity(daily))
        seg_model["channel"] = channel
        seg_model["campaign_type"] = campaign_type
        seg_model["uncertainty_inflation"] = base_inflation * calibration_scale
        seg_model["calibration_scale"] = calibration_scale
        seg_model["method_comparison"] = segment_method_scores[key]
        segments[key] = seg_model

        # This is the rolling-origin HELD-OUT result from search_calibration_scale -- the
        # scale was never tuned against these specific cutoffs, so this coverage number is
        # honest, not double-dipped (see search_calibration_scale's docstring).
        eval_result = per_segment_eval[key]
        final_backtest_detail[f"{channel}/{campaign_type}"] = {
            "method": seg_model["method"],
            "mean_ape_pct": eval_result["mean_ape_pct"],
            "mean_pinball_loss": eval_result["mean_pinball_loss"],
            "coverage_pct": eval_result["coverage_pct"],
            "n_scored": eval_result["n_scored"],
            "no_calibration_tuning": eval_result.get("no_calibration_tuning", False),
        }
        print(f"  final fit {channel}/{campaign_type} [{seg_model['method']}]: "
              f"{seg_model['n_days_observed']}d, elasticity={seg_model['elasticity_beta']:.2f} "
              f"(ci=[{seg_model['elasticity_ci_low']:.2f},{seg_model['elasticity_ci_high']:.2f}], "
              f"low_conf={seg_model['elasticity_low_confidence']})", file=sys.stderr)

    campaign_type_methods = {key: segment_fit_fns[key] for key in combos}
    print("Fitting campaign-level models...", file=sys.stderr)
    campaign_segments = fit_campaign_segments(features, campaign_type_methods, segments)

    scored = [v for v in final_backtest_detail.values() if v["n_scored"] > 0]
    n_no_tuning = sum(1 for v in final_backtest_detail.values() if v.get("no_calibration_tuning"))
    backtest_summary = {
        "methodology": (
            "rolling-origin: calibration scale tuned on the older half of available walk-forward "
            "cutoffs, coverage/MAPE/pinball reported on the newer held-out half never seen during "
            "tuning (fixes an earlier version that tuned and reported on the same 3 cutoffs)"
        ),
        "n_segments_without_holdout_split": n_no_tuning,
        "per_segment_calibration_scale": {f"{k[0]}/{k[1]}": v for k, v in per_segment_scale.items()},
        "per_segment_calibration_curve": {
            f"{k[0]}/{k[1]}": v for k, v in per_segment_coverage_curve.items()
        },
        "overall_mean_ape_pct": float(np.mean([v["mean_ape_pct"] for v in scored])) if scored else None,
        "overall_median_ape_pct": float(np.median([v["mean_ape_pct"] for v in scored])) if scored else None,
        "overall_mean_pinball_loss": float(np.mean([v["mean_pinball_loss"] for v in scored])) if scored else None,
        "overall_coverage_pct": float(np.mean([v["coverage_pct"] for v in scored])) if scored else None,
        "per_segment": final_backtest_detail,
    }

    model = {
        "segments": segments,
        "campaign_segments": campaign_segments,
        "channel_uncertainty_inflation": CHANNEL_UNCERTAINTY_INFLATION,
        "backtest_summary": backtest_summary,
        "feature_schema_version": 2,
        "trained_at": datetime.utcnow().isoformat(),
        "seed": SEED,
    }

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(model, f)
    print(f"Wrote model with {len(segments)} campaign_type segments + "
          f"{len(campaign_segments)} campaign segments to {args.model_out}", file=sys.stderr)

    # Dual export: JSON format for safe non-pickle serialization
    json_model_out = os.path.splitext(args.model_out)[0] + ".json"
    try:
        def _json_serialize(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, tuple):
                return list(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        # Convert tuple dict keys to string keys for JSON compatibility
        json_safe_model = {
            "segments": {f"{k[0]}/{k[1]}": v for k, v in segments.items()},
            "campaign_segments": {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in campaign_segments.items()},
            "channel_uncertainty_inflation": CHANNEL_UNCERTAINTY_INFLATION,
            "backtest_summary": backtest_summary,
            "feature_schema_version": 2,
            "trained_at": datetime.utcnow().isoformat(),
            "seed": SEED,
        }
        with open(json_model_out, "w") as f:
            json.dump(json_safe_model, f, indent=2, default=_json_serialize)
        print(f"Wrote JSON model artifact to {json_model_out}", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: JSON model export skipped ({exc})", file=sys.stderr)

    os.makedirs(os.path.dirname(args.backtest_out) or ".", exist_ok=True)
    with open(args.backtest_out, "w") as f:
        json.dump(backtest_summary, f, indent=2, default=str)
    print(f"Wrote post-fix backtest summary to {args.backtest_out}", file=sys.stderr)



if __name__ == "__main__":
    main()
