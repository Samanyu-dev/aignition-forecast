"""
Walk-forward backtesting for the segment-level forecasting model. NOT part of
run.sh's critical path -- this is an offline validation tool. Run manually:

    python src/backtest.py --data-dir data --out docs/backtest_results.json

For each (channel, campaign_type) segment, refits the model at 3 walk-forward
cutoffs (using only data available before each cutoff -- no leakage) and scores
the 30-day-ahead forecast against the actual held-out revenue:

  - MAPE / MAE on the point forecast (P50)
  - Pinball loss at P10/P50/P90 (the correct scoring rule for quantile forecasts)
  - P10-P90 calibration coverage: is the actual value inside the interval?
    A well-calibrated 80% interval should cover ~80% of held-out points --
    this is the actual evidence for "appropriate handling of uncertainty",
    not just an assertion that intervals exist.

Segments without enough history before a cutoff are skipped for that cutoff
(and flagged) rather than backtested on too little data to be meaningful.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train import build_features, daily_segment_series, fit_trend_seasonal, fit_elasticity, CHANNEL_UNCERTAINTY_INFLATION
from forecasting import _simulate_segment_daily

SEED = 42
TEST_HORIZON_DAYS = 30
N_CUTOFFS = 3
MIN_TRAIN_DAYS = 45
N_SIMS = 2000


def pinball_loss(y_true: float, y_pred: float, tau: float) -> float:
    diff = y_true - y_pred
    return max(tau * diff, (tau - 1) * diff)


def backtest_segment(daily: pd.DataFrame, channel: str, campaign_type: str, rng: np.random.Generator) -> list:
    last_date = daily["date"].max()
    cutoffs = [last_date - pd.Timedelta(days=TEST_HORIZON_DAYS * i) for i in range(N_CUTOFFS, 0, -1)]

    results = []
    for cutoff in cutoffs:
        train_slice = daily[daily.date <= cutoff]
        test_slice = daily[(daily.date > cutoff) & (daily.date <= cutoff + pd.Timedelta(days=TEST_HORIZON_DAYS))]

        if len(train_slice) < MIN_TRAIN_DAYS or len(test_slice) < TEST_HORIZON_DAYS:
            results.append({
                "cutoff": str(cutoff.date()), "skipped": True,
                "reason": f"insufficient history ({len(train_slice)} train days, need >={MIN_TRAIN_DAYS})",
            })
            continue

        seg = fit_trend_seasonal(train_slice)
        seg["uncertainty_inflation"] = CHANNEL_UNCERTAINTY_INFLATION.get(channel, 1.0)

        sims = _simulate_segment_daily(seg, TEST_HORIZON_DAYS, N_SIMS, rng)
        revenue_sim = sims.sum(axis=1)
        p10, p50, p90 = np.percentile(revenue_sim, [10, 50, 90])
        actual = float(test_slice["revenue"].sum())

        ape = abs(actual - p50) / max(actual, 1.0) * 100
        pinball = {
            "p10": pinball_loss(actual, p10, 0.1),
            "p50": pinball_loss(actual, p50, 0.5),
            "p90": pinball_loss(actual, p90, 0.9),
        }
        covered = p10 <= actual <= p90

        results.append({
            "cutoff": str(cutoff.date()), "skipped": False,
            "actual": round(actual, 2), "p10": round(float(p10), 2),
            "p50": round(float(p50), 2), "p90": round(float(p90), 2),
            "ape_pct": round(ape, 2),
            "pinball_loss": {k: round(v, 2) for k, v in pinball.items()},
            "covered_by_p10_p90": bool(covered),
        })
    return results


def run_backtest(data_dir: str) -> dict:
    rng = np.random.default_rng(SEED)
    features = build_features(data_dir)

    segment_results = {}
    combos = features[["channel", "campaign_type"]].drop_duplicates().itertuples(index=False)
    for channel, campaign_type in combos:
        daily = daily_segment_series(features, channel, campaign_type)
        key = f"{channel}/{campaign_type}"
        segment_results[key] = backtest_segment(daily, channel, campaign_type, rng)

    scored = [r for cutoffs in segment_results.values() for r in cutoffs if not r["skipped"]]
    n_skipped = sum(1 for cutoffs in segment_results.values() for r in cutoffs if r["skipped"])

    summary = {
        "n_segments": len(segment_results),
        "n_scored_cutoffs": len(scored),
        "n_skipped_cutoffs": n_skipped,
        "overall_mape_pct": round(float(np.mean([r["ape_pct"] for r in scored])), 2) if scored else None,
        "overall_median_ape_pct": round(float(np.median([r["ape_pct"] for r in scored])), 2) if scored else None,
        "overall_mean_pinball_loss": round(
            float(np.mean([v for r in scored for v in r["pinball_loss"].values()])), 2
        ) if scored else None,
        "p10_p90_coverage_pct": round(
            float(np.mean([r["covered_by_p10_p90"] for r in scored]) * 100), 1
        ) if scored else None,
        "nominal_coverage_pct": 80.0,
    }

    per_segment_summary = {}
    for key, cutoffs in segment_results.items():
        scored_c = [r for r in cutoffs if not r["skipped"]]
        if not scored_c:
            per_segment_summary[key] = {"status": "insufficient_history"}
            continue
        per_segment_summary[key] = {
            "status": "scored",
            "n_cutoffs_scored": len(scored_c),
            "mean_ape_pct": round(float(np.mean([r["ape_pct"] for r in scored_c])), 2),
            "coverage_pct": round(float(np.mean([r["covered_by_p10_p90"] for r in scored_c]) * 100), 1),
        }

    return {
        "config": {
            "test_horizon_days": TEST_HORIZON_DAYS, "n_cutoffs": N_CUTOFFS,
            "min_train_days": MIN_TRAIN_DAYS, "n_sims": N_SIMS, "seed": SEED,
        },
        "summary": summary,
        "per_segment_summary": per_segment_summary,
        "detail": segment_results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="./docs/backtest_results.json")
    args = ap.parse_args()

    results = run_backtest(args.data_dir)

    print("\n=== BACKTEST SUMMARY ===")
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")

    print("\n=== PER-SEGMENT ===")
    for key, s in sorted(results["per_segment_summary"].items()):
        if s["status"] == "insufficient_history":
            print(f"  {key}: insufficient history for backtesting")
        else:
            print(f"  {key}: MAPE={s['mean_ape_pct']}%, coverage={s['coverage_pct']}% (n={s['n_cutoffs_scored']})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
