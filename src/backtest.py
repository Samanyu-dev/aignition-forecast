"""
Standalone walk-forward backtest CLI. NOT part of run.sh's critical path --
an offline validation/reporting tool. Run manually:

    python src/backtest.py --data-dir data --out docs/backtest_results.json

Compares the two candidate methods (src/train.py's "empirical" trend+seasonal
decomposition vs. statsmodels Holt-Winters) per (channel, campaign_type)
segment on identical walk-forward windows, at residual_scale=1.0 (i.e. before
the calibration-scale fix train.py applies -- see docs/TECHNICAL_DOC.md
sections 3.5/3.6 and docs/backtest_results_baseline.json /
docs/backtest_results_post_fix.json for the "before" and "after" snapshots).
This file is the "document the comparison, don't just swap blindly" artifact.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from train import build_features, daily_segment_series, fit_trend_seasonal, fit_holt_winters, CHANNEL_UNCERTAINTY_INFLATION
from backtest_core import evaluate_segment_method

SEED = 42


def run_comparison(data_dir: str) -> dict:
    rng = np.random.default_rng(SEED)
    features = build_features(data_dir)

    per_segment = {}
    combos = features[["channel", "campaign_type"]].drop_duplicates().itertuples(index=False)
    for channel, campaign_type in combos:
        daily = daily_segment_series(features, channel, campaign_type)
        base_inflation = CHANNEL_UNCERTAINTY_INFLATION.get(channel, 1.0)

        empirical = evaluate_segment_method(daily, fit_trend_seasonal, rng, base_uncertainty_inflation=base_inflation)
        holt_winters = evaluate_segment_method(daily, fit_holt_winters, rng, base_uncertainty_inflation=base_inflation)

        def summarize(r):
            if r["n_scored"] == 0:
                return {"status": "insufficient_history"}
            return {
                "status": "scored", "n_scored": r["n_scored"],
                "mean_ape_pct": round(r["mean_ape_pct"], 2),
                "mean_pinball_loss": round(r["mean_pinball_loss"], 2),
                "coverage_pct": round(r["coverage_pct"], 1),
            }

        emp_s, hw_s = summarize(empirical), summarize(holt_winters)
        if emp_s["status"] == hw_s["status"] == "scored":
            winner = "empirical" if emp_s["mean_pinball_loss"] <= hw_s["mean_pinball_loss"] else "holt_winters"
        elif emp_s["status"] == "scored":
            winner = "empirical"
        elif hw_s["status"] == "scored":
            winner = "holt_winters"
        else:
            winner = "empirical (default, insufficient history to backtest)"

        per_segment[f"{channel}/{campaign_type}"] = {"empirical": emp_s, "holt_winters": hw_s, "winner": winner}

    return {"config": {"seed": SEED, "note": "residual_scale=1.0, pre-calibration-fix"}, "per_segment": per_segment}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="./docs/backtest_results.json")
    args = ap.parse_args()

    results = run_comparison(args.data_dir)

    print("\n=== EMPIRICAL vs HOLT-WINTERS (walk-forward backtest) ===")
    n_empirical_wins = 0
    n_hw_wins = 0
    for key, r in sorted(results["per_segment"].items()):
        print(f"  {key}:")
        print(f"    empirical:    {r['empirical']}")
        print(f"    holt_winters: {r['holt_winters']}")
        print(f"    winner: {r['winner']}")
        if r["winner"].startswith("empirical"):
            n_empirical_wins += 1
        elif r["winner"] == "holt_winters":
            n_hw_wins += 1
    print(f"\nWins: empirical={n_empirical_wins}, holt_winters={n_hw_wins}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
