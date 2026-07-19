"""
Retrospective (walk-forward) validation of src/recommendations.py's core
premise. Not part of run.sh's critical path -- an offline validation tool.

We can't literally test "would this budget shift have beaten actual spend
allocation" -- that requires a counterfactual (an actual controlled
experiment) that doesn't exist in observational ad-platform data. What we
CAN honestly test: at several past decision points, using only data
available up to that point, identify the donor (lowest tight-CI elasticity)
and receiver (highest tight-CI elasticity) segment the engine would have
picked -- then check what ACTUALLY happened to each segment's realized ROAS
in the subsequent window. If the recommendation logic is sound, the
receiver segment (told "this scales efficiently") should realize better
subsequent ROAS than the donor segment (told "this doesn't") more often
than chance, out of sample. This validates the engine's directional
premise, not a specific dollar-shift outcome.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train import build_features, daily_segment_series, fit_elasticity

TEST_HORIZON_DAYS = 30
CUTOFF_OFFSETS_DAYS = [180, 150, 120, 90, 60]  # 5 as-of decision points, oldest first
MIN_TRAIN_DAYS = 90  # need reasonable history to fit an as-of-cutoff elasticity at all
MIN_DAILY_SPEND = 5.0
CANDIDATE_POOL_SIZE = 3  # match recommendations.py's donor/receiver pool size, not just the single most extreme pair


def _as_of_eligible(features_asof: pd.DataFrame, channel_types: list) -> list:
    """Same eligibility gate as recommendations.py's _eligible_segments, but
    fit using ONLY data available as of a past cutoff -- no lookahead."""
    eligible = []
    for channel, campaign_type in channel_types:
        seg = features_asof[(features_asof.channel == channel) & (features_asof.campaign_type == campaign_type)]
        if seg.empty:
            continue  # this campaign_type hadn't started yet as of this cutoff
        daily = daily_segment_series(features_asof, channel, campaign_type)
        if len(daily) < MIN_TRAIN_DAYS:
            continue
        elas = fit_elasticity(daily)
        avg_daily_spend = float(daily["spend"].tail(28).mean())
        if elas["elasticity_low_confidence"] or avg_daily_spend < MIN_DAILY_SPEND:
            continue
        eligible.append({"channel": channel, "campaign_type": campaign_type,
                          "elasticity_beta": elas["elasticity_beta"]})
    return eligible


def _realized_roas(features: pd.DataFrame, channel: str, campaign_type: str,
                    start: pd.Timestamp, end: pd.Timestamp):
    seg = features[(features.channel == channel) & (features.campaign_type == campaign_type)
                    & (features.date > start) & (features.date <= end)]
    spend, revenue = float(seg["spend"].sum()), float(seg["revenue"].sum())
    roas = (revenue / spend) if spend > 0 else None
    return roas, spend, revenue


def run_retrospective_validation(data_dir: str = "./data") -> dict:
    features = build_features(data_dir)
    last_date = features["date"].max()
    channel_types = list(features[["channel", "campaign_type"]].drop_duplicates().itertuples(index=False))

    results = []
    for offset in CUTOFF_OFFSETS_DAYS:
        cutoff = last_date - pd.Timedelta(days=offset)
        features_asof = features[features.date <= cutoff]

        eligible = _as_of_eligible(features_asof, channel_types)
        if len(eligible) < 2:
            results.append({"cutoff": str(cutoff.date()), "skipped": True,
                             "reason": "insufficient eligible segments as-of this cutoff"})
            continue

        # Match recommendations.py's actual mechanism: a pool of the most/least elastic
        # eligible segments, not just the single most extreme pair -- one recurring pair
        # dominating the sample would be thin, easily-misleading evidence.
        donors = sorted(eligible, key=lambda r: r["elasticity_beta"])[:CANDIDATE_POOL_SIZE]
        receivers = sorted(eligible, key=lambda r: -r["elasticity_beta"])[:CANDIDATE_POOL_SIZE]
        window_end = min(cutoff + pd.Timedelta(days=TEST_HORIZON_DAYS), last_date)

        cutoff_pairs = []
        for donor in donors:
            for receiver in receivers:
                if (donor["channel"], donor["campaign_type"]) == (receiver["channel"], receiver["campaign_type"]):
                    continue
                donor_roas, donor_spend, _ = _realized_roas(
                    features, donor["channel"], donor["campaign_type"], cutoff, window_end)
                receiver_roas, receiver_spend, _ = _realized_roas(
                    features, receiver["channel"], receiver["campaign_type"], cutoff, window_end)
                confirmed = None
                if donor_roas is not None and receiver_roas is not None:
                    confirmed = receiver_roas > donor_roas
                cutoff_pairs.append({
                    "donor": f"{donor['channel']}/{donor['campaign_type']}",
                    "donor_asof_elasticity": round(donor["elasticity_beta"], 2),
                    "receiver": f"{receiver['channel']}/{receiver['campaign_type']}",
                    "receiver_asof_elasticity": round(receiver["elasticity_beta"], 2),
                    "donor_realized_roas": round(donor_roas, 2) if donor_roas is not None else None,
                    "receiver_realized_roas": round(receiver_roas, 2) if receiver_roas is not None else None,
                    "donor_realized_spend": round(donor_spend, 2), "receiver_realized_spend": round(receiver_spend, 2),
                    "recommendation_confirmed": confirmed,
                })

        results.append({
            "cutoff": str(cutoff.date()), "skipped": False,
            "window_days": int((window_end - cutoff).days),
            "n_eligible_segments": len(eligible),
            "pairs": cutoff_pairs,
        })

    all_pairs = [p for r in results if not r["skipped"] for p in r["pairs"]]
    scored_pairs = [p for p in all_pairs if p["recommendation_confirmed"] is not None]
    # Also report the top-elasticity-spread pair per cutoff separately -- that's the one
    # recommendations.py would actually have surfaced as its #1 pick, vs. the full grid above.
    top_pick_per_cutoff = [r["pairs"][0] for r in results if not r["skipped"] and r["pairs"]]
    top_pick_scored = [p for p in top_pick_per_cutoff if p["recommendation_confirmed"] is not None]

    summary = {
        "n_cutoffs_tested": len(results),
        "full_candidate_grid": {
            "n_pairs_scored": len(scored_pairs),
            "n_confirmed": sum(1 for p in scored_pairs if p["recommendation_confirmed"]),
            "confirmation_rate_pct": (
                round(sum(1 for p in scored_pairs if p["recommendation_confirmed"]) / len(scored_pairs) * 100, 1)
                if scored_pairs else None
            ),
        },
        "top_pick_only": {
            "n_scored": len(top_pick_scored),
            "n_confirmed": sum(1 for p in top_pick_scored if p["recommendation_confirmed"]),
            "confirmation_rate_pct": (
                round(sum(1 for p in top_pick_scored if p["recommendation_confirmed"]) / len(top_pick_scored) * 100, 1)
                if top_pick_scored else None
            ),
        },
    }
    return {
        "config": {"test_horizon_days": TEST_HORIZON_DAYS, "cutoff_offsets_days": CUTOFF_OFFSETS_DAYS,
                    "min_train_days": MIN_TRAIN_DAYS, "candidate_pool_size": CANDIDATE_POOL_SIZE},
        "summary": summary,
        "detail": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="./docs/recommendation_validation.json")
    args = ap.parse_args()

    report = run_retrospective_validation(args.data_dir)

    print("=== RETROSPECTIVE RECOMMENDATION-ENGINE VALIDATION ===")
    print(json.dumps(report["summary"], indent=2))
    print()
    for r in report["detail"]:
        if r["skipped"]:
            print(f"{r['cutoff']}: skipped ({r['reason']})")
            continue
        print(f"{r['cutoff']} ({r['n_eligible_segments']} eligible segments):")
        for p in r["pairs"]:
            print(f"    {p}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
