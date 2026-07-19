"""
Budget-reallocation recommendations. Uses the already-fitted per-segment
spend elasticity (with bootstrap CI -- see src/train.py::fit_elasticity) to
find high-confidence "shift $X/day from segment A to segment B" moves, and
prices each candidate's expected revenue/ROAS impact with the same Monte
Carlo machinery src/forecasting.py already uses for budget scenarios.

Only segments with elasticity_low_confidence=False (tight bootstrap CI,
n_obs>=10) are eligible as either a donor (reduce spend) or a receiver
(increase spend) -- this directly reuses the reliability gating already
built for budget-scenario scaling (src/forecasting.py's
elasticity_beta_for_scenario fallback), rather than inventing a new
confidence threshold. Not part of run.sh's critical path -- offline/demo-
layer analysis only.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from forecasting import forecast, DEFAULT_N_SIMS, SEED

SHIFT_FRACTION = 0.20  # move 20% of a donor segment's average daily spend
MAX_RECEIVER_MULTIPLIER = 2.5  # cap how far a receiver's elasticity fit is extrapolated
MIN_DONOR_DAILY_SPEND = 5.0  # skip near-zero-spend segments as donors -- nothing meaningful to shift
CANDIDATE_POOL_SIZE = 3  # consider the 3 most/least elastic eligible segments as donor/receiver pools
TOP_N_RECOMMENDATIONS = 5
RECOMMENDATION_HORIZON_DAYS = 90
SEARCH_N_SIMS = 500  # fewer sims while ranking many candidate pairs
FINAL_N_SIMS = DEFAULT_N_SIMS  # full precision for the top-N that get surfaced


def _eligible_segments(model: dict) -> list:
    eligible = []
    for (channel, campaign_type), seg in model["segments"].items():
        if seg.get("elasticity_low_confidence", True):
            continue
        if seg["avg_daily_spend"] < MIN_DONOR_DAILY_SPEND:
            continue
        eligible.append((channel, campaign_type, seg))
    return eligible


def compute_segment_multipliers(donor: tuple, receiver: tuple, shift_fraction: float) -> tuple:
    """Shared math for pricing a donor->receiver shift: returns
    (segment_budget_multipliers dict, shift_dollars). Exposed separately from
    _price_candidate so callers (e.g. src/api.py's recommendation-forecast
    endpoint) can get the same multipliers used for pricing without
    re-deriving the formula."""
    d_channel, d_type, d_seg = donor
    r_channel, r_type, r_seg = receiver

    shift_dollars = d_seg["avg_daily_spend"] * shift_fraction
    donor_mult = 1.0 - shift_fraction
    receiver_new_daily_spend = r_seg["avg_daily_spend"] + shift_dollars
    receiver_mult = (
        receiver_new_daily_spend / r_seg["avg_daily_spend"] if r_seg["avg_daily_spend"] > 0 else 1.0
    )
    receiver_mult = min(receiver_mult, MAX_RECEIVER_MULTIPLIER)

    return {(d_channel, d_type): donor_mult, (r_channel, r_type): receiver_mult}, shift_dollars


def _price_candidate(model, donor: tuple, receiver: tuple, shift_fraction: float,
                      horizon: int, n_sims: int, seed: int) -> dict:
    d_channel, d_type, d_seg = donor
    r_channel, r_type, r_seg = receiver

    seg_mults, shift_dollars = compute_segment_multipliers(donor, receiver, shift_fraction)
    baseline = forecast(model, horizons=[horizon], n_sims=n_sims, seed=seed)
    scenario = forecast(model, horizons=[horizon], segment_budget_multipliers=seg_mults,
                         n_sims=n_sims, seed=seed)

    def _pick(df, metric):
        row = df[(df.channel == "blended") & (df.metric == metric)]
        return float(row["p50"].iloc[0]) if len(row) else None

    baseline_revenue = _pick(baseline, "revenue")
    scenario_revenue = _pick(scenario, "revenue")
    baseline_roas = _pick(baseline, "roas")
    scenario_roas = _pick(scenario, "roas")

    revenue_delta = scenario_revenue - baseline_revenue
    revenue_delta_pct = (revenue_delta / baseline_revenue * 100) if baseline_revenue else None

    return {
        "from": f"{d_channel}/{d_type}",
        "to": f"{r_channel}/{r_type}",
        "shift_daily_dollars": round(shift_dollars, 2),
        "from_elasticity": round(d_seg["elasticity_beta"], 2),
        "from_elasticity_ci": [round(d_seg["elasticity_ci_low"], 2), round(d_seg["elasticity_ci_high"], 2)],
        "to_elasticity": round(r_seg["elasticity_beta"], 2),
        "to_elasticity_ci": [round(r_seg["elasticity_ci_low"], 2), round(r_seg["elasticity_ci_high"], 2)],
        "horizon_days": horizon,
        "baseline_p50_revenue": round(baseline_revenue, 2),
        "scenario_p50_revenue": round(scenario_revenue, 2),
        "revenue_delta": round(revenue_delta, 2),
        "revenue_delta_pct": round(revenue_delta_pct, 1) if revenue_delta_pct is not None else None,
        "baseline_p50_roas": round(baseline_roas, 2) if baseline_roas is not None else None,
        "scenario_p50_roas": round(scenario_roas, 2) if scenario_roas is not None else None,
        "confidence": "high",  # both donor and receiver passed the elasticity_low_confidence gate
    }


def generate_reallocation_candidates(
    model: dict,
    shift_fraction: float = SHIFT_FRACTION,
    top_n: int = TOP_N_RECOMMENDATIONS,
    horizon: int = RECOMMENDATION_HORIZON_DAYS,
    seed: int = SEED,
) -> list:
    eligible = _eligible_segments(model)
    if len(eligible) < 2:
        return []

    # donors: lower elasticity (diminishing returns -- inefficient to keep scaling)
    # receivers: higher elasticity (near-linear -- efficient to scale further)
    donors = sorted(eligible, key=lambda r: r[2]["elasticity_beta"])[:CANDIDATE_POOL_SIZE]
    receivers = sorted(eligible, key=lambda r: -r[2]["elasticity_beta"])[:CANDIDATE_POOL_SIZE]

    scored = []
    for donor in donors:
        for receiver in receivers:
            if (donor[0], donor[1]) == (receiver[0], receiver[1]):
                continue
            scored.append(_price_candidate(model, donor, receiver, shift_fraction, horizon,
                                            SEARCH_N_SIMS, seed))

    scored.sort(key=lambda c: -(c["revenue_delta"] if c["revenue_delta"] is not None else -1e18))
    top = scored[:top_n]

    # Re-price the finalists at full precision so the surfaced numbers aren't
    # search-time-cheap estimates.
    final = []
    for c in top:
        d_channel, d_type = c["from"].split("/", 1)
        r_channel, r_type = c["to"].split("/", 1)
        donor = next(x for x in eligible if x[0] == d_channel and x[1] == d_type)
        receiver = next(x for x in eligible if x[0] == r_channel and x[1] == r_type)
        final.append(_price_candidate(model, donor, receiver, shift_fraction, horizon,
                                       FINAL_N_SIMS, seed))
    return final
