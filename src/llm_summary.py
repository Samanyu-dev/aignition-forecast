"""
AI-assisted causal/anomaly narrative layer. Not part of run.sh's critical path
(feature-gen + predict must work with zero network access) -- this module is
used only by the FastAPI/Streamlit demo layer.

Design: never ask the LLM to invent numbers. Pre-compute structured statistics
(period-over-period deltas, spend elasticity per segment with bootstrap CIs,
anomalous campaigns by ROAS z-score, budget-scenario deltas, per-segment
walk-forward backtest reliability, a structural zero-revenue-campaign check,
confidence-gated budget-reallocation candidates with their priced impact --
see src/recommendations.py -- and formal campaign-consistency validation
findings -- see src/validate_consistency.py) and ask Claude to interpret
them -- including turning them into a ranked, caveated recommendation, not
just a description of what already happened.

If ANTHROPIC_API_KEY is unset, falls back to a deterministic template built
from the same stats dict, so the app runs fully offline. The template path
was the one exercised in this environment (no live API key available at
build time); the Claude call path is implemented against the current
Messages API but was validated structurally, not against a live request --
see docs/TECHNICAL_DOC.md limitations.
"""
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from recommendations import generate_reallocation_candidates

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
RECENT_WINDOW_DAYS = 30
ANOMALY_Z_THRESHOLD = 2.0
ANOMALY_LOOKBACK_DAYS = 60
LOW_COVERAGE_THRESHOLD_PCT = 40.0  # walk-forward P10-P90 coverage below this = flag as low-reliability
HIGH_APE_THRESHOLD_PCT = 150.0  # walk-forward MAPE above this = flag as low-reliability
ZERO_REVENUE_CAMPAIGN_SHARE_THRESHOLD = 0.25  # fraction of a channel's campaigns with lifetime spend>0, revenue=0


def _period_deltas(features: pd.DataFrame, window_days: int = RECENT_WINDOW_DAYS) -> list:
    now = features["date"].max()
    recent_start = now - pd.Timedelta(days=window_days)
    prev_start = now - pd.Timedelta(days=2 * window_days)

    recent = features[features["date"] > recent_start]
    previous = features[(features["date"] > prev_start) & (features["date"] <= recent_start)]

    out = []
    for channel in sorted(features["channel"].unique()):
        r = recent[recent.channel == channel]
        p = previous[previous.channel == channel]
        r_rev, p_rev = r["revenue"].sum(), p["revenue"].sum()
        r_spend, p_spend = r["spend"].sum(), p["spend"].sum()
        out.append({
            "channel": channel,
            "recent_revenue": round(float(r_rev), 2),
            "previous_revenue": round(float(p_rev), 2),
            "revenue_pct_change": round(float((r_rev - p_rev) / p_rev * 100), 1) if p_rev > 0 else None,
            "recent_spend": round(float(r_spend), 2),
            "previous_spend": round(float(p_spend), 2),
            "spend_pct_change": round(float((r_spend - p_spend) / p_spend * 100), 1) if p_spend > 0 else None,
        })
    return out


def _elasticity_summary(model: dict) -> list:
    out = []
    for (channel, campaign_type), seg in model["segments"].items():
        out.append({
            "channel": channel,
            "campaign_type": campaign_type,
            "elasticity_beta": round(seg["elasticity_beta"], 2),
            "elasticity_ci": [round(seg.get("elasticity_ci_low", seg["elasticity_beta"]), 2),
                              round(seg.get("elasticity_ci_high", seg["elasticity_beta"]), 2)],
            "n_obs": seg["elasticity_n_obs"],
            "low_confidence": seg.get("elasticity_low_confidence", seg["elasticity_n_obs"] < 10),
            "used_for_budget_scenarios": not seg.get("elasticity_low_confidence", seg["elasticity_n_obs"] < 10),
        })
    return sorted(out, key=lambda r: r["elasticity_beta"])


def _forecast_reliability(model: dict) -> list:
    """Per-segment walk-forward backtest reliability (see docs/TECHNICAL_DOC.md
    3.7-3.8) -- lets the narrative say *how much to trust* a forecast, not
    just what the forecast is."""
    summary = model.get("backtest_summary", {}).get("per_segment", {})
    out = []
    for key, s in summary.items():
        if s.get("n_scored", 0) == 0:
            continue
        out.append({
            "segment": key,
            "method": s["method"],
            "mean_ape_pct": round(s["mean_ape_pct"], 1) if s["mean_ape_pct"] is not None else None,
            "coverage_pct": round(s["coverage_pct"], 1) if s["coverage_pct"] is not None else None,
            "low_reliability": (
                (s["coverage_pct"] is not None and s["coverage_pct"] < LOW_COVERAGE_THRESHOLD_PCT)
                or (s["mean_ape_pct"] is not None and s["mean_ape_pct"] > HIGH_APE_THRESHOLD_PCT)
            ),
        })
    return sorted(out, key=lambda r: (r["coverage_pct"] if r["coverage_pct"] is not None else 999))


def _structural_risk_campaigns(features: pd.DataFrame) -> list:
    """Flags channels where a large share of campaigns have spent money but
    generated zero lifetime revenue -- a structural failure (e.g. broken
    tracking, a dead campaign left running), not routine variance. Computed
    live from whatever data is loaded, not hardcoded to any one channel."""
    lifetime = features.groupby(["channel", "campaign_id", "campaign_name"]).agg(
        revenue=("revenue", "sum"), spend=("spend", "sum")
    ).reset_index()

    out = []
    for channel, group in lifetime.groupby("channel"):
        spending = group[group.spend > 0]
        if len(spending) == 0:
            continue
        zero_rev = spending[spending.revenue == 0]
        share = len(zero_rev) / len(spending)
        if share >= ZERO_REVENUE_CAMPAIGN_SHARE_THRESHOLD:
            out.append({
                "channel": channel,
                "zero_revenue_campaigns": int(len(zero_rev)),
                "total_spending_campaigns": int(len(spending)),
                "zero_revenue_share_pct": round(share * 100, 1),
                "wasted_spend": round(float(zero_rev["spend"].sum()), 2),
            })
    return out


def _anomalous_campaigns(features: pd.DataFrame, z_thresh: float = ANOMALY_Z_THRESHOLD) -> list:
    now = features["date"].max()
    recent = features[features["date"] > now - pd.Timedelta(days=ANOMALY_LOOKBACK_DAYS)]
    agg = recent.groupby(["channel", "campaign_id", "campaign_name"]).agg(
        revenue=("revenue", "sum"), spend=("spend", "sum")
    ).reset_index()
    agg = agg[agg.spend > 0]
    agg["roas"] = agg["revenue"] / agg["spend"]

    out = []
    for channel, group in agg.groupby("channel"):
        if len(group) < 3 or group["roas"].std() == 0:
            continue
        z = (group["roas"] - group["roas"].mean()) / group["roas"].std()
        flagged = group[z.abs() > z_thresh].copy()
        flagged["z_score"] = z[z.abs() > z_thresh].round(2)
        for _, row in flagged.iterrows():
            out.append({
                "channel": channel,
                "campaign_name": row["campaign_name"],
                "roas": round(float(row["roas"]), 2),
                "z_score": float(row["z_score"]),
            })
    return sorted(out, key=lambda r: -abs(r["z_score"]))[:10]


def _budget_scenario_deltas(baseline_df: pd.DataFrame, scenario_df: pd.DataFrame) -> list:
    b = baseline_df[(baseline_df.channel == "blended") & (baseline_df.metric == "revenue")]
    s = scenario_df[(scenario_df.channel == "blended") & (scenario_df.metric == "revenue")]
    merged = b.merge(s, on="horizon_days", suffixes=("_baseline", "_scenario"))
    out = []
    for _, row in merged.iterrows():
        pct = (row.p50_scenario - row.p50_baseline) / row.p50_baseline * 100 if row.p50_baseline else None
        out.append({
            "horizon_days": int(row.horizon_days),
            "baseline_p50_revenue": round(float(row.p50_baseline), 2),
            "scenario_p50_revenue": round(float(row.p50_scenario), 2),
            "pct_change": round(float(pct), 1) if pct is not None else None,
        })
    return out


def _consistency_summary(validation_report: dict, top_n: int = 5) -> dict:
    """Condense the full campaign-consistency validation report (see
    src/validate_consistency.py) into something compact enough for the LLM
    prompt and narrative -- summary counts plus a handful of examples per
    check type, not the full per-campaign detail."""
    out = {"summary": validation_report["summary"], "examples": {}}
    for channel, c in validation_report["channels"].items():
        for check_name, issues in c["checks"].items():
            if not issues:
                continue
            out["examples"].setdefault(check_name, [])
            for issue in issues[:top_n]:
                out["examples"][check_name].append({"channel": channel, **issue})
    return out


def compute_stats(
    features: pd.DataFrame,
    model: dict,
    baseline_forecast: pd.DataFrame,
    scenario_forecast: Optional[pd.DataFrame] = None,
    include_recommendations: bool = True,
    validation_report: Optional[dict] = None,
) -> dict:
    stats = {
        "period_deltas": _period_deltas(features),
        "elasticity": _elasticity_summary(model),
        "anomalous_campaigns": _anomalous_campaigns(features),
        "forecast_reliability": _forecast_reliability(model),
        "structural_risk_campaigns": _structural_risk_campaigns(features),
    }
    if validation_report is not None:
        stats["consistency_validation"] = _consistency_summary(validation_report)
    if scenario_forecast is not None:
        stats["budget_scenario_deltas"] = _budget_scenario_deltas(baseline_forecast, scenario_forecast)
    if include_recommendations:
        stats["budget_reallocation_recommendations"] = generate_reallocation_candidates(model)
    return stats


def _template_narrative(stats: dict) -> dict:
    lines = []
    risk_flags = []

    deltas = sorted(stats["period_deltas"], key=lambda r: -(r["revenue_pct_change"] or 0))
    for d in deltas:
        if d["revenue_pct_change"] is None:
            continue
        direction = "up" if d["revenue_pct_change"] >= 0 else "down"
        lines.append(
            f"{d['channel'].capitalize()} revenue is {direction} "
            f"{abs(d['revenue_pct_change'])}% over the trailing {RECENT_WINDOW_DAYS} days "
            f"(${d['recent_revenue']:,.0f} vs ${d['previous_revenue']:,.0f})."
        )
        if d["revenue_pct_change"] < -15:
            risk_flags.append(f"{d['channel']} revenue declined more than 15% period-over-period.")

    scalable = [e for e in stats["elasticity"] if e["elasticity_beta"] >= 0.8 and not e["low_confidence"]]
    diminishing = [e for e in stats["elasticity"] if e["elasticity_beta"] < 0.5 and not e["low_confidence"]]
    if scalable:
        top = max(scalable, key=lambda e: e["elasticity_beta"])
        lines.append(
            f"{top['channel'].capitalize()}/{top['campaign_type']} shows near-linear returns to spend "
            f"(elasticity {top['elasticity_beta']}), suggesting room to scale budget efficiently."
        )
    if diminishing:
        bottom = min(diminishing, key=lambda e: e["elasticity_beta"])
        lines.append(
            f"{bottom['channel'].capitalize()}/{bottom['campaign_type']} shows strong diminishing returns "
            f"(elasticity {bottom['elasticity_beta']}) -- additional spend there is less efficient."
        )

    low_conf = [e for e in stats["elasticity"] if e["low_confidence"]]
    if low_conf:
        risk_flags.append(
            f"{len(low_conf)} segment(s) have too few spend/revenue observations for a reliable "
            f"elasticity estimate; treat their budget-scenario response as low-confidence."
        )

    for a in stats["anomalous_campaigns"][:5]:
        direction = "outperforming" if a["z_score"] > 0 else "underperforming"
        lines.append(
            f"Campaign '{a['campaign_name']}' ({a['channel']}) is {direction} its channel's typical "
            f"ROAS (ROAS {a['roas']}, z={a['z_score']})."
        )
        if a["z_score"] < -ANOMALY_Z_THRESHOLD:
            risk_flags.append(f"'{a['campaign_name']}' is a significant ROAS underperformer -- investigate.")

    unreliable = [r for r in stats.get("forecast_reliability", []) if r["low_reliability"]]
    if unreliable:
        names = ", ".join(r["segment"] for r in unreliable[:5])
        lines.append(
            f"Walk-forward backtesting flags {len(unreliable)} segment(s) as lower-reliability "
            f"forecasts (wide backtested error or thin interval coverage): {names}. Treat their "
            f"P10-P90 ranges as more approximate than the rest of the portfolio."
        )
        risk_flags.append(
            f"{len(unreliable)} segment(s) failed to backtest reliably: {names}. "
            f"See docs/TECHNICAL_DOC.md sec 3.8 for the walk-forward evidence."
        )

    for sr in stats.get("structural_risk_campaigns", []):
        lines.append(
            f"{sr['channel'].capitalize()} has {sr['zero_revenue_campaigns']} of "
            f"{sr['total_spending_campaigns']} spending campaigns ({sr['zero_revenue_share_pct']}%) "
            f"with zero lifetime revenue, representing ${sr['wasted_spend']:,.0f} of spend with no "
            f"measured return -- a structural issue (tracking, targeting, or a dead campaign left "
            f"running), not routine variance."
        )
        risk_flags.append(
            f"{sr['channel']}: {sr['zero_revenue_share_pct']}% of spending campaigns show zero "
            f"lifetime revenue (${sr['wasted_spend']:,.0f} wasted spend) -- investigate before "
            f"trusting this channel's forecast."
        )

    cv = stats.get("consistency_validation")
    if cv:
        s = cv["summary"]
        lines.append(
            f"Formal campaign-consistency validation flagged {s['total_issues_flagged']} issue(s) "
            f"across {s['total_campaigns']} campaigns: {s['by_check_type'].get('zero_revenue_with_spend', 0)} "
            f"zero-revenue-with-spend, {s['by_check_type'].get('budget_exceeded', 0)} daily-budget "
            f"overspend, {s['by_check_type'].get('date_coverage_gaps', 0)} date-coverage gaps "
            f">14 days, {s['by_check_type'].get('conversions_exceed_clicks', 0)} conversions-exceed-"
            f"clicks anomalies. Budget overspend is common in real ad-platform data because a stated "
            f"daily_budget is typically a pacing average a platform can exceed on any single day, "
            f"not a hard cap -- flagged for visibility, not automatically treated as an error."
        )

    recs = stats.get("budget_reallocation_recommendations", [])
    if recs:
        top = recs[0]
        lines.append(
            f"Recommended: shift ~${top['shift_daily_dollars']:.0f}/day from {top['from']} "
            f"(elasticity {top['from_elasticity']}, CI {top['from_elasticity_ci']}) to {top['to']} "
            f"(elasticity {top['to_elasticity']}, CI {top['to_elasticity_ci']}) -- estimated "
            f"{top['horizon_days']}-day revenue impact {top['revenue_delta_pct']:+.1f}% "
            f"(${top['baseline_p50_revenue']:,.0f} -> ${top['scenario_p50_revenue']:,.0f}), blended ROAS "
            f"{top['baseline_p50_roas']:.2f}x -> {top['scenario_p50_roas']:.2f}x. Confidence: "
            f"{top['confidence']} (both segments have a tight, backtested-eligible elasticity CI)."
        )
        if len(recs) > 1:
            others = "; ".join(
                f"{r['from']}->{r['to']} ({r['revenue_delta_pct']:+.1f}%)" for r in recs[1:4]
            )
            lines.append(f"Other confidence-gated shifts worth considering: {others}.")
        lines.append(
            "These are the only reallocations surfaced across the eligible segment pool -- every "
            "candidate was screened by elasticity confidence before being priced, so segments with "
            "too little data to trust are never recommended, only reported as unreliable (above)."
        )

    if "budget_scenario_deltas" in stats:
        for bs in stats["budget_scenario_deltas"]:
            if bs["pct_change"] is None:
                continue
            lines.append(
                f"Under the simulated budget scenario, {bs['horizon_days']}-day revenue moves "
                f"{bs['pct_change']:+.1f}% (${bs['baseline_p50_revenue']:,.0f} -> ${bs['scenario_p50_revenue']:,.0f})."
            )

    if not lines:
        lines.append("Insufficient history to compute period-over-period deltas or anomalies.")

    return {"text": " ".join(lines), "risk_flags": risk_flags}


def _call_claude(stats: dict, api_key: str, model_name: str) -> dict:
    import json
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "You are a marketing analytics assistant advising an e-commerce agency. Below is "
        "pre-computed JSON data about their blended Google/Bing/Meta performance: "
        "period-over-period revenue and spend deltas, per-segment spend elasticity with "
        "bootstrap confidence intervals (used_for_budget_scenarios=false means the CI was too "
        "wide to trust for budget scaling), anomalous campaigns by ROAS z-score, per-segment "
        "walk-forward backtest reliability (mean_ape_pct and coverage_pct from held-out "
        "validation -- low_reliability=true means treat that segment's forecast with more "
        "caution), a structural check for channels with a high share of zero-lifetime-revenue "
        "campaigns, formal campaign-consistency validation findings (consistency_validation -- "
        "budget overspend is normal in ad-platform data since daily_budget is usually a pacing "
        "average, not a hard cap; weight the other check types more heavily), confidence-gated "
        "budget-reallocation candidates with their priced revenue/ROAS impact "
        "(budget_reallocation_recommendations -- every candidate here already passed an "
        "elasticity-confidence screen, so all of them are trustworthy enough to recommend; rank "
        "and caveat them, don't re-litigate their eligibility), and (if present) a budget-scenario "
        "comparison.\n\n"
        "Do not invent or restate numbers beyond what's given. When forecast_reliability or "
        "structural_risk_campaigns entries are present, factor them explicitly into your "
        "narrative and risk flags -- these are honest reliability signals, not just "
        "performance metrics, and should shape how confidently you phrase recommendations. "
        "Write:\n"
        "1. A 3-5 sentence narrative summary of what's happening and why it matters operationally.\n"
        "2. If budget_reallocation_recommendations is non-empty, 1-2 sentences making the "
        "single best recommendation concrete and actionable ('shift $X/day from A to B because "
        "Y, expect roughly Z% revenue lift') -- this should read as advice, not just a "
        "description of a number that exists.\n"
        "3. A short bulleted list of risk flags or caveats (including anything that should "
        "limit confidence in the recommendation above, e.g. its own segment's backtest "
        "reliability).\n\n"
        f"DATA:\n{json.dumps(stats, indent=2)}"
    )
    response = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    # Narrative now spans two numbered items (summary + concrete recommendation) before the
    # bulleted risk list, so split on "first bulleted line" rather than "first blank line" --
    # a fixed split point would silently swallow the recommendation sentence into nowhere.
    narrative_lines, risk_flags = [], []
    in_risk_section = False
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*")):
            in_risk_section = True
            risk_flags.append(stripped.lstrip("-* ").strip())
        elif not in_risk_section and stripped:
            narrative_lines.append(stripped)
    narrative = " ".join(narrative_lines)
    return {"text": narrative, "risk_flags": risk_flags}


def generate_causal_summary(
    features: pd.DataFrame,
    model: dict,
    baseline_forecast: pd.DataFrame,
    scenario_forecast: Optional[pd.DataFrame] = None,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    include_recommendations: bool = True,
    validation_report: Optional[dict] = None,
) -> dict:
    stats = compute_stats(features, model, baseline_forecast, scenario_forecast,
                           include_recommendations=include_recommendations,
                           validation_report=validation_report)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        try:
            narrative = _call_claude(stats, api_key, model_name)
            source = "llm"
        except Exception as exc:  # noqa: BLE001 -- any API/network failure falls back safely
            narrative = _template_narrative(stats)
            narrative["text"] += f" [LLM call failed, using template fallback: {exc}]"
            source = "template_fallback"
    else:
        narrative = _template_narrative(stats)
        source = "template"

    return {
        "stats": stats,
        "narrative": narrative["text"],
        "risk_flags": narrative["risk_flags"],
        "source": source,
    }


def main():
    import argparse
    import json
    import pickle
    from forecasting import forecast

    ap = argparse.ArgumentParser(description="Generate AI-assisted causal business narrative and risk report")
    ap.add_argument("--features", required=True, help="Path to normalized parquet features")
    ap.add_argument("--model", required=True, help="Path to model pkl")
    ap.add_argument("--out-json", help="Path to output insights.json")
    ap.add_argument("--out-txt", help="Path to output insights.txt")
    args = ap.parse_args()

    features = pd.read_parquet(args.features)
    model = {}
    if os.path.exists(args.model):
        try:
            with open(args.model, "rb") as f:
                model = pickle.load(f)
        except Exception:
            model = {}

    baseline_fc = forecast(model, eval_features=features)
    summary = generate_causal_summary(features, model, baseline_fc)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote AI insights JSON to {args.out_json}", file=sys.stderr)

    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
        with open(args.out_txt, "w") as f:
            f.write(f"=== AIGNITION BUSINESS NARRATIVE ({summary['source'].upper()}) ===\n\n")
            f.write(summary["narrative"] + "\n\n")
            if summary["risk_flags"]:
                f.write("RISK FLAGS & CAVEATS:\n")
                for r in summary["risk_flags"]:
                    f.write(f" - {r}\n")
        print(f"Wrote AI insights text to {args.out_txt}", file=sys.stderr)


if __name__ == "__main__":
    main()

