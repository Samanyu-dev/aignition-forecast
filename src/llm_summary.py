"""
AI-assisted causal/anomaly narrative layer. Not part of run.sh's critical path
(feature-gen + predict must work with zero network access) -- this module is
used only by the FastAPI/Streamlit demo layer.

Design: never ask the LLM to invent numbers. Pre-compute structured statistics
(period-over-period deltas, spend elasticity per segment, anomalous campaigns
by ROAS z-score, and budget-scenario deltas) and ask Claude to interpret them.

If ANTHROPIC_API_KEY is unset, falls back to a deterministic template built
from the same stats dict, so the app runs fully offline. The template path
was the one exercised in this environment (no live API key available at
build time); the Claude call path is implemented against the current
Messages API but was validated structurally, not against a live request --
see docs/TECHNICAL_DOC.md limitations.
"""
import os
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
RECENT_WINDOW_DAYS = 30
ANOMALY_Z_THRESHOLD = 2.0
ANOMALY_LOOKBACK_DAYS = 60


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
            "n_obs": seg["elasticity_n_obs"],
            "low_confidence": seg["elasticity_n_obs"] < 10,
        })
    return sorted(out, key=lambda r: r["elasticity_beta"])


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


def compute_stats(
    features: pd.DataFrame,
    model: dict,
    baseline_forecast: pd.DataFrame,
    scenario_forecast: Optional[pd.DataFrame] = None,
) -> dict:
    stats = {
        "period_deltas": _period_deltas(features),
        "elasticity": _elasticity_summary(model),
        "anomalous_campaigns": _anomalous_campaigns(features),
    }
    if scenario_forecast is not None:
        stats["budget_scenario_deltas"] = _budget_scenario_deltas(baseline_forecast, scenario_forecast)
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
        "You are a marketing analytics assistant. Below is pre-computed JSON data about an "
        "e-commerce advertiser's blended Google/Bing/Meta performance: period-over-period "
        "revenue and spend deltas, per-segment spend elasticity, anomalous campaigns by ROAS "
        "z-score, and (if present) a budget-scenario comparison.\n\n"
        "Do not invent or restate numbers beyond what's given. Write:\n"
        "1. A 3-5 sentence narrative summary of what's happening and why it matters operationally.\n"
        "2. A short bulleted list of risk flags or operational recommendations.\n\n"
        f"DATA:\n{json.dumps(stats, indent=2)}"
    )
    response = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    parts = text.split("\n\n", 1)
    narrative = parts[0].strip()
    risk_section = parts[1].strip() if len(parts) > 1 else ""
    risk_flags = [
        line.lstrip("-* ").strip()
        for line in risk_section.splitlines()
        if line.strip().startswith(("-", "*"))
    ]
    return {"text": narrative, "risk_flags": risk_flags}


def generate_causal_summary(
    features: pd.DataFrame,
    model: dict,
    baseline_forecast: pd.DataFrame,
    scenario_forecast: Optional[pd.DataFrame] = None,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    stats = compute_stats(features, model, baseline_forecast, scenario_forecast)
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
