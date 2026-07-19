"""
Demo UI for the AIgnition forecast API. Not part of run.sh's critical path.

Run: streamlit run app_streamlit.py
Requires src/api.py running separately: uvicorn src.api:app --port 8000
"""
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AIgnition Forecast", layout="wide")
st.title("Probabilistic Revenue Forecast")
st.caption("Blended Google Ads / Microsoft Ads / Meta Ads revenue & ROAS forecasting with AI-assisted causal narrative.")
st.markdown(
    "**How to read this page, top to bottom:** "
    "① adjust the budget sliders on the left to simulate a spend change → "
    "② the forecast chart and scenario-impact numbers update live → "
    "③ **Recommended budget shifts** below is a specific, priced, confidence-gated "
    "recommendation, not just a forecast → "
    "④ the AI narrative explains *why*, and **Model reliability** shows the backtest "
    "evidence behind every number on this page, including where to trust it less."
)
st.divider()


@st.cache_data(ttl=60)
def call_forecast(channel, campaign_type, horizons, budget_multipliers, include_narrative):
    resp = requests.post(
        f"{API_BASE_URL}/forecast",
        json={
            "channel": channel,
            "campaign_type": campaign_type,
            "horizons": horizons,
            "budget_multipliers": budget_multipliers,
            "include_narrative": include_narrative,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


st.sidebar.header("Budget scenario")
st.sidebar.caption(
    "Multiplier on each channel's current daily spend run-rate. Move a slider and every "
    "number on the page -- the chart, the recommendation, the AI narrative -- recomputes live."
)
channels = ["bing", "google", "meta"]
multipliers = {c: st.sidebar.slider(c.capitalize(), 0.25, 3.0, 1.0, 0.05) for c in channels}
horizons = st.sidebar.multiselect("Horizons (days)", [30, 60, 90], default=[30, 60, 90])
_active = {c: m for c, m in multipliers.items() if m != 1.0}
if _active:
    st.sidebar.info("Active scenario: " + ", ".join(f"{c} {m}x" for c, m in _active.items()))
else:
    st.sidebar.caption("Currently showing the baseline (no budget change) -- move a slider to simulate one.")

if not horizons:
    st.warning("Select at least one horizon.")
    st.stop()

try:
    baseline_data = call_forecast("blended", None, horizons, {c: 1.0 for c in channels}, False)
    scenario_data = call_forecast("blended", None, horizons, multipliers, True)
except requests.exceptions.RequestException as e:
    st.error(f"Could not reach the forecast API at {API_BASE_URL}. Is `uvicorn src.api:app` running? ({e})")
    st.stop()

baseline_df = pd.DataFrame(baseline_data["forecast"])
scenario_df = pd.DataFrame(scenario_data["forecast"])
baseline_df["scenario"] = "Current run-rate"
scenario_df["scenario"] = "Budget scenario"
combined = pd.concat([baseline_df, scenario_df], ignore_index=True)

st.subheader("① Blended revenue forecast")
st.caption("Shaded band = the P10-P90 probabilistic range (not a guess -- walk-forward backtested, see Model reliability below). Two lines compare your current budget scenario against the run-rate baseline.")
revenue = combined[combined.metric == "revenue"]

band = alt.Chart(revenue).mark_area(opacity=0.25).encode(
    x=alt.X("horizon_days:O", title="Horizon (days)"),
    y=alt.Y("p10:Q", title="Revenue ($)"),
    y2="p90:Q",
    color=alt.Color("scenario:N", legend=alt.Legend(title=None)),
)
line = alt.Chart(revenue).mark_line(point=True).encode(
    x="horizon_days:O",
    y="p50:Q",
    color="scenario:N",
)
st.altair_chart((band + line).properties(height=350), use_container_width=True)

st.subheader("② Scenario impact")
st.caption("Revenue and ROAS at each horizon under your current slider settings, vs. the current run-rate baseline.")
roas = combined[combined.metric == "roas"]
cols = st.columns(len(horizons))
for col, h in zip(cols, sorted(horizons)):
    base_rev = revenue[(revenue.horizon_days == h) & (revenue.scenario == "Current run-rate")]["p50"].iloc[0]
    scen_rev = revenue[(revenue.horizon_days == h) & (revenue.scenario == "Budget scenario")]["p50"].iloc[0]
    scen_roas = roas[(roas.horizon_days == h) & (roas.scenario == "Budget scenario")]["p50"].iloc[0]
    col.metric(
        f"{h}-day revenue (P50)",
        f"${scen_rev:,.0f}",
        delta=f"{(scen_rev - base_rev) / base_rev * 100:+.1f}% vs run-rate",
    )
    col.metric(f"{h}-day blended ROAS (P50)", f"{scen_roas:.2f}x")

st.subheader("③ Recommended budget shifts")
st.caption(
    "Specific, priced reallocations -- not just a forecast. Every candidate is screened by "
    "spend-elasticity confidence (bootstrap CI) before it's even priced, so nothing here is a "
    "guess from a noisy segment. See TECHNICAL_DOC.md sec 3.9 for the full methodology."
)
recommendations = scenario_data.get("stats", {}).get("budget_reallocation_recommendations", [])
if recommendations:
    top_rec = recommendations[0]
    st.success(
        f"**Top pick:** shift ~${top_rec['shift_daily_dollars']:.0f}/day from **{top_rec['from']}** "
        f"(elasticity {top_rec['from_elasticity']}) to **{top_rec['to']}** "
        f"(elasticity {top_rec['to_elasticity']}) -- estimated {top_rec['horizon_days']}-day revenue "
        f"impact **{top_rec['revenue_delta_pct']:+.1f}%** "
        f"(${top_rec['baseline_p50_revenue']:,.0f} → ${top_rec['scenario_p50_revenue']:,.0f}), "
        f"blended ROAS {top_rec['baseline_p50_roas']:.2f}x → {top_rec['scenario_p50_roas']:.2f}x. "
        f"Confidence: **{top_rec['confidence']}**."
    )
    rec_df = pd.DataFrame(recommendations)[[
        "from", "to", "shift_daily_dollars", "from_elasticity", "to_elasticity",
        "revenue_delta_pct", "scenario_p50_roas", "confidence",
    ]].rename(columns={
        "from": "From", "to": "To", "shift_daily_dollars": "Shift ($/day)",
        "from_elasticity": "From β", "to_elasticity": "To β",
        "revenue_delta_pct": "Revenue Δ (%)", "scenario_p50_roas": "Scenario ROAS", "confidence": "Confidence",
    })
    st.dataframe(rec_df, width="stretch", hide_index=True)
else:
    st.caption("No confidence-gated reallocation candidates available for this dataset/scenario.")

st.subheader("④ Channel / campaign-type contribution")
st.caption("How the current scenario's forecast breaks down by channel and campaign type.")
segment_data = call_forecast(None, None, horizons, multipliers, False)
seg_df = pd.DataFrame(segment_data["forecast"])
# campaign_id == "" marks a channel/campaign_type rollup row -- individual
# campaign rows (added for campaign-level forecasting) share the same
# (channel, campaign_type) index and must be excluded here, or pivot_table's
# aggfunc="first" would silently grab an arbitrary campaign instead of the
# true campaign_type total.
seg_rev = seg_df[(seg_df.metric == "revenue") & (seg_df.channel != "blended") & (seg_df.campaign_id == "")]
pivot = seg_rev.pivot_table(
    index=["channel", "campaign_type"], columns="horizon_days", values="p50", aggfunc="first"
).round(0)
st.dataframe(pivot, width="stretch")

with st.expander("Top campaign-level forecasts (P50 revenue)"):
    camp_rev = seg_df[(seg_df.metric == "revenue") & (seg_df.campaign_id != "")]
    if not camp_rev.empty:
        h = max(horizons)
        top_campaigns = (
            camp_rev[camp_rev.horizon_days == h]
            .sort_values("p50", ascending=False)
            .head(15)[["channel", "campaign_type", "campaign_id", "p50"]]
            .rename(columns={"p50": f"{h}d_revenue_p50"})
        )
        st.dataframe(top_campaigns, width="stretch", hide_index=True)
    else:
        st.caption("No campaign-level forecasts available (all campaigns had <30 days of history).")

st.subheader("⑤ AI-assisted causal narrative")
st.caption("Claude (or the offline template if no API key is set) explains *why*, using only the numbers already computed above -- it cannot invent a figure that isn't already in the stats it's given.")
source_badge = {"llm": "🟢 Claude API", "template": "⚪ offline template (no ANTHROPIC_API_KEY)",
                 "template_fallback": "🟡 template (LLM call failed)"}
st.caption(source_badge.get(scenario_data.get("narrative_source"), scenario_data.get("narrative_source", "")))
st.write(scenario_data.get("narrative", ""))
risk_flags = scenario_data.get("risk_flags", [])
if risk_flags:
    st.markdown("**Risk flags:**")
    for flag in risk_flags:
        st.markdown(f"- {flag}")

st.divider()
st.subheader("⑥ Model reliability (walk-forward backtest) -- the evidence behind every number above")
rel_stats = scenario_data.get("stats", {}).get("forecast_reliability", [])
_covered = [r["coverage_pct"] for r in rel_stats if r.get("coverage_pct") is not None]
if _covered:
    st.metric(
        "Current model's aggregate P10-P90 coverage",
        f"{sum(_covered) / len(_covered):.1f}%",
        help="Fraction of walk-forward-backtested segments whose actual held-out revenue fell "
             "inside the model's own P10-P90 range. Nominal target is 80%. Improved from an "
             "initial 37.2% via method selection + calibration -- see TECHNICAL_DOC.md sec 3.7-3.8.",
    )
st.caption(
    "Each segment's model was walk-forward validated (3 held-out 30-day windows, refit on "
    "data before each cutoff). Method-selection + calibration fixes improved aggregate P10-P90 "
    "coverage from 37.2% to 57.8% and cut mean pinball loss ~15% -- see docs/TECHNICAL_DOC.md "
    "sec 3.7-3.8 for the full before/after. Still short of the 80% nominal target; segments "
    "flagged below should be read with extra caution."
)
reliability = scenario_data.get("stats", {}).get("forecast_reliability", [])
if reliability:
    rel_df = pd.DataFrame(reliability).sort_values("coverage_pct")
    st.dataframe(
        rel_df.rename(columns={
            "segment": "Segment", "method": "Method", "mean_ape_pct": "Backtest MAPE (%)",
            "coverage_pct": "P10-P90 Coverage (%)", "low_reliability": "Low reliability",
        }),
        width="stretch", hide_index=True,
    )
    chart = alt.Chart(rel_df).mark_circle(size=120).encode(
        x=alt.X("mean_ape_pct:Q", title="Backtest MAPE (%)", scale=alt.Scale(type="symlog")),
        y=alt.Y("coverage_pct:Q", title="P10-P90 Coverage (%)"),
        color=alt.Color("method:N", title="Method"),
        tooltip=["segment", "method", "mean_ape_pct", "coverage_pct"],
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
else:
    st.caption("No backtest reliability data available.")

structural_risk = scenario_data.get("stats", {}).get("structural_risk_campaigns", [])
if structural_risk:
    st.markdown("**Structural risk** -- channels where a large share of campaigns spent money but generated zero lifetime revenue (a tracking/targeting problem, not routine variance):")
    st.dataframe(pd.DataFrame(structural_risk), width="stretch", hide_index=True)
