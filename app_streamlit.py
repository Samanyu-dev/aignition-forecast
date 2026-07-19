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
st.sidebar.caption("Multiplier on each channel's current daily spend run-rate.")
channels = ["bing", "google", "meta"]
multipliers = {c: st.sidebar.slider(c.capitalize(), 0.25, 3.0, 1.0, 0.05) for c in channels}
horizons = st.sidebar.multiselect("Horizons (days)", [30, 60, 90], default=[30, 60, 90])

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

st.subheader("Blended revenue forecast")
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

st.subheader("Scenario impact")
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

st.subheader("Channel / campaign-type contribution")
segment_data = call_forecast(None, None, horizons, multipliers, False)
seg_df = pd.DataFrame(segment_data["forecast"])
seg_rev = seg_df[(seg_df.metric == "revenue") & (seg_df.channel != "blended")]
pivot = seg_rev.pivot_table(
    index=["channel", "campaign_type"], columns="horizon_days", values="p50", aggfunc="first"
).round(0)
st.dataframe(pivot, width="stretch")

st.subheader("AI-assisted causal narrative")
source_badge = {"llm": "🟢 Claude API", "template": "⚪ offline template (no ANTHROPIC_API_KEY)",
                 "template_fallback": "🟡 template (LLM call failed)"}
st.caption(source_badge.get(scenario_data.get("narrative_source"), scenario_data.get("narrative_source", "")))
st.write(scenario_data.get("narrative", ""))
risk_flags = scenario_data.get("risk_flags", [])
if risk_flags:
    st.markdown("**Risk flags:**")
    for flag in risk_flags:
        st.markdown(f"- {flag}")
