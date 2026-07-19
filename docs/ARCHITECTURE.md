# Architecture Overview

## Diagram

```
                         ┌────────────────────────┐
data/*.csv  ───────────▶ │ src/generate_features.py│
(Bing/Google/Meta)       │  normalize + Meta       │
                         │  data-validation logic  │
                         └───────────┬─────────────┘
                                     │ features.parquet
                                     ▼
                         ┌────────────────────────┐
                         │   src/train.py          │  (offline, not in run.sh)
                         │  backtest_core.py walk- │
                         │  forward selects method │
                         │  (empirical vs Holt-    │
                         │  Winters) + calibration │
                         │  scale per segment, then│
                         │  fits campaign_type AND │
                         │  campaign-level models  │
                         └───────────┬─────────────┘
                                     │ pickle/model.pkl  (committed)
                                     ▼
        run.sh   ──────▶ ┌────────────────────────┐
   (critical path,       │   src/predict.py         │──▶ output/predictions.csv
    no network)          │  loads model, calls      │
                         │  forecasting.forecast()  │
                         └───────────┬─────────────┘
                                     │
                                     ▼
                         ┌────────────────────────┐
                         │  src/forecasting.py      │  shared core:
                         │  Monte Carlo aggregation │  Monte Carlo sampling,
                         │  + budget elasticity     │  budget-multiplier scenarios,
                         └───────────┬─────────────┘  P10/P50/P90 rollups
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                        ▼
      output/predictions.csv                   ┌────────────────────┐
      (baseline, multiplier=1.0)                │   src/api.py         │  FastAPI
                                                 │   POST /forecast     │  (demo layer,
                                                 └──────────┬──────────┘   not in run.sh)
                                                            │
                                        ┌───────────────────┴───────────────────┐
                                        ▼                                        ▼
                             ┌────────────────────┐                 ┌────────────────────┐
                             │ src/llm_summary.py   │                │ app_streamlit.py     │
                             │ compute_stats() +    │                │ forecast chart,      │
                             │ Claude API narrative │◀───────────────│ budget sliders,       │
                             │ (template fallback   │   narrative    │ contribution table,   │
                             │  if no API key)       │   JSON         │ AI narrative panel    │
                             └───────────┬──────────┘                └────────────────────┘
                                         │
                                         ▼
                              Anthropic Messages API
                              (claude-opus-4-8, configurable
                               via ANTHROPIC_MODEL)
```

## Frontend

**Streamlit** (`app_streamlit.py`). Chosen over a custom React/Next.js app
for time-to-demo: one file, no build step, native support for the
interaction pattern this challenge needs (sliders → re-fetch → re-render).
Renders:
- Blended revenue forecast chart with a shaded P10–P90 band (Altair
  `mark_area` + `mark_line`), baseline vs. budget-scenario overlay.
- Per-horizon revenue/ROAS delta metrics (scenario vs. current run-rate).
- A channel/campaign-type contribution table (pivoted P50 revenue), plus an
  expandable top-15 individual-campaign table.
- The AI-generated causal narrative + risk-flag list, with a visible badge
  indicating whether it came from the live Claude API or the offline
  template fallback.
- A **model reliability panel**: the same per-segment walk-forward backtest
  table (method, MAPE, P10–P90 coverage, low-reliability flag) that drives
  the narrative's caution language, plus a MAPE-vs-coverage scatter chart —
  so a judge can see the calibration evidence directly, not just take the
  narrative's word for it.

## Backend

**FastAPI** (`src/api.py`), a single `POST /forecast` endpoint parameterized
by channel/campaign_type filters, horizon list, per-channel budget
multipliers, and an `include_narrative` flag. Loads the pickled model and
the normalized feature table once at startup; each request calls the shared
`forecasting.forecast()` core (no re-fitting). This is a demo/API layer only
— it is not part of `run.sh`'s critical path and is never required for
`predictions.csv` to be produced.

## Forecasting pipeline

`src/generate_features.py` → `src/train.py` (offline; uses
`src/backtest_core.py` for method selection and calibration) →
`src/forecasting.py` (shared by `predict.py` and `api.py`) → `src/predict.py`
(writes `predictions.csv`). `src/backtest.py` is a separate standalone CLI
for the empirical-vs-Holt-Winters comparison report — offline tooling, not
imported by `run.sh`, `train.py`, or `predict.py`'s critical path (`train.py`
uses `backtest_core.py` directly). See `docs/TECHNICAL_DOC.md` §3 for the
full modeling methodology: dual-method selection by walk-forward pinball
loss, trend + day-of-week + holiday-window seasonality, empirical residual
bootstrap Monte Carlo, log-log spend elasticity with bootstrap CI, and the
per-segment coverage-calibration search.

## LLM integration workflow

`src/llm_summary.py` sits strictly downstream of the numeric forecasting
pipeline and is called only from `src/api.py` (never from `run.sh` /
`predict.py`, satisfying the no-network-at-grading-time constraint).
`compute_stats()` pre-computes period-over-period deltas, per-segment
elasticity with bootstrap CIs, anomalous campaigns by ROAS z-score,
per-segment walk-forward backtest reliability, a live structural
zero-revenue-campaign check, and budget-scenario deltas as structured JSON;
`generate_causal_summary()` sends that JSON — never raw numbers for Claude to
reconstruct or invent — to the Anthropic Messages API for a narrative +
risk-flag list, falling back to a deterministic template built from the same
stats when no API key is present or the call fails for any reason. This
design means the LLM can never fabricate a number that isn't already in
`compute_stats()`'s output, and it can explicitly reason about *how much to
trust* a given segment's forecast, not just what the forecast says.
