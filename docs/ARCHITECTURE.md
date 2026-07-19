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
                         │  trend + seasonality +  │
                         │  elasticity per segment │
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
- A channel/campaign-type contribution table (pivoted P50 revenue).
- The AI-generated causal narrative + risk-flag list, with a visible badge
  indicating whether it came from the live Claude API or the offline
  template fallback.

## Backend

**FastAPI** (`src/api.py`), a single `POST /forecast` endpoint parameterized
by channel/campaign_type filters, horizon list, per-channel budget
multipliers, and an `include_narrative` flag. Loads the pickled model and
the normalized feature table once at startup; each request calls the shared
`forecasting.forecast()` core (no re-fitting). This is a demo/API layer only
— it is not part of `run.sh`'s critical path and is never required for
`predictions.csv` to be produced.

## Forecasting pipeline

`src/generate_features.py` → `src/train.py` (offline) → `src/forecasting.py`
(shared by `predict.py` and `api.py`) → `src/predict.py` (writes
`predictions.csv`). See `docs/TECHNICAL_DOC.md` §3 for the modeling
methodology (trend + day-of-week seasonality + empirical residual bootstrap
Monte Carlo, log-log spend elasticity).

## LLM integration workflow

`src/llm_summary.py` sits strictly downstream of the numeric forecasting
pipeline and is called only from `src/api.py` (never from `run.sh` /
`predict.py`, satisfying the no-network-at-grading-time constraint).
`compute_stats()` pre-computes period-over-period deltas, per-segment
elasticity with confidence flags, anomalous campaigns by ROAS z-score, and
budget-scenario deltas as structured JSON; `generate_causal_summary()` sends
that JSON — never raw numbers for Claude to reconstruct or invent — to the
Anthropic Messages API for a narrative + risk-flag list, falling back to a
deterministic template built from the same stats when no API key is present
or the call fails for any reason. This design means the LLM can never
fabricate a number that isn't already in `compute_stats()`'s output.
