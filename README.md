# AIgnition 3.0 — Probabilistic Revenue Forecasting

An AI-assisted forecasting utility for digital marketing agencies: ingests
Google Ads, Microsoft (Bing) Ads, and Meta Ads data and produces genuinely
probabilistic (P10/P50/P90) revenue and ROAS forecasts — blended, per
channel, per campaign_type, and per individual campaign — with budget
scenario simulation and an AI-generated causal narrative. **The two things
that make it credible, not just functional:** it caught that Meta's revenue
column is silently mislabeled as a conversion count (proven statistically,
not assumed — see §2.2 of `docs/TECHNICAL_DOC.md`), and its "probabilistic"
claim is walk-forward backtested rather than asserted — including the
honest finding that the first model version was badly overconfident (37.2%
actual P10–P90 coverage vs. 80% nominal), and later catching a
double-dipping flaw in our own calibration validation that had made a fix
look better than it was (an optimistic 57.8% became an honest 37.8% once
evaluated on genuinely held-out data — §3.7–3.8).

See `docs/DEMO_WORKFLOW.md` for an end-to-end walkthrough with real captured
output (data ingestion → forecast → budget simulation → AI insight).

## Python version

Python 3.9 (developed and tested on 3.9.6).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./run.sh                                        # defaults: ./data ./pickle/model.pkl ./output/predictions.csv
# or explicitly:
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

This generates features from `data/` and writes predictions to
`output/predictions.csv` using the pre-trained model committed at
`pickle/model.pkl`. No retraining happens at run time and no network access
is required — see [Data & Model Contract](#data--model-contract) below.

### Optional: interactive demo (budget simulation + AI narrative)

```bash
# terminal 1
uvicorn src.api:app --reload --port 8000

# terminal 2
streamlit run app_streamlit.py
```

Set `ANTHROPIC_API_KEY` in the environment to get live Claude-generated
causal narratives; without it, the app falls back to a deterministic
template built from the same underlying statistics (see
`src/llm_summary.py`). Either way the app runs fully offline-safe.

## Repo layout

```
aignition-forecast/
├── run.sh                    # required entry point: feature-gen + predict, no retraining, no network
├── requirements.txt          # pinned dependencies
├── data/                     # sample CSVs (overwritten with held-out data at grading time)
├── pickle/model.pkl          # pre-trained model artifact (committed)
├── src/
│   ├── generate_features.py  # ingest + normalize the 3 channel CSVs
│   ├── train.py               # fits pickle/model.pkl — NOT called by run.sh
│   │                          #   (method selection, holiday/clipping, elasticity CI,
│   │                          #    per-segment calibration, campaign-level fitting)
│   ├── backtest_core.py       # shared walk-forward evaluation core (used by train.py + backtest.py)
│   ├── backtest.py            # standalone empirical-vs-Holt-Winters backtest CLI (offline tool)
│   ├── forecasting.py         # Monte Carlo aggregation + budget elasticity (shared by predict.py and api.py)
│   ├── predict.py             # run.sh's second step: loads model, writes predictions.csv
│   ├── llm_summary.py         # Claude-assisted causal narrative (demo layer only)
│   └── api.py                 # FastAPI /forecast endpoint (demo layer only)
├── app_streamlit.py           # Streamlit demo UI (demo layer only)
├── output/predictions.csv     # generated fresh each run.sh invocation
└── docs/
    ├── TECHNICAL_DOC.md               # methodology, backtest results, assumptions, limitations, AI strategy
    ├── ARCHITECTURE.md                # frontend/backend/pipeline/LLM workflow
    ├── DEMO_WORKFLOW.md               # ingestion -> forecast -> budget sim -> AI insight, with real captured output
    ├── backtest_results_baseline.json # pre-Step-2 walk-forward backtest (single method)
    ├── backtest_results.json          # empirical-vs-Holt-Winters method comparison
    └── backtest_results_post_fix.json # post-fix (winning method + calibration) walk-forward results
```

## Data & model contract

- `run.sh`'s critical path is **feature generation + prediction only** — no
  retraining, no network calls. `src/train.py` produced the committed
  `pickle/model.pkl` and is not invoked at run/grading time.
- `src/generate_features.py` reads `data/` **by filename pattern**
  (case-insensitive substring match on `bing`/`google`/`meta`), not by
  hardcoded row counts, so it works unchanged against held-out data with the
  same schema.
- `output/predictions.csv` columns:
  `channel,campaign_type,campaign_id,horizon_days,metric,p10,p50,p90`
  (`metric` ∈ `{revenue, roas}`; blank `campaign_type`/`campaign_id` mark
  channel/blended rollups). Confirmed against the actual AIgnition brief: the
  "Date Link — AIgnition_dataset" resource is the same three CSVs already in
  `data/`, not a separate output-schema spec — the brief specifies required
  outputs qualitatively (channel/campaign_type/campaign-level revenue + ROAS
  ranges, probabilistic not deterministic) and this schema satisfies that. No
  external format to mismatch against.
- Random seed is fixed (`seed=42` in `src/forecasting.py`, `src/train.py`)
  everywhere Monte Carlo sampling occurs.

See `docs/TECHNICAL_DOC.md` for methodology, the Meta data-validation
finding, and full assumptions/limitations.
