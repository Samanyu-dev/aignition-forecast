# AIgnition 3.0 — Probabilistic Revenue Forecasting

An AI-assisted forecasting utility for digital marketing agencies. Ingests
Google Ads, Microsoft (Bing) Ads, and Meta Ads campaign data and produces
probabilistic (P10/P50/P90) revenue and ROAS forecasts over 30/60/90-day
horizons, at blended / channel / campaign-type granularity, with budget
scenario simulation and an AI-generated causal narrative.

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
│   ├── forecasting.py         # Monte Carlo aggregation + budget elasticity (shared by predict.py and api.py)
│   ├── predict.py             # run.sh's second step: loads model, writes predictions.csv
│   ├── llm_summary.py         # Claude-assisted causal narrative (demo layer only)
│   └── api.py                 # FastAPI /forecast endpoint (demo layer only)
├── app_streamlit.py           # Streamlit demo UI (demo layer only)
├── output/predictions.csv     # generated fresh each run.sh invocation
└── docs/
    ├── TECHNICAL_DOC.md        # methodology, assumptions, limitations, AI strategy
    └── ARCHITECTURE.md         # frontend/backend/pipeline/LLM workflow
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
  channel/blended rollups). **This schema was not published in the materials
  available at build time — verify against the official launch schema before
  the real submission and adjust `src/forecasting.py`'s output columns if
  needed.**
- Random seed is fixed (`seed=42` in `src/forecasting.py`, `src/train.py`)
  everywhere Monte Carlo sampling occurs.

See `docs/TECHNICAL_DOC.md` for methodology, the Meta data-validation
finding, and full assumptions/limitations.
