# Demo Workflow

This walkthrough follows the brief's four required beats — **data ingestion
→ forecast generation → budget simulation → AI-generated business insight**
— using real, reproducible output captured directly from this repo (not
mocked). Every command below can be re-run verbatim from a fresh clone.

A live, interactive version of steps 2–4 is also available:

```bash
uvicorn src.api:app --reload --port 8000   # terminal 1
streamlit run app_streamlit.py             # terminal 2
```

That gives budget-simulation sliders, a live forecast chart, the AI
narrative panel, and the model-reliability panel described below, all in
one page. This document is the fallback walkthrough — no screen-recording
tooling was available in the build environment to produce a video, so this
is the "short markdown/slide walkthrough with screenshots" form of the
deliverable instead.

---

## 1. Data ingestion

```bash
python src/generate_features.py --data-dir data --out /tmp/features.parquet
```

**Actual output:**

```
Loading Bing from data/bing_campaign_stats.csv
Loading Google from data/google_ads_campaign_stats.csv
Loading Meta from data/meta_ads_campaign_stats.csv
Wrote 25562 rows to /tmp/features.parquet
```

This step ingests all three raw CSVs (Bing: 2,873 rows, Google: 19,272 rows,
Meta: 3,417 rows), normalizes each platform's schema into one long table,
fixes Google's cost-micros scaling, and applies the data-validation finding
from `docs/TECHNICAL_DOC.md` §2.2: Meta's `conversion` column is treated as
revenue directly, based on evidence (fractional currency-like values,
`sum(conversion) > sum(clicks)`, and a matching ROAS-distribution shape
against Bing/Google) rather than assumption. 25,562 total normalized rows
come out the other end, ready for modeling.

---

## 2. Forecast generation

```bash
./run.sh
```

**Actual output:**

```
Loading Bing from ./data/bing_campaign_stats.csv
Loading Google from ./data/google_ads_campaign_stats.csv
Loading Meta from ./data/meta_ads_campaign_stats.csv
Wrote 25562 rows to /var/folders/.../aignition_features_XXXXXX.parquet
Loading features from /var/folders/.../aignition_features_XXXXXX.parquet
Loading model from ./pickle/model.pkl
Forecasting horizons (30, 60, 90) (seed=42)
Wrote 741 rows to ./output/predictions.csv
Done. Predictions written to ./output/predictions.csv
```

This is the actual submission entry point — feature generation and
prediction in one invocation, no retraining, no network access, using the
pre-trained `pickle/model.pkl` (17 campaign_type segments + 104 individual
campaigns, each fit with whichever of two methods — a custom trend/seasonal
decomposition or Holt-Winters — backtested better; see
`docs/TECHNICAL_DOC.md` §3). The blended (all-channel) forecast rows:

```
channel,campaign_type,campaign_id,horizon_days,metric,p10,p50,p90
blended,,,30,revenue,297496.86,357208.23,426096.97
blended,,,30,roas,3.84,4.61,5.49
blended,,,60,revenue,645456.08,731498.56,827823.38
blended,,,60,roas,4.16,4.72,5.34
blended,,,90,revenue,1012068.54,1124128.17,1241520.89
blended,,,90,roas,4.35,4.83,5.34
```

Reading this: 90-day blended revenue is expected around **$1.12M (P50)**,
with a genuine (walk-forward-validated, not asserted) probabilistic range of
**$1.01M–$1.24M**, at a blended ROAS around **4.8x**.

---

## 3. Budget simulation

Using `forecasting.forecast()` directly (the same function `src/api.py`
calls) with a budget scenario — Meta spend ×2.0, Google spend ×1.2, Bing
unchanged:

```python
from forecasting import forecast
baseline = forecast(model)
scenario = forecast(model, budget_multipliers={"meta": 2.0, "google": 1.2})
```

**Actual output:**

```
Scenario: Meta budget x2.0, Google budget x1.2, Bing unchanged

 horizon    baseline P50    scenario P50      delta
      30         357,208         457,377      28.0%
      60         731,499         929,800      27.1%
      90       1,124,128       1,428,208      27.1%
```

Note this is **not** a linear rescale — doubling Meta spend doesn't double
Meta's contribution, because the scaling uses each segment's fitted spend→
revenue elasticity (with a bootstrap confidence interval; segments where
the CI is too wide fall back to a conservative linear assumption rather than
trusting a noisy estimate — see `docs/TECHNICAL_DOC.md` §3.4). The ~27–28%
revenue lift from a much larger spend increase is the diminishing-returns
curve showing up correctly.

---

## 4. AI-generated business insight

Same scenario, run through `src/llm_summary.py::generate_causal_summary()`
(offline template path — no `ANTHROPIC_API_KEY` was available in the build
environment; the live Claude API path is implemented and used automatically
when a key is present, with this template as the always-available fallback):

**Actual output:**

> Bing revenue is up 41.8% over the trailing 30 days ($6,105 vs $4,304). Meta
> revenue is up 31.6% over the trailing 30 days ($43,398 vs $32,976). Google
> revenue is down 13.5% over the trailing 30 days ($293,298 vs $339,059).
> Google/VIDEO shows near-linear returns to spend (elasticity 1.1),
> suggesting room to scale budget efficiently. Bing/Search shows strong
> diminishing returns (elasticity 0.11) — additional spend there is less
> efficient. Campaign 'Shopping_NTM_Campaign_01' (bing) is outperforming its
> channel's typical ROAS (ROAS 18.77, z=2.02). Walk-forward backtesting
> flags 7 segment(s) as lower-reliability forecasts (wide backtested error
> or thin interval coverage): bing/Shopping, google/PERFORMANCE_MAX,
> google/VIDEO, meta/Prospecting_DPA, meta/Remarketing_Brand. Treat their
> P10-P90 ranges as more approximate than the rest of the portfolio. Bing
> has 18 of 28 spending campaigns (64.3%) with zero lifetime revenue,
> representing $2,594 of spend with no measured return — a structural issue
> (tracking, targeting, or a dead campaign left running), not routine
> variance. Under the simulated budget scenario, 30-day revenue moves
> +28.0% ($357,208 → $457,377)...

**Risk flags:**
- 5 segment(s) have too few spend/revenue observations for a reliable
  elasticity estimate; treat their budget-scenario response as low-confidence.
- 7 segment(s) failed to backtest reliably — see `docs/TECHNICAL_DOC.md` §3.8.
- Bing: 64.3% of spending campaigns show zero lifetime revenue ($2,594
  wasted spend) — investigate before trusting this channel's forecast.

Nothing in this narrative is invented — `compute_stats()` computes every
number above from the actual data before either the LLM or the template
ever sees it (see `docs/ARCHITECTURE.md` § LLM integration workflow). The
64.3%-zero-revenue-campaigns finding for Bing is computed live from
whatever data is loaded, not hardcoded — it independently rediscovers a
finding first surfaced while reviewing an earlier exploration of this same
dataset, purely from the numbers.

---

## What this demonstrates end-to-end

1. **Ingestion** correctly normalizes three genuinely different ad-platform
   schemas and catches a real data-quality issue (Meta's mislabeled column)
   through statistical evidence, not assumption.
2. **Forecasting** produces genuinely probabilistic (not just point-estimate)
   ranges, validated by walk-forward backtesting rather than asserted (see
   `docs/TECHNICAL_DOC.md` §3.7–3.8 for the honest 37.2%→57.8% calibration
   story).
3. **Budget simulation** responds non-linearly and per-segment, using
   confidence-gated elasticity rather than a single global multiplier.
4. **AI insight** synthesizes all of the above — deltas, elasticity,
   anomalies, backtest reliability, and a structural risk check — into an
   operator-readable narrative with explicit risk flags, entirely from
   pre-computed numbers the LLM cannot hallucinate around.
