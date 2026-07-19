# Technical Documentation

## 1. Problem framing

Forecast blended e-commerce revenue and ROAS across Google Ads, Microsoft
(Bing) Ads, and Meta Ads, as aggregate-period probabilistic ranges (30/60/90
days), at blended / channel / campaign-type granularity, supporting
what-if budget simulation and an AI-generated causal narrative. Per the
challenge brief, existing per-channel attribution is treated as ground
truth — this is explicitly not an attribution-engine or MMM exercise.

## 2. Data preprocessing

### 2.1 Source schemas

| Channel | Rows | Date range | Revenue field | Spend field |
|---|---|---|---|---|
| Bing | 2,873 | 2024-05-25 → 2026-06-05 | `Revenue` | `Spend` |
| Google | 19,272 | 2024-01-01 → 2026-06-04 | `metrics_conversions_value` | `metrics_cost_micros` (÷1,000,000) |
| Meta | 3,417 | 2024-05-23 → 2026-06-05 | `conversion` (see §2.2) | `spend` |

All three are normalized (`src/generate_features.py`) into one long table:
`date, channel, campaign_id, campaign_name, campaign_type, funnel_stage,
spend, revenue, conversions, revenue_is_imputed`.

### 2.2 Data validation finding: Meta's `conversion` field

Meta's CSV has no column named revenue or conversion-value, only a column
named `conversion`. Naively treated as a conversion **count**, this produces
a nonsensical result: the sum of `conversion` (1.66M) exceeds total clicks
(540K), which is impossible for a count of converting users, and interpreting
it as a count and multiplying by a derived AOV yields an 800x+ implied ROAS.

Evidence that `conversion` is actually **conversion value (revenue)**,
mislabeled:

1. Values are fractional currency-like amounts (e.g. `34.99`, `40.99`,
   `120.00`) — not something an integer count column would ever produce.
2. `sum(conversion) > sum(clicks)` — impossible for a count.
3. Row-level `conversion / spend` produces a distribution (median ≈4.1,
   mean ≈11.2) in the same range and shape as Bing's and Google's **measured**
   `revenue / spend` ROAS (median 0–2.6, mean 7.8–9.8), whereas a genuine
   conversion-count-per-dollar rate would be one to two orders of magnitude
   smaller.

The pipeline therefore treats `meta.conversion` as revenue directly (no
transformation). This is documented here as a dataset-specific assumption,
adjustable if an updated data dictionary becomes available. Meta has no
genuine conversion-*count* field in this dataset; `conversions` is left
`NaN` for Meta rows rather than fabricated.

With this treatment, blended ROAS across the three channels lands in a
consistent, plausible range: Bing 4.36x, Google 4.76x, Meta 8.44x (full
history). No AOV-imputation step is needed.

### 2.3 Funnel-stage inference

`infer_funnel_stage()` buckets every row into `high_intent` (branded/
trademark search — name contains `_tm_` — or remarketing/retargeting) vs.
`prospecting` (everything else), derived from campaign-name conventions
shared across all three platforms (`Search_TM_...` / `Search_NTM_...` on
Bing/Google; `Remarketing_...` / `Prospecting_...` on Meta). This was
originally built to segment Meta's AOV imputation; it's retained as a
`funnel_stage` column even though imputation was dropped (§2.2), since it's
a useful, low-cost segment for downstream analysis.

## 3. Forecasting methodology

### 3.1 Why not Prophet / ARIMA

At the (channel × campaign_type) daily grain, several of the 17 segments are
sparse and zero-inflated (e.g. `bing/Audience` has 54 observed days, several
with zero spend). MLE-based seasonal models (Prophet, SARIMAX/Holt-Winters)
are numerically unstable on series like this within a hackathon timeframe —
convergence warnings, degenerate seasonal components. Given the time
constraint, robustness was prioritized over model sophistication.

### 3.2 Model: trend + day-of-week seasonality + empirical residual bootstrap

Per (channel, campaign_type) segment (`src/train.py`):

1. Aggregate to a daily series over the segment's full observed date range
   (missing days filled with 0 revenue/spend — a campaign that didn't run
   that day, not missing data).
2. Fit a linear trend (`numpy.polyfit`, degree 1) over the trailing 120 days
   (or all available days if fewer).
3. Compute a day-of-week seasonal multiplier as `mean(revenue | dow) /
   mean(revenue)` over the segment's full history.
4. Compute in-sample residuals (`actual − trend×seasonal`) over the trend
   window; store the residual array as an empirical noise pool.
5. Fit a log-log spend→revenue elasticity (`log(revenue) = α + β·log(spend)`
   via OLS on days with spend>0 and revenue>0), clipped to `[0, 2]`. Segments
   with <10 qualifying days fall back to `β=1.0` (linear pass-through, flagged
   as low-confidence downstream).

Forecasting (`src/forecasting.py`) is Monte Carlo, not closed-form: for each
future day in the horizon, draw `point = trend(day) × seasonal(day)` and add
a residual **bootstrapped with replacement from that segment's empirical
residual pool** (not assumed Gaussian — this handles the zero-inflation and
skew directly from the observed data). 2,000 simulated paths per segment per
horizon; daily values are summed across the horizon and across segments
(assuming independence — see Limitations), then P10/P50/P90 are taken from
the resulting distribution. `numpy.random.default_rng(seed=42)` is seeded
throughout for reproducibility.

**Uncertainty inflation:** Meta's residual bootstrap noise is scaled ×1.3
relative to Bing/Google. Even though §2.2 treats Meta's field as measured
revenue rather than imputed, there remains residual doubt about the schema
interpretation itself (a data dictionary that clarified the field would
retire this) — the inflation factor keeps that risk visible in Meta's
forecast intervals rather than hiding it behind a point estimate with the
same confidence as Bing/Google's directly-labeled revenue.

### 3.3 Budget scenario simulation

A per-channel budget multiplier (default 1.0 = "continue current run-rate")
scales the segment's simulated revenue by `multiplier ** elasticity_beta`
(the log-log elasticity's direct implication for a proportional spend
change) and scales planned spend by the same multiplier. This is a
diminishing-returns model, not a linear rescale: a channel with β<1 shows
revenue growing more slowly than spend under a budget increase, and vice
versa for β>1. `run.sh`'s output (`predictions.csv`) always uses the
default multiplier (1.0, i.e. baseline/no scenario) — the interactive
what-if capability is exposed through `src/api.py` / `app_streamlit.py`
only, per the no-network-in-run.sh constraint below.

### 3.4 Aggregation levels

`forecast()` returns rows for: each (channel, campaign_type) segment, each
channel rollup (sum across that channel's segments), and the blended total
(sum across all channels) — for each horizon × metric (`revenue`, `roas`).
Segments are summed elementwise per Monte Carlo draw, so the resulting
distributions properly propagate uncertainty rather than just summing point
estimates.

## 4. Assumptions

1. Meta's `conversion` field is conversion value, not count (§2.2).
2. Segments (channel × campaign_type combinations) are simulated
   independently — no cross-segment correlation is modeled. In reality,
   e.g. a platform-wide seasonal spike likely correlates Google and Bing
   Search simultaneously; treating them as independent likely
   **understates** the width of the blended P10–P90 band.
3. Elasticity is estimated from **historical, not experimental**, spend/
   revenue covariation (no randomized budget experiments in the data) — it
   captures correlation, not necessarily a causal spend effect. Segments
   with <10 qualifying observations fall back to β=1.0 and are flagged
   low-confidence in the narrative layer.
4. Day-of-week seasonality only; no monthly/holiday seasonality modeling,
   given the aggregate 30/60/90-day forecast horizon specified by the brief.
5. `output/predictions.csv`'s column schema (§below) is our own proposal —
   the exact schema announced at the AIgnition launch was not available in
   the materials provided at build time. **This must be verified against the
   official schema before final submission.**

## 5. Limitations

- No cross-segment/cross-channel correlation in the Monte Carlo aggregation
  (assumption 2 above) — blended intervals are likely somewhat too narrow.
- Elasticity is correlational, estimated on ~90–580 days of naturally
  varying spend per segment, not from held-out or experimental data.
- Campaign-level (not just campaign_type-level) forecasts and anomaly
  detection were not built given the time budget; `src/llm_summary.py`'s
  anomaly detector operates at the individual-campaign grain for the
  narrative layer only, not in `predictions.csv`.
- The LLM causal-narrative layer (`src/llm_summary.py`) was validated in
  this environment against its **deterministic template fallback path**
  only (no `ANTHROPIC_API_KEY` was available at build time) — the
  Claude-API call path (`_call_claude`) is implemented against the current
  Messages API (model configurable via `ANTHROPIC_MODEL`, default
  `claude-opus-4-8`) but was not exercised against a live request. It is
  wrapped in a broad exception handler that falls back to the template on
  any failure, so this does not risk `run.sh` or the demo layer breaking —
  but it should be smoke-tested with a real key before relying on it in a
  live demo.
- No holdout/backtest evaluation of forecast accuracy was performed —
  reasonable given the historical data spans into "future" dates relative
  to a typical evaluation cutoff, but worth calling out as unverified.

## 6. AI integration strategy

The LLM (Claude, via the Anthropic Python SDK) is deliberately **not** asked
to forecast numbers — it is asked to *interpret pre-computed statistics*, to
avoid hallucinated figures:

- Period-over-period (trailing 30 days vs. prior 30 days) revenue/spend
  deltas per channel.
- Per-segment spend elasticity, with a low-confidence flag for
  under-observed segments.
- Anomalous campaigns by ROAS z-score within their channel (60-day lookback).
- Budget-scenario deltas (baseline vs. simulated blended revenue per
  horizon), when a scenario is active.

`src/llm_summary.py`'s `compute_stats()` produces this structured JSON;
`generate_causal_summary()` either sends it to Claude for a narrative +
risk-flag list, or — if `ANTHROPIC_API_KEY` is unset, or the API call fails
for any reason — falls back to a deterministic template built from the
same stats dict. The app therefore always runs, fully offline if needed,
with the LLM as a strict enhancement layer rather than a dependency.
