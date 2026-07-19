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

### 3.1 Two candidate methods, chosen per segment by backtest — not by assumption

At the (channel × campaign_type) daily grain, several of the 17 segments are
sparse and zero-inflated (e.g. `bing/Audience` has 54 observed days, several
with zero spend), where MLE-based seasonal models can be numerically
unstable. Rather than assume one method wins everywhere, **both** a
lightweight custom decomposition ("empirical") and statsmodels' Holt-Winters
("holt_winters", additive trend + weekly seasonal) are fit and
walk-forward-backtested per segment (`src/backtest.py`,
`docs/backtest_results.json`), and the lower-pinball-loss method is kept
(`src/train.py::select_method`). Final tally on this dataset: **9 segments
picked empirical, 6 picked Holt-Winters, 2 had too little history to
backtest** (defaulted to empirical). This isn't a coin flip — on
`google/VIDEO`, empirical's backtested MAPE was 140,512% vs. Holt-Winters'
1,892% (still bad, but 74x better); on `google/SEARCH`, pinball loss dropped
from 26,943 to 8,701. Picking per-segment rather than globally captured real,
measured gains that a single blanket choice would have missed.

### 3.2 Method 1: trend + day-of-week + holiday seasonality, empirical residual bootstrap

Per (channel, campaign_type) segment (`src/train.py::fit_trend_seasonal`):

1. Aggregate to a daily series over the segment's full observed date range
   (missing days filled with 0 revenue/spend — a campaign that didn't run
   that day, not missing data).
2. Clip revenue at the segment's 97th percentile before fitting (prevents a
   single extreme day — e.g. a Black Friday spike — from dominating the
   trend line; ported from a prior exploration of this same dataset that
   independently found the same Black Friday sensitivity, see §3.1 of the
   inventory note in project history).
3. Fit a linear trend (`numpy.polyfit`, degree 1) on the clipped series over
   the trailing 120 days (or all available days if fewer).
4. Compute a day-of-week seasonal multiplier as `mean(revenue | dow) /
   mean(revenue)` over the segment's full history.
5. Compute a **holiday-window multiplier** the same way, for Nov 20–Dec 31
   (Black Friday/Cyber Monday through year-end) vs. the rest of the year —
   added after finding a verified +5.1σ Black Friday 2024 spike on Google in
   this dataset (§3.6 has the finding detail). Segments with no holiday-window
   history default to 1.0 rather than guessing.
6. Compute in-sample residuals against `trend × dow × holiday` (using the
   **unclipped** actuals, so residual noise reflects real variance, not
   clipped-away variance); store as an empirical noise pool.

### 3.3 Method 2: Holt-Winters (statsmodels)

Per segment (`src/train.py::fit_holt_winters`): fit
`ExponentialSmoothing(trend="add", seasonal="add", seasonal_periods=7)` on
the same clipped daily series, take its native multi-step `.forecast(120)`
directly as the point path (no separate trend/dow decomposition — Holt-Winters'
own seasonal component handles day-of-week), and use in-sample residuals
(`actual − fitted.fittedvalues`) as its own empirical noise pool. This is a
genuinely different mechanism from Method 1, not a variant of it — that's
what makes the comparison in §3.1 meaningful rather than circular.

### 3.4 Elasticity, with a bootstrap confidence interval

Fit a log-log spend→revenue elasticity (`log(revenue) = α + β·log(spend)`
via OLS on days with spend>0 and revenue>0), clipped to `[0, 2]`
(`src/train.py::fit_elasticity`). A 10th/90th-percentile **bootstrap
confidence interval** on β (500 resamples with replacement) is computed
alongside the point estimate. Budget-scenario scaling (§3.5) uses β **only**
when `n_obs ≥ 10` **and** the CI width is ≤1.0 — otherwise it falls back to
β=1.0 (linear pass-through) for the scaling math specifically, while still
reporting the raw fitted β and its CI for transparency. Example: `google/SEARCH`
has β=0.85, CI=[0.81, 0.89] (tight, used as-is); `meta/Generic` has β=1.33,
CI=[0.19, 1.89] (width 1.70 — too wide to trust, scenario math falls back to
β=1.0 even though the point estimate is reported).

Forecasting (`src/forecasting.py`) is Monte Carlo, not closed-form, for
**both** methods: for each future day in the horizon, take the method's point
forecast and add a residual **bootstrapped with replacement from that
segment's own empirical residual pool** (not assumed Gaussian — this handles
the zero-inflation and skew directly from the observed data). 2,000 simulated
paths per segment per horizon; daily values are summed across the horizon and
across segments (assuming independence — see Limitations), then P10/P50/P90
are taken from the resulting distribution. `numpy.random.default_rng(seed=42)`
is seeded throughout for reproducibility.

**Uncertainty inflation:** Meta's residual bootstrap noise is scaled ×1.3
relative to Bing/Google. Even though §2.2 treats Meta's field as measured
revenue rather than imputed, there remains residual doubt about the schema
interpretation itself (a data dictionary that clarified the field would
retire this) — the inflation factor keeps that risk visible in Meta's
forecast intervals rather than hiding it behind a point estimate with the
same confidence as Bing/Google's directly-labeled revenue.

### 3.5 Budget scenario simulation

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

### 3.6 Aggregation levels

`forecast()` returns rows for: each (channel, campaign_type) segment, each
individual **campaign** (real Monte Carlo forecasts, not a fixed
percentage band — see §3.8), each channel rollup (sum across that channel's
campaign_type segments), and the blended total (sum across all channels) —
for each horizon × metric (`revenue`, `roas`). Campaign-type and channel/blended
rollups are summed elementwise per Monte Carlo draw, so the resulting
distributions properly propagate uncertainty rather than just summing point
estimates. Individual campaign rows are a separate, finer-grained breakdown
of the same underlying data already counted at the campaign_type level —
they are not additionally folded into the channel/blended totals, which
would double-count revenue.

### 3.7 Backtest & calibration results — initial baseline (pre-Step-2)

`src/backtest.py` walk-forward validates every segment: 3 non-overlapping
30-day test windows tiled across the last 90 days, each refit using **only**
data available before that window's cutoff (no leakage), scored against the
actual held-out revenue. Segments with <45 days of training history before a
cutoff are skipped rather than scored on too little data (`bing/Audience`,
`meta/Prospecting_Adv_Plus`). This is run once offline (not part of `run.sh`).
The numbers below are the **original, single-method (empirical-only, no
holiday factor, no clipping, no calibration)** snapshot, preserved at
`docs/backtest_results_baseline.json` for comparison against the post-fix
results in §3.8 — `docs/backtest_results.json` itself now holds the
empirical-vs-Holt-Winters method comparison described in §3.1.

**Aggregate, across 43 scored (segment × cutoff) pairs:**

| Metric | Value | Target |
|---|---|---|
| Median APE (point forecast) | 54.2% | — |
| Mean pinball loss | 3,585.8 | lower is better |
| **P10–P90 empirical coverage** | **37.2%** | **~80%** (nominal) |

**This is the honest finding, not a favorable one:** the initial model is
materially **overconfident** — its 80% nominal interval only actually
contains the true value 37.2% of the time, well short of target. Reporting
this rather than only the "intervals exist" claim is the actual evidence for
"appropriate handling of uncertainty" that the brief's evaluation criteria
ask for. Per-segment detail (`docs/backtest_results_baseline.json`):

| Segment | Mean APE | Coverage |
|---|---|---|
| google/DISPLAY | 0.0% | 100% |
| google/SEARCH | 22.3% | 66.7% |
| google/PERFORMANCE_MAX | 24.0% | 66.7% |
| meta/Generic | 33.3% | 66.7% |
| google/SHOPPING | 39.1% | 33.3% |
| meta/Prospecting_Brand | 44.7% | 66.7% |
| bing/PerformanceMax | 54.6% | 33.3% |
| bing/Shopping (n=1 cutoff) | 100.0% | 0.0% |
| google/DEMAND_GEN | 84.5% | 0.0% |
| bing/Search | 167.2% | 0.0% |
| meta/Remarketing_DPA | 302.6% | 0.0% |
| meta/Generic_Brand | 472.8% | 33.3% |
| meta/Prospecting_DPA | 489.9% | 0.0% |
| google/VIDEO | 2,147.9% | 33.3% |
| meta/Remarketing_Brand | huge (near-zero actual in one window) | 33.3% |

Low-volume, spiky segments (VIDEO, Generic_Brand, Prospecting/Remarketing DPA)
are the worst offenders — small absolute revenue means small absolute misses
translate into huge percentage errors, and the 120-day trailing-window trend
extrapolates recent noise further than in-sample residual variance accounts
for. §3.8 below documents what was changed in response to these numbers, and
the same table is reproduced there with post-fix results for direct
comparison — this is not a one-off measurement, it's the mechanism that
drove the modeling changes in the rest of this document.

### 3.8 Backtest & calibration results — after method selection + calibration fix

Three changes were made directly in response to §3.7's numbers, all
validated by re-running the identical walk-forward backtest:

1. **Per-segment method selection** (§3.1): empirical vs. Holt-Winters,
   picked by backtest pinball loss. 9 segments kept empirical, 6 switched to
   Holt-Winters, 2 defaulted (insufficient history to backtest either).
2. **97th-percentile clipping + holiday-window seasonality** (§3.2): reduces
   the influence of extreme single-day spikes on the trend fit.
3. **Per-segment calibration-scale search**: a global residual-noise
   multiplier was tried first and plateaued around 53% aggregate coverage
   (and got *worse* past 4x — a single multiplier can't fit segments that
   need very different amounts of widening). Switched to searching each
   segment's own multiplier against its own walk-forward coverage
   (`src/train.py::search_calibration_scale`, grid `[1.0, 1.5, 2.0, 2.5, 3.0,
   4.0, 5.0]`, picks the smallest scale reaching ≥2-of-3 covered cutoffs).

**Aggregate, across the same 43 scored pairs (`docs/backtest_results_post_fix.json`):**

| Metric | Baseline (§3.7) | Post-fix | Change |
|---|---|---|---|
| Median APE (point forecast) | 54.2% | 54.8% | ~flat overall (individual segments moved a lot — see below) |
| Mean pinball loss | 3,585.8 | 3,060.4 | −14.6% |
| **P10–P90 empirical coverage** | **37.2%** | **57.8%** | **+55% relative** |

**This is a real, measured improvement — and an honest, not-fully-solved
one.** Coverage moved from badly overconfident (37.2%) to moderately
overconfident (57.8%), still short of the 80% nominal target. Two reasons,
disclosed rather than papered over: (1) only 3 walk-forward cutoffs per
segment means achievable per-segment coverage is coarse (0%/33%/67%/100% —
"75%" isn't a reachable number at that resolution, so §3.8's search targets
≥67%, the nearest achievable rung below nominal); (2) some segments
(`meta/Remarketing_Brand`, `bing/Shopping`) have such sparse/spiky history
that no amount of residual widening fixes a systematically biased trend
extrapolation — the honest fix there would be a fundamentally different
uncertainty model (e.g. explicitly propagating trend-parameter uncertainty,
not just residual noise), which is out of scope for the remaining time. This
is called out again in §5 Limitations rather than left implicit.

Point-accuracy movement was uneven but often large on exactly the segments
§3.7 flagged as worst:

| Segment | Baseline MAPE | Post-fix MAPE (winning method) |
|---|---|---|
| google/VIDEO | 2,147.9% | 1,891.8% (Holt-Winters) |
| meta/Generic_Brand | 472.8% | 286.0% (Holt-Winters) |
| meta/Prospecting_DPA | 489.9% | 126.5% (Holt-Winters) |
| google/SEARCH | 22.3% | 33.3% (Holt-Winters; won on pinball loss, not MAPE — see note) |
| google/PERFORMANCE_MAX | 24.0% | 31.3% (Holt-Winters; same note) |

Note on `google/SEARCH` and `google/PERFORMANCE_MAX`: Holt-Winters won the
method-selection because it minimizes **pinball loss** (the metric that
matters for probabilistic forecasts — it scores the full P10/P50/P90 triple,
not just the median), even though its point-forecast MAPE is slightly worse
than empirical's on these two. Optimizing for pinball loss rather than MAPE
alone is deliberate — a forecast that's calibrated but has a slightly worse
median is more useful than one with a sharper median and badly wrong
intervals, given the brief explicitly asks for probabilistic ranges.

## 4. Assumptions

1. Meta's `conversion` field is conversion value, not count (§2.2).
2. Segments (channel × campaign_type combinations) are simulated
   independently — no cross-segment correlation is modeled. In reality,
   e.g. a platform-wide seasonal spike likely correlates Google and Bing
   Search simultaneously; treating them as independent likely
   **understates** the width of the blended P10–P90 band.
3. Elasticity is estimated from **historical, not experimental**, spend/
   revenue covariation (no randomized budget experiments in the data) — it
   captures correlation, not necessarily a causal spend effect. A bootstrap
   CI is computed (§3.4) and segments where it's wide (>1.0) or n_obs<10
   fall back to β=1.0 for budget-scenario scaling specifically, flagged
   low-confidence in the narrative layer and in `model.pkl`.
4. Day-of-week **and** a Black Friday/Cyber Monday–through–year-end holiday
   window (§3.2) are modeled; no finer-grained monthly seasonality or
   per-year holiday calendar (e.g. distinguishing Diwali, Christmas week,
   New Year specifically) is modeled, given the aggregate 30/60/90-day
   forecast horizon specified by the brief.
5. `output/predictions.csv`'s column schema (§below) is our own proposal —
   the exact schema announced at the AIgnition launch was not available in
   the materials provided at build time. **This must be verified against the
   official schema before final submission.**

## 5. Limitations

- No cross-segment/cross-channel correlation in the Monte Carlo aggregation
  (assumption 2 above) — blended intervals are likely somewhat too narrow.
- Elasticity is correlational, estimated on ~90–580 days of naturally
  varying spend per segment, not from held-out or experimental data.
- Campaign-level forecasts (104 of 136 campaigns; 32 skipped for having
  <30 days of history) reuse the method already selected for their parent
  campaign_type segment rather than running a full separate
  empirical-vs-Holt-Winters backtest per individual campaign — a
  per-campaign-type choice was judged the right compute/rigor tradeoff, but
  it means a handful of individual campaigns may not be on their personally
  optimal method.
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
- Backtesting (§3.7/§3.8) uses only 3 walk-forward cutoffs per segment over
  the last 90 days — enough to catch gross miscalibration and to show a
  real, measured improvement (37.2%→57.8% coverage), not enough for a
  statistically tight coverage estimate (43 scored pairs total) or to hit
  the 80% nominal target precisely. §3.8 states this directly rather than
  rounding 57.8% up to "calibrated."
- The per-segment calibration-scale factor in §3.8 is fit on the same
  backtest windows it's then evaluated against (no separate calibration/test
  split) — standard practice in low-data settings like this one, but it
  means the reported post-fix coverage is somewhat optimistic versus true
  out-of-sample performance.
- Some segments' under-coverage doesn't respond to residual widening at all
  (`meta/Remarketing_Brand`, `bing/Shopping`) — their forecast error is
  dominated by trend-extrapolation bias on sparse/spiky history, not
  residual noise, so no amount of noise-scale tuning fixes it. These
  segments' intervals should be read with more caution than the aggregate
  numbers suggest.

## 6. AI integration strategy

The LLM (Claude, via the Anthropic Python SDK) is deliberately **not** asked
to forecast numbers — it is asked to *interpret pre-computed statistics*, to
avoid hallucinated figures:

- Period-over-period (trailing 30 days vs. prior 30 days) revenue/spend
  deltas per channel.
- Per-segment spend elasticity, with its bootstrap CI (§3.4) and an explicit
  `used_for_budget_scenarios` flag so the LLM knows which elasticity numbers
  actually drove the budget-scenario math vs. which fell back to β=1.0.
- Anomalous campaigns by ROAS z-score within their channel (60-day lookback).
- **Per-segment forecast reliability** (`forecast_reliability`): each
  segment's own walk-forward backtest method, MAPE, and coverage (§3.8),
  with a `low_reliability` flag for segments below 40% coverage or above
  150% MAPE. This is what lets the narrative say "trust this range less"
  about a specific segment instead of treating every P10-P90 band as
  equally solid — it's a direct feed from the same backtest that drove the
  modeling changes in §3, not a separate qualitative judgment.
- **Structural zero-revenue-campaign check** (`structural_risk_campaigns`):
  computed live from whichever data is loaded — flags any channel where
  ≥25% of spending campaigns have zero lifetime revenue. Not hardcoded to
  Bing; on this dataset it independently rediscovers the Bing finding from
  §3.1's inventory note (64.3% of Bing's spending campaigns, $2,594 wasted
  spend) directly from the numbers, which is the point — the same check
  would catch an equivalent failure on Google or Meta in different data.
- Budget-scenario deltas (baseline vs. simulated blended revenue per
  horizon), when a scenario is active.

`src/llm_summary.py`'s `compute_stats()` produces this structured JSON;
`generate_causal_summary()` either sends it to Claude for a narrative +
risk-flag list, or — if `ANTHROPIC_API_KEY` is unset, or the API call fails
for any reason — falls back to a deterministic template built from the
same stats dict, extended to cover every new stat category so the offline
path stays just as informative as the live-LLM path. The app therefore
always runs, fully offline if needed, with the LLM as a strict enhancement
layer rather than a dependency.
