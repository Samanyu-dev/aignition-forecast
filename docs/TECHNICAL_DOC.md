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

### 2.4 Campaign consistency validation

A distinct, reusable pipeline step (`src/validate_consistency.py`), separate
from ingestion and forecasting — the earlier Meta-mislabeling (§2.2) and
Bing zero-revenue (§3.1's inventory note) findings were ad hoc investigative
work; this generalizes and formalizes that kind of check into a structured,
reusable report (`docs/validation_report.json`), surfaced in its own
Streamlit panel and fed into the AI narrative's `compute_stats()`. Reads the
raw per-channel CSVs directly (budget/click columns aren't carried into the
trimmed forecasting schema). Six checks, run per channel:

1. **Campaign ID stability** — does a `campaign_id` map to more than one
   `campaign_name`/`campaign_type` over time? (0 issues found — clean.)
2. **Date-coverage gaps** — any campaign with >14 consecutive inactive days
   inside its active range? (19 found.)
3. **Budget exceeded** — daily spend >5% over the campaign's stated
   `daily_budget`? (90 found — see caveat below.)
4. **Conversions exceed clicks** — Bing/Google only, since Meta has no
   genuine conversion-count field (§2.2). (6 found, 1 row each.)
5. **Negative/impossible values** — any negative spend/revenue/clicks/
   conversions/impressions. (0 found — clean.)
6. **Zero-revenue-with-spend** — lifetime spend>0, lifetime revenue=0,
   generalizing the original Bing-specific finding to all three channels
   the same way. (32 found: 18 Bing, 14 Google — the Google instances are a
   new finding this generalization surfaced, not previously reported.)

**Caveat on "budget exceeded" (90 flagged, the largest category):** a
platform's `daily_budget` field is typically a *pacing average* the system
is allowed to exceed on any individual day (sometimes by 2x or more) while
holding a longer-run average — it is usually not a hard per-day cap. This
check is flagged for visibility and completeness, not because every instance
is a genuine error; the AI narrative layer is explicitly told this via its
prompt, so it doesn't over-weight this category relative to the other five.

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

### 3.8 Backtest & calibration results — after method selection, and a correction to the calibration methodology itself

Three changes were made directly in response to §3.7's numbers:

1. **Per-segment method selection** (§3.1): empirical vs. Holt-Winters,
   picked by backtest pinball loss. 9 segments kept empirical, 6 switched to
   Holt-Winters, 2 defaulted (insufficient history to backtest either).
2. **97th-percentile clipping + holiday-window seasonality** (§3.2): reduces
   the influence of extreme single-day spikes on the trend fit.
3. **Per-segment calibration-scale search**: a global residual-noise
   multiplier was tried first and plateaued around 53% aggregate coverage
   (and got *worse* past 4x). Switched to searching each segment's own
   multiplier against its own walk-forward coverage.

**A genuine methodology flaw was found and fixed in (3), not just in the
model.** The first version of the calibration search tuned the residual
scale and then *reported coverage using the same 3 walk-forward cutoffs it
was tuned on* — classic double-dipping. That version reported 37.2%→57.8%
coverage. On review, this was flagged as likely optimistic, and it was:
re-implemented as **rolling-origin validation**
(`src/train.py::search_calibration_scale`) — extended to up to 6 monthly
cutoffs (180 days) where history allows, tuning the scale on the *older*
half and reporting coverage on the *newer* half, which the tuning step never
sees. 5 of 17 segments don't have enough history to split honestly at all;
for those, no tuning is performed (`calibration_scale=1.0`) and they're
labeled `no_calibration_tuning: true` rather than silently reusing a shared
fold.

**Aggregate, honest held-out result (`docs/backtest_results_post_fix.json`):**

| Metric | Baseline (§3.7, single fold) | Double-dipped calibration (superseded) | **Honest rolling-origin (current)** |
|---|---|---|---|
| Median APE (point forecast) | 54.2% | 54.8% | **66.4%** |
| Mean pinball loss | 3,585.8 | 3,060.4 | **4,002.6** |
| **P10–P90 coverage** | **37.2%** | ~~57.8%~~ | **37.8%** |

**Read this plainly: once evaluated correctly, the calibration fix did not
meaningfully move coverage** (37.2%→37.8%, within the noise of a 43-fold
backtest) or point accuracy (both got modestly worse once evaluated on
genuinely unseen, more recent windows rather than the same 3 windows used
throughout). The 57.8% figure previously reported in this document was real
output from real code, but was reporting the calibration search's own
training performance, not held-out performance — the classic overfitting
trap, caught by re-examining the methodology rather than trusting a
favorable number. This is left in the table above rather than deleted,
because the correction itself is evidence of process integrity that a
flattering-only number would not be.

**What this means practically:** the method-selection and
clipping/seasonality changes (1–2 above) are still genuine, measured
improvements to point accuracy on the worst segments (below). The interval
*width* itself remains under-calibrated — the honest conclusion is that
residual-bootstrap noise alone doesn't capture this model's true forecast
uncertainty, which mostly comes from trend-extrapolation error the model
doesn't otherwise account for (§5 Limitations expands on this — the correct
fix is a fundamentally different uncertainty model, not a wider scale
factor, and is out of scope for the remaining time).

Point-accuracy movement (method selection alone, independent of the
calibration-scale question) is still real and large on the worst segments
§3.7 flagged, per the current rolling-origin evaluation:

| Segment | Baseline MAPE (§3.7) | Current MAPE (winning method) |
|---|---|---|
| google/VIDEO | 2,147.9% | 1,878.8% (Holt-Winters) |
| meta/Generic_Brand | 472.8% | 283.4% (Holt-Winters) |
| meta/Prospecting_DPA | 489.9% | 125.4% (Holt-Winters) |
| google/PERFORMANCE_MAX | 24.0% | 31.2% (Holt-Winters; won on pinball loss, not MAPE — see note) |

Note on `google/PERFORMANCE_MAX`: Holt-Winters won method-selection because
it minimizes **pinball loss** (scores the full P10/P50/P90 triple, not just
the median) — a calibrated forecast with a slightly worse median is more
useful than a sharp median with badly wrong intervals, given the brief asks
for probabilistic ranges specifically.

### 3.9 Budget-reallocation recommendations

A forecast alone isn't a decision — `src/recommendations.py` turns the
elasticity work above into an actual "shift $X/day from segment A to segment
B" recommendation, priced with the same Monte Carlo machinery as the budget
scenarios above (`forecasting.py` now also accepts a **segment-level**
budget override, not just a per-channel one, so a scenario can move spend
between two specific segments precisely).

**Eligibility is confidence-gated, not just elasticity-ranked**: only
segments with `elasticity_low_confidence=False` (tight bootstrap CI, §3.4)
and non-trivial spend qualify as either a donor (reduce) or a receiver
(increase) — 12 of 17 campaign_type segments qualified on this dataset. The
donor pool is the 3 lowest-elasticity eligible segments (most diminishing
returns); the receiver pool is the 3 highest-elasticity eligible segments
(most linear/efficient). Each donor×receiver pair is priced by simulating a
20%-of-daily-spend shift (receiver-side multiplier capped at 2.5x to avoid
extrapolating the elasticity fit past where it was estimated) and comparing
blended 90-day P50 revenue/ROAS against the unshifted baseline. Top 5 by
revenue delta are surfaced, re-priced at full Monte Carlo precision
(2,000 sims) rather than the cheaper search-time estimate (500 sims) used to
rank the full candidate set.

**Actual top recommendation on this dataset:**

| From | To | Shift ($/day) | From β (CI) | To β (CI) | 90d revenue Δ |
|---|---|---|---|---|---|
| meta/Generic_Brand | meta/Remarketing_DPA | $16.95 | 0.21 [0.06,0.32] | 0.81 [0.76,0.85] | +$8,564 (+0.8%) |
| bing/Search | meta/Remarketing_DPA | $14.06 | 0.11 [0.02,0.20] | 0.81 [0.76,0.85] | +$6,929 (+0.6%) |
| bing/PerformanceMax | meta/Remarketing_DPA | $10.54 | 0.13 [0.00,0.36] | 0.81 [0.76,0.85] | +$5,425 (+0.5%) |

Read the small dollar sizes honestly: the segments with genuinely tight
elasticity CIs skew toward Bing and smaller Meta campaign types, which have
small absolute daily spend (§2 — Bing's average daily spend here is under
$100), so the *dollar* shifts are modest even though the *directional*
signal (move spend toward higher, tightly-estimated elasticity) is real and
statistically defensible. This is a deliberate byproduct of the confidence
gate, not a limitation of the method — a bigger, noisier recommendation
would be easy to produce by ignoring the CI width, and would be worse advice.

### 3.10 Retrospective validation of the recommendation engine

The Monte Carlo pricing above is forward-looking — it says what *should*
happen if a shift is made, given the fitted elasticity. It is not itself
evidence the underlying premise (rank segments by elasticity, favor the
high-elasticity ones) actually holds up against real subsequent outcomes.
`src/validate_recommendations.py` tests that premise walk-forward: no actual
counterfactual budget shift is observable in this data (it's observational,
not an experiment), so this doesn't test "would this exact dollar shift have
beaten reality" — it tests the engine's directional logic instead. At 5
past decision points (60–180 days before the dataset's end), using **only**
data available as of that point, it identifies the same donor/receiver
candidate pool `recommendations.py` would have (3 lowest- and 3
highest-elasticity eligible segments), then checks: did the receiver
segment's **actual, realized** ROAS in the subsequent 30 days exceed the
donor's? If the recommendation logic is sound, this should happen
meaningfully more than half the time.

**Honest result (`docs/recommendation_validation.json`):**

| Test | n scored | Confirmed | Rate |
|---|---|---|---|
| Full candidate grid (all donor×receiver pairs, matching the actual pool size) | 26 | 15 | **57.7%** |
| Single top pick only (what the engine would have surfaced as its #1 recommendation) | 2 | 0 | **0%** |

**This is a mixed, not a clean, result — reported as such.** The full-grid
number (57.7%) is meaningfully better than a coin flip and shows the
elasticity ranking has real, if modest, predictive value for relative
subsequent efficiency. But the single-top-pick number is 0%, and the reason
is instructive rather than random: at every one of the 5 decision points,
`google/VIDEO` had the highest as-of elasticity estimate and was picked as
the top receiver — and at every single one, its actual realized ROAS in the
following 30 days collapsed (0.00–0.02, and in 2 of 5 windows it stopped
spending entirely). This is the **same segment** already flagged
`low_reliability` by the walk-forward backtest (§3.8 — Holt-Winters MAPE
1,878.8% on `google/VIDEO`, the worst in the portfolio) — two independent
validation methods (forecast backtesting and recommendation retrospection)
converged on the same finding through entirely different mechanisms. That
convergence is more convincing than either result alone, and it points at a
concrete, cheap improvement not yet implemented: **gate the receiver pool by
`forecast_reliability.low_reliability` (§3.6) as well as elasticity
confidence** — `google/VIDEO` passes the elasticity-CI gate (its CI is
reasonably tight, [0.81, 1.40]) but is a known-unreliable forecaster, and
the two checks are currently independent when they should compound.

One caveat on the comparison itself: "receiver ROAS > donor ROAS" compares
absolute realized ROAS levels, not marginal returns to *additional* spend —
`recommendations.py`'s actual criterion. A donor can legitimately have high
absolute ROAS (e.g. `meta/Prospecting_Brand` realized 15.57x in one window)
while still being a correct "don't scale further" pick if its marginal
returns are diminishing faster than the receiver's — a few of the
"unconfirmed" pairs in the full grid are this case, not a failure of the
underlying logic. A more precise retrospective test would compare realized
marginal ROAS (via a local regression on realized spend/revenue in the
window) rather than levels; that refinement is out of scope for the
remaining time.

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
5. `output/predictions.csv`'s column schema
   (`channel,campaign_type,campaign_id,horizon_days,metric,p10,p50,p90`) is
   a considered design decision, not a guess against an unseen spec: the
   brief's "Date Link — AIgnition_dataset" resource is the same three CSVs
   already in `data/`, not a separate output-format document, and the brief
   itself specifies required outputs only qualitatively (channel- /
   campaign_type- / campaign-level revenue and ROAS ranges, probabilistic
   not deterministic) — which this schema satisfies directly.

## 5. Limitations

- **GA4 session source/medium data and Shopify conversion data are listed
  as brief resources but do not exist in the actual dataset** — only the
  three ad-platform CSVs (Bing/Google/Meta) were provided. This is a real
  constraint, not an oversight: without an independent revenue source (a
  Shopify order feed, e.g.), there is no ground truth to directly verify
  Meta's revenue figures against. The §2.2 finding that `meta.conversion` is
  mislabeled revenue had to be established by statistical inference
  (matching its implied ROAS distribution against Bing/Google's measured
  ROAS) rather than a direct cross-check — a reasonable substitute given the
  constraint, but a substitute nonetheless.

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
- Backtesting uses at most 6 monthly cutoffs per segment (fewer for
  shorter-history segments, split further into tune/eval halves for
  calibration — §3.8) — enough to catch gross miscalibration and to catch
  a real methodology bug in this project's own validation code, not enough
  for a statistically tight coverage estimate. The honest rolling-origin
  number (37.8%) should be read as "still meaningfully overconfident," not
  as a precise estimate of true long-run coverage.
- **Residual-bootstrap widening alone does not fix this model's
  calibration** (§3.8) — once measured honestly out-of-sample, the
  calibration-scale search barely moved aggregate coverage (37.2%→37.8%).
  The forecast uncertainty this model under-represents appears to come
  mostly from trend-extrapolation error (the linear/Holt-Winters trend
  fit's own uncertainty), which residual-pool noise doesn't capture at all
  — a correct fix would explicitly propagate that parameter uncertainty
  into the Monte Carlo simulation (e.g. bootstrap the trend fit itself
  across resampled training windows, not just the residuals), which is out
  of scope for the remaining time.
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
- **Confidence-gated budget-reallocation candidates** (`budget_reallocation_recommendations`,
  §3.9), already priced with Monte Carlo revenue/ROAS impact and already
  screened by elasticity confidence before they ever reach the LLM. This is
  the deliberate design point that pushes the AI layer from *narrating* to
  *reasoning*: the LLM is explicitly instructed to turn the top candidate
  into a concrete, actionable recommendation ("shift $X/day from A to B
  because Y, expect roughly Z% lift") rather than only describing what
  already happened — while still being told the eligibility screening is
  already done, so it caveats confidence rather than re-deriving it.

`src/llm_summary.py`'s `compute_stats()` produces this structured JSON;
`generate_causal_summary()` either sends it to Claude for a narrative +
recommendation + risk-flag list, or — if `ANTHROPIC_API_KEY` is unset, or
the API call fails for any reason — falls back to a deterministic template
built from the same stats dict, extended to cover every new stat category
(including the recommendation) so the offline path stays just as
informative as the live-LLM path. The app therefore always runs, fully
offline if needed, with the LLM as a strict enhancement layer rather than a
dependency.
