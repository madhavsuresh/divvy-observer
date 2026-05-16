# Calibration & forecast-display design

Design document for the visualization and dashboard layer of divvy-observer
([src/divvy/dashboard.py](src/divvy/dashboard.py),
[src/divvy/viz.py](src/divvy/viz.py),
[src/divvy/dashboard_metrics.py](src/divvy/dashboard_metrics.py)).

The structure and naming convention here intentionally mirror
transit-observer's `CALIBRATION_VIZ_DESIGN.md`, then diverge where the task
differs. Read the transit-observer doc first if you want the underlying
literature; this one explains *what changes when the forecast is a binary
event probability + a count PMF instead of a continuous duration quantile*.

## Why this exists

divvy-observer produces, at every prediction call, a bundle of quantities
per (station, horizon, model):

- `p_has_ebike` — probability ≥ 1 eBike is available
- `p_zero` — probability the station is empty
- `p_appears` / `p_survives` — conditional appearance / survival probabilities
- `expected_ebikes` — point forecast of the count
- `p_count_ebikes_json` — a full PMF over `{0, 1, 2, 3, 4, ≥5}` eBikes
- `p_capacity_violation`, `p_dock_constrained_arrival` — coherence checks
- `walk_adjusted_score`, `arrival_time_minutes`, `reliable_probability_lcb`
  — decision-layer scores used by the recommendation API

These resolve to:

- `observed_has_ebike` (boolean), `observed_ebikes` (count),
  `observed_total_bikes`, `observed_docks`.

Before this work, the dashboard had ~4,000 lines of rendering with a dozen
overlapping panels: a Brier table, a per-horizon table, a multi-bike
table, a threshold-k table, a top-k recommendation table, and a survival
table — each computing its own metrics inline, each rendering its own
loosely-styled Altair chart. The two questions that actually drive the
project got buried:

1. **Is the kernel calibrated and discriminating?** When the model says
   "80% chance the dock has an eBike in 10 min", does that happen 80% of
   the time? And are the predictions sharp enough to actually rank
   stations?
2. **Would a rider understand the forecast?** If a person opens the
   dashboard to decide which station to walk to, can they read off the
   answer in ≤ 5 seconds?

These are two different problems with two different literatures. We
adopted the standards from each, in some cases identical to transit-observer
and in some cases swapped for the binary / count analogue.

## Two audiences, two chart families

The single most-load-bearing decision: **the maintainer view and the
rider view use different chart types and should not be conflated.**
Padilla, Kay, and Hullman's
[*Uncertainty Visualization* handbook chapter (2022)](http://space.ucmerced.edu/Downloads/publications/Uncertainty_Visualization_Padilla_Kay_Hullman_2022.pdf)
makes the point: diagnostic charts (reliability, sharpness, decomposition)
are for people who already understand probabilistic forecasts; user-facing
charts must support frequency-counting cognition, which is how
non-statisticians reason about probability.

| Audience | Question they ask | Chart we use |
|---|---|---|
| Rider | "Will I find a bike if I walk to *this* station?" | Icon array (frequency dot grid) |
| Rider | "Which of these candidates is the safer bet?" | Side-by-side icon arrays + map |
| Rider | "I need k bikes — how do I split the trip?" | Combined-success dot grid |
| Maintainer | "Is the model calibrated overall?" | Reliability diagram with Wilson CIs |
| Maintainer | "Can the model tell good stations from bad?" | Score-distribution histogram split by outcome |
| Maintainer | "*Where* is the model broken — which buckets?" | Coverage heatmap, hour × day, faceted by model |
| Maintainer | "Are we sharp where we're accurate?" | Sharpness ↔ ECE scatter |
| Maintainer | "Does the count PMF match reality?" | Randomized PIT histogram (Czado et al.) |
| Maintainer | "Did the recommendation actually help?" | Top-k hit-rate bar + regret distribution |

The first three go in the **Find a Bike** tab — the product surface. The
rest go in **Model Performance** and **Calibration** tabs.

## Choice 1: Icon-array dot grids for end-user displays

### What we chose

A 10×10 grid of 100 dots (`n=100` by default; tunable). Filled dots
represent "out of 100 trips like this, X find a bike." Built in
[viz.py:dot_grid_chart](src/divvy/viz.py).

### Why dots, not a percentage label

This is the binary analogue of transit-observer's quantile dotplot, and
the underlying citation is identical: the icon-array / dot-grid literature
sits inside the same frequency-framing tradition that the Kay & Fernandes
quantile-dotplot work drew on.

- Galesic, Garcia-Retamero, Gigerenzer — [*Using icon arrays to communicate
  medical risks* (Health Psychology 2009)](https://pubmed.ncbi.nlm.nih.gov/19290708/)
  — showed that icon arrays outperformed numerical-only displays for risk
  reasoning in low-numeracy populations.
- Spiegelhalter, Pearson, Short — [*Visualizing Uncertainty About the Future*
  (Science 2011)](https://www.science.org/doi/10.1126/science.1191181)
  — argues for frequency framing as the dominant communication
  paradigm for forecasts to lay audiences.
- Hullman, Resnick, Adar —
  [*Hypothetical Outcome Plots Outperform Error Bars and Violin Plots*
  (PLOS ONE 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142444)
  — same cognitive grounding (countable discrete instances), different
  output channel (animation vs. static array).

We chose static dot grids over HOPs (animated) because the user is
making a one-shot decision (which station to walk to), not estimating
a trend. The dot count is the dominant cue; we do not jitter or animate.

### Why 100, not 20 or 1,000

Spiegelhalter et al. point to "1 in 100" as the sweet spot for risk
communication: granular enough to express common probabilities (one
percentage point), coarse enough to count visually on a phone screen.
20 dots is too coarse (each dot worth 5%); 1,000 dots loses the
"countable" property entirely. We expose `n` so the Cozy Fox-style iOS
port can tune for screen size if needed.

### Why we did *not* use a probability label alone

A single label ("82%") tests poorly for non-statistical audiences in
every cited study. The label appears as a small subtitle under the
dot grid — useful, not load-bearing.

### Why we did *not* use the NYT-style jittering needle

Same reason transit-observer didn't. See
[FlowingData's *Needle of uncertainty*](https://flowingdata.com/2018/03/14/needle-of-uncertainty/)
and the [Fast Company critique](https://www.fastcompany.com/90459366/the-most-hated-data-visualization-in-politics-is-back-to-spike-your-blood-pressure):
animated jitter raises anxiety without communicating useful information.

### Multi-bike plan: combined-probability dot grid

When a rider needs k bikes, the recommendations layer convolves per-station
PMFs to compute P(plan succeeds). We render that combined probability
the same way — one dot grid for the whole plan — to keep the visual
language consistent. The per-station dots become small thumbnails next
to the combined grid.

## Choice 2: Reliability diagram for the binary target

### What we chose

For each `(model, horizon)` slice, bin `p_has_ebike` into deciles, count
the empirical positive rate per bin, and plot against the predicted bin
mean with **Wilson 95% confidence intervals** sized by sqrt(n). Reference
line is `y = x`. Implemented in
[dashboard_metrics.py:reliability_curve](src/divvy/dashboard_metrics.py)
and [viz.py:reliability_diagram_chart](src/divvy/viz.py).

### Why binning-and-counting, not isotonic regression (CORP)

Dimitriadis, Gneiting, Jordan — [*Stable Reliability Diagrams for
Probabilistic Classifiers* (PNAS 2021)](https://www.pnas.org/doi/10.1073/pnas.2016191118)
— introduced the CORP form using isotonic regression, which gives tighter
diagnostics in low-sample regimes. We use the classical binning form
because:

1. The Divvy collector produces ~20k–50k resolved outcomes per day per
   horizon, so bins are well-populated and the classical estimator is
   essentially identical to CORP.
2. CORP requires scipy (or a custom isotonic implementation); we want
   the visualization layer scipy-free.
3. The Wilson interval on the binomial-positive-rate-per-bin is a
   well-understood error bar and tells the reader the same story as the
   CORP band would.

Wilson intervals (Wilson 1927; recommended in
[Brown, Cai, DasGupta — *Interval Estimation for a Binomial Proportion*
(Statistical Science 2001)](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full))
beat the naive Wald interval at small per-bin n, which matters when we
slice by `(model, horizon, hour)`.

### Why we did *not* use a quantile-based reliability diagram

That was transit-observer's design choice, and it's correct *for a
continuous-quantile forecast* (`(p50, p80, p90)` of a duration). Divvy's
forecast is fundamentally a binary classifier on `has eBike`. The native
diagnostic for a binary classifier is "of the predictions in bucket
[0.7, 0.8], what fraction had `y=1`?" — that is the classical reliability
diagram.

For the *count* forecast, we use a different diagnostic — see Choice 4.

## Choice 3: Score-distribution histogram for discrimination

### What we chose

Two stacked density-normalized histograms per `(model, horizon)`: the
distribution of predicted `p_has_ebike` conditional on `y=1`, overlaid
on the distribution conditional on `y=0`. A well-discriminating model
separates the two; a poor one piles them on top of each other.
Implemented in
[dashboard_metrics.py:score_distribution](src/divvy/dashboard_metrics.py)
and [viz.py:score_distribution_chart](src/divvy/viz.py).

### Why this and not ROC

Both ROC curves and score-distribution overlays answer "is the model
discriminating?". The score-distribution view is more diagnostically
useful here because:

1. It shows *where* in the probability range the model lives. If both
   distributions hug 0.5 you have a model that doesn't predict anything;
   if they're bimodal and well-separated you have a confident model;
   if they're separated but compressed (say both fall in 0.3–0.7) you
   have a calibrated-but-blunt model. ROC compresses all that into a
   single curve and a single number.
2. It pairs naturally with the reliability diagram on the same axis
   (predicted probability). The maintainer sees calibration and
   discrimination side by side without context-switching the x-axis.

This is the binary/probabilistic analogue of the PIT histogram in
transit-observer. The PIT framework
([Gneiting, Balabdaoui, Raftery — *Probabilistic forecasts, calibration
and sharpness* (JRSS B 2007)](https://doi.org/10.1111/j.1467-9868.2007.00587.x))
is undefined for binary outcomes (no continuous CDF to plug into); the
score-distribution view is the standard alternative, used as the basis
of Krzanowski–Hand's measure of discrimination and surfaced in every
modern probabilistic-classification text (e.g.
[Niculescu-Mizil & Caruana — *Predicting Good Probabilities with
Supervised Learning* (ICML 2005)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)).

## Choice 4: Randomized PIT for the count PMF

### What we chose

For each resolved forecast, compute the PIT-equivalent value for the
discrete count `observed_ebikes` using the **randomized PIT**: draw
`u ~ Uniform(F(observed - 1), F(observed))` where `F` is the predicted
count CDF. Histogram those values per model. Implemented in
[dashboard_metrics.py:count_pit_values](src/divvy/dashboard_metrics.py)
and [viz.py:count_pit_histogram_chart](src/divvy/viz.py).

### Why randomized

The naive PIT `F(observed)` for a discrete variable is not uniform under
a perfectly-calibrated model — it has gaps at the support points.
Czado, Gneiting, Held — [*Predictive Model Assessment for Count Data*
(Biometrics 2009)](https://doi.org/10.1111/j.1541-0420.2009.01191.x)
— introduced the **randomized PIT** for exactly this case: it restores
uniformity under perfect calibration so the same diagnostic vocabulary
applies (flat = calibrated, U = underdispersed, ∩ = overdispersed).

The reading is identical to transit-observer's PIT histogram, with
the same diagnostic table:

| Histogram shape | Diagnosis |
|---|---|
| Flat / uniform | Calibrated |
| U-shape (peaks at 0 and 1) | Underdispersion — PMF too tight |
| ∩-shape (peak in middle) | Overdispersion — PMF too wide |
| Left-skewed | Counts predicted too low |
| Right-skewed | Counts predicted too high |

### Why we did *not* discretize / cumulative-uniform-test instead

Czado et al. test those alternatives and show randomized PIT dominates
on power-to-detect-miscalibration across discrete-forecast benchmarks.

## Choice 5: Coverage heatmap, faceted by model

### What we chose

A small-multiples grid: one panel per model, each panel a 24×7
heatmap where rows are `weekday | weekend` (or day-of-week), columns
are `hour_of_day`, and cell colour is `ECE` on a sequential red scale.
Implemented in [viz.py:coverage_heatmap_chart](src/divvy/viz.py).

### Why

Same justification as transit-observer's per-line heatmap, transposed
to our task: this is the highest-signal-density chart for "*where* is
the kernel broken?". Each cell encodes the exact calibration error for
a specific time-of-week — morning commute, late-night service decay,
weekend-vs-weekday demand shifts. The maintainer scans for hotspots
without reading numbers.

The small-multiples convention is from Tufte's *Envisioning Information*
and used identically in
[*The Economist*'s state-by-state election forecasts](https://statmodeling.stat.columbia.edu/2021/08/11/forecast-displays-that-emphasize-uncertainty/).

### Why hour × day, not station × model

We tried slicing by station and the chart became a 700-station grid
with too much sparsity per cell to interpret. Hour × day-of-week is the
natural seasonality cycle that the predictor explicitly models
(`is_commute_hour`, `dow` features in `predictor.py`); failures cluster
on those axes, not on station identity.

We still surface the *station-hour worst-offenders* as a sortable table
([dashboard_metrics.worst_station_hours](src/divvy/dashboard_metrics.py))
for drilling in.

## Choice 6: Sharpness ↔ calibration scatter

### What we chose

One point per `(model, horizon, hour-band)` bucket. x = mean predicted
probability variance `p(1−p)` (the binary analogue of `p80 − p50`
spread), y = Expected Calibration Error (ECE) for that bucket, point
size scaled to sample count, colour to model. Implemented in
[viz.py:sharpness_ece_chart](src/divvy/viz.py).

### Why

The Gneiting et al. (2007) sharpness principle: among all calibrated
forecasts, prefer the sharpest. For binary forecasts the textbook
sharpness measure is the variance of the prediction — a model that
always says 0.5 has maximum variance (0.25) and is useless; a model
that says 0.95 when it's right and 0.05 when it's wrong has variance
near 0.05 and is excellent.

The target zone is **low x, low y** — sharp *and* calibrated.
The worst quadrant is **low x, high y** — confident *and* wrong.
A model with high x (blunt, near 0.5 everywhere) and low y is
calibrated-but-useless: it's hitting the marginal rate but giving the
rider no rank ordering.

This is the exact same chart as transit-observer's
"sharpness ↔ coverage" scatter, with sharpness measured in the
binary-appropriate way (variance) and calibration measured by ECE
(instead of P80 coverage deviation, which doesn't apply).

## Choice 7: Brier decomposition + skill score on the leaderboard

### What we chose

For each model in the leaderboard:

- **Brier score** (the headline, lower-is-better)
- **Brier skill score (BSS)** vs the empirical-bayes baseline — `1 − BS_model / BS_baseline`
- **Reliability** component of Brier (Murphy decomposition)
- **Resolution** component (sharpness × discrimination)
- **ECE** as an independent calibration error
- **Log loss** (rank-aware, less generous to confident wrong calls)
- **Decision rank loss** (the composite the active-model selector uses)

Implemented in [dashboard_metrics.py:brier_decomposition](src/divvy/dashboard_metrics.py).

### Why decomposition

Murphy (1973) showed that `BS = Reliability − Resolution + Uncertainty`,
where Uncertainty is the irreducible variance of the outcome. The
reliability term is the calibration error; the resolution term is what
the model adds *over and above the marginal rate*; uncertainty is fixed
by the data and equals `ȳ(1 − ȳ)`. A model with low Brier can have low
Reliability and high Resolution (good — calibrated and discriminating),
or it can ride the marginal rate (bad — calibrated but useless). The
decomposition lets the maintainer tell those apart on the same
leaderboard.

References:
- Murphy — [*A new vector partition of the probability score*
  (J. Applied Meteorology 1973)](https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml)
- Bröcker — [*Reliability, sufficiency, and the decomposition of proper
  scores* (QJRMS 2009)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.456)

### Why skill score against the empirical-bayes baseline

The absolute Brier number is hard to interpret in isolation — it depends
on the marginal positive rate, which drifts seasonally. The skill score
normalises against the dumbest non-trivial baseline (the empirical
prior of "this station, this hour-of-week, this season"), so
`+10%` means *the model adds 10% over knowing the prior*. Same convention
as numerical-weather-prediction skill scores
([Wilks — *Statistical Methods in the Atmospheric Sciences*](https://www.elsevier.com/books/statistical-methods-in-the-atmospheric-sciences/wilks/978-0-12-385022-5)).

## Choice 8: Decision-impact panel for the recommendation layer

### What we chose

A separate panel that scores not the *probabilities* but the *decisions*
they drive:

- **Top-k recommendation hit rate** per model — did the model's top-k
  list contain the actually-best station?
- **Distance-adjusted regret distribution** — how much utility did
  riders give up by walking to the recommended station vs. the oracle
  best?
- **Active-vs-best gap** — is the currently-promoted model winning, or
  is something else catching up?

Implemented as separate `viz.py` charts (`topk_hitrate_chart`,
`regret_distribution_chart`) reading from `recommendation_outcomes` /
`model_metrics`.

### Why this is a separate concern

The Brier / ECE / skill-score panel asks "is the kernel well-calibrated?"
The decision-impact panel asks "did the wrapping decision logic
actually help users?" These can diverge: a well-calibrated kernel
combined with a misweighted utility function can score *worse* than a
miscalibrated kernel with a well-tuned utility. The
`recommendations.py` walk-adjusted score has knobs (distance penalty,
LCB offset) that don't show up in Brier; surfacing the decision metrics
separately keeps the audit honest.

## Library choice: Altair

Already a project dependency (see `pyproject.toml`). Declarative,
JSON-serializable, small-multiples and faceted layered charts are
~15 lines each. We avoided Plotly to keep the dependency footprint
lean.

Same constraint as transit-observer: Altair's faceted layered charts
must declare the shared dataset at the facet level rather than per-layer
or `to_dict()` raises a schema-validation error. The chart constructors
in [viz.py](src/divvy/viz.py) handle this by passing `data=df` at the
`.facet()` call.

## Dashboard structure

```
Sidebar: window selector (24h / 7d / 30d), refresh button

Tabs (top-level):
  1. Find a Bike     — rider product surface
  2. Station         — single-station deep dive
  3. Performance     — maintainer leaderboard, reliability, discrimination
  4. Calibration     — coverage heatmap, sharpness scatter, count PIT
  5. Decisions       — top-k hit rate, regret, active-vs-best
  6. System          — collector ticks, queue depth, replica freshness
```

The first two tabs target the rider; the next three target the
maintainer; the last is operational. Splitting them lets each chart
type stay in its native idiom — no frequency dots in the Performance
tab, no reliability diagrams in Find a Bike.

## What's deliberately out of scope

- **Per-station calibration cards.** With ~700 stations, no individual
  station has enough resolved outcomes to compute a stable per-station
  reliability curve. We aggregate to (hour × day-of-week) instead and
  expose `worst_station_hours` as a drill-in table.
- **Geographic choropleth of calibration error.** Cool, would require
  H3 binning + interpolation; the value over the hour × day-of-week
  heatmap is marginal for a city-wide model.
- **Per-rider personalized forecast displays.** Single-shot icon array
  is the right level of abstraction for the project; "personalized"
  would mean tracking individual usage which is out of scope for this
  observatory.
- **CORP / isotonic reliability bands.** Would tighten the diagnostic
  but requires scipy; the Wilson-CI binning form is sufficient at the
  resolved-outcomes counts we see.
- **Animated needles / HOPs.** Per the FlowingData / Fast Company
  critique, explicitly not appropriate for a "should I walk to this
  station?" decision.

## Citation index

- Galesic, Garcia-Retamero, Gigerenzer — [*Using icon arrays to communicate medical risks* (Health Psychology 2009)](https://pubmed.ncbi.nlm.nih.gov/19290708/)
- Spiegelhalter, Pearson, Short — [*Visualizing Uncertainty About the Future* (Science 2011)](https://www.science.org/doi/10.1126/science.1191181)
- Kay, Kola, Hullman, Munson — [*When (ish) is My Bus?* (CHI 2016)](https://dl.acm.org/doi/10.1145/2858036.2858558) — adjacent work; foundation for transit-observer's quantile dotplot choice
- Fernandes, Walls, Munson, Hullman, Kay — [*Uncertainty Displays Using Quantile Dotplots or CDFs Improve Transit Decision-Making* (CHI 2018)](https://idl.uw.edu/papers/uncertainty-bus)
- Hullman, Resnick, Adar — [*Hypothetical Outcome Plots Outperform Error Bars and Violin Plots* (PLOS ONE 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0142444)
- Padilla, Kay, Hullman — [*Uncertainty Visualization* handbook chapter (2022)](http://space.ucmerced.edu/Downloads/publications/Uncertainty_Visualization_Padilla_Kay_Hullman_2022.pdf)
- Murphy — [*A new vector partition of the probability score* (J. Applied Meteorology 1973)](https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml)
- Niculescu-Mizil & Caruana — [*Predicting Good Probabilities with Supervised Learning* (ICML 2005)](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- Gneiting, Balabdaoui, Raftery — [*Probabilistic forecasts, calibration and sharpness* (JRSS B 2007)](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
- Czado, Gneiting, Held — [*Predictive Model Assessment for Count Data* (Biometrics 2009)](https://doi.org/10.1111/j.1541-0420.2009.01191.x)
- Dimitriadis, Gneiting, Jordan — [*Stable Reliability Diagrams for Probabilistic Classifiers* (PNAS 2021)](https://www.pnas.org/doi/10.1073/pnas.2016191118)
- Bröcker — [*Reliability, sufficiency, and the decomposition of proper scores* (QJRMS 2009)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.456)
- Brown, Cai, DasGupta — [*Interval Estimation for a Binomial Proportion* (Statistical Science 2001)](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full)
- Yau — [*Needle of uncertainty* (FlowingData 2018)](https://flowingdata.com/2018/03/14/needle-of-uncertainty/)
- Schiller — [*The most hated data visualization in politics is back* (Fast Company 2020)](https://www.fastcompany.com/90459366/the-most-hated-data-visualization-in-politics-is-back-to-spike-your-blood-pressure)
- Gelman / Morris — [*Forecast displays that emphasize uncertainty* (2020)](https://statmodeling.stat.columbia.edu/2021/08/11/forecast-displays-that-emphasize-uncertainty/)
