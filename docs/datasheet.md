# Datasheet: BBO capstone, function by function

This datasheet follows the course template. It documents my optimisation decisions, learning and reasoning for each of the eight black-box functions. Answers shared by all functions are given once in Part A. Part B answers the template questions for each function. All numbers are from `data/queries.csv` and `notebooks/BBO_capstone_final.ipynb`.

**Scope and status.** Weeks 1 to 13: 13 queries were submitted per function. The week-12 results have not been received, so 12 results are known per function (the week-12 inputs are in the data with a blank output).


---

## Part A: Answers common to all functions

### Nature of the data
- **Structure.** For function *f* with *d* inputs: `X` has shape (n, d) with every value in [0, 1], and `Y` has shape (n,). The course supplied the first rows (week 0); each of my queries added one row.
- **Evolution.** One new point per function per week. Early weeks spread points out (broad coverage). Middle weeks tested model settings. Late weeks concentrated points near the best region (small steps or a trust region). Sampling is therefore **adaptive**: later points depend on earlier results and are not independent.

| Function | Inputs | Course points | Results known (week 0 to 13) | Output range |
|---|---|---|---|---|
| 1 | 2 | 10 | 22 | -0.0036 to 1.4e-8 |
| 2 | 2 | 10 | 22 | -0.069 to 0.611 |
| 3 | 3 | 15 | 27 | -0.399 to -0.0007 |
| 4 | 4 | 30 | 42 | -32.6 to 0.368 |
| 5 | 4 | 20 | 32 | 0.113 to 8,267 |
| 6 | 5 | 20 | 32 | -2.57 to -0.114 |
| 7 | 6 | 30 | 42 | 0.0027 to 3.25 |
| 8 | 8 | 40 | 52 | 5.59 to 9.998 |

### Optimisation strategy
- **Method.** Bayesian optimisation: a Gaussian-process (GP) surrogate with an upper confidence bound (UCB) rule, `mean + beta * std`. Settings (kernel, length-scale bound, noise level `alpha`) were ranked by log marginal likelihood. From week 11, TuRBO (a trust-region method) was used where a calibration check passed.
- **Exploration versus exploitation.** `beta` was compared at 1.0 (exploit) and 1.96 (explore) each week and chosen per function. Early weeks leaned to exploration; late weeks to exploitation, apart from one bold final query on F5.
- **Strategy changes over the weeks.** Week 2 to 3: length-scale bounds, then a systematic search over kernel, bound and `beta`. Weeks 4 to 6: `alpha` added as a fourth setting. Week 9: duplicate detector. Week 10: log transform for F5. Week 11: TuRBO and the calibration check.

### Data handling and preprocessing
- **Rescaling.** Inputs were already in [0, 1], so no rescaling. Outputs are standardised inside the GP (`normalize_y=True`).
- **Surrogates.** A GP for every function. I also compared linear regression, SVM, a small neural network and local models (quadratic, RBF interpolation, nearest neighbour) on F4 and F7. The GP predicted held-out points better or equally well, so I kept it.
- **Preprocessing.** Kernel choice, length-scale bound and noise level were tuned per function. The stored data is never altered; only F5 is fitted on `log(y)`.
- **Outliers.** I did not remove any point. Extreme values (for example F4's -32.6 in the course data) stay in the data. A few exact repeats were flagged and de-duplicated only when replaying the TuRBO trust-region history.

### Ethical, practical and general considerations
- **Real-world relevance.** The same setting occurs in hyperparameter tuning, experiment design and drug or process optimisation, where each evaluation is expensive and gradients are unavailable.
- **Limits of synthetic functions.** The functions are generated, so there are no real costs, safety limits or people affected. Real problems add constraints, drift over time and human review that this task does not have.
- **Scaling.** The calibration check, duplicate check and trust regions scale well. A plain GP with random candidate points does not scale beyond roughly 10 inputs without changes, and exact GP fitting becomes slow beyond a few thousand points.
- **Pitfalls for a future user.** (1) Queries are not independent, so standard train/test splits are misleading. (2) A function's best value may be a lucky draw if the function is noisy (F6). (3) Several model settings fit almost equally well, so results can change between machines. (4) Do not use this dataset to benchmark other optimisers: it records one path-dependent sequence of choices, including manual overrides.
- **Data availability.** The course data is provided under the course's terms. The queries and results are my own project work.

---

## Part B: Per-function answers

### Function 1

**Function overview**
- Function 1, 2 inputs, 10 course points; output is a tiny signal that is essentially zero almost everywhere.

**Nature of the data**
- `X` (22, 2), `Y` (22,). Most weekly outputs are either exactly 0 or smaller in size than 1e-20 (down to 1e-192). Only four are larger: 2.7e-9, 1.4e-8, -2.3e-4 and -6.5e-11.
- Noise: not measured. The learned noise level was 0.95 (standardised), meaning the model explains the data as noise.
- Shape: almost entirely flat with at most one faint feature near `[0.65, 0.70]` (a dip of -0.00023 at `[0.66, 0.69]` and a small positive 1.4e-8 at `[0.6939, 0.7143]`). Unimodal at best; the data cannot show more.

**Optimisation strategy**
- Manual first query at `[0.5, 0.5]`, then GP-UCB, then an alternating GP and RBF-interpolation hybrid, plus corner probes (`[0, 1]` in week 4, `[1, 1]` in week 8).
- Why: with so little signal, any smooth model collapsed, so I tried to add local structure by hand.
- Strategy changed from model-driven to manual and hybrid; none beat the first feature meaningfully.

**Data handling and preprocessing**
- Surrogate: GP (Matern 1.5, length-scale bound 0.05, `alpha` 0.1). The fitted length-scale sits **at the lower bound (0.05)**, a sign of collapse.
- I tested sign-log and rank transforms. One looked promising but inverted the ranking of outcomes near zero, so I rejected it.

**Weekly iteration and learning**
- Each new point changed little: the landscape stayed flat except near `[0.65, 0.70]`.
- Most informative: the points around `[0.66, 0.69]`, which found the only feature.
- If I restarted: put several queries in a tight cluster around that feature in the first weeks, and not spend weeks on global models.

**Performance and results**
- Best: about 1.4e-8 (practically zero) at `[0.6939, 0.7143]`, week 7.
- Confidence that this is near the global maximum: low. Large flat regions are unexplored.
- Expectation: consistent with a needle-in-a-haystack function, but I did not locate a strong peak. Not placed on the leaderboard.

### Function 2

**Function overview**
- Function 2, 2 inputs, 10 course points; output is a score where higher is better.

**Nature of the data**
- `X` (22, 2), `Y` (22,). Weekly results ranged from -0.07 to 0.59.
- Noise: not measured directly; learned noise 0.106, borderline.
- Shape: appears **multimodal**. High values (0.4 to 0.6) occur near `x2` between 0.90 and 0.96 (with `x1` about 0.67 to 0.71) and also near `x2` between 0.08 and 0.18 (`x1` about 0.67 to 0.73).

**Optimisation strategy**
- GP-UCB with weekly comparison of `beta` 1.0 and 1.96. Settings by LML (Matern 1.5, bound 0.05, `alpha` 0.001).
- Why: only two inputs, so a dense grid of 2,500 candidates can be scored.
- No major strategy change; I probed both high-value regions but never improved on the course data.

**Data handling and preprocessing**
- GP on raw outputs. Fitted length-scale 0.071, close to the lower bound 0.05.

**Weekly iteration and learning**
- The two-region pattern became clear by week 8.
- Most informative: `[0.6735, 0.0816]` (0.52), which revealed the second region.
- If I restarted: split the budget between the two regions early and place queries on their ridges.

**Performance and results**
- Best: 0.6112 at `[0.7026, 0.9266]`. This is a **course data point**; my 12 results peaked at 0.593.
- Confidence: low. Sparse data in 2D and no improvement on the start.
- Expectation: below. Not placed on the leaderboard; a top performer reached 0.814.

### Function 3

**Function overview**
- Function 3, 3 inputs, 15 course points; every output is negative (closer to zero is better).

**Nature of the data**
- `X` (27, 3), `Y` (27,). Range -0.399 to -0.0007.
- Noise: not measured; learned noise 0.011 (small).
- Shape: looks smooth with one good region near `[0.41, 0.52, 0.48]`; calibration was borderline (25% outside 2 sd).

**Optimisation strategy**
- GP-UCB, then small steps around the best. Settings by LML (RBF, bound 0.2, `alpha` 0.1).
- Why: three inputs and a smooth surface suit a GP.
- Changed from broad coverage to local refinement once the centre region appeared.

**Data handling and preprocessing**
- GP on raw outputs; fitted length-scale 0.24.

**Weekly iteration and learning**
- Boundary points in weeks 3 and 4 were poor (-0.13, -0.09), so the interior is better.
- Most informative: week 8 at `[0.4098, 0.5249, 0.4771]` (-0.0007) and week 9 at `[0.2452, 0.4563, 0.4916]` (-0.0049).
- If I restarted: sample the centre region earlier.

**Performance and results**
- Best: -0.00073 at `[0.4098, 0.5249, 0.4771]`, week 8 (initial -0.035).
- Confidence: moderate; the value is very close to zero, which may be the ceiling.
- Leaderboard: 3rd.

### Function 4

**Function overview**
- Function 4, 4 inputs, 30 course points; output is a score where higher is better and values go as low as -32.6.

**Nature of the data**
- `X` (42, 4), `Y` (42,). Range -32.6 to 0.368.
- Noise: **none seen**. The same input in weeks 6 and 7 returned exactly 0.36746.
- Shape: rough and multimodal. Points only 0.05 apart ranged from 0.37 to -0.87. Calibration poor (33% outside 2 sd).

**Optimisation strategy**
- GP-UCB, then small steps around the best point (about ±0.05).
- Why: the model could not be trusted for jumps, so I stayed local.
- Early on the default noise setting extrapolated to unsupported high values; adding `alpha` to the search fixed that.

**Data handling and preprocessing**
- GP on raw outputs (Matern 2.5, bound 0.05, `alpha` 0.001; length-scale 0.77).
- I tested linear, SVM and neural-network models; the GP's held-out error (about 1.2) was best.

**Weekly iteration and learning**
- The week 6 result (0.367) turned the function positive after an all-negative history.
- Most informative: week 6 `[0.3846, 0.429, 0.4098, 0.3929]`.
- If I restarted: use more exploratory methods (for example Thompson sampling) to look for other regions.

**Performance and results**
- Best: 0.3675 at `[0.3846, 0.429, 0.4098, 0.3929]`, week 6.
- Confidence: low; the best may be a lucky spot. My last small step returned 0.055.
- Leaderboard: 34th; a top performer reached 0.566.

### Function 5

**Function overview**
- Function 5, 4 inputs, 20 course points; output is a yield-like quantity (higher is better).

**Nature of the data**
- `X` (32, 4), `Y` (32,). Range 0.11 to 8,267, tens of thousands of times.
- Noise: **none seen**. Three submissions of the same input all returned 5,431.47.
- Shape: smooth with large values when all four inputs are close to their upper limit.

**Optimisation strategy**
- GP-UCB, then (from week 10) a log transform and TuRBO, then a final query at the GP's peak on the boundary.
- Why: very wide output range and a calibration failure (a prediction missed by 7.4 standard deviations).
- Strategy changed from raw GP-UCB to log transform plus TuRBO after repeated duplicate queries.

**Data handling and preprocessing**
- Fitted on `log(y)` (Matern 2.5, bound 0.2, `alpha` 1e-6; length-scale 0.58). After this, only 8% of predictions missed by more than 2 sd.

**Weekly iteration and learning**
- The best points always sat near the upper limit of every input.
- Most informative: weeks 3 to 5 (4,397 to 5,431, with the last three inputs near 0.9 to 1.0) and week 13 `[1.0, 1.0, 0.9787, 1.0]` (8,267), which was above the model's prediction (about 6,500).
- If I restarted: test boundary corners much earlier.

**Performance and results**
- Best: 8,267 at `[1.0, 1.0, 0.9787, 1.0]`, week 13 (initial 1,089).
- Confidence: moderate; the corner may be near the maximum, but the local shape is unexplored.
- Leaderboard: 23rd

### Function 6

**Function overview**
- Function 6, 5 inputs, 20 course points; output is a negative score (closer to zero is better).

**Nature of the data**
- `X` (32, 5), `Y` (32,). Range -2.57 to -0.114.
- Noise: **yes**. The input `[0.4402, 0.3901, 0.6726, 0.6905, 0.0459]` gave -0.1144 (week 5), -0.1608 (week 7) and -0.1681 (week 13); week 12 had the same input but no recorded result. The first pair implies a noise variance of about 0.001.
- Shape: noisy and probably smooth.

**Optimisation strategy**
- GP-UCB with a noise setting (finer `alpha` grid 0.0005 to 0.01), plus an attempt to repeat the best input to measure noise.
- Why: the repeated input showed real noise.
- Changed from noise-free assumptions to an explicit noise level.

**Data handling and preprocessing**
- GP on raw outputs (Matern 1.5, bound 0.05, `alpha` 0.001; length-scale 0.92).

**Weekly iteration and learning**
- The week 5 best was not reproducible: repeats gave about -0.16 to -0.17.
- Most informative: the repeats themselves.
- If I restarted: repeat one point in week 2 and average, and query several points near the best.

**Performance and results**
- Best recorded: -0.1144, probably lucky; the typical value at that input is about -0.15.
- Confidence: low to moderate.
- Leaderboard: 12th.

### Function 7

**Function overview**
- Function 7, 6 inputs, 30 course points; output is a performance score (higher is better).

**Nature of the data**
- `X` (42, 6), `Y` (42,). Range 0.0027 to 3.25.
- Noise: not measured. Calibration poor (67% outside 2 sd).
- Shape: smooth enough for small local steps; best region near `[0.2, 0.2, 0.43, 0.27, 0.29, 0.64]`.

**Optimisation strategy**
- GP-UCB, then small steps around the best point (the "box pick": the GP's highest mean within ±0.05).
- Why: the model was too poorly calibrated for big jumps, but local steps worked.
- Changed from global UCB to local exploitation late in the project.

**Data handling and preprocessing**
- GP on raw outputs (Matern 1.5, bound 0.2, `alpha` 1e-6; length-scale 0.68).

**Weekly iteration and learning**
- Week 6 gave 3.083; many later queries were lower (1.9 to 2.7), showing the surface is sensitive.
- Most informative: week 6 `[0.1682, 0.1899, 0.4607, 0.2863, 0.2475, 0.6452]` and week 13 `[0.2159, 0.1817, 0.4316, 0.2663, 0.2863, 0.6392]`.
- If I restarted: take small steps around the best point earlier.

**Performance and results**
- Best: 3.248 at `[0.2159, 0.1817, 0.4316, 0.2663, 0.2863, 0.6392]`, week 13 (initial 1.36).
- Confidence: moderate.
- Leaderboard: 7th

### Function 8

**Function overview**
- Function 8, 8 inputs, 40 course points; output is a performance score (higher is better).

**Nature of the data**
- `X` (52, 8), `Y` (52,). Range 5.59 to 9.998; all my weekly results are above 9.7.
- Noise: not measured. Calibration very good (0% outside 2 sd).
- Shape: smooth with a high plateau; best region near `[0.11, 0.15, 0.12, 0.17, 0.78, 0.49, 0.21, 0.52]`.

**Optimisation strategy**
- GP-UCB for ten weeks (results 9.74 to 9.91), then TuRBO from week 11.
- Why: in 8 dimensions a global search spreads too thin, and the calibration check passed.
- Changed to TuRBO after the check; two TuRBO queries both set new bests.

**Data handling and preprocessing**
- GP on raw outputs (Matern 2.5, bound 0.05, `alpha` 1e-6; length-scale 1.68). For TuRBO, an ARD GP with one length-scale per input.

**Weekly iteration and learning**
- TuRBO's week-13 prediction (9.998, range 9.996 to 9.9995) matched the result (9.9983).
- Most informative: week 11 `[0.1041, 0.1625, 0.1219, 0.157, 0.791, 0.5024, 0.2202, 0.5199]`.
- If I restarted: use TuRBO from about week 5.

**Performance and results**
- Best: 9.99833 at `[0.1118, 0.1486, 0.1223, 0.1653, 0.7849, 0.4948, 0.2110, 0.5226]`, week 13 (initial 9.598).
- Confidence: moderate to high locally, unknown globally; gains were already tiny.
- Leaderboard: 5th.
