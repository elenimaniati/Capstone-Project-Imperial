# Model Card: Adaptive GP-UCB with a calibration gate, for black-box optimisation

Following the course template and the idea of model cards (Mitchell et al., 2019). Numbers come from `notebooks/BBO_capstone_final.ipynb` and `data/queries.csv`, using results from weeks 0 to 13 (the week-12 results never arrived).

## Model Description

**Input.**
- The history for one function: input vectors `X` (every value in [0, 1], 2 to 8 inputs) and observed outputs `Y`. The history holds 22 to 52 points per function.
- Settings: kernel type, smallest allowed length-scale, noise level (`alpha`), exploration weight (`beta`), and whether to fit on `log(y)`.

**Output.**
- A prediction (mean and standard deviation) at any input.
- One proposed next input per function: the point with the highest `mean + beta * std` (UCB), or, for TuRBO, the centre of Thompson-sampling picks inside a trust region.
- Flags: a duplicate warning if the proposal is within 0.02 of an earlier query, and a calibration verdict per function.

**Model Architecture.**
1. **Surrogate:** a Gaussian process (scikit-learn `GaussianProcessRegressor`; kernels RBF, Matern 1.5 or Matern 2.5; standardised outputs; 10 optimiser restarts; seed 42).
2. **Acquisition:** upper confidence bound, scored on a 50 x 50 grid (2 inputs) or 20,000 random points (more inputs).
3. **Settings search:** every combination of kernel, length-scale bound and `alpha` is fitted and ranked by log marginal likelihood (LML).
4. **Calibration gate:** before trusting a model, refit on only the earlier data for each past query and check the misses against the model's stated uncertainty.
5. **TuRBO** (NumPy/SciPy, `scripts/utils/turbo_np.py`): an ARD-RBF GP with learned noise, a trust region around the best point that grows after successes and halves after failures, and Thompson sampling inside it. Used on F8 and F5.
6. **Safety checks:** duplicate detector (threshold 0.02) and a log transform for F5 only.

**Intended use.** Low-dimensional (2 to 8 inputs), continuous, expensive-to-evaluate functions with one query per round. **Not intended for** discrete or combinatorial spaces, batch (parallel) queries, more than about 10 inputs without changes, or fully automatic use without review.

## Performance

**How it was measured.**
- *Outcome:* best value found per function, over weeks 0 to 13.
- *Calibration:* for each of the 12 known weekly results, refit on earlier data only and compute z = (actual - predicted) / predicted std. Target: about 5% of |z| above 2.
- *Model comparison:* forward-chaining prediction error on F4 and F7 (data through week 10).
- *TuRBO backtests:* data before a week, suggest, then compare the prediction with the real result.

**Final results (best value found).**

| Function | Course data best | My best | Leaderboard place (of 40) |
|---|---|---|---|
| 1 | 7.7e-16 | 1.4e-8 | not listed |
| 2 | 0.611 | 0.611 | not listed |
| 3 | -0.0348 | -0.00073 | 3rd |
| 4 | -4.03 | 0.368 | 34th |
| 5 | 1,089 | 8,267 | 23rd |
| 6 | -0.714 | -0.114 | 12th |
| 7 | 1.36 | 3.248 | 7th |
| 8 | 9.598 | 9.998 | 5th |


![Best value found per function](../results/best_so_far.png)

**Calibration (12 checks per function).**

| Function | Share outside 2 sd | Learned noise | Verdict |
|---|---|---|---|
| 1 | 8% | 0.95 | not usable (explained as noise) |
| 2 | 17% | 0.11 | borderline |
| 3 | 25% | 0.01 | borderline |
| 4 | 33% | 0.003 | poorly calibrated |
| 5 (log scale) | 8% | 0.00 | acceptable |
| 6 | 25% | 0.01 | borderline |
| 7 | 67% | 0.00 | poorly calibrated |
| 8 | 0% | 0.00 | acceptable |

With only 12 checks, one extra miss changes the share by 8 points, so these are signals and not precise rates.

![Calibration check](../results/calibration.png)

**Other evidence.**
- *TuRBO on F8:* predicted 9.998 (range 9.996 to 9.9995) for week 13 and returned 9.9983, a new best. For week 11 it predicted 10.0006 and returned 9.9981, also a new best.
- *TuRBO on F5:* predictions for weeks 11 and 13 were about 5,600 and 6,500. Week 11 returned 5,856 (inside the range). Week 13 was above the range: I submitted the boundary corner instead, which returned 8,267, so the model under-estimated the boundary.
- *Surrogate comparison (F4 / F7 error through week 10):* GP 1.23 / 0.64; local quadratic 2.67 / 0.73; RBF interpolation 1.50 / 0.63; nearest neighbour 1.79 / 0.71. The GP won on F4 and tied on F7, and it also gives an uncertainty. Linear, SVM and neural-network surrogates were no better and gave no usable uncertainty.

## Limitations

- **Assumes one smoothness everywhere** (stationarity). F5 broke this (its output range is huge), which needed a log transform.
- **Assumes smooth, continuous functions.** F1 (a needle in a flat region) and F4 (rough) violate this. On F1 the length-scale collapsed to its lower bound (0.05) and the model explains the data as noise.
- **Assumes independent Gaussian noise.** Confirmed only for F6 (the same input returned -0.114, -0.161 and -0.168). Other functions were not tested by repeated measurement. F4 and F5 returned identical values for repeated inputs, so they look noise-free at those points.
- **Approximate acquisition.** UCB is maximised over a finite candidate set, which gets sparser as inputs increase.
- **Near-tied settings.** Several settings have almost identical LML, so the top-ranked one can change between machines or after one more point.
- **Lucky results.** F6's best is probably a lucky draw, and F4's best may be a lucky spot. A best value found does not prove the model understands the function.
- **Missing data and small samples.** The week-12 results never arrived. Calibration rests on 12 checks per function.
- **Failure modes met:** collapsed length-scale (F1, F2), repeated queries when exploit and explore picks agree, extreme-scale miscalibration (F5), and unreliable far-away proposals (for example, F5's pure-exploration proposal predicts about 745 against a best of 8,267).

## Trade-offs

- **Caution versus boldness.** Where the calibration check failed I used small steps near known good points. That protected the model-guided functions but may have cost F4, F2 and F5 early on. A single bold final query on F5 gave the largest gain of the project.
- **Simple versus sophisticated.** A plain GP beat or matched every alternative tried, while TuRBO helped only where the model was well calibrated (F8, F5).
- **Search cost.** Every extra setting multiplies the number of fits, so I kept the grid small and accepted near-ties.
- **Effort across functions.** Time spent rescuing F1 gave little. The calibration check showed early which functions deserved effort.
- **Reproducibility versus a forced answer.** I reported near-tied settings as ties and did not pick a "winner" that the data could not support.

## How the model changed

- **Weeks 1 to 3:** RBF kernel with UCB, then length-scale bounds and a systematic search over kernel, bound and `beta` by LML.
- **Weeks 4 to 6:** noise level (`alpha`) added after F4 extrapolated to unsupported values.
- **Weeks 7 to 9:** F1 hybrid (GP alternating with RBF interpolation), a noise setting for F6, and the duplicate detector.
- **Week 10:** log transform for F5.
- **Week 11 onward:** calibration gate and TuRBO for F8 and F5; small steps for F4 and F7.

## Ethical considerations

Every change was tied to a test that can be rerun (a calibration check, an LML comparison, a recorded reason for an override), so a reviewer can check the reasoning and not just trust it. Failures are reported, not hidden: the repeated queries, the miscalibration, the transform I rejected because it inverted the ranking of outcomes. The functions are synthetic, so no people or real resources are affected. Applying this to real experiments (drugs, processes) would need domain review, safety limits and a human decision on every query.

## References

- Mitchell, M. et al. (2019). Model cards for model reporting. *FAT\**.
- Rasmussen, C. and Williams, C. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Eriksson, D. et al. (2019). Scalable global optimization via local Bayesian optimization. *NeurIPS*.
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *JMLR*.
