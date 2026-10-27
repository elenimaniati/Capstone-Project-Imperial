"""
Helper functions for the BBO capstone: data loading, GP + UCB tuning,
forward-chaining calibration check and duplicate detection.

Everything here is used by notebooks/BBO_capstone_final.ipynb. The functions are
deliberately small and commented so each step of the method can be read on its own.

Conventions
-----------
* Every input lies in [0, 1]^d. Function f has d = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}[f].
* data/queries.csv holds ALL rows: week 0 = the initial data supplied by the course,
  weeks 1-13 = my weekly queries. The week-12 result never arrived, so its y is blank.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

warnings.filterwarnings("ignore")

SEED = 42
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

# The three kernels compared every week (all have a length-scale; Matern is rougher than RBF).
KERNELS = {
    "RBF": lambda ls, bounds: RBF(ls, bounds),
    "Matern_nu1.5": lambda ls, bounds: Matern(ls, bounds, nu=1.5),
    "Matern_nu2.5": lambda ls, bounds: Matern(ls, bounds, nu=2.5),
}


# --------------------------------------------------------------------------- data
def load_queries(path=None):
    """Read data/queries.csv (works from the repo root or from notebooks/)."""
    if path is None:
        here = Path(__file__).resolve()
        path = here.parents[2] / "data" / "queries.csv"
    return pd.read_csv(path)


def get_xy(df, f, last_week=None):
    """Inputs X and outputs Y for function f, using only rows whose result is known.

    last_week: keep rows up to and including this week (None = everything).
    """
    d = df[(df.function == f) & df.y.notna()]
    if last_week is not None:
        d = d[d.week <= last_week]
    d = d.sort_values("week", kind="stable")
    X = d[[f"x{i + 1}" for i in range(DIMS[f])]].to_numpy(float)
    return X, d.y.to_numpy(float), d.week.to_numpy(int)


# ------------------------------------------------------------------------ GP model
def fit_gp(X, Y, kernel="Matern_nu1.5", lower_bound=0.05, alpha=1e-6,
           log_y=False, n_restarts=5):
    """Fit a Gaussian-process surrogate.

    lower_bound : smallest allowed length-scale. Without it the length-scale can
                  collapse to ~0 on flat data, giving a useless flat posterior (F1, F2).
    alpha       : assumed noise variance added to the kernel diagonal (in standardised units).
    log_y       : fit on log(Y); only valid when all Y > 0 (used for F5, whose outputs span tens of thousands of times).
    """
    k = KERNELS[kernel](0.1, (lower_bound, 2.0))
    gp = GaussianProcessRegressor(kernel=k, alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, random_state=SEED)
    gp.fit(X, np.log(Y) if log_y else Y)
    return gp


def candidate_points(dim, n_random=20000, grid=50):
    """Points the acquisition function is scored on: a 50x50 grid in 2-D, random points otherwise."""
    if dim == 2:
        g = np.linspace(0, 1, grid)
        return np.array(np.meshgrid(g, g)).reshape(2, -1).T
    return np.random.RandomState(SEED).uniform(0, 1, (n_random, dim))


def tune_gp(X, Y, log_y=False, kernels=KERNELS.keys(), lower_bounds=(0.05, 0.2),
            alphas=(1e-6, 1e-3, 1e-1), n_restarts=3):
    """Try every (kernel, length-scale bound, alpha) and rank by log marginal likelihood (LML).

    LML rewards models that explain the data without being needlessly complex, so it is a
    fair way to choose settings without a separate validation set (we only have ~20-50 points).
    Returns a table sorted best-first.
    """
    rows = []
    for kern in kernels:
        for lb in lower_bounds:
            for a in alphas:
                gp = fit_gp(X, Y, kern, lb, a, log_y, n_restarts)
                ls = float(np.mean(np.atleast_1d(gp.kernel_.length_scale)))
                rows.append(dict(kernel=kern, lower_bound=lb, alpha=a,
                                 lml=gp.log_marginal_likelihood(gp.kernel_.theta),
                                 length_scale=ls, pinned=abs(ls - lb) < 1e-3))
    return pd.DataFrame(rows).sort_values("lml", ascending=False).reset_index(drop=True)


def ucb_pick(gp, dim, beta=1.96, log_y=False):
    """Upper confidence bound: score = mean + beta * std. Large beta = explore, small = exploit."""
    cand = candidate_points(dim)
    mean, std = gp.predict(cand, return_std=True)
    i = int(np.argmax(mean + beta * std))
    pred = float(np.exp(mean[i])) if log_y else float(mean[i])
    return cand[i], pred, float(std[i])


# ------------------------------------------------------------- duplicate detection
def nearest_prior(X, x_new):
    """Distance from a proposed point to the closest point already queried."""
    return float(np.linalg.norm(X - np.asarray(x_new), axis=1).min())


def is_duplicate(X, x_new, threshold=0.02):
    """True if the proposal is within `threshold` of an earlier query (wasted query)."""
    return nearest_prior(X, x_new) < threshold


# ----------------------------------------------------- forward-chaining calibration
def forward_chaining_z(X, Y, weeks, log_y=False, n_restarts=4):
    """Calibration check (the test I used to decide how far to trust each model).

    For every weekly query, refit the model on ONLY the data available before that
    query, predict it, and compute z = (actual - predicted) / predictive_std.
    If the model's uncertainty is honest, about 95% of |z| should be below 2.
    The predictive std includes the model's learned noise, as a fair test requires.

    Uses the ARD-RBF GP in turbo_np.py (one length-scale per input, learned noise),
    the same model TuRBO uses, so the check and the optimiser agree.
    Returns one row per weekly query plus the learned noise at the last fit.
    """
    import turbo_np as T
    yy = np.log(Y) if log_y else Y
    rows, noise = [], np.nan
    for i in np.where(weeks >= 1)[0]:
        gp = T.GP(X[:i], yy[:i]).fit(n_restarts=n_restarts)
        m, s = gp.predict(X[i:i + 1], include_noise=True)
        rows.append(dict(week=int(weeks[i]), actual=float(Y[i]),
                         predicted=float(np.exp(m[0]) if log_y else m[0]),
                         z=float((yy[i] - m[0]) / s[0])))
        noise = float(gp.noise)
    out = pd.DataFrame(rows)
    out.attrs["noise"] = noise
    return out


def calibration_summary(z, noise=0.0):
    """Fraction of |z|>2 (target ~5%), mean z, and a plain-English verdict.

    Verdict rules: learned noise above 0.5 (in standardised units) means the model
    blames most variation on noise, so it cannot guide search; more than 30% of
    predictions outside 2 sd = poorly calibrated; 15-30% = borderline.
    """
    frac = float((np.abs(z) > 2).mean())
    if noise > 0.5:
        verdict = "not usable (model explains the data as noise)"
    elif frac > 0.30:
        verdict = "poorly calibrated: do not trust model-guided jumps"
    elif frac > 0.15:
        verdict = "borderline: use small steps only"
    else:
        verdict = "acceptable: model-guided steps are reasonable"
    return frac, float(np.mean(z)), verdict


# --------------------------------------------------------------------------- TuRBO
def turbo_suggestion(X, Y, weeks, log_y=False):
    """Trust-region (TuRBO) suggestion from the current data.

    1. Replay the real history to get the trust-region size (grows after successes,
       halves after repeated failures). Exact repeats (same x and same y) are skipped.
    2. Fit the ARD GP on all data.
    3. Draw Thompson samples inside a box around the best point (box sides follow the
       learned length-scales), and return the centroid of 200 picks.
    Returns (x_next, predicted value, predicted +-1 sd, trust-region state, GP).
    """
    import turbo_np as T
    init = weeks == 0
    state, _ = T.replay_state(Y[init], X[~init], Y[~init], X.shape[1])
    yy = np.log(Y) if log_y else Y
    gp = T.GP(X, yy).fit()
    x_next, _, how, _ = T.next_query(state, gp, X, yy)
    m, s = gp.predict(x_next)
    f = np.exp if log_y else (lambda v: v)
    return x_next, float(f(m[0])), (float(f(m[0] - s[0])), float(f(m[0] + s[0]))), state, gp
