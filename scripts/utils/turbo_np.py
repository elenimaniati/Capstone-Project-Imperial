"""NumPy/SciPy re-implementation of the TuRBO notebook used in earlier weeks
(ARD-RBF GP with constant mean + output scale, SciPy L-BFGS-B fit, Thompson
sampling in a trust region, centroid of 200 picks). No torch needed."""
import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import qmc

LS_BOUNDS = (0.05, 10.0)
NOISE_LB = 1e-6


# ---------------- TuRBO state ----------------
@dataclass
class TurboState:
    dim: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max(4.0, float(self.dim)))


def update_state(s, y):
    if y > s.best_value + 1e-3 * abs(s.best_value):
        s.success_counter += 1
        s.failure_counter = 0
    else:
        s.success_counter = 0
        s.failure_counter += 1
    if s.success_counter == s.success_tolerance:
        s.length = min(2.0 * s.length, s.length_max)
        s.success_counter = 0
    elif s.failure_counter == s.failure_tolerance:
        s.length /= 2.0
        s.failure_counter = 0
    s.best_value = max(s.best_value, y)
    if s.length < s.length_min:
        s.restart_triggered = True
    return s


def replay_state(Y0, weekly_X, weekly_Y, dim):
    """Replay real history; exact repeats (same x AND same y) are skipped."""
    s = TurboState(dim=dim, best_value=float(np.max(Y0)))
    seen = []
    skipped = 0
    for x, y in zip(weekly_X, weekly_Y):
        if any(np.allclose(x, px) and y == py for px, py in seen):
            skipped += 1
            continue
        seen.append((np.array(x), y))
        s = update_state(s, float(y))
    return s, skipped


# ---------------- GP ----------------
def _rbf(A, B, ls):
    A = A / ls
    B = B / ls
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-0.5 * np.maximum(d2, 0.0))


class GP:
    def __init__(self, X, y_raw):
        self.X = np.asarray(X, float)
        self.mu_y = y_raw.mean()
        self.sd_y = y_raw.std(ddof=1)          # torch .std() is unbiased
        self.y = (y_raw - self.mu_y) / self.sd_y
        self.d = self.X.shape[1]

    # theta = [c, log ls (d), log outputscale, log noise]
    def _unpack(self, th):
        d = self.d
        return th[0], np.exp(th[1:1 + d]), np.exp(th[1 + d]), np.exp(th[2 + d])

    def nlml(self, th):
        c, ls, os_, nz = self._unpack(th)
        n = len(self.y)
        K = os_ * _rbf(self.X, self.X, ls) + (nz + 1e-8) * np.eye(n)
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        r = self.y - c
        a = cho_solve(cf, r)
        logdet = 2 * np.log(np.diag(cf[0])).sum()
        return 0.5 * r @ a + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)

    def nlml_grad(self, th):
        c, ls, os_, nz = self._unpack(th)
        n = len(self.y); d = self.d
        Kr = _rbf(self.X, self.X, ls)
        K = os_ * Kr + (nz + 1e-8) * np.eye(n)
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(th)
        r = self.y - c
        a = cho_solve(cf, r)
        Kinv = cho_solve(cf, np.eye(n))
        logdet = 2 * np.log(np.diag(cf[0])).sum()
        f = 0.5 * r @ a + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
        W = Kinv - np.outer(a, a)                 # dNLML/dK = 0.5 * W
        g = np.zeros_like(th)
        g[0] = -a.sum()
        OK = os_ * Kr
        for j in range(d):
            diff2 = (self.X[:, j][:, None] - self.X[:, j][None, :]) ** 2 / ls[j] ** 2
            g[1 + j] = 0.5 * np.sum(W * (OK * diff2))
        g[1 + d] = 0.5 * np.sum(W * OK)
        g[2 + d] = 0.5 * np.trace(W) * nz
        return f, g

    def fit(self, n_restarts=10, seed=0):
        d = self.d
        bounds = ([(-1e3, 1e3)] + [(np.log(LS_BOUNDS[0]), np.log(LS_BOUNDS[1]))] * d
                  + [(np.log(1e-4), np.log(1e8))] + [(np.log(NOISE_LB), np.log(10.0))])
        rng = np.random.default_rng(seed)
        starts = [np.concatenate([[0.0], np.full(d, np.log(0.693)), [np.log(0.693)], [np.log(0.693)]])]
        for _ in range(n_restarts):
            starts.append(np.array([rng.normal(0, 0.5)] +
                                   list(rng.uniform(np.log(0.1), np.log(5.0), d)) +
                                   [rng.uniform(np.log(0.05), np.log(50.0))] +
                                   [rng.uniform(np.log(1e-5), np.log(0.5))]))
        for _ in range(4):
            starts.append(np.array([rng.normal(0, 3.0)] + list(rng.uniform(np.log(1.5), np.log(8.0), d)) +
                                   [rng.uniform(np.log(5.0), np.log(1e4))] + [np.log(1e-5)]))
        best = None
        for s0 in starts:
            s0 = np.clip(s0, [b[0] for b in bounds], [b[1] for b in bounds])
            r = minimize(self.nlml_grad, s0, jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=2000, maxfun=50000, ftol=1e-12, gtol=1e-8))
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        self.c, self.ls, self.os, self.noise = self._unpack(best.x)
        self.lml = -best.fun            # total log marginal likelihood (standardised y)
        K = self.os * _rbf(self.X, self.X, self.ls) + (self.noise + 1e-8) * np.eye(len(self.y))
        self.cf = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.cf, self.y - self.c)
        return self

    def predict_std_units(self, Xs, full_cov=False):
        Ks = self.os * _rbf(Xs, self.X, self.ls)
        mean = self.c + Ks @ self.alpha
        v = cho_solve(self.cf, Ks.T)
        if full_cov:
            cov = self.os * _rbf(Xs, Xs, self.ls) - Ks @ v
            return mean, cov
        var = self.os - (Ks * v.T).sum(1)
        return mean, np.maximum(var, 1e-12)

    def predict(self, Xs, include_noise=False):
        Xs = np.atleast_2d(Xs)
        m, v = self.predict_std_units(Xs)
        if include_noise:
            v = v + self.noise
        return m * self.sd_y + self.mu_y, np.sqrt(v) * self.sd_y


# ---------------- candidates / sampling ----------------
def generate_batch(state, gp, X, y_for_best, n_draws=10, seed=0):
    dim = X.shape[1]
    n_cand = min(5000, max(2000, 200 * dim))
    x_center = X[np.argmax(y_for_best)].copy()
    w = gp.ls / gp.ls.mean()
    w = w / np.prod(w ** (1.0 / dim))
    lb = np.clip(x_center - w * state.length / 2.0, 0.0, 1.0)
    ub = np.clip(x_center + w * state.length / 2.0, 0.0, 1.0)
    sob = qmc.Sobol(d=dim, scramble=True, seed=seed)
    pert = lb + (ub - lb) * sob.random(n_cand)
    rng = np.random.default_rng(seed)
    prob = min(20.0 / dim, 1.0)
    mask = rng.random((n_cand, dim)) <= prob
    for i in np.where(mask.sum(1) == 0)[0]:
        mask[i, rng.integers(0, dim)] = True
    Xc = np.tile(x_center, (n_cand, 1))
    Xc[mask] = pert[mask]
    m, cov = gp.predict_std_units(Xc, full_cov=True)
    cov = 0.5 * (cov + cov.T)
    jit = 1e-8
    for _ in range(8):
        try:
            L = np.linalg.cholesky(cov + jit * np.eye(n_cand))
            break
        except np.linalg.LinAlgError:
            jit *= 10
    z = rng.standard_normal((n_cand, n_draws))
    samples = m[:, None] + L @ z
    return Xc[np.argmax(samples, axis=0)], (lb, ub)


def next_query(state, gp, X, y_for_best, n_sets=20, n_draws=10):
    picks = []
    for k in range(n_sets):
        x, box = generate_batch(state, gp, X, y_for_best, n_draws=n_draws, seed=k)
        picks.append(x)
    P = np.vstack(picks)
    cen = P.mean(0)
    pm, _ = gp.predict(P)
    cm, _ = gp.predict(cen)
    if cm[0] >= np.median(pm):
        return cen, P, "centroid of 200 Thompson picks", box
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    return P[np.argmin(d.sum(1))], P, "MEDOID (centroid scored below typical pick)", box


def maximise_posterior_mean(gp, lb, ub, n_starts=30, seed=0):
    rng = np.random.default_rng(seed)

    def f(v):
        return -gp.predict_std_units(v.reshape(1, -1))[0][0]

    best = None
    for _ in range(n_starts):
        v0 = lb + (ub - lb) * rng.random(len(lb))
        r = minimize(f, v0, method="L-BFGS-B", bounds=list(zip(lb, ub)))
        if best is None or r.fun < best.fun:
            best = r
    return best.x
