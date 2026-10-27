"""
Computes exploration-focused alternatives to the standard UCB pick, for the
functions where the standard pick deserves the most skepticism:

  Function 1 & 2: kernel is pinned at its length_scale lower bound (low
                  confidence away from existing points). Alternative =
                  the single grid point furthest (in Euclidean distance)
                  from every existing observation -- i.e. genuinely
                  unexplored territory, picked independently of the
                  (unreliable) GP posterior.

  Function 4:     kernel fit is fine, but beta=1.96 keeps exploiting right
                  next to the current best. Alternative = sweep beta to
                  find the smallest value that produces a genuinely
                  different candidate (not just a tiny nudge), and use
                  that as a moderate-exploration pick rather than jumping
                  straight to beta values so high they ignore the mean
                  entirely.

Edit week1_X/week1_Y/week2_X/week2_Y below to update with your latest
round's data before rerunning.
"""

import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

warnings.filterwarnings('ignore')
RNG_SEED = 42

week1_X = {1: [0.5, 0.5], 2: [0.816326, 0.959183], 4: [0.394868, 0.425405, 0.352448, 0.464929]}
week1_Y = {1: 2.6752879910742468e-9, 2: -0.06913022970475102, 4: -0.3932940148544044}
week2_X = {1: [0.6, 0.3], 2: [0.673469, 0.897959], 4: [0.451289, 0.457435, 0.214355, 0.422482]}
week2_Y = {1: 3.7759561298146587e-34, 2: 0.4900359752987052, 4: -3.2314120937555297}


def most_underexplored_point(function_id, week1_x, week2_x):
    """Grid point (2D only) furthest from every existing observation."""
    X0 = np.load(f'data/initial_data/function_{function_id}/initial_inputs.npy')
    X = np.vstack([X0, week1_x, week2_x])

    g = np.linspace(0, 1, 50)
    Xg, Yg = np.meshgrid(g, g)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])

    dists = np.min(np.linalg.norm(grid[:, None, :] - X[None, :, :], axis=2), axis=1)
    idx = np.argmax(dists)
    return grid[idx], dists[idx], X


def beta_sweep_first_shift(function_id, week1_x, week1_y, week2_x, week2_y,
                            n_dims, betas_to_try, length_scale_bound=(0.05, 2.0),
                            n_candidates=20000):
    """Fits once, then finds the smallest beta in `betas_to_try` whose pick
    genuinely differs from the beta=1.96 baseline pick."""
    X0 = np.load(f'data/initial_data/function_{function_id}/initial_inputs.npy')
    Y0 = np.load(f'data/initial_data/function_{function_id}/initial_outputs.npy')
    X = np.vstack([X0, week1_x, week2_x])
    Y = np.append(Y0, [week1_y, week2_y])

    kernel = RBF(length_scale=0.1, length_scale_bounds=length_scale_bound)
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                      n_restarts_optimizer=10, random_state=RNG_SEED)
    model.fit(X, Y)

    rng = np.random.RandomState(RNG_SEED)
    candidates = rng.uniform(0, 1, size=(n_candidates, n_dims))
    post_mean, post_std = model.predict(candidates, return_std=True)

    results = []
    for beta in betas_to_try:
        ucb = post_mean + beta * post_std
        idx = np.argmax(ucb)
        results.append({
            'beta': beta,
            'next_x': candidates[idx].round(4),
            'pred_mean': round(post_mean[idx], 4),
            'pred_std': round(post_std[idx], 4),
        })
    return results


if __name__ == '__main__':
    print("=== Function 1: most under-explored point ===")
    pt, dist, X = most_underexplored_point(1, week1_X[1], week2_X[1])
    print(f"  exploration alternative x = {pt.round(4)}  (distance {dist:.4f} from nearest sample)")

    print("\n=== Function 2: most under-explored point ===")
    pt, dist, X = most_underexplored_point(2, week1_X[2], week2_X[2])
    print(f"  exploration alternative x = {pt.round(4)}  (distance {dist:.4f} from nearest sample)")

    print("\n=== Function 4: beta sweep to find a moderate-exploration pick ===")
    results = beta_sweep_first_shift(4, week1_X[4], week1_Y[4], week2_X[4], week2_Y[4],
                                      n_dims=4, betas_to_try=[1.96, 2.5, 3.0, 3.2, 3.5, 4.0])
    for r in results:
        print(f"  beta={r['beta']:.2f}  next_x={r['next_x']}  "
              f"pred_mean={r['pred_mean']}  pred_std={r['pred_std']}")
    print("\n  Recommendation: use the smallest beta whose next_x first differs "
          "meaningfully from the beta=1.96 row -- that's the moderate-exploration "
          "pick (a big jump in pred_std without an extreme drop in pred_mean). "
          "Avoid jumping straight to the highest beta tried, since that tends to "
          "chase pure uncertainty regardless of expected value.")
