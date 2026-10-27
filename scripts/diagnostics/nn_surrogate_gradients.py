"""
Trains a small MLP (neural network) surrogate for one function, compares
its training R^2 against a plain linear regression on the same data (to
check whether the added flexibility is actually earning its keep), and
computes approximate gradients (via finite differences) at the current
best point -- the NN analogue of the GP partial-dependence sweep done
earlier in this project.

Run from the directory containing initial_data/.
"""

import numpy as np
import warnings
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

week1_X = {8: [0.158359, 0.089191, 0.006755, 0.068470, 0.779613, 0.339703, 0.109925, 0.073427]}
week1_Y = {8: 9.8674177220056}
week2_X = {8: [0.192332, 0.148385, 0.293081, 0.032111, 0.743423, 0.258716, 0.124238, 0.699129]}
week2_Y = {8: 9.8169820770504}
week3_X = {8: [0.0955, 0.2689, 0.0107, 0.4009, 0.9253, 0.8020, 0.0575, 0.7128]}
week3_Y = {8: 9.739235081}


def load_function_data(function_id):
    X0 = np.load(f'data/initial_data/function_{function_id}/initial_inputs.npy')
    Y0 = np.load(f'data/initial_data/function_{function_id}/initial_outputs.npy')
    X = np.vstack([X0, week1_X[function_id], week2_X[function_id], week3_X[function_id]])
    Y = np.concatenate([Y0, [week1_Y[function_id], week2_Y[function_id], week3_Y[function_id]]])
    return X, Y


def train_nn_surrogate_and_check_gradients(function_id, eps=0.01):
    X, Y = load_function_data(function_id)
    n_dims = X.shape[1]

    # --- NN surrogate ---
    mlp = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=5000, random_state=42, alpha=0.01)
    mlp.fit(X, Y)
    nn_r2 = mlp.score(X, Y)

    # --- linear regression, same data, for comparison ---
    lr = LinearRegression().fit(X, Y)
    lr_r2 = lr.score(X, Y)

    print(f"Function {function_id}: n_obs={len(Y)}, dims={n_dims}")
    print(f"  NN surrogate train R^2:     {nn_r2:.4f}")
    print(f"  Linear regression train R^2: {lr_r2:.4f}")
    print(f"  (NN uses ~{n_dims*16 + 16 + 16*16 + 16 + 16 + 1} parameters vs. "
          f"linear regression's {n_dims+1} -- worth comparing the R^2 gain against that cost)\n")

    # --- gradients at the current best point, via finite differences ---
    baseline = X[np.argmax(Y)].copy()
    grads = []
    for dim in range(n_dims):
        x_plus = baseline.copy(); x_plus[dim] = min(1.0, x_plus[dim] + eps)
        x_minus = baseline.copy(); x_minus[dim] = max(0.0, x_minus[dim] - eps)
        grad = (mlp.predict([x_plus])[0] - mlp.predict([x_minus])[0]) / (x_plus[dim] - x_minus[dim])
        grads.append(grad)
        print(f"  dim {dim+1}: gradient={grad:.3f}")

    order = np.argsort(-np.abs(grads))
    print(f"\n  Dimensions ranked by |gradient| (most influential first): "
          f"{[d+1 for d in order]}")

    return mlp, grads


if __name__ == '__main__':
    train_nn_surrogate_and_check_gradients(function_id=8)
