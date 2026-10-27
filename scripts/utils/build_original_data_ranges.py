"""
Builds a summary of the ORIGINAL provided data only (initial_inputs.npy /
initial_outputs.npy), excluding any of your own query rounds -- i.e. what
the domain and output range looked like before you started querying.

Run from the directory containing initial_data/ (same as your other scripts).
"""

import numpy as np
import pandas as pd

rows = []
for i in range(1, 9):
    X0 = np.load(f'data/initial_data/function_{i}/initial_inputs.npy')
    Y0 = np.load(f'data/initial_data/function_{i}/initial_outputs.npy')

    rows.append({
        'function': i,
        'dims': X0.shape[1],
        'n_obs_original': X0.shape[0],
        'x_min_per_dim': [round(v, 4) for v in X0.min(axis=0)],
        'x_max_per_dim': [round(v, 4) for v in X0.max(axis=0)],
        'y_min': round(Y0.min(), 6),
        'y_max': round(Y0.max(), 6),
    })

df = pd.DataFrame(rows)
df.to_csv('original_data_ranges.csv', index=False)

pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 80)
print(df.to_string(index=False))
print("\nSaved original_data_ranges.csv")
