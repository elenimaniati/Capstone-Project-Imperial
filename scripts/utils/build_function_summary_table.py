"""
Builds a summary table across all 8 functions: dimensions, total observed
Y range (original data + all 3 query rounds), and each round's query point
and result.

Run from the directory containing initial_data/ (same as your other scripts).
"""

import numpy as np
import pandas as pd

week1_X = {
    1: [0.500000, 0.500000], 2: [0.816326, 0.959183], 3: [0.406027, 0.410592, 0.490850],
    4: [0.394868, 0.425405, 0.352448, 0.464929], 5: [0.323763, 0.780786, 0.952277, 0.952084],
    6: [0.345264, 0.328436, 0.412802, 0.727459, 0.096332],
    7: [0.057185, 0.424604, 0.394749, 0.207966, 0.379147, 0.653218],
    8: [0.158359, 0.089191, 0.006755, 0.068470, 0.779613, 0.339703, 0.109925, 0.073427],
}
week1_Y = {1: 2.6752879910742468e-9, 2: -0.06913022970475102, 3: -0.027253680430408482,
           4: -0.3932940148544044, 5: 1703.1512205236875, 6: -0.41036378008423247,
           7: 2.0679435112701356, 8: 9.8674177220056}

week2_X = {
    1: [0.600000, 0.300000], 2: [0.673469, 0.897959], 3: [0.377732, 0.089220, 0.423517],
    4: [0.451289, 0.457435, 0.214355, 0.422482], 5: [0.532623, 0.594318, 0.990109, 0.985016],
    6: [0.313541, 0.259869, 0.535241, 0.987761, 0.214623],
    7: [0.142611, 0.367475, 0.491490, 0.152728, 0.361910, 0.685650],
    8: [0.192332, 0.148385, 0.293081, 0.032111, 0.743423, 0.258716, 0.124238, 0.699129],
}
week2_Y = {1: 3.7759561298146587e-34, 2: 0.4900359752987052, 3: -0.0665391852481089,
           4: -3.2314120937555297, 5: 1880.1864969021228, 6: -0.6644176392227166,
           7: 2.3848190050793328, 8: 9.8169820770504}

week3_X = {
    1: [0.775500, 0.795900], 2: [0.673500, 0.959200], 3: [0.649100, 0.969600, 0.004500],
    4: [0.368200, 0.492200, 0.413200, 0.455800], 5: [0.595600, 0.993700, 0.987300, 0.981700],
    6: [0.283200, 0.570800, 0.059500, 0.960000, 0.109900],
    7: [0.101800, 0.291000, 0.575600, 0.016000, 0.482400, 0.625700],
    8: [0.095500, 0.268900, 0.010700, 0.400900, 0.925300, 0.802000, 0.057500, 0.712800],
}
week3_Y = {1: 5.706193265431438e-35, 2: 0.5556031562372118, 3: -0.12661734906119,
           4: -1.4732579244500381, 5: 4397.233847397341, 6: -1.141864063971705,
           7: 1.443148014381498, 8: 9.739235081}

rows = []
for i in range(1, 9):
    X0 = np.load(f'data/initial_data/function_{i}/initial_inputs.npy')
    Y0 = np.load(f'data/initial_data/function_{i}/initial_outputs.npy')
    dims = X0.shape[1]

    # combine original data with all 3 rounds of queries to get the
    # overall observed Y range so far
    Y_all = np.concatenate([Y0, [week1_Y[i], week2_Y[i], week3_Y[i]]])

    rows.append({
        'function': i,
        'dims': dims,
        'n_obs_total': len(Y_all),
        'y_min': round(Y_all.min(), 6),
        'y_max': round(Y_all.max(), 6),
        'week1_x': [round(v, 4) for v in week1_X[i]],
        'week1_y': round(week1_Y[i], 6),
        'week2_x': [round(v, 4) for v in week2_X[i]],
        'week2_y': round(week2_Y[i], 6),
        'week3_x': [round(v, 4) for v in week3_X[i]],
        'week3_y': round(week3_Y[i], 6),
    })

df = pd.DataFrame(rows)
df.to_csv('function_summary_weeks1to3.csv', index=False)

pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)
print(df.to_string(index=False))
print("\nSaved function_summary_weeks1to3.csv")
