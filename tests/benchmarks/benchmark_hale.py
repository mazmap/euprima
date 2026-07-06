import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import euprima
import utils.benchmark_utils as benchmark_utils

T = 255
N = 20000
STEPS = 2000
START = 6000
SAMPLES_PER_N = 1

filename = "benchmark_hale_new"
csv_file = f"{filename}.csv"
results_dir = "results/"

def append_column_to_csv(col_name, col_data, index_args):
    """Reads the current CSV, adds the newly computed column, and overwrites."""
    if os.path.exists(results_dir+csv_file):
        df = pd.read_csv(results_dir+csv_file)
    else:
        # If this is the first summary, initialize the DataFrame with the 'N' values
        df = pd.DataFrame({"Image Dimension N": list(index_args)})
    
    df[col_name] = col_data
    df.to_csv(results_dir+csv_file, index=False)

if os.path.exists(results_dir+csv_file):
    os.remove(results_dir+csv_file)

def input_generator(n):
    return [
        (np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         T, T, T)
        for _ in range(SAMPLES_PER_N+1)
    ]

ig_args = range(START, N+1, STEPS)

summary1 = benchmark_utils.benchmark(euprima.ecp_hale_t_2d3c, input_generator, ig_args)
append_column_to_csv("HALE:T(2,3) [euprima]", summary1, ig_args)

summary2 = benchmark_utils.benchmark(euprima.ecp_hale_v_2d3c, input_generator, ig_args)
append_column_to_csv("HALE:V(2,3) [euprima]", summary2, ig_args)
