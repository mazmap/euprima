import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import build.euprima as euprima

import testing_utils

T = 255
N = 5000
STEPS = 1000
START = 1000
SAMPLES_PER_N = 1

filename = "benchmark_all"
csv_file = f"{filename}.csv"

def append_column_to_csv(col_name, col_data, index_args):
    """Reads the current CSV, adds the newly computed column, and overwrites."""
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # If this is the first summary, initialize the DataFrame with the 'N' values
        df = pd.DataFrame({"Image Dimension N": list(index_args)})
    
    df[col_name] = col_data
    df.to_csv(f"{filename}_appended.csv", index=False)

def input_generator(n):
    return [
        (np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         T, T, T)
        for _ in range(SAMPLES_PER_N+1)
    ]

ig_args = range(START, N+1, STEPS)

#summary = testing_utils.benchmark(euprima.ecp_2d3c_hale_v, input_generator, ig_args)
summary = testing_utils.benchmark(euprima.ecp_2d3c_hl, input_generator, ig_args)
#append_column_to_csv("HALE:V(2,3) [euprima]", summary, ig_args)
append_column_to_csv("HALE:T(2,3) [euprima]", summary, ig_args)
