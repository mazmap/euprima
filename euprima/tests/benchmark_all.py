import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eulearning.descriptors import compute_euler_profile

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
    df.to_csv(csv_file, index=False)

# Ensure a fresh CSV at the start of a new run to avoid appending to old data
if os.path.exists(csv_file):
    os.remove(csv_file)

def input_generator(n):
    return [
        (np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         T, T, T)
        for _ in range(SAMPLES_PER_N+1)
    ]

ig_args = range(START, N+1, STEPS)

summary1 = testing_utils.benchmark(euprima.ecp_hale_t_2d3c, input_generator, ig_args)
append_column_to_csv("HALE:T(2,3) [euprima]", summary1, ig_args)

summary2 = testing_utils.benchmark(euprima.ecp_hale_v_2d3c, input_generator, ig_args)
append_column_to_csv("HALE:V(2,3) [euprima]", summary1, ig_args)

summary3 = testing_utils.benchmark(euprima.ecp_hale_i_2d3c, input_generator, ig_args)
append_column_to_csv("HALE:I(2,3) [euprima]", summary2, ig_args)

summary4 = testing_utils.benchmark(euprima.ecp_hewa_2d3c, input_generator, ig_args)
append_column_to_csv("HEWA(2,3) [euprima]", summary3, ig_args)

def eulearning(img_c1, img_c2, img_c3, T1, T2, T3): 
    contributions = euprima.list_of_minimal_grades(img_c1, img_c2, img_c3)
    return compute_euler_profile(contributions, [(0,T1,T1+1), (0,T2,T2+1), (0,T3,T3+1)])

def ecp_hale_eulearning_2d3c_contributions(img_c1, img_c2, img_c3, T1, T2, T3): 
    contributions = euprima.list_of_minimal_grades(img_c1, img_c2, img_c3)
    return euprima.ecp_hale_eulearning_2d3c(contributions, T1, T2, T3)

summary5 = testing_utils.benchmark(eulearning, input_generator, ig_args)
append_column_to_csv("HALE:T(2,3) [eulearning]", summary4, ig_args)

summary6 = testing_utils.benchmark(ecp_hale_eulearning_2d3c_contributions, input_generator, ig_args)
append_column_to_csv("HALE:T(2,3) like eulearning [euprima]", summary5, ig_args)

euler_changes = euprima.belt_2d_euler_changes()
def input_generator2(n):
    return [
        (np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         euler_changes,
         T, T, T)
        for _ in range(SAMPLES_PER_N+1)
    ]

summary7 = testing_utils.benchmark(euprima.ecp_belt_2d3c, input_generator2, ig_args)
append_column_to_csv("BELT(2,3) [euprima]", summary6, ig_args)


# Plotting logic remains the same (pulling from the in-memory lists)
plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t for t in summary1], label='HALE:T(2,3) [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary2], label='HALE:V(2,3) [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary3], label='HALE:I(2,3) [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary4], label='HEWA(2,3) [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary5], label='HALE:T(2,3) [eulearning]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary6], label='HALE:T(2,3) like eulearning [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary7], label='BELT(2,3) [euprima]', marker='o', markersize=2)

plt.xlabel('Image Dimension N')
plt.ylabel('Avg Execution Time (s)')
plt.title(f"$T_1 = T_2 = T_3 = {T}$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig(filename+".png")
print("Benchmark complete. Data successfully saved to CSV incrementally, and graph saved as PNG.")
