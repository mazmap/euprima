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
STEPS = 500
START = 500
SAMPLES_PER_N = 1

def input_generator(n):
    return [
        (np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         T, T, T)
        for _ in range(SAMPLES_PER_N+1)
    ]

ig_args = range(START,N+1,STEPS)

summary1 = testing_utils.benchmark(euprima.ecp_2d3c_hl, input_generator, ig_args)

def eulearning(img_c1, img_c2, img_c3, T1, T2, T3): 
    contributions = euprima.compute_contributions(img_c1, img_c2, img_c3);
    return compute_euler_profile(contributions, [(0,T1,T1+1), (0,T2,T2+1), (0,T3,T3+1)])

def ecp_2d3c_hl_contributions(img_c1, img_c2, img_c3, T1, T2, T3): 
    contributions = euprima.compute_contributions(img_c1, img_c2, img_c3);
    return euprima.ecp_2d3c_hl_contributions(contributions, T1, T2, T3)

summary2 = testing_utils.benchmark(eulearning, input_generator, ig_args)
summary3 = testing_utils.benchmark(ecp_2d3c_hl_contributions, input_generator, ig_args)


# TODO: Save summaries to csv
data = {
    "ecp_2d3c_hl": summary1,
    "eulearning": summary2,
    "ecp_2d3c_hl_contributions": summary3,
}

filename = "benchmark_eulearning"

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv(filename+".csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t for t in summary1], label='ecp_2d3c_hl [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary2], label='eulearning [eulearning]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary3], label='ecp_2d3c_hw_contributions [euprima]', marker='o', markersize=2)

plt.xlabel('Image Dimension N')
plt.ylabel('Avg Execution Time (s)')
plt.title(f"$T_1 = T_2 = T_3 = {T}$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig(filename+".png")
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")
