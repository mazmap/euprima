import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import testing_utils

import build.euprima as euprima

T = 255
N = 10000
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
summary2 = testing_utils.benchmark(euprima.ecp_2d3c_hw_optimized, input_generator, ig_args)

# TODO: Save summaries to csv
data = {
    "ecp_2d3c_hl": summary1,
    "ecp_2d3c_hw": summary2,
}

filename = "benchmark_hl_top_cell"

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv(filename+".csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t*1000 for t in summary1], label='ecp_2d3c_hl [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary2], label='ecp_2d3c_hw [euprima]', marker='o', markersize=2)

plt.xlabel('Image Dimension N')
plt.ylabel('Avg Execution Time (ms)')
plt.title(f"$T_1 = T_2 = T_3 = {T}$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig(filename+".png")
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")
