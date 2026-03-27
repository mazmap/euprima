import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import testing_utils

import build.euprima as euprima
import euchar.utils, euchar.surface

euler_changes = euchar.utils.vector_all_euler_changes_in_2D_images()

T = 255

samples_per_n = 1
def input_generator(n):
    return [
        (np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         euler_changes,
         T,
         T,
         T)
        for _ in range(samples_per_n+1)
    ]

N = 300
ig_args = range(1,N+1)

summary1 = testing_utils.benchmark(euprima.ecp_2d3c, input_generator, ig_args)
summary2 = testing_utils.benchmark(euprima.ecp_2d3c_optimized, input_generator, ig_args)

def input_generator2(n):
    return [
        (np.random.randint(0, T, (n,n), dtype=np.uint8),
         np.random.randint(0, T, (n,n), dtype=np.uint8),
         np.random.randint(0, T, (n,n), dtype=np.uint8),
         T,
         T,
         T)
        for _ in range(samples_per_n+1)
    ] 
summary3 = testing_utils.benchmark(euprima.ecp_2d3c_hw_optimized, input_generator2, ig_args)
#summary4 = testing_utils.benchmark(euprima.ecp_2d2c_naive, input_generator2, ig_args)

# TODO: Save summaries to csv
data = {
    "ecp_2d3c": summary1,
    "ecp_2d3c_hw_optimized2": summary3,
    "ecp_2d3c_optimized": summary2,
}

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv(f"benchmark_ecp_2d3c_T{T}_o_N{N}.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t*1000 for t in summary1], label='ecp_2d3c [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary3], label='ecp_2d3c_hw_optimized2 [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary2], label='ecp_2d3c_optimized [euprima]', marker='o', markersize=2)
#plt.plot(ig_args, [t*1000 for t in summary4], label='ecp_2d2c_naive [euprima]', marker='o', markersize=2)

plt.xlabel('Image Dimension (N x N)')
plt.ylabel('Avg Execution Time (ms)')
plt.title(f"$T_1 = T_2 = T_3 = {T}$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig(f"benchmark_ecp_2d3c_T{T}_o_N{N}.png")
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")
