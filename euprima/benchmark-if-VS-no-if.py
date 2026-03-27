import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import testing_utils

import build.euprima as euprima


T = 255
samples_per_n = 1
N = 100
ig_args = range(1,N+1)

def input_generator(n):
    return [
        (np.random.randint(0, T, (n,n), dtype=np.uint8),
         np.random.randint(0, T, (n,n), dtype=np.uint8),
         np.random.randint(0, T, (n,n), dtype=np.uint8),
         T,
         T,
         T)
        for _ in range(samples_per_n+1)
    ] 
summary1 = testing_utils.benchmark(euprima.ecp_2d3c_hw_optimized, input_generator, ig_args)
summary2 = testing_utils.benchmark(euprima.ecp_2d3c_hw_optimized_test, input_generator, ig_args)

# TODO: Save summaries to csv
data = {
    "ecp_2d3c_optimized": summary1,
    "ecp_2d3c_optimized_test": summary2
}

file_name = f"benchmark_if_VS_no_if_T{T}"

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv(file_name+".csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t*1000 for t in summary1], label='ecp_2d3c_hw_optimized [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary2], label='ecp_2d3c_hw_optimized_test [euprima]', marker='o', markersize=2)

plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('Avg Execution Time (ms)')
plt.title(f"$T_1 = T_2 = T_3 = {T}$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig(file_name+".png")
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")
