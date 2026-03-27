import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import testing_utils

import build.euprima as euprima

euler_changes = euprima.euler_changes_2d()

T = 255
samples_per_n = 1
N = 10
ig_args = range(1,N+1)

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

summary1 = testing_utils.benchmark(euprima.ecp_2d3c, input_generator, ig_args)

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
summary2 = testing_utils.benchmark(euprima.ecp_2d3c_naive, input_generator2, ig_args)

data = {
    "ecp_2d3c": summary1,
    "ecp_2d3c_naive": summary2,
}

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv("benchmark_ecp_2d3c_naive_vs_generalized.csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t for t in summary1], label='ecp_2d3c [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary2], label='ecp_2d3c_naive [euprima]', marker='o', markersize=2)

plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('Avg Execution Time (s)')
plt.title('beltramo vs. heiss/wagner')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig("benchmark_ecp_2d3c_naive_vs_generalized.png")
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")

