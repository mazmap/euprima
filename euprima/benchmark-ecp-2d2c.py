import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import testing_utils

import build.euprima as euprima
import euchar.utils, euchar.surface

euler_changes = euchar.utils.vector_all_euler_changes_in_2D_images()

samples_per_n = 10
def input_generator(n):
    return [
        (np.random.randint(0, 256, (n,n), dtype=np.uint8),
         np.random.randint(0, 256, (n,n), dtype=np.uint8),
         euler_changes,
         255,
         255)
        for _ in range(samples_per_n+1)
    ]

N = 50
ig_args = range(1,N+1)

summary1 = testing_utils.benchmark(euprima.ecp_2d2c, input_generator, ig_args)
summary2 = testing_utils.benchmark(euchar.surface.images_2D, input_generator, ig_args)

def input_generator2(n):
    return [
        (np.random.randint(0, 256, (n,n), dtype=np.uint8),
         np.random.randint(0, 256, (n,n), dtype=np.uint8),
         255,
         255)
        for _ in range(samples_per_n+1)
    ] 
summary3 = testing_utils.benchmark(euprima.ecp_2d2c_hw, input_generator2, ig_args)
#summary4 = testing_utils.benchmark(euprima.ecp_2d2c_naive, input_generator2, ig_args)

# TODO: Save summaries to csv
data = {
    "ecp_2d2c": summary1,
    "images_2D": summary2,
    "ecp_2d2c_hw": summary3,
    #"ecp_2d2c_naive": summary4
}

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv('benchmark_ecp_2d2c.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t*1000 for t in summary1], label='ecp_2d2c [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary2], label='images_2D [euchar]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary3], label='ecp_2d2c_hw [euprima]', marker='o', markersize=2)
#plt.plot(ig_args, [t*1000 for t in summary4], label='ecp_2d2c_naive [euprima]', marker='o', markersize=2)

plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('Avg Execution Time (Microseconds)')
plt.title('euprima vs. euchar')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('benchmark_ecp_2d2c.png')
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")
