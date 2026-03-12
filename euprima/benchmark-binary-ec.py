import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import testing_utils

import build.euprima as euprima
import euchar.utils, euchar.surface

samples_per_n = 10
def input_generator(n):
    return [
        [np.random.randint(0, 2, (n,n), dtype=np.uint8)]
        for _ in range(samples_per_n+1)
    ]

ig_args = range(1,501)

summary1 = testing_utils.benchmark(euprima.ec_binary_image_2d_naive, input_generator, ig_args)
summary2 = testing_utils.benchmark(euchar.surface.char_binary_image_2d, input_generator, ig_args)
summary3 = testing_utils.benchmark(euprima.ec_binary_image_2d_gray, input_generator, ig_args)

# TODO: Save summaries to csv
data = {
    "ec_binar_image_2d_naive": summary1,
    "char_binary_image_2d": summary2,
    "ec_binary_image_2d_gray": summary3
}

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv('benchmark_binary_ec.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t*1000 for t in summary1], label='ec_binary_image_2d_naive [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary2], label='char_binary_image_2d [euchar]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary3], label='ec_binary_image_2d_gray [euprima]', marker='o', markersize=2)

plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('Avg Execution Time (Microseconds)')
plt.title('euprima vs. euchar')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig('benchmark_binary_ec.png')
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")
