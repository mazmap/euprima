import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import build.euprima as euprima

import testing_utils

def pad_image(image, value):
    return np.pad(image, ((1, 1), (1, 1), (0, 0)), constant_values=value)

def compute_RGB_contributions(image, inf_value=np.inf):
    # pad image
    image = pad_image(image, inf_value)

    # compute contributions of all cells,
    # starting from bottom left
    # uses lowert star filtration

    contributions = dict()

    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            # lets track all the contributions
            # from cell i,j

            # itself, 2d cell
            f = tuple(image[i, j])
            contributions[f] = contributions.get(f, 0) + 1

            # 0d cell SW
            f = tuple(
                np.fmin(
                    image[i, j],
                    np.fmin(
                        image[i - 1, j - 1], np.fmin(image[i - 1, j], image[i, j - 1])
                    ),
                )
            )
            contributions[f] = contributions.get(f, 0) + 1

            # 1d cell W
            f = tuple(np.fmin(image[i, j], image[i, j - 1]))
            contributions[f] = contributions.get(f, 0) - 1

            # 1d cell S
            f = tuple(np.fmin(image[i, j], image[i - 1, j]))
            contributions[f] = contributions.get(f, 0) - 1

    # remove the contributions that are 0
    to_del = []
    for key in contributions:
        if contributions[key] == 0:
            to_del.append(key)
    for key in to_del:
        del contributions[key]

    return sorted(list(contributions.items()), key=lambda x: x[0])[:-1]


T = 255
N = 5000
STEPS = 1000
START = 1000
SAMPLES_PER_N = 1
ig_args = range(START,N+1,STEPS)

def input_generator1(n):
    return [np.random.randint(0, T+1, (n,n,3), dtype=np.uint8)]

summary1 = testing_utils.benchmark(compute_RGB_contributions, input_generator1, ig_args)

def input_generator2(n):
    return [
        (np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8),
         np.random.randint(0, T+1, (n,n), dtype=np.uint8))
        for _ in range(SAMPLES_PER_N+1)
    ]

summary2 = testing_utils.benchmark(euprima.compute_contributions, input_generator2, ig_args)

data = {
    "Dlotko/Gurnari": summary1,
    "euprima": summary2,
}

filename = "benchmark_contributions"

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv(filename+".csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t for t in summary1], label='Dlotko/Gurnari', marker='o', markersize=2)
plt.plot(ig_args, [t for t in summary2], label='euprima', marker='o', markersize=2)

plt.xlabel('Image Dimension N')
plt.ylabel('Avg Execution Time (s)')
plt.title(f"$T_1 = T_2 = T_3 = {T}$")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig(filename+".png")
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")
