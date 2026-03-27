import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import testing_utils

import build.euprima as euprima

from itertools import product

from tqdm import tqdm

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


def EC_at_RGB_value(contributions, r, g, b):
    return sum(
        [
            c[1]
            for c in contributions
            if (c[0][0] <= r) and (c[0][1] <= g) and (c[0][2] <= b)
        ]
    )

def ecp_2d3c_dg(image, inf_value=np.inf):
    contributions = compute_RGB_contributions(image, inf_value)

    ecp = np.zeros((256, 256, 256))

    for r in range(256):
        for g in range(256):
            for b in range(256):
                ecp[(r,g,b)] = EC_at_RGB_value(contributions, r, g, b)

    # for r,g,b in tqdm(product(range(256), range(256), range(256))): 
    #     ecp[(r,g,b)] = EC_at_RGB_value(contributions, r, g, b)

    return ecp

random_rgb = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)

samples_per_n = 1
N = 50
ig_args = range(1,N+1)

def input_generator(n):
    return [
        (np.random.randint(0, 256, (n,n), dtype=np.uint8),
         np.random.randint(0, 256, (n,n), dtype=np.uint8),
         np.random.randint(0, 256, (n,n), dtype=np.uint8),
         255, 255, 255)
        for _ in range(samples_per_n+1)
    ] 

def input_generator2(n):
    return [
        np.random.randint(0, 256, (n,n,3), dtype=np.uint8)
        for _ in range(samples_per_n+1)
    ]

summary2 = testing_utils.benchmark(ecp_2d3c_dg, input_generator2, ig_args)
summary1 = testing_utils.benchmark(euprima.ecp_2d3c_hw_optimized, input_generator, ig_args)

# TODO: Save summaries to csv
data = {
    "ecp_2d3c_hw_optimized": summary1,
    "ecp_2d3c_dg": summary2,
}

file_name = "benchmark_dlotko_gurnari"

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv(file_name+".csv", index=False)

plt.figure(figsize=(10, 6))
plt.plot(ig_args, [t*1000 for t in summary1], label='ecp_2d3c_hw_optimized [euprima]', marker='o', markersize=2)
plt.plot(ig_args, [t*1000 for t in summary2], label='ecp_2d3c_dg ["pyEulerCurves"]', marker='o', markersize=2)

plt.xlabel('Image Dimension (N x N)')
plt.ylabel('Avg Execution Time (ms)')
plt.title('euprima vs. euchar')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig(file_name+".png")
print("Benchmark complete. Data saved to CSV and graph saved as PNG.")
