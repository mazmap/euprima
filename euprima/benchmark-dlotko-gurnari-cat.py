from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import testing_utils

import build.euprima as euprima

from itertools import product

from tqdm import tqdm

import time

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

    return list(contributions.items())
    #return sorted(list(contributions.items()), key=lambda x: x[0])[:-1]

cat_img = Image.open("tigger.jpg").convert("RGB")

# t_start = time.perf_counter()
# contributions = compute_RGB_contributions(cat_img_np) # contains tuples of the form ((r,g,b), ec)
# t_end = time.perf_counter()
# t_dur = t_end - t_start
# print("Dlotko & Gurnari:", t_dur)
#print(contributions)

cat_img_np = np.array(cat_img)
cat_img_c1 = cat_img_np[:,:,0]
cat_img_c2 = cat_img_np[:,:,1]
cat_img_c3 = cat_img_np[:,:,2]

width = cat_img_np.shape[0]
height = cat_img_np.shape[1]

t_start = time.perf_counter()
euprima.ecp_2d3c_hw_optimized(cat_img_c1, cat_img_c2, cat_img_c3, 255, 255, 255)
t_end = time.perf_counter()
t_dur = t_end - t_start
print("HW:", t_dur)

# rand_img_c1 = np.random.randint(0,256,(width,height))
# rand_img_c2 = np.random.randint(0,256,(width,height))
# rand_img_c3 = np.random.randint(0,256,(width,height))
# 
# t_start = time.perf_counter()
# euprima.ecp_2d3c_hw_optimized(rand_img_c1, rand_img_c2, rand_img_c3, 255, 255, 255)
# t_end = time.perf_counter()
# t_dur = t_end - t_start
# print("random:", t_dur)

t_start = time.perf_counter()
euprima.ecp_2d3c_hl(cat_img_c1, cat_img_c2, cat_img_c3, 255, 255, 255)
t_end = time.perf_counter()
t_dur = t_end - t_start
print("HL:", t_dur)
