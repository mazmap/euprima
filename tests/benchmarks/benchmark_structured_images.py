from PIL import Image
import numpy as np
import pandas as pd
from tabulate import tabulate
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import build.euprima as euprima

filename = "benchmark_structured_images-2"
csv_file = f"{filename}.csv"

if os.path.exists(csv_file):
    os.remove(csv_file)

def append_column_to_csv(col_name, col_data): 
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame({"Algorithms": ["HEWA(2,3)", "BELT(2,3)", "HALE:T(2,3)", "HALE:V(2,3)", "HALE:I(2,3)"]})
    df[col_name] = col_data
    df.to_csv(csv_file, index=False)

def benchmark_img(img_folder, img_file):
    img = Image.open(img_folder + img_file).convert("RGB")
    img_np = np.array(img)

    img_c1 = img_np[:,:,0]
    img_c2 = img_np[:,:,1]
    img_c3 = img_np[:,:,2]

    width = img_np.shape[0]
    height = img_np.shape[1]

    print("====", img_file, f"({width}x{height} = {width*height})", "====")

    t_start = time.perf_counter()
    euprima.ecp_hewa_2d3c(img_c1, img_c2, img_c3, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_1 = t_end - t_start

    t_start = time.perf_counter()
    euler_changes = euprima.belt_2d_euler_changes()
    euprima.ecp_belt_2d3c(img_c1, img_c2, img_c3, euler_changes, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_2 = t_end - t_start

    t_start = time.perf_counter()
    euprima.ecp_hale_t_2d3c(img_c1, img_c2, img_c3, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_3 = t_end - t_start

    t_start = time.perf_counter()
    euprima.ecp_hale_v_2d3c(img_c1, img_c2, img_c3, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_4 = t_end - t_start

    t_start = time.perf_counter()
    euprima.ecp_hale_i_2d3c(img_c1, img_c2, img_c3, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_5 = t_end - t_start

    append_column_to_csv(img_file, [t_dur_1, t_dur_2, t_dur_3, t_dur_4, t_dur_5])

IMG_FOLDER = "test-images/"
IMG_FILES = [
    "night-sky-lines.jpg",
    "mountains-silhouette.jpg",
]

for img_file in IMG_FILES: 
    benchmark_img(IMG_FOLDER, img_file)
