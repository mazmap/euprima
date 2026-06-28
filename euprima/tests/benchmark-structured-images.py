from PIL import Image
import numpy as np
from tabulate import tabulate
import time

import build.euprima as euprima

def benchmark_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    img_c1 = img_np[:,:,0]
    img_c2 = img_np[:,:,1]
    img_c3 = img_np[:,:,2]

    width = img_np.shape[0]
    height = img_np.shape[1]

    print("====", img_path, f"({width}x{height} = {width*height})", "====")

    t_start = time.perf_counter()
    euprima.ecp_2d3c_hw_optimized(img_c1, img_c2, img_c3, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_1 = t_end - t_start

    t_start = time.perf_counter()
    euler_changes = euprima.euler_changes_2d()
    euprima.ecp_2d3c_optimized(img_c1, img_c2, img_c3, euler_changes, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_2 = t_end - t_start

    t_start = time.perf_counter()
    euprima.ecp_2d3c_hl(img_c1, img_c2, img_c3, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_3 = t_end - t_start

    t_start = time.perf_counter()
    euprima.ecp_2d3c_hl_mc(img_c1, img_c2, img_c3, 255, 255, 255)
    t_end = time.perf_counter()
    t_dur_4 = t_end - t_start

    table_data = []
    headers = ["", "Execution time (in s)"]
    table_data.append(["HEWA3", round(t_dur_1,1)])
    table_data.append(["BELT3", round(t_dur_2,1)])
    table_data.append(["HALE3t", round(t_dur_3,1)])
    table_data.append(["HALE3i", round(t_dur_4,1)])

    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

IMG_PATHS = [
    "tigger.jpg",
    "ship.jpg"
]

for img_path in IMG_PATHS: 
    benchmark_img(img_path)
