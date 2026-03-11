import numpy as np
import time
from tabulate import tabulate

import build.euprima as euprima
import euchar.utils

samples = 1000

t1_start = time.perf_counter()
for i in range(0,samples):
    euprima.euler_changes_2d()
t1_end = time.perf_counter()
avg_t1 = (t1_end - t1_start) / samples

t2_start = time.perf_counter()
for i in range(0,samples):
    euchar.utils.vector_all_euler_changes_in_2D_images()
t2_end = time.perf_counter()
avg_t2 = (t2_end - t2_start) / samples

headers = ["euprima", "euchar", "improvement"]
table_data = [[f"{avg_t1*1e6:.3f}μs", f"{avg_t2*1e6:.3f}μs", f"{abs(avg_t1-avg_t2)/min(avg_t1,avg_t2)*100:.0f}%"]]

print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
