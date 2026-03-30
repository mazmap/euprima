import numpy as np
import build.euprima as euprima
import euchar.utils, euchar.surface
from tabulate import tabulate
import testing_utils

img_c1 = np.array([[1,2]])
img_c2 = np.array([[2,1]])

ecs_hw = euprima.ecp_2d2c_hw_optimized2(img_c1, img_c2, 2, 2)

print(ecs_hw)
