import numpy as np
import build.euprima as euprima

T = 255
N = 50

img_c1 = np.random.randint(0, T+1, (N,N), dtype=np.uint8)
img_c2 = np.random.randint(0, T+1, (N,N), dtype=np.uint8)
img_c3 = np.random.randint(0, T+1, (N,N), dtype=np.uint8)

# print(img_c1)
# print(img_c2)
# print(img_c3)

ecp_hl = euprima.ecp_2d3c_hl(img_c1, img_c2, img_c3, T, T, T)
