import numpy as np
import build.euprima as euprima
import euchar.utils, euchar.surface
from tabulate import tabulate
import testing_utils

#binary_image = np.random.randint(0, 2, (6,6), dtype=np.uint8)
# binary_image = np.array([
#     [1,0,0,1],
#     [0,0,0,1],
#     [1,0,1,1],
#     [0,0,0,1]
# ]);

#ec = euprima.ec_binary_image_2d_naive(binary_image);
#ec_euchar = euprima.char_binary_image_2d(binary_image);
#ec_gray = euprima.ec_binary_image_2d_gray(binary_image);
#ec_yao = euprima.ec_binary_image_2d_yao(binary_image);

#print(binary_image);
#print(ec);
#print(ec_euchar);
#print(ec_gray);
#print(ec_yao);

T = 10
N = 10

img_c1 = np.random.randint(0, T+1, (N,N), dtype=np.uint8)
img_c2 = np.random.randint(0, T+1, (N,N), dtype=np.uint8)
img_c3 = np.random.randint(0, T+1, (N,N), dtype=np.uint8)

print(img_c1);

euler_changes = euchar.utils.vector_all_euler_changes_in_2D_images()
euler_changes_euprima = euprima.euler_changes_2d();

ecp = euprima.ecp_2d2c(img_c1, img_c2, euler_changes, T, T);
ecp_euchar = euchar.surface.images_2D(img_c1, img_c2, euler_changes, T, T);
ecp_naive = euprima.ecp_2d2c_naive(img_c1, img_c2, T, T);
ecp_hw = euprima.ecp_2d2c_hw_optimized2(img_c1, img_c2, T, T);
ecp_optimized = euprima.ecp_2d2c_optimized(img_c1,img_c2, euler_changes, T,T);


print("ECP_2d2c")
testing_utils.check_pairwise_equality([ecp,ecp_optimized,ecp_euchar,ecp_hw,ecp_naive], labels=["euprima","euprima (o)", "euchar","heiss/wagner","euprima_naive"])

print("Euler changes")
testing_utils.check_pairwise_equality([euler_changes, euler_changes_euprima], labels=["euchar", "euprima"])

ecp3_naive = euprima.ecp_2d3c_naive(img_c1, img_c2, img_c3, T, T, T)
ecp3 = euprima.ecp_2d3c(img_c1, img_c2, img_c3, euler_changes, T, T, T)
ecp3_o = euprima.ecp_2d3c_optimized(img_c1, img_c2, img_c3, euler_changes, T, T, T)
ecp3_hw = euprima.ecp_2d3c_hw_optimized(img_c1, img_c2, img_c3, T, T, T)
ecp3_hw_test = euprima.ecp_2d3c_hw_optimized_test(img_c1, img_c2, img_c3, T, T, T)
ecp3_hl_mc = euprima.ecp_2d3c_hl_mc(img_c1, img_c2, img_c3, T, T, T)

print("ECP_2d3c")
testing_utils.check_pairwise_equality([ecp3_naive, ecp3, ecp3_o, ecp3_hw, ecp3_hw_test, ecp3_hl_mc], labels=["ecp3_naive", "ecp3","ecp3_o","ecp3_hw", "ecp3_hw_test", "ecp3_hl_mc"])

#print(ecp3_naive)
#print(ecp3)
