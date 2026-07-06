import numpy as np
# import euchar.utils, euchar.surface
from tabulate import tabulate
import testing_utils

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import build.euprima as euprima

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
N = 3

img_c1 = np.random.randint(0, T+1, (N,N), dtype=np.int32)
img_c2 = np.random.randint(0, T+1, (N,N), dtype=np.int32)
img_c3 = np.random.randint(0, T+1, (N,N), dtype=np.int32)

euler_changes = euprima.belt_2d_euler_changes();

ecp_naive = euprima.ecp_ind_2d3c_naive(img_c1, img_c2, img_c3, T, T, T)
ecp_belt = euprima.ecp_belt_2d3c(img_c1, img_c2, img_c3, euler_changes, T, T, T)
ecp_hewa = euprima.ecp_hewa_2d3c(img_c1, img_c2, img_c3, T, T, T)
ecp_hale_i = euprima.ecp_hale_i_2d3c(img_c1, img_c2, img_c3, T, T, T)

print("ECP of the filtration induced by top-cells of 2-dimensional 3-channel images")
testing_utils.check_pairwise_equality([ecp_naive, ecp_belt, ecp_hewa, ecp_hale_i], labels=["Naive", "BELT(2,3)", "HEWA(2,3)", "HALE:I(2,3)"])

print("ECP of the top-cell filtration of 2-dimensional 3-channel images")
# Example from Figure 2.4
img1_c1 = np.array([[1,2]], dtype=np.int32)
img1_c2 = np.array([[2,1]], dtype=np.int32)
img1_c3 = np.array([[0,0]], dtype=np.int32)
# Example from Figure 2.5
img2_c1 = np.array([[2,2,1], [2,2,2], [1,2,2]], dtype=np.int32)
img2_c2 = np.array([[2,2,2], [2,2,1], [1,2,2]], dtype=np.int32)
img2_c3 = np.array([[0,0,0], [0,0,0], [0,0,0]], dtype=np.int32)

ecp_hale_t_img1 = euprima.ecp_hale_t_2d3c(img1_c1, img1_c2, img1_c3, 2, 2, 2)
ecp_hale_i_img1 = euprima.ecp_hale_i_2d3c(img1_c1, img1_c2, img1_c3, 2, 2, 2)
ecp_naive_img1 = euprima.ecp_ind_2d3c_naive(img1_c1, img1_c2, img1_c3, 2, 2, 2)

print(ecp_hale_t_img1[0][0][0], ecp_hale_t_img1[0][1][0], ecp_hale_t_img1[0][2][0])
print(ecp_hale_t_img1[1][0][0], ecp_hale_t_img1[1][1][0], ecp_hale_t_img1[1][2][0])
print(ecp_hale_t_img1[2][0][0], ecp_hale_t_img1[2][1][0], ecp_hale_t_img1[2][2][0])
print()
print(ecp_hale_i_img1[0][0][0], ecp_hale_i_img1[0][1][0], ecp_hale_i_img1[0][2][0])
print(ecp_hale_i_img1[1][0][0], ecp_hale_i_img1[1][1][0], ecp_hale_i_img1[1][2][0])
print(ecp_hale_i_img1[2][0][0], ecp_hale_i_img1[2][1][0], ecp_hale_i_img1[2][2][0])
print()
print(ecp_naive_img1[0][0][0], ecp_naive_img1[0][1][0], ecp_naive_img1[0][2][0])
print(ecp_naive_img1[1][0][0], ecp_naive_img1[1][1][0], ecp_naive_img1[1][2][0])
print(ecp_naive_img1[2][0][0], ecp_naive_img1[2][1][0], ecp_naive_img1[2][2][0])

print(np.array_equal(euprima.ecp_ind_2d3c_naive(img2_c1, img2_c2, img2_c3, 2, 2, 2), euprima.ecp_hale_i_2d3c(img2_c1, img2_c2, img2_c3, 2, 2, 2)))

T=2
N=3
img_c1 = np.random.randint(0, T+1, (N,N), dtype=np.int32)
img_c2 = np.random.randint(0, T+1, (N,N), dtype=np.int32)
img_c3 = np.zeros((N,N), dtype=np.int32)
ecp_hale_v_img1 = euprima.ecp_hale_v_2d3c(img_c1, img_c2, img_c3, 2, 2, 2)
print("ECP of vertex filtration")
print(np.stack((img_c1, img_c2), axis=-1))
print(ecp_hale_v_img1[0][0][0], ecp_hale_v_img1[0][1][0], ecp_hale_v_img1[0][2][0])
print(ecp_hale_v_img1[1][0][0], ecp_hale_v_img1[1][1][0], ecp_hale_v_img1[1][2][0])
print(ecp_hale_v_img1[2][0][0], ecp_hale_v_img1[2][1][0], ecp_hale_v_img1[2][2][0])
