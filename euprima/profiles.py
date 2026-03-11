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

img_c1 = np.random.randint(0, 256, (10,10), dtype=np.uint8)
img_c2 = np.random.randint(0, 256, (10,10), dtype=np.uint8)

euler_changes = euchar.utils.vector_all_euler_changes_in_2D_images()

ecp = euprima.ecp_2d2c(img_c1, img_c2, euler_changes, 255, 255);
ecp_euchar = euchar.surface.images_2D(img_c1, img_c2, euler_changes, 255, 255);
ecp_naive = euprima.ecp_2d2c_naive(img_c1, img_c2, 255, 255);
euler_changes_euprima = euprima.euler_changes_2d();

testing_utils.check_pairwise_equality([ecp,ecp_euchar,ecp_naive], labels=["euprima","euchar","euprima_naive"])

testing_utils.check_pairwise_equality([euler_changes, euler_changes_euprima], labels=["euchar", "euprima"])

print(euler_changes)
print(euler_changes_euprima)
