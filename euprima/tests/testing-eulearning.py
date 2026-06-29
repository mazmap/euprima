import numpy as np

from eulearning.descriptors import compute_euler_profile

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import build.euprima as euprima

import testing_utils

# TODO: From image to vectorized simplex tree

N = 10
img_c1 = np.random.randint(0, 256, (N,N) , dtype=np.uint8)
img_c2 = np.random.randint(0, 256, (N,N) , dtype=np.uint8)
img_c3 = np.random.randint(0, 256, (N,N) , dtype=np.uint8)

list_of_minimal_grades_top_cell = euprima.list_of_minimal_grades_top_cell(img_c1, img_c2, img_c3);

ecp_hale_t_eulearning = compute_euler_profile(list_of_minimal_grades_top_cell, [(0,255,256), (0,255,256), (0,255,256)])
ecp_hale_t_euprima = euprima.ecp_hale_t_2d3c(img_c1, img_c2, img_c3, 255, 255, 255)
ecp_hale_t_eulearning_euprima = euprima.ecp_hale_eulearning_2d3c(list_of_minimal_grades_top_cell, 255, 255, 255)

testing_utils.check_pairwise_equality([ecp_hale_t_eulearning, ecp_hale_t_euprima, ecp_hale_t_eulearning_euprima], ["HALE:T(2,3) [eulearning]", "HALE:T(2,3) [euprima]", "HALT:T(2,3) eulearning [euprima]"])

list_of_minimal_grades_vertex = euprima.list_of_minimal_grades_vertex(img_c1, img_c2, img_c3);

ecp_hale_v_eulearning = compute_euler_profile(list_of_minimal_grades_vertex, [(0,255,256), (0,255,256), (0,255,256)])
ecp_hale_v_euprima = euprima.ecp_hale_v_2d3c(img_c1, img_c2, img_c3, 255, 255, 255)
ecp_hale_v_eulearning_euprima = euprima.ecp_hale_eulearning_2d3c(list_of_minimal_grades_vertex, 255, 255, 255)

testing_utils.check_pairwise_equality([ecp_hale_v_eulearning, ecp_hale_v_euprima, ecp_hale_v_eulearning_euprima], ["HALE:V(2,3) [eulearning]", "HALE:V(2,3) [euprima]", "HALE:V(2,3) eulearning [euprima]"])
