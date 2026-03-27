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

contributions = euprima.compute_contributions(img_c1, img_c2, img_c3);

ecp_hl = compute_euler_profile(contributions, [(0,255,256), (0,255,256), (0,255,256)])
ecp_euprima_hl = euprima.ecp_2d3c_hl(img_c1, img_c2, img_c3, 255, 255, 255)
ecp_euprima_hl_c = euprima.ecp_2d3c_hl_contributions(contributions, 255, 255, 255)

testing_utils.check_pairwise_equality([ecp_hl, ecp_euprima_hl, ecp_euprima_hl_c], ["ecp_hl", "ecp_euprima_hl", "ecp_euprima_hl_c"])
