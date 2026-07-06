import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../eulearning')))
from eulearning.descriptors import compute_euler_profile

import euprima
import utils.testing_utils as testing_utils

def compute_euler_profile_vectorized(vec_st, bin_sizes):
    # (Assumes assertions are handled or kept)
    num_filts = len(bin_sizes)
    
    t_mins = np.array([size[0] for size in bin_sizes])
    t_maxs = np.array([size[1] for size in bin_sizes])
    resolutions = np.array([size[-1] for size in bin_sizes]).astype(int)
    scales = (resolutions - 1) / (t_maxs - t_mins)
    
    # Extract signs and filtration values globally
    sgns = vec_st[:, 0]
    filtrations = vec_st[:, 1:]
    
    # Vectorized calculation of all grid indices
    # Shape: (n_splx, num_filts)
    inds = np.maximum(np.ceil((filtrations - t_mins) * scales), 0).astype(int)
    
    # Filter out simplices that fall outside the resolution grid boundary
    valid_mask = np.all(inds < resolutions, axis=1)
    inds = inds[valid_mask]
    sgns = sgns[valid_mask]
    
    # Initialize the grid
    ecp = np.zeros(resolutions)
    
    # Unbuffered in-place addition to handle potential coordinate duplicates
    # We transpose 'inds' so it can be unpacked into a tuple of index arrays
    np.add.at(ecp, tuple(inds.T), sgns)
    
    # Compute the cumulative sum across all filtration dimensions
    for k in range(num_filts):
        ecp = np.cumsum(ecp, axis=k)
        
    return ecp

N = 10
img_c1 = np.random.randint(0, 256, (N,N) , dtype=np.uint8)
img_c2 = np.random.randint(0, 256, (N,N) , dtype=np.uint8)
img_c3 = np.random.randint(0, 256, (N,N) , dtype=np.uint8)

list_of_minimal_grades_top_cell = euprima.list_of_minimal_grades_top_cell(img_c1, img_c2, img_c3);

ecp_hale_t_eulearning = compute_euler_profile(list_of_minimal_grades_top_cell, [(0,255,256), (0,255,256), (0,255,256)]).transpose()
ecp_hale_t_euprima = euprima.ecp_hale_t_2d3c(img_c1, img_c2, img_c3, 255, 255, 255)
ecp_hale_t_eulearning_euprima = euprima.ecp_hale_eulearning_2d3c(list_of_minimal_grades_top_cell, 255, 255, 255)

testing_utils.check_pairwise_equality([ecp_hale_t_eulearning, ecp_hale_t_euprima, ecp_hale_t_eulearning_euprima], ["HALE:T(2,3) [eulearning]", "HALE:T(2,3) [euprima]", "HALT:T(2,3) eulearning [euprima]"])

list_of_minimal_grades_vertex = euprima.list_of_minimal_grades_vertex(img_c1, img_c2, img_c3);

ecp_hale_v_eulearning = compute_euler_profile(list_of_minimal_grades_vertex, [(0,255,256), (0,255,256), (0,255,256)]).transpose()
ecp_hale_v_euprima = euprima.ecp_hale_v_2d3c(img_c1, img_c2, img_c3, 255, 255, 255)
ecp_hale_v_eulearning_euprima = euprima.ecp_hale_eulearning_2d3c(list_of_minimal_grades_vertex, 255, 255, 255)

testing_utils.check_pairwise_equality([ecp_hale_v_eulearning, ecp_hale_v_euprima, ecp_hale_v_eulearning_euprima], ["HALE:V(2,3) [eulearning]", "HALE:V(2,3) [euprima]", "HALE:V(2,3) eulearning [euprima]"])
