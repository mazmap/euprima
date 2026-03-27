from gudhi import CubicalComplex
import numpy as np
cc = CubicalComplex(top_dimensional_cells=np.array([[ (1.,1.),  (8.,8.),  (7.,7.)],
                                                [ (4.,4.), (20.,20.),  (6.,6.)]]))

print(cc.all_cells())
