import os

import numpy as np

from Dual_SAE import load_data
from trajectory_features_and_segmentation import scale_segs

x_train_RP, x_test_RP, x_train_SF, x_test_SF, x_train_centroids, \
    x_test_centroids, y_train, y_test = load_data()
scaled = scale_segs(x_test_SF)
print(1)
a = np.array([[1,2], [3,4]])
b = np.array([[5,6], [7,8]])
c = a *b
print(c)


