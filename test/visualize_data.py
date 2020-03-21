import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from params import N_CLASS
from utils import visualizeData

trjs_segs_features = np.load('../data/geolife_features/trjs_segs_features.npy')
trjs_segs_features = np.reshape(trjs_segs_features, (trjs_segs_features.shape[0],-1))
labels = np.load('../data/geolife_features/trjs_segs_features_labels.npy')
x_features_series_train, x_features_series_test, \
y_train, y_test \
    = train_test_split(
    trjs_segs_features,
    labels,
    test_size=0.20, random_state=7, shuffle=True
)
np.save('../data/geolife_features/x_features_series_train.npy', x_features_series_train)
np.save('../data/geolife_features/x_features_series_test.npy', x_features_series_test)
np.save('../data/geolife_features/y_train.npy', y_train)
np.save('../data/geolife_features/y_test.npy', y_test)


visualizeData(x_features_series_test, y_test, N_CLASS,  'raw_data.png')