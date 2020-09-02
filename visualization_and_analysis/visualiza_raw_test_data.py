import torch
import numpy as np

from params import N_CLASS
from utils import visualize_data

dataset_name = 'geolife'
data_type = 'test'
multi_feature_segs = np.load(f'../data/{dataset_name}_features/multi_feature_segs_{data_type}_normalized.npy')
labels = np.load(f'../data/{dataset_name}_features/multi_feature_seg_labels_{data_type}.npy')
labels = np.argmax(labels, 1)
multi_feature_segs = np.reshape(multi_feature_segs, (multi_feature_segs.shape[0], -1))
visualize_data(multi_feature_segs, labels, N_CLASS, f'../results/{dataset_name}raw_test_data.png')
