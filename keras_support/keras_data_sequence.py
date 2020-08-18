import math
import threading

from tensorflow.python.keras.utils import data_utils

import MF_RP_mat
import utils
from utils import synchronized_open_file


class RP_Sequence(data_utils.Sequence):
    """
    for multi-channel RP mats, RP autoencoder
    """
    def __init__(self, n_samples, batch_size, RP_mats_h5array):
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.RP_mats_h5array = RP_mats_h5array

    def __getitem__(self, idx):
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = self.RP_mats_h5array[batch_slice, ...]
        batch_x = utils.scale_RP_each_feature(batch_x)
        batch_y = batch_x  # train autoecoder, input is equal to output
        return batch_x, batch_y

    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)

class FS_Sequence(data_utils.Sequence):
    """
    for multi-feature segs, FS autoencoder
    """
    def __init__(self, n_samples, batch_size, multi_feature_segs) -> None:
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.multi_feature_segs = multi_feature_segs

    def __getitem__(self, idx):
        batch_x = self.multi_feature_segs[idx * self.batch_size, (idx + 1) * self.batch_size]
        batch_x = utils.scale_segs_each_features(batch_x)
        batch_y = batch_x  # train autoecoder, input is equal to output
        return batch_x, batch_y

    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)

class RP_FS_Centroid_Sequence(data_utils.Sequence):
    """
    for multi-feature segs, Dual-CSA autoencoder
    """
    def __init__(self, n_samples, batch_size, RP_mats_h5array, multi_feature_segs, centroids, labels) -> None:
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.RP_mats_h5array = RP_mats_h5array
        self.multi_feature_segs = multi_feature_segs
        self.centroids = centroids
        self.labels = labels

    def __getitem__(self, idx):
        print('*enter get item ')
        batch_slice = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_x = [
            # utils.scale_RP_each_feature(self.multi_channel_RP_mats[batch_slice, ...]),
            self.RP_mats_h5array[batch_slice, ...],
            self.centroids[batch_slice],
            self.multi_feature_segs[batch_slice]
            # utils.scale_segs_each_features(self.multi_feature_segs[batch_slice])
                   ]

        batch_y = [
            # utils.scale_RP_each_feature(self.multi_channel_RP_mats[batch_slice, ...]),
            self.RP_mats_h5array[batch_slice, ...],
            self.labels[batch_slice],
            # utils.scale_segs_each_features(self.multi_feature_segs[batch_slice]),
            self.multi_feature_segs[batch_slice]
        ]
        print('*end get item ')

        return batch_x, batch_y

    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)

if __name__ == '__main__':
    lock = threading.Lock()

    RP_s = RP_Sequence(1000, 200, 'data\geolife_features\RP_mats_train.h5', lock)
    RP_s.__getitem__(2)
