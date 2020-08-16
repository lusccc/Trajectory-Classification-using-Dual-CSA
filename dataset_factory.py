import math

import numpy as np
import tables as tb

import MF_RP_mat


class Dataset:
    def __init__(self, name, n_parts) -> None:
        print(f'dataset:{name}')
        self.dataset = name
        self.n_parts = n_parts

        self.multi_channel_RP_mat_test = np.load(f'./data/{self.dataset}_features/RP_mats_test.npy')

        self.multi_feature_segment_train =np.load(f'./data/{self.dataset}_features/trjs_segs_features_train.npy', )
        self.multi_feature_segment_test = np.load(f'./data/{self.dataset}_features/trjs_segs_features_test.npy', )
        self.centroid_train =             np.load(f'./data/{self.dataset}_features/centroids_train.npy', )
        self.centroid_test =              np.load(f'./data/{self.dataset}_features/centroids_test.npy', )
        self.label_train =                np.load(f'./data/{self.dataset}_features/trjs_segs_features_labels_train.npy', )
        self.label_test =                 np.load(f'./data/{self.dataset}_features/trjs_segs_features_labels_test.npy', )

        self.n_samples_train = len(self.label_train)
        self.n_samples_test = len(self.label_test)

    def get_RP_train_part(self, part_index):
        return np.load(f'./data/{self.dataset}_features/RP_mats_train_p{part_index}.npy')

    def get_multi_feature_seg_train_part(self, part_size, part_index):
        return self.multi_feature_segment_train[part_index * part_size: (part_index + 1) * part_size]

    def get_centroid_train_part(self, part_size, part_index):
        return self.centroid_train[part_index * part_size: (part_index + 1) * part_size]

    def get_label_train_part(self, part_size, part_index):
        return self.label_train[part_index * part_size: (part_index + 1) * part_size]

if __name__ == '__main__':
    # data_set = Dataset('geolife')
    # x_RP_train = data_set.x_RP_train
    # x_RP_test = data_set.x_RP_test
    # x_features_series_train = data_set.x_features_series_train
    # x_features_series_test = data_set.x_features_series_test
    # x_centroids_train = data_set.x_centroids_train
    # x_centroids_test = data_set.x_centroids_test
    # y_train = data_set.y_train
    # y_test = data_set.y_test
    # RP_mats_h5file = synchronized_open_file('data\geolife_features\RP_mats_train.h5', mode='a')
    # RP_mats_h5array = RP_mats_h5file.get_node('/' + 'RP_data')
    # a = RP_mats_h5array[0]
    print()
