import numpy as np


class Dataset:
    def __init__(self, name) -> None:
        print(f'dataset:{name}')
        if name == 'geolife':
            self.x_RP_train = np.load('./data/geolife_features/RP_mats_train.npy', )
            self.x_RP_test = np.load('./data/geolife_features/RP_mats_test.npy', )
            self.x_features_series_train = np.load('./data/geolife_features/trjs_segs_features_train.npy', )
            self.x_features_series_test = np.load('./data/geolife_features/trjs_segs_features_test.npy', )
            self.x_centroids_train = np.load('./data/geolife_features/centroids_train.npy', )
            self.x_centroids_test = np.load('./data/geolife_features/centroids_test.npy', )
            self.y_train = np.load('./data/geolife_features/trjs_segs_features_labels_train.npy', )
            self.y_test = np.load('./data/geolife_features/trjs_segs_features_labels_test.npy', )
            print('centroids shape:{}'.format(self.x_centroids_train.shape))
        if name == 'SHL':
            self.x_RP_train = np.load('./data/SHL_features/RP_mats_train.npy', )
            self.x_RP_test = np.load('./data/SHL_features/RP_mats_test.npy', )
            self.x_features_series_train = np.load('./data/SHL_features/trjs_segs_features_train.npy', )
            self.x_features_series_test = np.load('./data/SHL_features/trjs_segs_features_test.npy', )
            self.x_centroids_train = np.load('./data/SHL_features/centroids_train.npy', )
            self.x_centroids_test = np.load('./data/SHL_features/centroids_test.npy', )
            self.y_train = np.load('./data/SHL_features/trjs_segs_features_labels_train.npy', )
            self.y_test = np.load('./data/SHL_features/trjs_segs_features_labels_test.npy', )
            print('centroids shape:{}'.format(self.x_centroids_train.shape))
