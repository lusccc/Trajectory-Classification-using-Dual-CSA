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

if __name__ == '__main__':
    data_set = Dataset('geolife')
    x_RP_train = data_set.x_RP_train
    x_RP_test = data_set.x_RP_test
    x_features_series_train = data_set.x_features_series_train
    x_features_series_test = data_set.x_features_series_test
    x_centroids_train = data_set.x_centroids_train
    x_centroids_test = data_set.x_centroids_test
    y_train = data_set.y_train
    y_test = data_set.y_test
    print()