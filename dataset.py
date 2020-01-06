import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from params import N_CLASS, features_set_2, features_set_1
from utils import scale_any_shape_data


def make_dataset():
    def scale_RP_each_feature(RP_all_features):
        scaler = StandardScaler()
        n_features = RP_all_features.shape[3]
        for i in range(n_features):
            RP_single_feature = RP_all_features[:, :, :, i]
            scaled = scale_any_shape_data(RP_single_feature, scaler)
            RP_all_features[:, :, :, i] = scaled
        return RP_all_features

    def scale_segs_each_features(segs_all_features):
        scaler = StandardScaler()
        n_features = segs_all_features.shape[3]
        for i in range(n_features):
            segs_single_feature = segs_all_features[:, :, :, i]
            scaled = scale_any_shape_data(segs_single_feature, scaler)
            segs_all_features[:, :, :, i] = scaled
        return segs_all_features

    RP_mats_clean_mf = np.load('./geolife_features/RP_mats_clean_mf.npy')[:, :, :, features_set_1]
    RP_mats_noise_mf = np.load('./geolife_features/RP_mats_noise_mf.npy')[:, :, :, features_set_1]
    trjs_segs_clean_features = np.load('./geolife_features/trjs_segs_clean_features.npy')[:, :, :,
                               features_set_2]  # (2454, 184, 10)
    trjs_segs_noise_features = np.load('./geolife_features/trjs_segs_noise_features.npy')[:, :, :, features_set_2]
    centroids = np.load('./geolife_features/centroids.npy')
    labels = np.load('./geolife_features/trjs_segs_features_labels.npy')

    x_RP_clean_train, x_RP_clean_test, \
    x_RP_noise_train, x_RP_noise_test, \
    x_features_series_clean_train, x_features_series_clean_test, \
    x_features_series_noise_train, x_features_series_noise_test, \
    x_centroids_train, x_centroids_test, \
    y_train, y_test \
        = train_test_split(
        RP_mats_clean_mf,
        RP_mats_noise_mf,
        trjs_segs_clean_features,
        trjs_segs_noise_features,
        centroids,
        labels,
        test_size=0.20, random_state=7, shuffle=True
    )
    y_train = to_categorical(y_train, num_classes=N_CLASS)
    y_test = to_categorical(y_test, num_classes=N_CLASS)

    x_RP_clean_train, x_RP_clean_test, \
    x_RP_noise_train, x_RP_noise_test, \
    x_features_series_clean_train, x_features_series_clean_test, \
    x_features_series_noise_train, x_features_series_noise_test = \
        scale_RP_each_feature(x_RP_clean_train), scale_RP_each_feature(x_RP_clean_test), \
        scale_RP_each_feature(x_RP_noise_train), scale_RP_each_feature(x_RP_noise_test), \
        scale_segs_each_features(x_features_series_clean_train), scale_segs_each_features(x_features_series_clean_test), \
        scale_segs_each_features(x_features_series_noise_train), scale_segs_each_features(x_features_series_noise_test)
    np.save('./geolife_features/x_RP_clean_train.npy', x_RP_clean_train)
    np.save('./geolife_features/x_RP_clean_test.npy', x_RP_clean_test)
    np.save('./geolife_features/x_RP_noise_train.npy', x_RP_noise_train)
    np.save('./geolife_features/x_RP_noise_test.npy', x_RP_noise_test)
    np.save('./geolife_features/x_features_series_clean_train.npy', x_features_series_clean_train)
    np.save('./geolife_features/x_features_series_clean_test.npy', x_features_series_clean_test)
    np.save('./geolife_features/x_features_series_noise_train.npy', x_features_series_noise_train)
    np.save('./geolife_features/x_features_series_noise_test.npy', x_features_series_noise_test)
    np.save('./geolife_features/x_centroids_train.npy', x_centroids_train)
    np.save('./geolife_features/x_centroids_test.npy', x_centroids_test)
    np.save('./geolife_features/y_train.npy', y_train)
    np.save('./geolife_features/y_test.npy', y_test)


# make_dataset()

x_RP_clean_train = np.load('./geolife_features/x_RP_clean_train.npy', )
x_RP_clean_test = np.load('./geolife_features/x_RP_clean_test.npy', )
x_RP_noise_train = np.load('./geolife_features/x_RP_noise_train.npy', )
x_RP_noise_test = np.load('./geolife_features/x_RP_noise_test.npy', )
x_features_series_clean_train = np.load('./geolife_features/x_features_series_clean_train.npy', )
x_features_series_clean_test = np.load('./geolife_features/x_features_series_clean_test.npy', )
x_features_series_noise_train = np.load('./geolife_features/x_features_series_noise_train.npy', )
x_features_series_noise_test = np.load('./geolife_features/x_features_series_noise_test.npy', )
x_centroids_train = np.load('./geolife_features/x_centroids_train.npy', )
x_centroids_test = np.load('./geolife_features/x_centroids_test.npy', )
y_train = np.load('./geolife_features/y_train.npy', )
y_test = np.load('./geolife_features/y_test.npy', )
