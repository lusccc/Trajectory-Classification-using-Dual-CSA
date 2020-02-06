import multiprocessing

import numpy as np
import tables
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from PEDCC import generate_center, u, v, G, repeat
from params import *

from utils import scale_any_shape_data


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


def make_dataset():
    RP_mats = np.load('./data/geolife_features/RP_mats.npy')[:, :, :, features_set_1]
    trjs_segs_features = np.load('./data/geolife_features/trjs_segs_features.npy')[:, :, :, features_set_2]
    centroids = np.load('./data/geolife_features/centroids.npy')
    labels = np.load('./data/geolife_features/trjs_segs_features_labels.npy')

    x_RP_train, x_RP_test, \
    x_features_series_train, x_features_series_test, \
    x_centroids_train, x_centroids_test, \
    y_train, y_test \
        = train_test_split(
        RP_mats,
        trjs_segs_features,
        centroids,
        labels,
        test_size=0.20, random_state=7, shuffle=True
    )
    y_train = to_categorical(y_train, num_classes=N_CLASS)
    y_test = to_categorical(y_test, num_classes=N_CLASS)

    n_cpus = multiprocessing.cpu_count()
    print('n_thread:{}'.format(n_cpus))
    pool = multiprocessing.Pool()
    tasks = []
    tasks.append(pool.apply_async(scale_RP_each_feature, (x_RP_train,)))
    tasks.append(pool.apply_async(scale_RP_each_feature, (x_RP_test,)))
    tasks.append(pool.apply_async(scale_segs_each_features, (x_features_series_train,)))
    tasks.append(pool.apply_async(scale_segs_each_features, (x_features_series_test,)))
    x_RP_train, x_RP_test, \
    x_features_series_train, x_features_series_test = [t.get() for t in tasks]

    # x_RP_train, x_RP_test, \
    # x_features_series_train, x_features_series_test, \
    #     scale_RP_each_feature(x_RP_train), scale_RP_each_feature(x_RP_test), \
    #     scale_segs_each_features(x_features_series_train), scale_segs_each_features(x_features_series_test), \
    np.save('./data/geolife_features/x_RP_train.npy', x_RP_train)
    np.save('./data/geolife_features/x_RP_test.npy', x_RP_test)
    np.save('./data/geolife_features/x_features_series_train.npy', x_features_series_train)
    np.save('./data/geolife_features/x_features_series_test.npy', x_features_series_test)
    np.save('./data/geolife_features/x_centroids_train.npy', x_centroids_train)
    np.save('./data/geolife_features/x_centroids_test.npy', x_centroids_test)
    np.save('./data/geolife_features/y_train.npy', y_train)
    np.save('./data/geolife_features/y_test.npy', y_test)


x_RP_train = np.load('./data/geolife_features/x_RP_train.npy', )
x_RP_test = np.load('./data/geolife_features/x_RP_test.npy', )
x_features_series_train = np.load('./data/geolife_features/x_features_series_train.npy', )
x_features_series_test = np.load('./data/geolife_features/x_features_series_test.npy', )
x_centroids_train = np.load('./data/geolife_features/x_centroids_train.npy', )
x_centroids_test = np.load('./data/geolife_features/x_centroids_test.npy', )
y_train = np.load('./data/geolife_features/y_train.npy', )
y_test = np.load('./data/geolife_features/y_test.npy', )


def regenerate_PEDCC():
    c = generate_center(u, v, G)
    RP_mats_mf_filtered = np.load('./data/geolife_features/RP_mats.npy')

    n_samples = RP_mats_mf_filtered.shape[0]

    # !!those data are generated, no real trajectory data involved!!
    scale = True
    centroids = repeat(c, n_samples, scale)
    np.save('./data/geolife_features/centroids.npy', centroids)

    x_centroids_train, x_centroids_test = train_test_split(centroids, test_size=0.20, random_state=7, shuffle=True)
    np.save('./data/geolife_features/x_centroids_train.npy', x_centroids_train)
    np.save('./data/geolife_features/x_centroids_test.npy', x_centroids_test)


if __name__ == '__main__':
    make_dataset()
    # regenerate_PEDCC()
    print()
