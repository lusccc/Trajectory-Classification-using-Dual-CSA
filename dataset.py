import argparse
import multiprocessing
from enum import Enum

import numpy as np
# from keras.utils import to_categorical
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from PEDCC import PEDDC
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
    print('make_dataset...')
    RP_mats = np.load('./data/geolife_features/RP_mats.npy')
    trjs_segs_features = np.load('./data/geolife_features/trjs_segs_features.npy')
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

    x_RP_train, x_RP_test, \
    x_features_series_train, x_features_series_test = \
        scale_RP_each_feature(x_RP_train), scale_RP_each_feature(x_RP_test), \
        scale_segs_each_features(x_features_series_train), scale_segs_each_features(x_features_series_test)
    np.save('./data/geolife_features/x_RP_train.npy', x_RP_train)
    np.save('./data/geolife_features/x_RP_test.npy', x_RP_test)
    np.save('./data/geolife_features/x_features_series_train.npy', x_features_series_train)
    np.save('./data/geolife_features/x_features_series_test.npy', x_features_series_test)
    np.save('./data/geolife_features/x_centroids_train.npy', x_centroids_train)
    np.save('./data/geolife_features/x_centroids_test.npy', x_centroids_test)
    np.save('./data/geolife_features/y_train.npy', y_train)
    np.save('./data/geolife_features/y_test.npy', y_test)


class Dataset(object):
    x_RP_train = np.load('./data/geolife_features/x_RP_train.npy', )
    x_RP_test = np.load('./data/geolife_features/x_RP_test.npy', )
    x_features_series_train = np.load('./data/geolife_features/x_features_series_train.npy', )
    x_features_series_test = np.load('./data/geolife_features/x_features_series_test.npy', )
    x_centroids_train = np.load('./data/geolife_features/x_centroids_train.npy', )
    x_centroids_test = np.load('./data/geolife_features/x_centroids_test.npy', )
    y_train = np.load('./data/geolife_features/y_train.npy', )
    y_test = np.load('./data/geolife_features/y_test.npy', )
    print('centroids shape:{}'.format(x_centroids_train.shape))

def make_dataset_for_clustering():
    print('make_dataset_for_clustering...')
    RP_mats = np.load('./data/geolife_features/RP_mats.npy')
    labels = np.load('./data/geolife_features/trjs_segs_features_labels.npy')
    x_RP_scaled = scale_RP_each_feature(RP_mats)
    np.save('./data/geolife_features/x_RP_scaled.npy', x_RP_scaled)
    np.save('./data/geolife_features/y_RP.npy', labels)


def regenerate_PEDCC(EMBEDDING_DIM):
    print('regenerate_PEDCC..EMBEDDING_DIM:{}'.format(EMBEDDING_DIM))
    pedcc = PEDDC(EMBEDDING_DIM)
    c = pedcc.generate_center()
    n_samples = np.load('./data/geolife_features/trjs_segs_features_labels.npy').shape[0]
    # !!those data are generated, no real trajectory data involved!!
    scale = True
    centroids = pedcc.repeat(c, n_samples, scale)
    np.save('./data/geolife_features/centroids.npy', centroids)
    print('centroids shape:{}'.format(centroids.shape))

    x_centroids_train, x_centroids_test = train_test_split(centroids, test_size=0.20, random_state=7, shuffle=True)
    np.save('./data/geolife_features/x_centroids_train.npy', x_centroids_train)
    np.save('./data/geolife_features/x_centroids_test.npy', x_centroids_test)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('--DIM', type=int)
    args = parser.parse_args()
    DIM = args.DIM
    if DIM:
        regenerate_PEDCC(DIM)
    else:
        make_dataset()
        # make_dataset_for_clustering()
