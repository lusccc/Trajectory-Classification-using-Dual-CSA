import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from params import N_CLASS
from trajectory_segmentation_and_features import other_features
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


RP_mats_clean_mf = np.load('./geolife_features/RP_mats_clean_mf.npy')
RP_mats_noise_mf = np.load('./geolife_features/RP_mats_noise_mf.npy')
trjs_segs_clean_features = np.load('./geolife_features/trjs_segs_clean_features.npy')[:, :, :,
                           other_features]  # (2454, 184, 10)
trjs_segs_noise_features = np.load('./geolife_features/trjs_segs_noise_features.npy')[:, :, :, other_features]
centroids = np.load('./geolife_features/centroids.npy')

labels = np.load('./geolife_features/trjs_segs_features_labels.npy')

x_RP_clean_mf_train, x_RP_clean_mf_test, \
x_RP_noise_mf_train, x_RP_noise_mf_test, \
x_trj_seg_clean_of_train, x_trj_seg_clean_of_test, \
x_trj_seg_noise_of_train, x_trj_seg_noise_of_test, \
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

x_RP_clean_mf_train, x_RP_clean_mf_test, \
x_RP_noise_mf_train, x_RP_noise_mf_test, \
x_trj_seg_clean_of_train, x_trj_seg_clean_of_test, \
x_trj_seg_noise_of_train, x_trj_seg_noise_of_test = \
    scale_RP_each_feature(x_RP_clean_mf_train), scale_RP_each_feature(x_RP_clean_mf_test), \
    scale_RP_each_feature(x_RP_noise_mf_train), scale_RP_each_feature(x_RP_noise_mf_test), \
    scale_segs_each_features(x_trj_seg_clean_of_train), scale_segs_each_features(x_trj_seg_clean_of_test), \
    scale_segs_each_features(x_trj_seg_noise_of_train), scale_segs_each_features(x_trj_seg_noise_of_test)

print('load data done')


# for i in range(trjs_segs_features.shape[3]):
#     f = trjs_segs_features[:, :, :, i]
#     max = np.max(f)
#     min = np.min(f)
#     print('features value range:', min, max)


def load_data():
    return x_RP_clean_mf_train, x_RP_clean_mf_test, \
           x_RP_noise_mf_train, x_RP_noise_mf_test, \
           x_trj_seg_clean_of_train, x_trj_seg_clean_of_test, \
           x_trj_seg_noise_of_train, x_trj_seg_noise_of_test, \
           x_centroids_train, x_centroids_test, \
           y_train, y_test
