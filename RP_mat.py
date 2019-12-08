from math import cos, pi

import numpy as np
from numpy.linalg import norm
from pyts.image.recurrence import _trajectories
from scipy.spatial.distance import pdist, squareform

#  Settings for the embedding
from sklearn.preprocessing import StandardScaler

from utils import scale_1d_data

DIM = 5  # Embedding dimension
TAU = 5  # Embedding delay

# Distance metric in phase space ->
# Possible choices ("manhattan","euclidean","supremum")
METRIC = "euclidean"

EPS = 0.05  # Fixed recurrence threshold


def gen_multiple_RP_mats(series, scale=False):
    phase_trjss = _trajectories(series, dimension=DIM, time_delay=TAU)
    print('phase_trjss.shape:{}'.format(phase_trjss.shape))

    mat_size = phase_trjss.shape[1]

    _mats = []
    for phase_trjs in phase_trjss:
        # https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
        distances = pdist(phase_trjs, )
        sign_distances = []
        k = 0
        for i in range(DIM):
            for j in range(DIM):
                if i == j:
                    continue
                k += 1
                # equation (5) in "Robust Single Accelerometer-Based Activity Recognition
                # Using Modified Recurrence Plot"
                sign_value = sign(phase_trjs[i], phase_trjs[j])
                sign_distances.append(sign_value * distances[k])
        distances_mat = squareform(distances)
        _mats.append(distances_mat)

    if scale:
        _mats = scale_1d_data(_mats)
    return np.array(_mats)


def sign(m, n):
    # equation (5) in "Robust Single Accelerometer-Based Activity Recognition
    # Using Modified Recurrence Plot"
    m = np.array(m)
    n = np.array(n)
    dim = m.shape[0] # vector dim
    v = np.repeat(1, dim) #base vec
    cos_angle = np.dot(m - n, v) / (norm(m - n) * norm(v))
    if cos_angle < cos(3. / 4. * pi):
        return -1
    else:
        return 1


if __name__ == '__main__':

    scale_each_feature = True
    scale_all = False

    data_type = {'train', 'test'}

    for type in data_type:
        features_segments = np.load('./geolife/{}_features_segments.npy'.format(type))
        features_segments_labels = np.load('./geolife/{}_segments_labels.npy'.format(type))
        for i in range(2):
            f = features_segments[:, :, :, i]
            max = np.max(f)
            min = np.min(f)
            print('features value range:', min, max)

        # (106026, 1, 48, 3) (106026,)
        print(features_segments.shape, features_segments_labels.shape)

        features_RP_mats = []
        n_features = features_segments.shape[3]
        features_segments = np.squeeze(features_segments)
        for i in range(n_features):
            single_feature_segs = features_segments[:, :, i]  # (n, 48)
            feature_RP_mats = gen_multiple_RP_mats(single_feature_segs, scale=scale_each_feature)
            feature_RP_mats = np.expand_dims(feature_RP_mats, axis=3)
            features_RP_mats.append(feature_RP_mats)
        features_RP_mats = np.concatenate(features_RP_mats, axis=3)
        print(features_RP_mats.shape)

        if scale_all:
            features_RP_mats = scale_1d_data(features_RP_mats)
        max = np.max(features_RP_mats)
        min = np.min(features_RP_mats)
        print('RP mat value range:', min, max)
        np.save('./geolife/{}_features_RP_mats.npy'.format(type), features_RP_mats)