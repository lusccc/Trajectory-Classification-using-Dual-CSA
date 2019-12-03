
import numpy as np
from pyts.image.recurrence import _trajectories
from scipy.spatial.distance import pdist, squareform

#  Settings for the embedding
DIM = 3  # Embedding dimension
TAU = 4  # Embedding delay

# Distance metric in phase space ->
# Possible choices ("manhattan","euclidean","supremum")
METRIC = "euclidean"

EPS = 0.05  # Fixed recurrence threshold

def gen_multiple_RP_mats(series):
    phase_trjss = _trajectories(series, dimension=DIM, time_delay=TAU)
    print(
    'phase_trjss.shape:{}'.format(phase_trjss.shape))

    mat_size = phase_trjss.shape[1]

    _mats = []
    for phase_trjs in phase_trjss:
        distances = pdist(phase_trjs, )
        distances_mat = squareform(distances)
        _mats.append(distances_mat)
    return np.array(_mats)



if __name__ == '__main__':
    features_segments = np.load('./geolife/features_segments.npy')
    features_segments_labels = np.load('./geolife/features_segments_labels.npy')
    #(32197, 3, 200)(32197, )
    print(features_segments.shape, features_segments_labels.shape)

    features_RP_mats = []

    for i in range(features_segments.shape[1]):
        single_feature_segs = features_segments[:, i, :]#(n, 48)
        feature_RP_mats = gen_multiple_RP_mats(single_feature_segs)
        feature_RP_mats = np.expand_dims(feature_RP_mats, axis=3)
        features_RP_mats.append(feature_RP_mats)
    features_RP_mats = np.concatenate(features_RP_mats, axis=3)
    print(features_RP_mats.shape)
    np.save('./geolife/features_RP_mats.npy', features_RP_mats)


