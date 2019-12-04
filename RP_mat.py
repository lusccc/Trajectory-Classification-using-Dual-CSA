
import numpy as np
from pyts.image.recurrence import _trajectories
from scipy.spatial.distance import pdist, squareform

#  Settings for the embedding
from sklearn.preprocessing import StandardScaler

DIM = 3  # Embedding dimension
TAU = 4  # Embedding delay

# Distance metric in phase space ->
# Possible choices ("manhattan","euclidean","supremum")
METRIC = "euclidean"

EPS = 0.05  # Fixed recurrence threshold

def gen_multiple_RP_mats(series, scale=False):
    phase_trjss = _trajectories(series, dimension=DIM, time_delay=TAU)
    print(
    'phase_trjss.shape:{}'.format(phase_trjss.shape))

    mat_size = phase_trjss.shape[1]

    _mats = []
    for phase_trjs in phase_trjss:
        distances = pdist(phase_trjs, )
        distances_mat = squareform(distances)
        _mats.append(distances_mat)

    if scale:
        _mats = scale_data(_mats)
    return np.array(_mats)

def scale_data(data, scaler=StandardScaler()):

    data = np.array(data)
    n = np.where(np.isnan(data))
    f = np.where(np.isfinite(data))
    shape_ = data.shape
    data = data.reshape((-1, 1))
    # scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = np.reshape(data, shape_)
    return data



if __name__ == '__main__':
    features_segments = np.load('./geolife/features_segments.npy')
    features_segments_labels = np.load('./geolife/features_segments_labels.npy')
    # (106026, 1, 48, 3) (106026,)
    print(features_segments.shape, features_segments_labels.shape)

    features_RP_mats = []
    n_features = features_segments.shape[3]
    features_segments = np.squeeze(features_segments)
    for i in range(n_features):
        single_feature_segs = features_segments[:, :, i]  # (n, 48)
        feature_RP_mats = gen_multiple_RP_mats(single_feature_segs, scale=True)
        feature_RP_mats = np.expand_dims(feature_RP_mats, axis=3)
        features_RP_mats.append(feature_RP_mats)
    features_RP_mats = np.concatenate(features_RP_mats, axis=3)
    print(features_RP_mats.shape)
    np.save('./geolife/features_RP_mats.npy', features_RP_mats)


